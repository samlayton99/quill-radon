"""Bounded-feature-memory residual products for native directional operators.

The engine stores points, residual vectors, and O(Q) pointwise sensitivities.
It never requests a full feature matrix. Feature work and callback autograd
are restricted to point batches, including in the right preconditioner.
Dense user-supplied aggregation matrices retain their own storage cost.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .general_residual import (_Engine, _NonfiniteResidual, ResidualLinearization,
                              infer_field_parity_shifts, _ideal_correction_operator)


def _state_array_bytes(states):
    """Count unique NumPy arrays in optional prepared backend states."""
    seen = set()
    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(v) for v in value)
        if hasattr(value, '__dict__'):
            return visit(vars(value))
        return 0
    return visit(states)


class _StreamedEngine(_Engine):
    def __init__(self, problem, validate_locality=True, batch_size=128):
        began = time.perf_counter()
        for name in ('forward_jets', 'adjoint_jets', 'selected_columns', 'forward'):
            if not callable(getattr(problem.features, name, None)):
                raise TypeError(f"streamed execution requires features.{name}; no dense fallback is used")
        self.problem = problem
        self.p, self.f = int(problem.features.size), problem.fields
        self.h = len(problem.parameter_initial)
        self.n = self.p*self.f+self.h
        self.validate_locality = validate_locality
        self.execution, self.batch_size = 'streamed', batch_size
        self.prepared_forward = all(callable(getattr(problem.features, name, None))
            for name in ('prepare_forward', 'forward_prepared_jets'))
        self.prepared_adjoint = all(callable(getattr(problem.features, name, None))
            for name in ('prepare_adjoint', 'accumulate_adjoint_jets', 'finish_adjoint'))
        self.multifield = self.f>1 and all(callable(getattr(problem.features,name,None)) for name in
            ('prepare_forward_fields','forward_prepared_fields_jets','prepare_adjoint_fields',
             'accumulate_adjoint_fields_jets','finish_adjoint_fields'))
        self.evaluations = self.linearizations = 0
        self.cache, self.cache_bytes = {}, 0
        self.product_counts = dict(matvec_calls=0, rmatvec_calls=0,
            column_gram_calls=0, column_panel_bytes_max=0,
            selected_feature_panel_bytes_max=0, forward_batches=0,
            adjoint_batches=0, callback_rows_max=0,
            callback_jet_bytes_max=0, feature_rows_max=0,
            prepared_forward_calls=0, prepared_adjoint_calls=0,
            prepared_forward_bytes_max=0, prepared_adjoint_bytes_max=0)
        self.setup_seconds = time.perf_counter()-began

    def slices(self, rows, limit=None):
        size = self.batch_size if limit is None else min(self.batch_size, limit)
        for start in range(0, rows, size):
            yield slice(start, min(start+size, rows))

    def prepare_forward(self, coefficients):
        if not self.prepared_forward:
            return None
        states = (self.problem.features.prepare_forward_fields(coefficients) if self.multifield else
                  [self.problem.features.prepare_forward(coefficients[:, field]) for field in range(self.f)])
        self.product_counts['prepared_forward_calls'] += 1 if self.multifield else self.f
        self.product_counts['prepared_forward_bytes_max'] = max(
            self.product_counts['prepared_forward_bytes_max'], _state_array_bytes(states))
        return states

    def jets(self, coefficients, points, derivatives, prepared=None):
        result = {d: np.empty((len(points), self.f)) for d in derivatives}
        for rows in self.slices(len(points)):
            self.product_counts['feature_rows_max'] = max(
                self.product_counts['feature_rows_max'], rows.stop-rows.start)
            if self.multifield and prepared is not None:
                values=self.problem.features.forward_prepared_fields_jets(prepared,points[rows],derivatives)
                self.product_counts['forward_batches']+=1
                for d in derivatives:
                    value=np.asarray(values[d],float)
                    if value.shape!=(rows.stop-rows.start,self.f) or not np.all(np.isfinite(value)):
                        raise _NonfiniteResidual('Feature forward_prepared_fields_jets must return finite Q-by-fields arrays')
                    result[d][rows]=value
                continue
            for field in range(self.f):
                values = (self.problem.features.forward_jets(coefficients[:, field], points[rows], derivatives)
                          if prepared is None else
                          self.problem.features.forward_prepared_jets(prepared[field], points[rows], derivatives))
                self.product_counts['forward_batches'] += 1
                for d in derivatives:
                    value = np.asarray(values[d], float)
                    if value.shape != (rows.stop-rows.start,) or not np.all(np.isfinite(value)):
                        raise _NonfiniteResidual('Feature forward_jets must return finite Q-vectors')
                    result[d][rows, field] = value
        return result

    def callback(self, block, rows, coefficients, parameters, derivatives, prepared=None):
        points = block.points[rows]
        values = self.jets(coefficients, points, block.derivatives, prepared)
        jets = {d: torch.tensor(v, dtype=torch.float64, requires_grad=derivatives)
                for d, v in values.items()}
        parameter_tensor = torch.tensor(np.broadcast_to(parameters, (len(points), self.h)).copy(),
                                        dtype=torch.float64, requires_grad=derivatives)
        self.product_counts['callback_rows_max'] = max(
            self.product_counts['callback_rows_max'], len(points))
        self.product_counts['callback_jet_bytes_max'] = max(
            self.product_counts['callback_jet_bytes_max'],
            sum(v.nbytes for v in values.values())+len(points)*self.h*8)
        with torch.set_grad_enabled(derivatives):
            x = torch.as_tensor(points, dtype=torch.float64)
            raw = (block.function(x, jets, parameter_tensor) if block.batch_function is None else
                   block.batch_function(x, jets, parameter_tensor,
                                        np.arange(rows.start, rows.stop)))
            if not isinstance(raw, torch.Tensor):
                raise TypeError(f'{block.name}: callback must return a torch.Tensor')
            if raw.ndim == 1:
                raw = raw[:, None]
            if raw.ndim != 2 or raw.shape[0] != len(points) or raw.shape[1] < 1:
                raise ValueError(f'{block.name}: callback must return batch rows or rows-by-equations; '
                                 'use batch_function with row indices for captured per-point arrays')
            local = raw.detach().cpu().numpy().astype(float, copy=True)
            if not np.all(np.isfinite(local)):
                raise _NonfiniteResidual(f'{block.name}: callback returned nonfinite residuals')
            if not derivatives:
                return local, None, None
            inputs = list(jets.values())+[parameter_tensor]
            channels = []
            for channel in range(raw.shape[1]):
                gradients = (torch.autograd.grad(raw[:, channel].sum(), inputs,
                    allow_unused=True, retain_graph=True) if raw.requires_grad else [None]*len(inputs))
                channels.append([np.zeros(tuple(v.shape)) if g is None else g.detach().cpu().numpy().copy()
                                 for g, v in zip(gradients, inputs)])
            sensitivity = np.transpose(np.asarray([g[:-1] for g in channels]), (2, 0, 1, 3))
            parameter_sensitivity = np.transpose(np.asarray([g[-1] for g in channels]), (1, 0, 2))
            if not np.all(np.isfinite(sensitivity)) or not np.all(np.isfinite(parameter_sensitivity)):
                raise _NonfiniteResidual(f'{block.name}: nonfinite local derivatives')
            if self.validate_locality and raw.requires_grad:
                weights = np.random.default_rng(24017+rows.start).normal(size=tuple(raw.shape))
                pullback = torch.autograd.grad((raw*torch.as_tensor(weights, dtype=raw.dtype)).sum(),
                                               inputs, allow_unused=True, retain_graph=False)
                for j, (actual, tensor) in enumerate(zip(pullback, inputs)):
                    actual = np.zeros(tuple(tensor.shape)) if actual is None else actual.detach().cpu().numpy()
                    expected = (np.einsum('qef,qe->qf', sensitivity[:, :, j], weights)
                                if j < len(jets) else
                                np.einsum('qeh,qe->qh', parameter_sensitivity, weights))
                    if not np.allclose(actual, expected, rtol=2e-7, atol=2e-10):
                        raise ValueError(f'{block.name}: callback couples rows; use pointwise callbacks and aggregation instead')
        return local, sensitivity, parameter_sensitivity

    def evaluate(self, x, derivatives=False):
        self.evaluations += 1
        if derivatives:
            self.linearizations += 1
        coefficients, parameters = self.unpack(x)
        prepared = self.prepare_forward(coefficients)
        residuals, block_metrics, locals_ = [], [], []
        for block in self.problem.blocks:
            local_values = sensitivity = parameter_sensitivity = None
            # A one-row derivative cannot expose row coupling. Validate a tiny
            # two-row probe in that exceptional setting; feature batches stay
            # bounded because the probe is assembled from one-row forwards.
            if derivatives and self.validate_locality and self.batch_size == 1 and len(block.points) > 1:
                self.callback(block, slice(0, 2), coefficients, parameters, True, prepared)
            for rows in self.slices(len(block.points)):
                values, ds, dp = self.callback(block, rows, coefficients, parameters, derivatives, prepared)
                if local_values is None:
                    local_values = np.empty((len(block.points), values.shape[1]))
                    if derivatives:
                        sensitivity = np.empty((len(block.points),)+ds.shape[1:])
                        parameter_sensitivity = np.empty((len(block.points),)+dp.shape[1:])
                if values.shape[1] != local_values.shape[1]:
                    raise ValueError(f'{block.name}: callback equation count changes across batches')
                local_values[rows] = values
                if derivatives:
                    sensitivity[rows], parameter_sensitivity[rows] = ds, dp
            aggregated = local_values if block.aggregation is None else block.aggregation @ local_values
            active = (np.ones_like(aggregated, dtype=bool) if block.relation == 'eq' else
                      aggregated > 0 if block.relation == 'le' else aggregated < 0)
            constrained = aggregated*active
            scale = np.broadcast_to(block.scale, (local_values.shape[1],))
            factor = np.sqrt(block.weight/len(aggregated))/scale
            residuals.append((constrained*factor).ravel())
            scaled_rms = np.sqrt(np.mean((constrained/scale)**2, axis=0))
            block_metrics.append(dict(name=block.name, input_rows=len(block.points),
                output_rows=len(aggregated), equations=local_values.shape[1],
                rms=float(np.sqrt(np.mean(constrained**2))),
                pre_relation_rms=float(np.sqrt(np.mean(aggregated**2))),
                relation=block.relation, active_entries=int(np.count_nonzero(active)),
                scaled_rms=scaled_rms.tolist(), maximum_scaled_rms=float(np.max(scaled_rms)),
                weighted_norm=float(np.linalg.norm(constrained*factor))))
            if derivatives:
                locals_.append((block, factor, sensitivity, parameter_sensitivity, active))
        residual = np.concatenate(residuals)
        if not np.all(np.isfinite(residual)):
            raise _NonfiniteResidual('Nonfinite aggregated or scaled residual')
        if not derivatives:
            return residual, block_metrics
        counts = dict(matvec_calls=0, rmatvec_calls=0)

        def matvec(delta):
            counts['matvec_calls'] += 1
            self.product_counts['matvec_calls'] += 1
            dc, dp = self.unpack(np.asarray(delta).reshape(-1))
            prepared_delta = self.prepare_forward(dc)
            outputs = []
            for block, factor, sensitivity, parameter_sensitivity, active in locals_:
                value = np.empty((len(block.points), len(factor)))
                for rows in self.slices(len(block.points)):
                    jets = self.jets(dc, block.points[rows], block.derivatives, prepared_delta)
                    value[rows] = sum(np.einsum('qef,qf->qe', sensitivity[rows, :, j], jets[d])
                                      for j, d in enumerate(block.derivatives))
                    value[rows] += np.einsum('qeh,h->qe', parameter_sensitivity[rows], dp)
                if block.aggregation is not None:
                    value = block.aggregation @ value
                outputs.append((value*active*factor).ravel())
            return np.concatenate(outputs)

        def rmatvec(vector):
            counts['rmatvec_calls'] += 1
            self.product_counts['rmatvec_calls'] += 1
            gc, gp, offset = np.zeros((self.p, self.f)), np.zeros(self.h), 0
            accumulators = (self.problem.features.prepare_adjoint_fields(self.f) if self.multifield else
                            [self.problem.features.prepare_adjoint() for _ in range(self.f)]
                            if self.prepared_adjoint else None)
            if accumulators is not None:
                self.product_counts['prepared_adjoint_calls'] += 1 if self.multifield else self.f
                self.product_counts['prepared_adjoint_bytes_max'] = max(
                    self.product_counts['prepared_adjoint_bytes_max'], _state_array_bytes(accumulators))
            for block, factor, sensitivity, parameter_sensitivity, active in locals_:
                size = active.size
                v = np.asarray(vector[offset:offset+size]).reshape(active.shape)*factor*active
                offset += size
                if block.aggregation is not None:
                    v = block.aggregation.T @ v
                for rows in self.slices(len(block.points)):
                    if self.multifield:
                        cotangents={d:np.einsum('qef,qe->qf',sensitivity[rows,:,j],v[rows])
                                    for j,d in enumerate(block.derivatives)}
                        self.problem.features.accumulate_adjoint_fields_jets(cotangents,block.points[rows],accumulators)
                        self.product_counts['adjoint_batches']+=1
                        gp+=np.einsum('qeh,qe->h',parameter_sensitivity[rows],v[rows])
                        continue
                    for field in range(self.f):
                        cotangents = {d: np.einsum('qe,qe->q', sensitivity[rows, :, j, field], v[rows])
                                      for j, d in enumerate(block.derivatives)}
                        self.product_counts['adjoint_batches'] += 1
                        if accumulators is None:
                            pulled = np.asarray(self.problem.features.adjoint_jets(cotangents, block.points[rows]), float)
                            if pulled.shape != (self.p,) or not np.all(np.isfinite(pulled)):
                                raise _NonfiniteResidual('Feature adjoint_jets must return a finite coefficient vector')
                            gc[:, field] += pulled
                        else:
                            self.problem.features.accumulate_adjoint_jets(
                                cotangents, block.points[rows], accumulators[field])
                    gp += np.einsum('qeh,qe->h', parameter_sensitivity[rows], v[rows])
            if self.multifield:
                gc=np.asarray(self.problem.features.finish_adjoint_fields(accumulators),float)
                if gc.shape!=(self.p,self.f) or not np.all(np.isfinite(gc)):
                    raise _NonfiniteResidual('Feature finish_adjoint_fields must return finite coefficients-by-fields')
            elif accumulators is not None:
                for field in range(self.f):
                    pulled = np.asarray(self.problem.features.finish_adjoint(accumulators[field]), float)
                    if pulled.shape != (self.p,) or not np.all(np.isfinite(pulled)):
                        raise _NonfiniteResidual('Feature finish_adjoint must return a finite coefficient vector')
                    gc[:, field] = pulled
            return np.r_[gc.ravel(), gp]

        def column_gram(indices, scales, chunk_rows=None, *, ideal=False):
            indices, scales = np.asarray(indices, int), np.asarray(scales, float)
            if (indices.ndim != 1 or len(indices) < 1 or scales.shape != indices.shape or
                    np.any(indices < 0) or np.any(indices >= self.n) or
                    (chunk_rows is not None and (int(chunk_rows) != chunk_rows or chunk_rows < 1))):
                raise ValueError('Invalid column indices, scales, or row chunk size')
            self.product_counts['column_gram_calls'] += 1
            if ideal:
                self.product_counts['approximate_column_gram_calls'] = self.product_counts.get('approximate_column_gram_calls', 0)+1
            selector = (self.problem.features.selected_ideal_columns if ideal else
                        self.problem.features.selected_columns)
            cp = np.flatnonzero(indices < self.p*self.f)
            pp = np.flatnonzero(indices >= self.p*self.f)
            fi, fields = indices[cp]//self.f, indices[cp] % self.f
            pi = indices[pp]-self.p*self.f
            gram = np.zeros((len(indices), len(indices)))
            for block, factor, sensitivity, parameter_sensitivity, active in locals_:
                equations = len(factor)
                def local_panel(rows):
                    panel = np.zeros((rows.stop-rows.start, equations, len(indices)))
                    if len(cp):
                        basis = selector(block.points[rows], fi, block.derivatives)
                        self.product_counts['selected_feature_panel_bytes_max'] = max(
                            self.product_counts['selected_feature_panel_bytes_max'],
                            sum(np.asarray(basis[d]).nbytes for d in block.derivatives))
                        for j, d in enumerate(block.derivatives):
                            values = np.asarray(basis[d], float)
                            if values.shape != (rows.stop-rows.start, len(cp)) or not np.all(np.isfinite(values)):
                                raise _NonfiniteResidual('selected_columns must return finite batch-by-selected arrays')
                            panel[:, :, cp] += sensitivity[rows, :, j][:, :, fields]*values[:, None, :]
                    panel[:, :, pp] = parameter_sensitivity[rows, :, pi]
                    panel *= scales[None, None, :]
                    self.product_counts['column_panel_bytes_max'] = max(
                        self.product_counts['column_panel_bytes_max'], panel.nbytes)
                    return panel
                for output_rows in self.slices(len(active), chunk_rows):
                    if block.aggregation is None:
                        panel = local_panel(output_rows)
                    else:
                        panel = np.zeros((output_rows.stop-output_rows.start, equations, len(indices)))
                        for input_rows in self.slices(len(block.points), chunk_rows):
                            local = local_panel(input_rows)
                            panel += (block.aggregation[output_rows, input_rows] @
                                      local.reshape(input_rows.stop-input_rows.start, -1)).reshape(panel.shape)
                    panel *= active[output_rows, :, None]*factor[None, :, None]
                    columns = panel.reshape(-1, len(indices))
                    gram += columns.T @ columns
                    self.product_counts['column_panel_bytes_max'] = max(
                        self.product_counts['column_panel_bytes_max'], panel.nbytes)
            return gram

        operator = LinearOperator((len(residual), self.n), matvec=matvec, rmatvec=rmatvec, dtype=float)
        operator.column_gram = column_gram
        if callable(getattr(self.problem.features, 'selected_ideal_columns', None)):
            operator.approximate_column_gram = lambda indices, scales, chunk_rows=None: column_gram(
                indices, scales, chunk_rows, ideal=True)
        operator.field_parity_inference = infer_field_parity_shifts(
            self.f, self.problem.features.dimension,
            [(block, sensitivity) for block, _, sensitivity, _, _ in locals_])
        operator.column_row_chunk_size = self.batch_size
        operator.linearization_basis = 'actual'
        if callable(getattr(operator, 'approximate_column_gram', None)):
            operator.ideal_operator = _ideal_correction_operator(self.problem, locals_, operator,
                self.product_counts, self.batch_size)
        local_bytes = sum(s.nbytes+p.nbytes+a.nbytes for _, _, s, p, a in locals_)
        aggregations = {id(b.aggregation): b.aggregation for b in self.problem.blocks if b.aggregation is not None}
        aggregation_bytes = sum((a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
                                 if sparse.issparse(a) else a.nbytes) for a in aggregations.values())
        input_bytes = sum(p.nbytes for p in {id(b.points): b.points for b in self.problem.blocks}.values())
        return ResidualLinearization(operator, residual, block_metrics, dict(
            execution='streamed', batch_size=self.batch_size, basis_cache_bytes=0,
            local_derivative_bytes=local_bytes, residual_vector_bytes=residual.nbytes,
            coefficient_vector_bytes=self.n*8, input_point_bytes=input_bytes,
            prepared_forward_active=self.prepared_forward, prepared_adjoint_active=self.prepared_adjoint,
            shared_multifield_active=self.multifield,
            supplied_aggregation_bytes=aggregation_bytes,
            unassembled_dense_jacobian_bytes=8*len(residual)*self.n,
            storage_note='No Q-by-coefficient feature cache; O(Q) local sensitivities and residual/Krylov vectors retained; bounded point/column panels; optional prepared neuron readouts/adjoints use O(neurons * fields) storage, counted in product_counts; supplied dense aggregation is counted separately. Byte metrics are array accounting, not process peak RSS.',
            product_counts=self.product_counts, last_linearization_product_counts=counts,
            unknowns=self.n, residual_entries=len(residual)))
