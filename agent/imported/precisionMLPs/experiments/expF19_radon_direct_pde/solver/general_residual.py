"""Generic, matrix-free nonlinear residual solve over constructed features.

Callbacks are pointwise across rows. Differential identities, boundary/initial
conditions, data, gauges, and priors are supplied only through ResidualBlocks.
Optional dense/sparse aggregation implements weak or integral residuals after
the pointwise callback. This is an iterative global coefficient solve, not a
closed-form PDE solver or a method that avoids solving for its readout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
import torch
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, lsmr


class _NonfiniteResidual(FloatingPointError):
    pass


@dataclass
class ResidualBlock:
    name: str
    points: np.ndarray
    function: object
    derivatives: tuple
    weight: float = 1.
    scale: object = 1.
    aggregation: object = None
    relation: str = 'eq'
    batch_function: object = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or not len(self.points) or not np.all(np.isfinite(self.points)):
            raise ValueError('ResidualBlock.points must be a nonempty finite Q-by-d array')
        if not callable(self.function):
            raise ValueError('ResidualBlock.function must be callable')
        if self.batch_function is not None and not callable(self.batch_function):
            raise ValueError('ResidualBlock.batch_function must be callable or None')
        if self.relation not in ('eq', 'le', 'ge'):
            raise ValueError("relation must be 'eq', 'le', or 'ge'")
        self.derivatives = tuple(tuple(d) for d in self.derivatives)
        if not self.derivatives or len(set(self.derivatives)) != len(self.derivatives):
            raise ValueError('Declare a nonempty tuple of distinct derivative multiindices')
        for derivative in self.derivatives:
            if len(derivative) != self.points.shape[1] or any(int(k) != k or k < 0 for k in derivative):
                raise ValueError('Derivative multiindices must have dimension d and nonnegative integer entries')
        self.derivatives = tuple(tuple(int(k) for k in d) for d in self.derivatives)
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Block weight must be finite and positive')
        self.scale = np.asarray(self.scale, dtype=float)
        if self.scale.ndim > 1 or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError('Block scale must be a positive finite scalar or per-equation vector')
        if self.aggregation is not None:
            if sparse.issparse(self.aggregation):
                self.aggregation = self.aggregation.astype(float).tocsr()
                finite = np.all(np.isfinite(self.aggregation.data))
            else:
                self.aggregation = np.asarray(self.aggregation, dtype=float)
                finite = np.all(np.isfinite(self.aggregation))
            if self.aggregation.ndim != 2 or self.aggregation.shape[1] != len(self.points) or self.aggregation.shape[0] < 1 or not finite:
                raise ValueError('aggregation must be a finite M-by-Q dense/sparse matrix with M>=1')


@dataclass
class ResidualProblem:
    features: object
    blocks: list
    fields: int = 1
    parameter_initial: object = ()
    parameter_bounds: object = None
    execution: str = 'cached'
    batch_size: int = 128

    def __post_init__(self):
        _validate_execution(self.execution, self.batch_size)
        if int(self.fields) != self.fields or self.fields < 1:
            raise ValueError('fields must be a positive integer')
        self.fields = int(self.fields)
        self.blocks = tuple(self.blocks)
        if not self.blocks or any(not isinstance(b, ResidualBlock) for b in self.blocks):
            raise ValueError('blocks must contain at least one ResidualBlock')
        if len({b.name for b in self.blocks}) != len(self.blocks):
            raise ValueError('ResidualBlock names must be distinct')
        if any(b.points.shape[1] != self.features.dimension for b in self.blocks):
            raise ValueError('Block point dimensions must match features.dimension')
        self.parameter_initial = np.asarray(self.parameter_initial, dtype=float).reshape(-1)
        if not np.all(np.isfinite(self.parameter_initial)):
            raise ValueError('Initial parameters must be finite')
        if self.parameter_bounds is None:
            self.parameter_lower = np.full(len(self.parameter_initial), -np.inf)
            self.parameter_upper = np.full(len(self.parameter_initial), np.inf)
        else:
            if len(self.parameter_bounds) != 2:
                raise ValueError('parameter_bounds must be (lower, upper), broadcastable to the parameter vector')
            self.parameter_lower = np.broadcast_to(np.asarray(self.parameter_bounds[0], float), self.parameter_initial.shape).copy()
            self.parameter_upper = np.broadcast_to(np.asarray(self.parameter_bounds[1], float), self.parameter_initial.shape).copy()
            if np.any(np.isnan(self.parameter_lower)) or np.any(np.isnan(self.parameter_upper)) or np.any(self.parameter_lower >= self.parameter_upper):
                raise ValueError('Each parameter lower bound must be strictly less than its upper bound')
        if np.any(self.parameter_initial < self.parameter_lower) or np.any(self.parameter_initial > self.parameter_upper):
            raise ValueError('parameter_initial must satisfy parameter_bounds')


@dataclass
class ResidualSolution:
    problem: ResidualProblem
    coefficients: np.ndarray
    parameters: np.ndarray
    metrics: dict
    history: list = field(default_factory=list)

    @property
    def status(self):
        return self.metrics['status']

    def evaluate(self, points, derivative=None):
        if self.metrics.get('execution', self.problem.execution) == 'streamed':
            points = np.asarray(points, float)
            batch_size = self.metrics.get('batch_size', self.problem.batch_size)
            output = np.empty((len(points), self.problem.fields))
            features = self.problem.features
            use_prepared = (callable(getattr(features, 'prepare_forward', None)) and
                            callable(getattr(features, 'forward_prepared_jets', None)))
            use_shared=(self.problem.fields>1 and callable(getattr(features,'prepare_forward_fields',None))
                        and callable(getattr(features,'forward_prepared_fields_jets',None)))
            prepared = (features.prepare_forward_fields(self.coefficients) if use_shared else
                        [features.prepare_forward(self.coefficients[:, field])
                         for field in range(self.problem.fields)] if use_prepared else None)
            order = (0,)*features.dimension if derivative is None else tuple(derivative)
            for start in range(0, len(points), batch_size):
                stop = min(start+batch_size, len(points))
                if use_shared:
                    output[start:stop]=features.forward_prepared_fields_jets(prepared,points[start:stop],(order,))[order]
                    continue
                for field in range(self.problem.fields):
                    output[start:stop, field] = (features.forward(
                        self.coefficients[:, field], points[start:stop], derivative=derivative)
                        if prepared is None else features.forward_prepared_jets(
                            prepared[field], points[start:stop], (order,))[order])
            return output
        return np.asarray(self.problem.features.evaluate(points, derivative=derivative), float) @ self.coefficients


@dataclass
class ResidualLinearization:
    operator: LinearOperator
    residual: np.ndarray
    block_metrics: list
    metrics: dict


def _ideal_correction_operator(problem, locals_, actual_operator, product_counts,
                               row_batch_size, column_batch_size=64):
    """Approximate Jacobian at an ACTUAL neural state, with bounded panels.

    locals_ contains (block, weight_factor, actual_jet_sensitivity,
    actual_parameter_sensitivity, active_constraint_mask). Only the linear
    correction basis is idealized. Neither the primal field nor the residual
    is evaluated here. Parameter columns retain their exact local derivatives.
    """
    features, fields = problem.features, problem.fields
    size = int(features.size)
    parameters = len(problem.parameter_initial)
    unknowns = size*fields+parameters
    tensor_planner=getattr(features,'ideal_tensor_plan',None)
    orders=tuple(dict.fromkeys(d for block,*_ in locals_ for d in block.derivatives))
    tensor_plan=(tensor_planner(fields,row_batch_size,orders) if callable(tensor_planner)
                 else dict(enabled=False,reason='feature_family_has_no_tensor_protocol'))
    use_tensor=tensor_plan['enabled']
    if use_tensor:
        row_batch_size=tensor_plan['rows']
        product_counts['ideal_tensor_workspace_plan']=tensor_plan

    def rows(count):
        for start in range(0, count, row_batch_size):
            yield slice(start, min(start+row_batch_size, count))

    def panels(points, derivatives):
        prepare = getattr(features, 'prepare_ideal_columns', None)
        prepared = prepare(points, derivatives) if callable(prepare) else None
        if prepared is not None:
            product_counts['ideal_correction_table_bytes_max'] = max(
                product_counts.get('ideal_correction_table_bytes_max', 0), prepared.get('retained_bytes', 0))
        for start in range(0, size, column_batch_size):
            stop = min(start+column_batch_size, size)
            indices = np.arange(start, stop)
            panel = (features.selected_ideal_columns(points, indices, derivatives, prepared=prepared)
                     if prepared is not None else features.selected_ideal_columns(points, indices, derivatives))
            for derivative in derivatives:
                value = np.asarray(panel[derivative], float)
                if value.shape != (len(points), stop-start) or not np.all(np.isfinite(value)):
                    raise _NonfiniteResidual('Ideal correction jets must be finite bounded point-by-column panels')
            product_counts['ideal_correction_rows_max'] = max(product_counts.get('ideal_correction_rows_max', 0), len(points))
            product_counts['ideal_correction_columns_max'] = max(product_counts.get('ideal_correction_columns_max', 0), stop-start)
            product_counts['ideal_correction_panel_bytes_max'] = max(product_counts.get('ideal_correction_panel_bytes_max', 0),
                sum(np.asarray(panel[d]).nbytes for d in derivatives))
            yield slice(start, stop), panel

    def matvec(delta):
        product_counts['ideal_matvec_calls'] = product_counts.get('ideal_matvec_calls', 0)+1
        delta = np.asarray(delta, float).reshape(-1)
        coefficients, dp = delta[:size*fields].reshape(size, fields), delta[size*fields:]
        tensor=features.prepare_ideal_tensor_forward(coefficients) if use_tensor else None
        outputs = []
        for block, factor, sensitivity, parameter_sensitivity, active in locals_:
            value = np.empty((len(block.points), len(factor)))
            for batch in rows(len(block.points)):
                if use_tensor:
                    jets=features.forward_ideal_tensor_jets(tensor,block.points[batch],block.derivatives)
                    if any(np.asarray(jets[d]).shape!=(batch.stop-batch.start,fields) or
                           not np.all(np.isfinite(jets[d])) for d in block.derivatives):
                        raise _NonfiniteResidual('Ideal tensor jets must be finite batch-by-field arrays')
                    product_counts['ideal_tensor_forward_batches']=product_counts.get('ideal_tensor_forward_batches',0)+1
                else:
                    jets = {d: np.zeros((batch.stop-batch.start, fields)) for d in block.derivatives}
                    for columns, panel in panels(block.points[batch], block.derivatives):
                        for derivative in block.derivatives:
                            jets[derivative] += panel[derivative] @ coefficients[columns]
                value[batch] = sum(np.einsum('qef,qf->qe', sensitivity[batch, :, j], jets[d])
                                   for j, d in enumerate(block.derivatives))
                value[batch] += np.einsum('qeh,h->qe', parameter_sensitivity[batch], dp)
            if block.aggregation is not None:
                value = block.aggregation @ value
            outputs.append((value*active*factor).ravel())
        result=np.concatenate(outputs)
        if not np.all(np.isfinite(result)):
            raise _NonfiniteResidual('Nonfinite ideal correction product')
        return result

    def rmatvec(vector):
        product_counts['ideal_rmatvec_calls'] = product_counts.get('ideal_rmatvec_calls', 0)+1
        gc, gp, offset = np.zeros((size, fields)), np.zeros(parameters), 0
        tensor=features.prepare_ideal_tensor_adjoint(fields) if use_tensor else None
        for block, factor, sensitivity, parameter_sensitivity, active in locals_:
            count = active.size
            cotangent = np.asarray(vector[offset:offset+count]).reshape(active.shape)*active*factor
            offset += count
            if block.aggregation is not None:
                cotangent = block.aggregation.T @ cotangent
            for batch in rows(len(block.points)):
                jets = {d: np.einsum('qef,qe->qf', sensitivity[batch, :, j], cotangent[batch])
                        for j, d in enumerate(block.derivatives)}
                if use_tensor:
                    features.accumulate_ideal_tensor_jets(jets,block.points[batch],tensor)
                    product_counts['ideal_tensor_adjoint_batches']=product_counts.get('ideal_tensor_adjoint_batches',0)+1
                else:
                    for columns, panel in panels(block.points[batch], block.derivatives):
                        gc[columns] += sum(panel[d].T @ jets[d] for d in block.derivatives)
                gp += np.einsum('qeh,qe->h', parameter_sensitivity[batch], cotangent[batch])
        if use_tensor:gc+=features.finish_ideal_tensor_adjoint(tensor)
        result=np.r_[gc.ravel(), gp]
        if not np.all(np.isfinite(result)):
            raise _NonfiniteResidual('Nonfinite ideal correction transpose product')
        return result

    operator = LinearOperator(actual_operator.shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
    operator.column_gram = actual_operator.approximate_column_gram
    operator.approximate_column_gram = actual_operator.approximate_column_gram
    operator.field_parity_inference = actual_operator.field_parity_inference
    operator.column_row_chunk_size = row_batch_size
    operator.linearization_basis = 'ideal'
    operator.ideal_column_batch_size = column_batch_size
    return operator


class _Engine:
    def __init__(self, problem, validate_locality=True):
        began = time.perf_counter()
        self.problem = problem
        self.p, self.f = int(problem.features.size), problem.fields
        self.h = len(problem.parameter_initial)
        self.n = self.p*self.f+self.h
        self.validate_locality = validate_locality
        self.evaluations = self.linearizations = 0
        self.product_counts = dict(matvec_calls=0, rmatvec_calls=0,
                                   column_gram_calls=0, column_panel_bytes_max=0)
        self.cache, self.blocks = {}, []
        for block in problem.blocks:
            matrices = []
            missing=[derivative for derivative in block.derivatives
                     if (id(block.points),derivative) not in self.cache]
            shared=(problem.features.evaluate_many(block.points,missing)
                    if missing and hasattr(problem.features,'evaluate_many') else None)
            for derivative in block.derivatives:
                key = (id(block.points), derivative)
                if key not in self.cache:
                    basis = np.asarray(shared[derivative] if shared is not None else
                                       problem.features.evaluate(block.points, derivative=derivative), dtype=float)
                    if basis.shape != (len(block.points), self.p) or not np.all(np.isfinite(basis)):
                        raise ValueError('Feature evaluation must return finite Q-by-P arrays')
                    self.cache[key] = basis
                matrices.append(self.cache[key])
            self.blocks.append((block, matrices, torch.as_tensor(block.points, dtype=torch.float64)))
        self.cache_bytes = sum(b.nbytes for b in self.cache.values())
        self.setup_seconds = time.perf_counter()-began

    def unpack(self, x):
        return x[:self.p*self.f].reshape(self.p, self.f), x[self.p*self.f:]

    def project(self, x):
        result = x.copy()
        result[self.p*self.f:] = np.clip(result[self.p*self.f:], self.problem.parameter_lower, self.problem.parameter_upper)
        return result

    def evaluate(self, x, derivatives=False):
        self.evaluations += 1
        if derivatives:
            self.linearizations += 1
        coefficients, parameters = self.unpack(x)
        residuals, block_metrics, locals_ = [], [], []
        for block, bases, points in self.blocks:
            jets = {d: torch.tensor(basis @ coefficients, dtype=torch.float64, requires_grad=derivatives)
                    for d, basis in zip(block.derivatives, bases)}
            parameter_tensor = torch.tensor(np.broadcast_to(parameters, (len(points), self.h)).copy(),
                                            dtype=torch.float64, requires_grad=derivatives)
            with torch.set_grad_enabled(derivatives):
                raw = block.function(points, jets, parameter_tensor)
                if not isinstance(raw, torch.Tensor):
                    raise TypeError(f'{block.name}: callback must return a torch.Tensor')
                if raw.ndim == 1:
                    raw = raw[:, None]
                if raw.ndim != 2 or raw.shape[0] != len(points) or raw.shape[1] < 1:
                    raise ValueError(f'{block.name}: callback must return Q or Q-by-equations residuals')
                local_values = raw.detach().cpu().numpy().astype(float, copy=False)
                if not np.all(np.isfinite(local_values)):
                    raise _NonfiniteResidual(f'{block.name}: callback returned nonfinite residuals')
                aggregated = local_values if block.aggregation is None else block.aggregation @ local_values
                active = (np.ones_like(aggregated, dtype=bool) if block.relation == 'eq' else
                          aggregated > 0 if block.relation == 'le' else aggregated < 0)
                constrained = aggregated*active
                scale = np.broadcast_to(block.scale, (raw.shape[1],))
                factor = np.sqrt(block.weight/len(aggregated))/scale
                residuals.append((constrained*factor).ravel())
                scaled_rms = np.sqrt(np.mean((constrained/scale)**2, axis=0))
                block_metrics.append(dict(name=block.name, input_rows=len(points), output_rows=len(aggregated),
                                          equations=raw.shape[1], rms=float(np.sqrt(np.mean(constrained**2))),
                                          pre_relation_rms=float(np.sqrt(np.mean(aggregated**2))),
                                          relation=block.relation, active_entries=int(np.count_nonzero(active)),
                                          scaled_rms=scaled_rms.tolist(),
                                          maximum_scaled_rms=float(np.max(scaled_rms)),
                                          weighted_norm=float(np.linalg.norm(constrained*factor))))
                if not derivatives:
                    continue
                inputs = list(jets.values())+[parameter_tensor]
                channels = []
                for channel in range(raw.shape[1]):
                    if raw.requires_grad:
                        gradients = torch.autograd.grad(raw[:, channel].sum(), inputs,
                                                        allow_unused=True, retain_graph=True)
                    else:
                        gradients = [None]*len(inputs)
                    channels.append([np.zeros(tuple(v.shape)) if g is None else g.detach().cpu().numpy().copy()
                                     for g,v in zip(gradients, inputs)])
                sensitivities = np.transpose(np.asarray([g[:-1] for g in channels]), (2, 0, 1, 3))
                parameter_sensitivities = np.transpose(np.asarray([g[-1] for g in channels]), (1, 0, 2))
                if not np.all(np.isfinite(sensitivities)) or not np.all(np.isfinite(parameter_sensitivities)):
                    raise _NonfiniteResidual(f'{block.name}: nonfinite local derivatives')
                if self.validate_locality and raw.requires_grad:
                    weights = np.random.default_rng(24017).normal(size=tuple(raw.shape))
                    pullback = torch.autograd.grad((raw*torch.as_tensor(weights, dtype=raw.dtype)).sum(), inputs,
                                                   allow_unused=True, retain_graph=False)
                    for j, (actual, tensor) in enumerate(zip(pullback, inputs)):
                        actual = np.zeros(tuple(tensor.shape)) if actual is None else actual.detach().cpu().numpy()
                        expected = (np.einsum('qef,qe->qf', sensitivities[:, :, j], weights) if j < len(bases)
                                    else np.einsum('qeh,qe->qh', parameter_sensitivities, weights))
                        if not np.allclose(actual, expected, rtol=2e-7, atol=2e-10):
                            raise ValueError(f'{block.name}: callback couples rows; use pointwise callbacks and aggregation instead')
                locals_.append((block, bases, factor, sensitivities, parameter_sensitivities, active))
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
            outputs = []
            for block, bases, factor, sensitivity, parameter_sensitivity, active in locals_:
                value = sum(np.einsum('qef,qf->qe', sensitivity[:, :, j], basis @ dc)
                            for j, basis in enumerate(bases))
                value += np.einsum('qeh,h->qe', parameter_sensitivity, dp)
                if block.aggregation is not None:
                    value = block.aggregation @ value
                outputs.append((value*active*factor).ravel())
            return np.concatenate(outputs)
        def rmatvec(vector):
            counts['rmatvec_calls'] += 1
            self.product_counts['rmatvec_calls'] += 1
            gc, gp, offset = np.zeros((self.p, self.f)), np.zeros(self.h), 0
            for block, bases, factor, sensitivity, parameter_sensitivity, active in locals_:
                rows = len(active)
                size = rows*len(factor)
                v = np.asarray(vector[offset:offset+size]).reshape(rows, len(factor))*factor*active
                offset += size
                if block.aggregation is not None:
                    v = block.aggregation.T @ v
                for j, basis in enumerate(bases):
                    gc += basis.T @ np.einsum('qef,qe->qf', sensitivity[:, :, j], v)
                gp += np.einsum('qeh,qe->h', parameter_sensitivity, v)
            return np.r_[gc.ravel(), gp]
        def column_gram(indices, scales, chunk_rows=1024, *, ideal=False):
            """Exact small Gram block, accumulated from bounded local panels.

            This never builds columns for unselected coefficients. Aggregation
            is applied before active masks and weights, as in matvec/rmatvec.
            The full residual Jacobian and its global Gram remain unassembled.
            """
            indices = np.asarray(indices, dtype=int)
            scales = np.asarray(scales, dtype=float)
            if (indices.ndim != 1 or len(indices) < 1 or scales.shape != indices.shape or
                    np.any(indices < 0) or np.any(indices >= self.n) or
                    int(chunk_rows) != chunk_rows or chunk_rows < 1):
                raise ValueError('Invalid column indices, scales, or row chunk size')
            self.product_counts['column_gram_calls'] += 1
            if ideal:
                self.product_counts['approximate_column_gram_calls'] = self.product_counts.get('approximate_column_gram_calls', 0)+1
            coefficient_positions = np.flatnonzero(indices < self.p*self.f)
            parameter_positions = np.flatnonzero(indices >= self.p*self.f)
            feature_indices = indices[coefficient_positions]//self.f
            field_indices = indices[coefficient_positions] % self.f
            parameter_indices = indices[parameter_positions]-self.p*self.f
            gram = np.zeros((len(indices), len(indices)))
            for block, bases, factor, sensitivity, parameter_sensitivity, active in locals_:
                equations = len(factor)
                def local_panel(start, stop):
                    panel = np.zeros((stop-start, equations, len(indices)))
                    ideal_columns = (self.problem.features.selected_ideal_columns(
                        block.points[start:stop], feature_indices, block.derivatives) if ideal else None)
                    for j, basis in enumerate(bases):
                        values = (ideal_columns[block.derivatives[j]] if ideal else
                                  basis[start:stop, feature_indices])
                        panel[:, :, coefficient_positions] += (
                            sensitivity[start:stop, :, j][:, :, field_indices]*
                            values[:, None, :])
                    panel[:, :, parameter_positions] = parameter_sensitivity[start:stop, :, parameter_indices]
                    panel *= scales[None, None, :]
                    self.product_counts['column_panel_bytes_max'] = max(
                        self.product_counts['column_panel_bytes_max'], panel.nbytes)
                    return panel
                for start in range(0, len(active), chunk_rows):
                    stop = min(start+chunk_rows, len(active))
                    if block.aggregation is None:
                        panel = local_panel(start, stop)
                    else:
                        panel = np.zeros((stop-start, equations, len(indices)))
                        aggregate_rows = block.aggregation[start:stop]
                        for input_start in range(0, len(block.points), chunk_rows):
                            input_stop = min(input_start+chunk_rows, len(block.points))
                            local = local_panel(input_start, input_stop)
                            panel += (aggregate_rows[:, input_start:input_stop] @
                                      local.reshape(input_stop-input_start, -1)).reshape(panel.shape)
                    panel *= active[start:stop, :, None]*factor[None, :, None]
                    columns = panel.reshape(-1, len(indices))
                    gram += columns.T @ columns
                    self.product_counts['column_panel_bytes_max'] = max(
                        self.product_counts['column_panel_bytes_max'], panel.nbytes)
            return gram
        operator = LinearOperator((len(residual), self.n), matvec=matvec, rmatvec=rmatvec, dtype=float)
        operator.column_gram = column_gram
        if callable(getattr(self.problem.features, 'selected_ideal_columns', None)):
            operator.approximate_column_gram = lambda indices, scales, chunk_rows=1024: column_gram(
                indices, scales, chunk_rows, ideal=True)
        operator.field_parity_inference = infer_field_parity_shifts(
            self.f, self.problem.features.dimension,
            [(block, sensitivity) for block, _, _, sensitivity, _, _ in locals_])
        operator.linearization_basis = 'actual'
        if callable(getattr(operator, 'approximate_column_gram', None)):
            operator.ideal_operator = _ideal_correction_operator(self.problem,
                [(block, factor, sensitivity, parameters, active)
                 for block, _, factor, sensitivity, parameters, active in locals_],
                operator, self.product_counts, getattr(self, 'batch_size', self.problem.batch_size))
        local_bytes = sum(s.nbytes+p.nbytes+a.nbytes for _, _, _, s, p, a in locals_)
        return ResidualLinearization(operator, residual, block_metrics,
                                     dict(basis_cache_bytes=self.cache_bytes, local_derivative_bytes=local_bytes,
                                          unassembled_dense_jacobian_bytes=8*len(residual)*self.n,
                                          storage_note='Basis jets are cached densely and may exceed the hypothetical global Jacobian size; global residual Jacobian and Gram matrix are not assembled',
                                          product_counts=self.product_counts,
                                          last_linearization_product_counts=counts,
                                          unknowns=self.n, residual_entries=len(residual)))


def _initial_vector(problem, coefficients=None, parameters=None):
    c = np.zeros((problem.features.size, problem.fields)) if coefficients is None else np.asarray(coefficients, float)
    p = problem.parameter_initial if parameters is None else np.asarray(parameters, float)
    if c.shape != (problem.features.size, problem.fields) or p.shape != problem.parameter_initial.shape:
        raise ValueError('Coefficient or parameter array has an incompatible shape')
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(p)):
        raise ValueError('Initial coefficients and parameters must be finite')
    return np.r_[c.ravel(), p]


def _validate_execution(execution, batch_size):
    if execution not in ('cached', 'streamed'):
        raise ValueError("execution must be 'cached' or 'streamed'")
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')


def _make_engine(problem, validate_locality, execution=None, batch_size=None):
    execution = problem.execution if execution is None else execution
    batch_size = problem.batch_size if batch_size is None else batch_size
    _validate_execution(execution, batch_size)
    if execution == 'streamed':
        from .streamed_residual import _StreamedEngine
        return _StreamedEngine(problem, validate_locality, int(batch_size))
    engine = _Engine(problem, validate_locality)
    engine.execution, engine.batch_size = execution, int(batch_size)
    return engine


def linearize_residual(problem, coefficients, parameters=None, validate_locality=True,
                       execution=None, batch_size=None):
    """Expose matrix-free products for adjoint/finite-difference verification."""
    engine = _make_engine(problem, validate_locality, execution, batch_size)
    return engine.evaluate(_initial_vector(problem, coefficients, parameters), derivatives=True)


def residual_vector(problem, coefficients, parameters=None, execution=None, batch_size=None):
    """Weighted residual vector; weights multiply mean squared residuals."""
    return _make_engine(problem, False, execution, batch_size).evaluate(
        _initial_vector(problem, coefficients, parameters))[0]


def infer_field_parity_shifts(fields, dimension, block_sensitivities,
                             relative_tolerance=1e-10):
    """Infer coefficient-parity coupling from sampled local linearizations.

    ``block_sensitivities`` yields (ResidualBlock, Q-by-E-by-J-by-F) pairs.
    A mode with parity alpha in field f is grouped using alpha XOR sigma_f.
    Two constant-coefficient terms D^a u_f and D^b u_g in one equation
    therefore require sigma_f XOR sigma_g = a XOR b (componentwise mod 2).

    This is a grouping heuristic, not a PDE symmetry certificate. Spatially
    varying sampled coefficients, integral aggregation, inequalities, and
    singleton samples supply no constraints. Conflicting constraints disable
    only the corresponding coordinate parity bit. In particular a time
    derivative plus diffusion conflicts in time but preserves spatial bits.
    No feature values, target labels, Jacobian columns or Gram are requested.
    """
    if (int(fields) != fields or fields < 1 or int(dimension) != dimension or dimension < 1
            or not np.isfinite(relative_tolerance) or relative_tolerance < 0):
        raise ValueError('Invalid field count, dimension, or parity tolerance')
    fields, dimension = int(fields), int(dimension)
    edges = []
    counts = dict(blocks_ignored=0, equations_considered=0,
                  constant_terms=0, varying_terms_ignored=0)
    for block, sensitivity in block_sensitivities:
        sensitivity = np.asarray(sensitivity, float)
        expected = (len(block.points), len(block.derivatives), fields)
        if (sensitivity.ndim != 4 or (sensitivity.shape[0], sensitivity.shape[2], sensitivity.shape[3]) != expected
                or not np.all(np.isfinite(sensitivity))):
            raise ValueError('Parity sensitivities must be finite Q-by-equations-by-derivatives-by-fields arrays')
        if block.aggregation is not None or block.relation != 'eq' or len(block.points) < 2:
            counts['blocks_ignored'] += 1
            continue
        average = sensitivity.mean(axis=0)
        variation = np.max(np.abs(sensitivity-average[None, ...]), axis=0)
        maximum = np.max(np.abs(sensitivity), axis=0)
        parity = np.asarray(block.derivatives, int) % 2
        for equation in range(sensitivity.shape[1]):
            counts['equations_considered'] += 1
            floor = 64*np.finfo(float).eps*max(float(np.max(maximum[equation])), np.finfo(float).tiny)
            active = maximum[equation] > floor
            constant = active & (variation[equation] <= floor+relative_tolerance*maximum[equation])
            counts['constant_terms'] += int(np.count_nonzero(constant))
            counts['varying_terms_ignored'] += int(np.count_nonzero(active & ~constant))
            terms = np.argwhere(constant)
            if len(terms) < 2:
                continue
            derivative, field = map(int, terms[0])
            for other_derivative, other_field in terms[1:]:
                edges.append((field, int(other_field), parity[derivative] ^ parity[int(other_derivative)]))
    shifts = np.zeros((fields, dimension), dtype=int)
    axis_mask = np.ones(dimension, dtype=bool)
    for axis in range(dimension):
        graph = [[] for _ in range(fields)]
        for first, second, difference in edges:
            graph[first].append((second, int(difference[axis])))
            graph[second].append((first, int(difference[axis])))
        seen = np.zeros(fields, dtype=bool)
        consistent = True
        for root in range(fields):
            if seen[root]:
                continue
            seen[root] = True
            stack = [root]
            while stack:
                first = stack.pop()
                for second, difference in graph[first]:
                    required = shifts[first, axis] ^ difference
                    if seen[second]:
                        consistent &= bool(shifts[second, axis] == required)
                    else:
                        shifts[second, axis] = required
                        seen[second] = True
                        stack.append(second)
        if not consistent:
            axis_mask[axis] = False
            shifts[:, axis] = 0
    return dict(shifts=shifts.tolist(), axis_mask=axis_mask.tolist(),
                inconsistent_axes=np.flatnonzero(~axis_mask).tolist(),
                constraint_edges=len(edges), **counts,
                interpretation='Sampled constant local-sensitivity XOR constraints; varying terms ignored; conflicting coordinate bits disabled; not a proven symmetry')


def _coordinate_parities(features):
    """Coordinate parities where the feature family explicitly identifies them."""
    modes = getattr(features, 'multiindices', None)
    if modes is not None:
        return np.asarray(modes, int) % 2
    supplied = getattr(features, 'coordinate_parities', None)
    if supplied is not None:
        return np.asarray(supplied, int) % 2
    # Disk modes are radial-even polynomials times cos(k phi) or sin(k phi).
    # Reflection x->-x supplies parity k+sine; y->-y supplies parity sine.
    if (features.dimension == 2 and hasattr(features, 'harmonics')
            and hasattr(features, 'is_sine')):
        sine = np.asarray(features.is_sine, int)
        return np.column_stack(((np.asarray(features.harmonics, int)+sine) % 2, sine))
    return None


class _RightPreconditioner:
    """Bounded parity blocks; no global Jacobian or global Gram is assembled."""
    def __init__(self, problem, operator, inverse_scale, block_size=64, ridge=1e-10,
                 field_parity='none', basis='actual'):
        began = time.perf_counter()
        self.inverse_scale = inverse_scale.copy()
        if basis not in ('actual', 'ideal'):
            raise ValueError("Preconditioner basis must be 'actual' or 'ideal'")
        n = len(inverse_scale)
        # Keep the factors genuinely local even on a small test problem.
        limit = min(block_size, max(1, n//2))
        dimension = problem.features.dimension
        shifts = np.zeros((problem.fields, dimension), dtype=int)
        axis_mask = np.ones(dimension, dtype=bool)
        parity_info = dict(mode='none')
        if isinstance(field_parity, str):
            if field_parity not in ('none', 'auto'):
                raise ValueError("field_parity must be 'none', 'auto', or an F-by-d binary array")
            if field_parity == 'auto':
                inferred = getattr(operator, 'field_parity_inference', None)
                parity_info = dict(mode='auto', metadata_available=inferred is not None)
                if inferred is not None:
                    shifts = np.asarray(inferred['shifts'], int)
                    axis_mask = np.asarray(inferred['axis_mask'], bool)
                    parity_info.update(inferred)
            # Preserve old grouping exactly unless users opt into inference.
            modes = (getattr(problem.features, 'multiindices', None) if field_parity == 'none'
                     else _coordinate_parities(problem.features))
        else:
            shifts = np.asarray(field_parity)
            parity_info = dict(mode='explicit')
            modes = _coordinate_parities(problem.features)
        if (shifts.shape != (problem.fields, dimension)
                or not np.issubdtype(shifts.dtype, np.integer) or np.any((shifts != 0) & (shifts != 1))
                or axis_mask.shape != (dimension,)):
            raise ValueError('Field parity shifts must be an F-by-d binary integer array')
        if modes is not None and np.asarray(modes).shape != (problem.features.size, dimension):
            raise ValueError('Coordinate parity labels must have feature-by-dimension shape')
        grouped = {}
        if modes is None:
            grouped[()] = list(range(problem.features.size*problem.fields))
        else:
            for index, mode in enumerate(modes):
                for field_index in range(problem.fields):
                    key = tuple((np.asarray(mode, int) % 2 ^ shifts[field_index])[axis_mask])
                    grouped.setdefault(key, []).append(index*problem.fields+field_index)
        parameter_start = problem.features.size*problem.fields
        if parameter_start < n:
            grouped[('parameters',)] = list(range(parameter_start, n))
        groups = [np.asarray(indices[start:start+limit], int)
                  for indices in grouped.values() for start in range(0, len(indices), limit)]
        self.factors = []
        column_method = getattr(operator, 'approximate_column_gram' if basis == 'ideal' else 'column_gram', None)
        direct_columns = callable(column_method)
        if basis == 'ideal' and not direct_columns:
            raise ValueError('Ideal preconditioning requires explicit selected_ideal_columns support')
        for indices in groups:
            if direct_columns:
                gram = column_method(indices, inverse_scale[indices])
            else:
                gram = np.empty((len(indices), len(indices)))
                # Fallback for external operators without local column access.
                for column, index in enumerate(indices):
                    delta = np.zeros(n)
                    delta[index] = inverse_scale[index]
                    gram[:, column] = inverse_scale[indices]*operator.rmatvec(operator.matvec(delta))[indices]
            gram = (gram+gram.T)/2
            if not np.all(np.isfinite(gram)):
                raise _NonfiniteResidual('Nonfinite block-preconditioner Gram')
            magnitude = float(np.max(np.diag(gram)))
            if magnitude <= 1e-25:
                factor = np.eye(len(indices))
            else:
                regularization = ridge*max(1., magnitude)
                try:
                    factor = np.linalg.cholesky(gram+regularization*np.eye(len(indices)))
                except np.linalg.LinAlgError:
                    eigenvalues, vectors = np.linalg.eigh(gram)
                    repaired = (vectors*np.maximum(eigenvalues, regularization)) @ vectors.T
                    factor = np.linalg.cholesky(repaired+regularization*np.eye(len(indices)))
            self.factors.append((indices, factor))
        self.metrics = dict(kind='parity_blocks', blocks=len(groups),
                            maximum_block_size=max(map(len, groups)),
                            factor_bytes=sum(f.nbytes for _,f in self.factors),
                            setup_seconds=time.perf_counter()-began,
                            ridge_relative=ridge,
                            basis=basis,
                            linearization_basis=getattr(operator, 'linearization_basis', 'actual'),
                            field_parity=parity_info,
                            field_parity_shifts=shifts.tolist(),
                            parity_axis_mask=axis_mask.tolist(),
                            setup_matvec_calls=0 if direct_columns else n,
                            setup_rmatvec_calls=0 if direct_columns else n,
                            setup_column_gram_calls=len(groups) if direct_columns else 0,
                            column_row_chunk_size=getattr(operator, 'column_row_chunk_size', 1024) if direct_columns else None,
                            construction='local_column_panels' if direct_columns else 'operator_products',
                            interpretation='Small regularized Gram blocks; inverse-triangular right preconditioning, reused across nonlinear steps; local row chunks avoid global Jacobian assembly')

    def apply(self, vector):
        out = np.empty_like(vector)
        for indices, factor in self.factors:
            out[indices] = self.inverse_scale[indices]*solve_triangular(factor.T, vector[indices], lower=False, check_finite=False)
        return out

    def transpose(self, vector):
        out = np.empty_like(vector)
        for indices, factor in self.factors:
            out[indices] = solve_triangular(factor, self.inverse_scale[indices]*vector[indices], lower=True, check_finite=False)
        return out


def _linear_stationarity(operator, residual, operator_norm):
    """Recheck least-squares orthogonality with matrix-vector products only.

    For a linear model e = r + A z, its first-order condition is A.T e = 0.
    LSMR's accumulated norm estimate supplies a scale, not a certified bound.
    Consequently this is numerical stationarity of this particular model;
    it is not a residual-tolerance or representation-error certificate.
    """
    residual_norm = float(np.linalg.norm(residual))
    normal_norm = float(np.linalg.norm(operator.rmatvec(residual)))
    scale = float(operator_norm)*residual_norm
    threshold = 64*np.finfo(float).eps*scale
    return dict(residual_norm=residual_norm, normal_residual_norm=normal_norm,
                operator_norm_estimate=float(operator_norm),
                normal_residual_threshold=threshold,
                normal_residual_relative=normal_norm/max(scale, np.finfo(float).tiny),
                orthogonal_to_working_precision=bool(normal_norm <= threshold))


def solve_residual(problem, initial_coefficients=None, max_iterations=30, tolerance=1e-8,
                   damping=1e-6, lsmr_tolerance=1e-7, lsmr_max_iterations=None,
                   gradient_tolerance=1e-11, step_tolerance=1e-12,
                   max_line_search=12, scaling_probes=8, validate_locality=True,
                   max_seconds=None, max_residual_evaluations=500,
                   preconditioner='diagonal', preconditioner_block_size=64,
                   preconditioner_refresh=0, preconditioner_ridge=1e-10,
                   preconditioner_field_parity='none',
                   preconditioner_basis='actual',
                   high_accuracy=False, lsmr_condition_limit=None,
                   linear_refinement_steps=None, iteration_callback=None,
                   inexact_newton=False,
                   linearization_basis='actual',
                   refine_on_ideal_stall=False,
                   execution=None, batch_size=None):
    """Damped Gauss–Newton/LSMR with projected bounds and backtracking.

    No global residual Jacobian or Gram matrix is formed. Diagonal scaling is
    a deterministic Hutchinson estimate from Jacobian transpose products.
    ``converged`` means all declared scaled block RMS targets were reached;
    it says nothing about unprovided constraints or unsampled physical error.

    ``high_accuracy=True`` tightens inner solves, removes absolute gradient and
    coefficient-step cutoffs, lets damping vanish near a root, and applies two
    linear defect-correction passes by default when useful. Explicit
    ``linear_refinement_steps=0`` disables them; ``None`` selects two in high
    accuracy mode and zero otherwise. The requested
    residual tolerance is never relaxed. This remains float64 arithmetic: a
    small sampled residual does not certify coefficient or continuum accuracy,
    and an unattainable target can return ``precision_limited``. A completed
    linear least-squares solve with explicitly checked orthogonality, negligible
    predicted improvement, and no meaningful accepted objective step can instead return
    ``linearized_stationary``. This is not convergence to the residual target
    or proof of a global representation floor. Krylov iteration-limit exits
    alone never establish this status.

    ``iteration_callback(coefficients, parameters, record)`` receives copies
    at the start of each nonlinear iteration, after residual evaluation. It
    may save checkpoints; modifying these copies does not change the solve.

    ``inexact_newton=True`` optionally avoids solving early linear models to
    final precision. It schedules LSMR tolerances from the actual residual,
    capped at .05, and restores the existing tight tolerance near a root.
    Coarse solves skip defect correction and cannot establish numerical
    stationarity. A rejected/tiny coarse direction is retried tightly before
    precision/stationarity decisions. An accepted coarse step with negligible
    objective improvement above tolerance requests one tight actual-Jacobian
    check at the new state; the usual stationarity evidence is still required.
    This is a practical inner-tolerance
    schedule, not a proof that the classical inexact-Newton forcing inequality
    holds for an inconsistent least-squares model. Physical tolerance is never
    changed. Default False preserves the previous inner-solve policy.

    ``linearization_basis='ideal'`` optionally uses ideal-coordinate jets in
    the correction Jacobian, at the ACTUAL neural state and with actual local
    PDE/parameter sensitivities. Corrections are generated with bounded row
    and coordinate panels, never a global feature cache. Residual evaluation,
    line-search objective/gradient, convergence, and export remain neural.
    Rejected/tiny/invalid approximate corrections fall back to the actual
    Jacobian before precision or stationarity conclusions. Default 'actual'
    preserves the exact neural Jacobian path.

    ``refine_on_ideal_stall=True`` lets a resolution driver explore a richer
    model instead of paying for an actual-Jacobian diagnosis after a negligible
    or rejected ideal correction above tolerance. It returns the last accepted
    actual-neural state with ``approximate_model_stalled``; this is explicitly
    NOT convergence, stationarity, or evidence of a representation limit.
    The driver must still validate subsequent states with the actual network.
    Default False retains the actual-Jacobian fallback. Invalid/nonfinite ideal
    operators retain that safety fallback even when this option is enabled.

    ``preconditioner_field_parity='auto'`` optionally groups fields using
    parity shifts inferred from constant sampled local PDE sensitivities.
    This lets a differentiated field couple to the appropriate neighboring
    coordinate parity of another field. Inconsistent axes are dropped; this
    heuristic is neither a continuum symmetry claim nor a convergence
    guarantee. The backward-compatible default ``'none'`` keeps the original
    groups. An explicit fields-by-dimension binary array is also accepted.

    ``preconditioner_basis='ideal'`` optionally substitutes analytically known
    coordinate jets only while constructing small preconditioner Gram blocks.
    Local physics sensitivities, residuals, all Jacobian products, stopping
    checks and exported inference remain those of the actual neural model.
    The default ``'actual'`` uses actual tanh jets throughout. This approximation
    changes conditioning only and cannot silently change the solved equation.

    ``execution='streamed'`` recomputes features in bounded point batches;
    it retains residual vectors and pointwise callback derivatives, but no
    Q-by-coefficient feature matrices. It requires the feature forward_jets,
    adjoint_jets, and selected_columns protocol. Callbacks must accept any
    point batch; a ResidualBlock.batch_function additionally receives original
    row indices for prescribed per-point data. None inherits problem settings.

    ``max_seconds`` is a soft budget between indivisible numerical calls. If
    a completed Krylov solve overruns it, one objective trial of its direction
    is allowed before returning ``budget_exhausted``; a useful step is kept.
    ``max_residual_evaluations`` remains a hard evaluation-count limit.
    """
    began = time.perf_counter()
    for name, value in [('max_iterations', max_iterations), ('max_line_search', max_line_search),
                        ('scaling_probes', scaling_probes), ('max_residual_evaluations', max_residual_evaluations)]:
        if int(value) != value or value < (0 if name == 'max_iterations' else 1):
            raise ValueError(f'{name} must be a valid nonnegative/positive integer')
    if any(not np.isfinite(v) or v <= 0 for v in (tolerance, damping, lsmr_tolerance, gradient_tolerance, step_tolerance)):
        raise ValueError('Solver tolerances and damping must be positive and finite')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive and finite or None')
    if preconditioner not in ('diagonal', 'block', 'auto'):
        raise ValueError("preconditioner must be 'diagonal', 'block', or 'auto'")
    if preconditioner_basis not in ('actual', 'ideal'):
        raise ValueError("preconditioner_basis must be 'actual' or 'ideal'")
    if linearization_basis not in ('actual', 'ideal'):
        raise ValueError("linearization_basis must be 'actual' or 'ideal'")
    if linearization_basis == 'ideal' and not callable(getattr(problem.features, 'selected_ideal_columns', None)):
        raise ValueError('Ideal correction linearization requires selected_ideal_columns support')
    if isinstance(preconditioner_field_parity, str):
        if preconditioner_field_parity not in ('none', 'auto'):
            raise ValueError("preconditioner_field_parity must be 'none', 'auto', or an F-by-d binary array")
    else:
        shifts = np.asarray(preconditioner_field_parity)
        if (shifts.shape != (problem.fields, problem.features.dimension)
                or not np.issubdtype(shifts.dtype, np.integer) or np.any((shifts != 0) & (shifts != 1))):
            raise ValueError('preconditioner_field_parity must have binary integer shape fields-by-dimension')
    if (int(preconditioner_block_size) != preconditioner_block_size or preconditioner_block_size < 1 or
            int(preconditioner_refresh) != preconditioner_refresh or preconditioner_refresh < 0 or
            not np.isfinite(preconditioner_ridge) or preconditioner_ridge <= 0):
        raise ValueError('Preconditioner block size, refresh interval, or ridge is invalid')
    if lsmr_max_iterations is not None and (int(lsmr_max_iterations) != lsmr_max_iterations or lsmr_max_iterations < 1):
        raise ValueError('lsmr_max_iterations must be a positive integer or None')
    if not isinstance(high_accuracy, (bool, np.bool_)):
        raise ValueError('high_accuracy must be a boolean')
    if not isinstance(inexact_newton, (bool, np.bool_)):
        raise ValueError('inexact_newton must be a boolean')
    if not isinstance(refine_on_ideal_stall, (bool, np.bool_)):
        raise ValueError('refine_on_ideal_stall must be a boolean')
    if iteration_callback is not None and not callable(iteration_callback):
        raise ValueError('iteration_callback must be callable or None')
    if linear_refinement_steps is not None and (
            int(linear_refinement_steps) != linear_refinement_steps or linear_refinement_steps < 0):
        raise ValueError('linear_refinement_steps must be a nonnegative integer or None')
    if lsmr_condition_limit is not None and (not np.isfinite(lsmr_condition_limit) or lsmr_condition_limit < 0):
        raise ValueError('lsmr_condition_limit must be finite and nonnegative or None; zero disables this stopping condition')
    epsilon = np.finfo(float).eps
    inner_tolerance = (max(epsilon, min(lsmr_tolerance, .01*tolerance))
                       if high_accuracy else lsmr_tolerance)
    condition_limit = (0. if high_accuracy else 1e8) if lsmr_condition_limit is None else float(lsmr_condition_limit)
    refinement_steps = (2 if high_accuracy else 0) if linear_refinement_steps is None else int(linear_refinement_steps)
    engine = _make_engine(problem, validate_locality, execution, batch_size)
    x = _initial_vector(problem, initial_coefficients)
    history, status, blocks = [], 'iteration_limit', []
    lm, latest, objective, total_lsmr = float(damping), None, np.inf, 0
    block_preconditioner, preconditioner_builds = None, []
    preconditioner_switches, stationarity_polish_used = [], False
    completed_direction_trials_after_time_budget = 0
    initial_residual_scale = None
    ideal_model_stalled = False
    tight_actual_check_pending = False
    iteration = 0
    def evaluation_budget_exhausted():
        return engine.evaluations >= max_residual_evaluations
    def time_budget_exhausted():
        return max_seconds is not None and time.perf_counter()-began >= max_seconds
    def budget_exhausted():
        return evaluation_budget_exhausted() or time_budget_exhausted()
    for iteration in range(max_iterations+1):
        if budget_exhausted():
            status = 'budget_exhausted'
            break
        try:
            latest = engine.evaluate(x, derivatives=True)
        except _NonfiniteResidual:
            status = 'nonfinite_residual'
            break
        r, actual_operator, blocks = latest.residual, latest.operator, latest.block_metrics
        tight_actual_check = tight_actual_check_pending
        tight_actual_check_pending = False
        active_linearization_basis = ('actual' if ideal_model_stalled or tight_actual_check else linearization_basis)
        operator = actual_operator.ideal_operator if active_linearization_basis == 'ideal' else actual_operator
        ideal_basis_invalid = False
        objective = .5*float(r @ r)
        target = max(b['maximum_scaled_rms'] for b in blocks)
        if initial_residual_scale is None:
            initial_residual_scale = max(1., target)
        tight_switch = max(10*tolerance, np.sqrt(epsilon)*initial_residual_scale)
        solve_tolerance = (max(inner_tolerance, min(.05, .1*np.sqrt(target/initial_residual_scale)))
                           if inexact_newton and target > tight_switch and not tight_actual_check else inner_tolerance)
        if high_accuracy:
            # LM stabilizes large nonlinear updates. It must not impose a
            # fixed penalty when the objective is already close to a root.
            lm = 0. if target <= np.sqrt(epsilon) else min(lm, target**2)
        record = dict(iteration=iteration, objective=objective,
                      maximum_scaled_block_rms=target, damping=lm,
                      scheduled_inner_tolerance=float(solve_tolerance),
                      tight_inner_switch_rms=float(tight_switch),
                      linearization_basis=active_linearization_basis,
                      elapsed_seconds=time.perf_counter()-began)
        if ideal_model_stalled:
            record['linearization_fallbacks'] = ['previous_ideal_step_negligible']
        if tight_actual_check:
            record['tight_actual_model_check'] = 'previous_accepted_inexact_step_negligible'
        history.append(record)
        if iteration_callback is not None:
            callback_coefficients, callback_parameters = engine.unpack(x)
            iteration_callback(callback_coefficients.copy(), callback_parameters.copy(), dict(record))
        if target <= tolerance:
            status = 'converged'
            break
        if iteration == max_iterations:
            status = 'iteration_limit'
            break
        if budget_exhausted():
            # A streamed actual linearization or checkpoint callback can use
            # the remaining budget. Preserve that evaluated state and avoid
            # starting another expensive adjoint/Gram construction afterward.
            status = 'budget_exhausted'
            break
        random = np.random.default_rng(18431)
        diagonal = np.zeros(engine.n)
        try:
            for _ in range(scaling_probes):
                probe = random.choice((-1., 1.), size=operator.shape[0])
                diagonal += operator.rmatvec(probe)**2/scaling_probes
            if not np.all(np.isfinite(diagonal)):
                raise _NonfiniteResidual('Nonfinite correction-Jacobian diagonal estimate')
        except (_NonfiniteResidual, FloatingPointError):
            if active_linearization_basis != 'ideal':
                raise
            operator, active_linearization_basis = actual_operator, 'actual'
            ideal_basis_invalid = True
            diagonal[:] = 0.
            for _ in range(scaling_probes):
                probe = random.choice((-1., 1.), size=operator.shape[0])
                diagonal += operator.rmatvec(probe)**2/scaling_probes
            record.setdefault('linearization_fallbacks', []).append('invalid_ideal_scaling')
        floor = max(float(np.max(diagonal))*1e-16, 1e-30)
        inverse_scale = 1/np.sqrt(np.maximum(diagonal, floor))
        block_preconditioning_allowed = not (ideal_basis_invalid and preconditioner_basis == 'ideal')
        if not block_preconditioning_allowed:
            block_preconditioner = None
            record.setdefault('preconditioner_fallbacks', []).append('invalid_ideal_basis_use_diagonal')
        def build_block(reason):
            nonlocal block_preconditioner, block_preconditioning_allowed
            nonlocal operator, active_linearization_basis, solve_tolerance
            nonlocal apply_preconditioner, transpose_preconditioner
            # Drop both bound-method references before constructing replacement
            # factors. Refresh must not retain two O(unknowns*block) sets.
            block_preconditioner = None
            apply_preconditioner = lambda v: inverse_scale*v
            transpose_preconditioner = lambda v: inverse_scale*v
            try:
                block_preconditioner = _RightPreconditioner(problem, actual_operator, inverse_scale,
                                                           preconditioner_block_size, preconditioner_ridge,
                                                           field_parity=preconditioner_field_parity,
                                                           basis=preconditioner_basis)
            except (_NonfiniteResidual, FloatingPointError, np.linalg.LinAlgError):
                if preconditioner_basis != 'ideal':
                    raise
                # Actual directional column panels can be much larger than
                # ideal panels. Do not silently construct them after a budget
                # approved only ideal storage. Exact-J products with diagonal
                # scaling provide the memory-bounded safety path.
                block_preconditioner, block_preconditioning_allowed = None, False
                record.setdefault('preconditioner_fallbacks', []).append('invalid_ideal_gram_use_diagonal')
                if active_linearization_basis == 'ideal':
                    operator, active_linearization_basis = actual_operator, 'actual'
                    solve_tolerance = inner_tolerance
                    record.setdefault('linearization_fallbacks', []).append('invalid_ideal_gram_use_actual_jacobian')
                return False
            build_metrics=dict(block_preconditioner.metrics)
            build_metrics.update(linearization_basis=active_linearization_basis,
                                 local_sensitivity_basis='actual_neural')
            preconditioner_builds.append(dict(iteration=iteration, reason=reason,**build_metrics))
            if preconditioner == 'auto' and not preconditioner_switches:
                preconditioner_switches.append(dict(iteration=iteration, reason=reason))
            return True
        if block_preconditioning_allowed and (preconditioner == 'block' or block_preconditioner is not None):
            sensitivity_change = False
            if block_preconditioner is not None:
                old_diagonal = 1/block_preconditioner.inverse_scale**2
                new_diagonal = np.maximum(diagonal, floor)
                sensitivity_change = bool(np.max(np.maximum(old_diagonal/new_diagonal,
                                                             new_diagonal/old_diagonal)) > 1e3)
            if block_preconditioner is None or sensitivity_change or (preconditioner_refresh and iteration % preconditioner_refresh == 0):
                build_block('sensitivity_change' if sensitivity_change else 'initial_or_periodic')
        if block_preconditioner is not None:
            apply_preconditioner = block_preconditioner.apply
            transpose_preconditioner = block_preconditioner.transpose
        else:
            apply_preconditioner = lambda v: inverse_scale*v
            transpose_preconditioner = lambda v: inverse_scale*v
        gradient = actual_operator.rmatvec(r)
        projected_gradient = gradient.copy()
        offset = engine.p*engine.f
        at_lower = x[offset:] <= problem.parameter_lower+1e-13
        at_upper = x[offset:] >= problem.parameter_upper-1e-13
        projected_gradient[offset:][at_lower & (gradient[offset:] > 0)] = 0
        projected_gradient[offset:][at_upper & (gradient[offset:] < 0)] = 0
        scaled_gradient = float(np.max(abs(transpose_preconditioner(projected_gradient))))
        record['scaled_projected_gradient_inf'] = scaled_gradient
        gradient_threshold = (max(epsilon, min(gradient_tolerance, .01*tolerance))*np.linalg.norm(r)
                              if high_accuracy else gradient_tolerance*max(1., np.linalg.norm(r)))
        record['scaled_gradient_threshold'] = float(gradient_threshold)
        if scaled_gradient <= gradient_threshold:
            if (preconditioner == 'auto' and block_preconditioner is None and block_preconditioning_allowed
                    and build_block('stationary_above_residual_tolerance')):
                apply_preconditioner, transpose_preconditioner = block_preconditioner.apply, block_preconditioner.transpose
                scaled_gradient = float(np.max(abs(transpose_preconditioner(projected_gradient))))
                record['scaled_projected_gradient_inf'] = scaled_gradient
            if scaled_gradient <= gradient_threshold:
                # One generic extra inexact-Newton attempt prevents a first-order
                # threshold from prematurely deciding an attainable residual goal.
                if preconditioner == 'auto' and scaled_gradient > 0 and not stationarity_polish_used:
                    stationarity_polish_used = True
                    record['stationarity_polish'] = True
                else:
                    if not high_accuracy:
                        status = 'stationary'
                        break
                    # Check the actual linear least-squares model below even
                    # when its current projected gradient is already tiny.
                    record['small_gradient_requires_linear_model_check'] = True
        scaled_operator = LinearOperator(operator.shape,
                                          matvec=lambda z: operator.matvec(apply_preconditioner(z)),
                                          rmatvec=lambda v: transpose_preconditioner(operator.rmatvec(v)), dtype=float)
        accepted, tiny, linearized_stationary = False, False, False
        approximate_refinement_requested = False
        for _ in range(4):
            if budget_exhausted():
                break
            try:
                linear = lsmr(scaled_operator, -r, damp=np.sqrt(lm),
                              atol=solve_tolerance, btol=solve_tolerance, conlim=condition_limit,
                              maxiter=lsmr_max_iterations or min(4*engine.n, 1000))
                if not np.all(np.isfinite(linear[0])):
                    raise _NonfiniteResidual('Nonfinite correction step')
            except (_NonfiniteResidual, FloatingPointError):
                if active_linearization_basis != 'ideal':
                    raise
                operator, active_linearization_basis = actual_operator, 'actual'
                solve_tolerance = inner_tolerance
                record.setdefault('linearization_fallbacks', []).append('invalid_ideal_correction')
                continue
            total_lsmr += linear[2]
            record.setdefault('linear_solves', []).append(dict(
                preconditioner='block' if block_preconditioner is not None else 'diagonal',
                stop=int(linear[1]), iterations=int(linear[2]),
                linearization_basis=active_linearization_basis,
                tolerance=float(solve_tolerance), accurate_tolerance=bool(solve_tolerance <= inner_tolerance)))
            if (preconditioner == 'auto' and block_preconditioner is None and block_preconditioning_allowed and not budget_exhausted() and
                    (linear[1] == 7 or linear[2] >= max(8, 2*engine.n))):
                if build_block('krylov_iteration_cost'):
                    apply_preconditioner, transpose_preconditioner = block_preconditioner.apply, block_preconditioner.transpose
                # Closures use the new preconditioner, or the guarded actual-J
                # fallback if ideal Gram setup failed. Recompute the step in
                # that model rather than interpreting the previous one as it.
                continue
            z = linear[0]
            if refinement_steps and solve_tolerance > inner_tolerance:
                record['linear_refinement_skipped_for_inexact_step'] = True
            if refinement_steps and solve_tolerance <= inner_tolerance:
                # Correct the explicitly recomputed defect, not LSMR's
                # recursively estimated residual. For LM include the penalty
                # rows so correction solves the same augmented least squares.
                root_lm = np.sqrt(lm)
                augmented = (LinearOperator((operator.shape[0]+engine.n, engine.n),
                    matvec=lambda v: np.r_[scaled_operator.matvec(v), root_lm*v],
                    rmatvec=lambda v: scaled_operator.rmatvec(v[:operator.shape[0]])+root_lm*v[operator.shape[0]:],
                    dtype=float) if lm else scaled_operator)
                rhs = np.r_[-r, np.zeros(engine.n)] if lm else -r
                defect = rhs-augmented.matvec(z)
                record.setdefault('linear_defect_corrections', [])
                for refinement in range(refinement_steps):
                    if budget_exhausted():
                        record['linear_refinement_stopped_by_budget'] = True
                        break
                    old_norm = float(np.linalg.norm(defect))
                    if old_norm <= inner_tolerance*np.linalg.norm(rhs):
                        break
                    # An inconsistent least-squares system leaves an
                    # orthogonal residual. Solving that residual repeatedly
                    # cannot improve the coefficients. Check A.T e rather
                    # than treating all of ||e|| as an unresolved correction.
                    defect_stationarity = _linear_stationarity(augmented, defect, linear[5])
                    record['linear_defect_stationarity'] = defect_stationarity
                    if (linear[1] in (0, 1, 2, 4, 5) and
                            defect_stationarity['orthogonal_to_working_precision']):
                        record['linear_refinement_stopped_by_orthogonality'] = True
                        break
                    correction = lsmr(augmented, defect, atol=inner_tolerance,
                                      btol=inner_tolerance, conlim=condition_limit,
                                      maxiter=lsmr_max_iterations or min(4*engine.n, 1000))
                    total_lsmr += correction[2]
                    updated = z+correction[0]
                    updated_defect = rhs-augmented.matvec(updated)
                    new_norm = float(np.linalg.norm(updated_defect))
                    improves = new_norm < old_norm
                    record['linear_defect_corrections'].append(dict(
                        iteration=refinement, stop=int(correction[1]), iterations=int(correction[2]),
                        tolerance=float(inner_tolerance),
                        linearization_basis=active_linearization_basis,
                        defect_before=old_norm, defect_after=new_norm, accepted=improves))
                    if not improves:
                        break
                    z, defect = updated, updated_defect
                record['linear_augmented_defect_norm'] = float(np.linalg.norm(defect))
            step = apply_preconditioner(z)
            record.update(lsmr_stop=int(linear[1]), lsmr_iterations=int(linear[2]),
                          estimated_condition=float(linear[6]))
            if high_accuracy:
                linear_defect = r+operator.matvec(step)
                record['linearized_residual_norm'] = float(np.linalg.norm(linear_defect))
                record['linearized_residual_relative'] = float(np.linalg.norm(linear_defect)/max(np.linalg.norm(r), np.finfo(float).tiny))
                stationarity = _linear_stationarity(scaled_operator, linear_defect, linear[5])
                stationarity['lsmr_completed'] = bool(linear[1] in (0, 1, 2, 4, 5))
                stationarity['inner_solve_accurate'] = bool(solve_tolerance <= inner_tolerance)
                stationarity['actual_jacobian'] = active_linearization_basis == 'actual'
                # sqrt(eps) describes negligible relative *model decrease*;
                # it does not replace the requested physical residual goal.
                relative_decrease = 1-record['linearized_residual_relative']
                stationarity['relative_model_decrease'] = relative_decrease
                stationarity['negligible_model_decrease_threshold'] = float(np.sqrt(epsilon))
                stationary_model = (stationarity['actual_jacobian'] and stationarity['inner_solve_accurate'] and stationarity['lsmr_completed'] and
                    stationarity['orthogonal_to_working_precision'] and
                    abs(relative_decrease) <= np.sqrt(epsilon))
                record['linearized_stationarity'] = stationarity
            for search in range(max_line_search):
                if evaluation_budget_exhausted():
                    break
                expired_time = time_budget_exhausted()
                if expired_time and search > 0:
                    break
                alpha = .5**search
                trial = engine.project(x+alpha*step)
                actual_step = trial-x
                if (np.array_equal(trial, x) if high_accuracy else
                        np.linalg.norm(actual_step) <= step_tolerance*(1+np.linalg.norm(x))):
                    tiny = True
                    break
                try:
                    if expired_time:
                        completed_direction_trials_after_time_budget += 1
                        record['completed_direction_trial_after_time_budget'] = True
                    trial_r, trial_blocks = engine.evaluate(trial)
                    trial_objective = .5*float(trial_r @ trial_r)
                except _NonfiniteResidual:
                    continue
                slope = float(gradient @ actual_step)
                if trial_objective < objective and trial_objective <= objective+1e-4*min(slope, 0.):
                    decrease=(objective-trial_objective)/max(objective,np.finfo(float).tiny)
                    x, accepted = trial, True
                    record.update(step_length=alpha, step_norm=float(np.linalg.norm(actual_step)),
                                  trial_objective=trial_objective,relative_objective_decrease=float(decrease))
                    if (inexact_newton and solve_tolerance > inner_tolerance and decrease <= np.sqrt(epsilon)
                            and max(b['maximum_scaled_rms'] for b in trial_blocks) > tolerance):
                        # Accept genuine progress, then spend one accurate
                        # actual-model solve deciding whether this is a floor
                        # or merely an insufficient inner solve. A coarse
                        # accepted direction alone cannot certify stationarity.
                        tight_actual_check_pending = True
                        record['requested_tight_actual_model_check'] = 'accepted_negligible_inexact_step'
                    if (active_linearization_basis == 'ideal' and decrease <= np.sqrt(epsilon)
                            and max(b['maximum_scaled_rms'] for b in trial_blocks) > tolerance):
                        # A formally accepted roundoff-sized improvement must
                        # not keep an inaccurate correction model alive forever.
                        # Preserve the progress and rebuild the actual model at
                        # the new state on the next outer iteration.
                        ideal_model_stalled = True
                        record.setdefault('linearization_fallbacks', []).append('accepted_negligible_ideal_step')
                    if (high_accuracy and stationary_model and decrease<=np.sqrt(epsilon)
                            and max(b['maximum_scaled_rms'] for b in trial_blocks)>tolerance):
                        linearized_stationary=True
                        record['linearized_stationarity']['accepted_negligible_objective_step']=True
                    lm = max(lm/3, 0. if high_accuracy else 1e-14)
                    objective, blocks = trial_objective, trial_blocks
                    break
            if (not accepted and active_linearization_basis == 'ideal' and not budget_exhausted()):
                if refine_on_ideal_stall:
                    approximate_refinement_requested = True
                    record['uncertified_refinement_request'] = (
                        'tiny_ideal_correction' if tiny else 'rejected_ideal_correction')
                    break
                record.setdefault('linearization_fallbacks', []).append(
                    'tiny_ideal_correction' if tiny else 'rejected_ideal_correction')
                operator, active_linearization_basis = actual_operator, 'actual'
                solve_tolerance = inner_tolerance
                tiny = False
                continue
            if (not accepted and solve_tolerance > inner_tolerance and not budget_exhausted()):
                record.setdefault('inner_tolerance_retries', []).append(dict(
                    before=float(solve_tolerance), after=float(inner_tolerance),
                    reason='tiny_coarse_direction' if tiny else 'rejected_coarse_direction'))
                solve_tolerance = inner_tolerance
                tiny = False
                continue
            if high_accuracy and not accepted and stationary_model and not budget_exhausted():
                linearized_stationary = True
                record['linearized_stationarity']['no_accepted_objective_step'] = True
                break
            if accepted or tiny:
                break
            lm = min(max(lm*10, epsilon if high_accuracy else 0.), 1e12)
        if (refine_on_ideal_stall and (approximate_refinement_requested or ideal_model_stalled)
                and not budget_exhausted()):
            record.setdefault('uncertified_refinement_request','accepted_negligible_ideal_step')
            status = 'approximate_model_stalled'
            break
        if linearized_stationary and not budget_exhausted():
            status='linearized_stationary'
            break
        if not accepted:
            status = ('budget_exhausted' if budget_exhausted() else
                      'linearized_stationary' if linearized_stationary else
                      'precision_limited' if high_accuracy and tiny else
                      'stagnated' if tiny else 'line_search_failed')
            break
    coefficients, parameters = engine.unpack(x)
    elapsed_seconds = time.perf_counter()-began
    metrics = dict(status=status, converged=status == 'converged',
                   execution=engine.execution, batch_size=engine.batch_size,
                   optimization_converged=status in ('converged', 'stationary'),
                   iterations=iteration, objective=objective if np.isfinite(objective) else None, tolerance=tolerance,
                   blocks=blocks, residual_evaluations=engine.evaluations,
                   linearizations=engine.linearizations, lsmr_iterations=total_lsmr,
                   seconds=elapsed_seconds, basis_setup_seconds=engine.setup_seconds,
                   basis_cache_bytes=engine.cache_bytes, unknowns=engine.n,
                   feature_metrics=getattr(problem.features, 'metrics', {}),
                   parameter_bounds=[problem.parameter_lower.tolist(), problem.parameter_upper.tolist()],
                   identifiability_checked=False,
                   preconditioner=preconditioner, preconditioner_builds=preconditioner_builds,
                   active_preconditioner='block' if block_preconditioner is not None else 'diagonal',
                   preconditioner_switches=preconditioner_switches,
                   preconditioner_setup_seconds=sum(b['setup_seconds'] for b in preconditioner_builds),
                   settings=dict(max_iterations=max_iterations, lsmr_tolerance=lsmr_tolerance,
                                 effective_lsmr_tolerance=inner_tolerance,
                                 lsmr_condition_limit=condition_limit,
                                 lsmr_max_iterations=lsmr_max_iterations,
                                 gradient_tolerance=gradient_tolerance, step_tolerance=step_tolerance,
                                 scaling_probes=scaling_probes, max_seconds=max_seconds,
                                 max_residual_evaluations=max_residual_evaluations,
                                 high_accuracy=bool(high_accuracy), linear_refinement_steps=refinement_steps,
                                 inexact_newton=bool(inexact_newton),
                                 linearization_basis=linearization_basis,
                                 refine_on_ideal_stall=bool(refine_on_ideal_stall),
                                 iteration_callback_active=iteration_callback is not None),
                   arithmetic=dict(dtype='float64', machine_epsilon=epsilon,
                                   requested_tolerance_below_machine_epsilon=bool(tolerance < epsilon),
                                   tolerance_was_relaxed=False,
                                   precision_scope='No certified rounding-error bound; conditioning, feature approximation, and callback cancellation may prevent the requested target'),
                   budget=dict(time_limit_seconds=max_seconds,
                               time_overshoot_seconds=max(0., elapsed_seconds-max_seconds) if max_seconds is not None else 0.,
                               completed_direction_trials_after_time_budget=completed_direction_trials_after_time_budget,
                               residual_evaluation_limit=max_residual_evaluations,
                               residual_evaluation_limit_exceeded=engine.evaluations > max_residual_evaluations,
                               semantics='Soft wall-clock budget between numerical calls, with one trial of an already-computed direction; hard residual evaluation-count limit'),
                   convergence_scope='Declared sampled/aggregated residual blocks only; no continuum error, uniqueness, identifiability, or omitted-condition certificate',
                   method='Damped Gauss-Newton, matrix-free LSMR, local pointwise Torch derivatives, diagonal/bounded-block right preconditioning with optional automatic selection, projected parameter bounds',
                   callback_contract='Callbacks must be pointwise across rows; explicit aggregation supplies weak/integral coupling; optional le/ge hinge follows aggregation')
    if latest is not None:
        metrics.update(latest.metrics)
    return ResidualSolution(problem, coefficients.copy(), parameters.copy(), metrics, history)
