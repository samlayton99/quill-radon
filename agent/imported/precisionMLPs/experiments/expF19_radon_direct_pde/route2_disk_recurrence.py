"""Opt-in bounded Jacobi recurrence for the disk's ideal correction jets.

Only analytical correction/preconditioner columns change implementation.
The actual tanh features, residual, coordinate-to-readout map, and exported
ordinary MLP are inherited unchanged from the existing disk operators.

For n=k+2m the complex disk mode is
    (x+iy)**k P_m**(0,k)(2*(x*x+y*y)-1).
The old evaluator separately evaluates a Jacobi polynomial for every mode.
Here all radial degrees share their three-term recurrence and cosine/sine
partners share one radial table. Building a batch costs O(rows * degree**2),
not O(rows * degree**3). Powers of x+iy also share a recurrence.

Prepared tables are local to ONE bounded point batch; no dataset-wide cache
is retained. Their construction and retained arrays have a conservative
16-MiB cap. Requested output panels are separate caller-owned arrays, as in
the original selected-column protocol. The solver already visits at most
64 correction columns at a time. No solver integration is changed here.
"""
from __future__ import annotations

import numpy as np

from route2_disk_profiles import DiskProfileOperator
from route2_affine_disk_profiles import AffineDiskProfileOperator


WORKSPACE_LIMIT = 16 * 1024**2


def _radial_layout(degree):
    counts = (degree - np.arange(degree + 1)) // 2 + 1
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    return offsets, int(counts.sum())


def _radial_family(t, degree, offsets, radial_count, derivative):
    """Return d**derivative/dt**derivative P_m^(0,k)(t) for every (k,m).

    For d=0,1,2, d^d P_m^(0,k)/dt^d = 2^-d (m+k+1)_d
    P_(m-d)^(d,k+d). Advance the latter degree simultaneously for all k;
    higher k drop out once k+2m exceeds the requested total degree.
    """
    table = np.zeros((len(t), radial_count))
    maximum_k = degree - 2 * derivative
    if maximum_k < 0:
        return table

    def store(values, jacobi_degree):
        k = np.arange(values.shape[1])
        original_m = jacobi_degree + derivative
        multiplier = np.ones(len(k))
        for step in range(derivative):
            multiplier *= .5 * (original_m + k + 1 + step)
        table[:, offsets[k] + original_m] = values * multiplier

    previous = np.ones((len(t), maximum_k + 1))
    store(previous, 0)
    maximum_order = degree // 2 - derivative
    if maximum_order < 1:
        return table
    k = np.arange(maximum_k - 1, dtype=float)
    current = .5 * (-k + (k + 2 * derivative + 2) * t[:, None])
    store(current, 1)
    alpha = float(derivative)
    for n in range(1, maximum_order):
        count = degree - 2 * (n + 1 + derivative) + 1
        k = np.arange(count, dtype=float)
        beta = k + derivative
        ab = alpha + beta
        twice = 2 * n + ab
        denominator = 2 * (n + 1) * (n + ab + 1) * twice
        first = (twice + 1) * (
            twice * (twice + 2) * t[:, None] + alpha**2 - beta**2)
        second = 2 * (n + alpha) * (n + beta) * (twice + 2)
        following = (first * current[:, :count] - second * previous[:, :count]) / denominator
        store(following, n + 1)
        previous, current = current, following
    return table


class DiskRecurrenceMixin:
    """Prepared ideal columns; every actual MLP path is deliberately unchanged.

    A one-off small Gram panel should not pay to build every radial mode.
    Calls without prepared tables therefore retain the original SciPy path.
    The existing correction-product protocol explicitly prepares/reuses a
    row batch, which is where the recurrence amortizes its setup cost.
    """

    ideal_internal_column_cap = 64

    def ideal_recurrence_plan(self, rows, derivatives):
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        highest = max(map(sum, derivatives), default=0)
        _, radial_count = _radial_layout(self.degree)
        # points copy + scaled xy + radius variable; complex powers; tables.
        retained_per_row = 40 + 16 * (self.degree + 1) + 8 * radial_count * (highest + 1)
        # Reserve the larger of recurrence construction scratch and the real/
        # complex gather/product-rule scratch for one <=64-column panel.
        # Requested outputs are caller-owned, as in the old protocol.
        column_cap = self.ideal_internal_column_cap
        scratch_per_row = max(64 * (self.degree + 1), 224 * min(column_cap, self.size))
        construction_per_row = retained_per_row + scratch_per_row
        fixed = 256 * (self.degree + 1) + 8 * self.size
        maximum_rows = max(0, (WORKSPACE_LIMIT - fixed) // construction_per_row)
        return dict(enabled=highest <= 2, rows=int(rows), highest_derivative=highest,
                    maximum_rows=int(maximum_rows), workspace_limit_bytes=WORKSPACE_LIMIT,
                    construction_workspace_bound=int(rows * construction_per_row + fixed),
                    retained_workspace_bound=int(rows * retained_per_row + fixed),
                    selected_internal_column_cap=column_cap,
                    output_panels_included=False,
                    note='One row batch only; selected output panels belong to caller and are counted separately.')

    def prepare_ideal_columns(self, points, derivatives):
        points = self.base._points(points)
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        plan = self.ideal_recurrence_plan(len(points), derivatives)
        if not plan['enabled']:
            # The existing independent angular identity handles arbitrary orders.
            return dict(owner=self, points=points.copy(), derivatives=derivatives,
                        fallback=True, retained_bytes=points.nbytes, plan=plan)
        if len(points) > plan['maximum_rows']:
            raise MemoryError('Disk recurrence requires bounded point batches: '
                              f'{len(points)} rows requested, at most {plan["maximum_rows"]} '
                              'fit its 16-MiB construction workspace.')
        points = points.copy()
        xy = (points - self.midpoint) * self.scale
        t = 2 * np.sum(xy**2, axis=1) - 1
        powers = np.empty((len(points), self.degree + 1), complex)
        powers[:, 0] = 1.
        z = xy[:, 0] + 1j * xy[:, 1]
        for k in range(1, self.degree + 1):
            powers[:, k] = powers[:, k - 1] * z
        offsets, radial_count = _radial_layout(self.degree)
        radial = tuple(_radial_family(t, self.degree, offsets, radial_count, d)
                       for d in range(plan['highest_derivative'] + 1))
        mode_indices = offsets[self.harmonics] + (self.degrees - self.harmonics) // 2
        arrays = (points, xy, t, powers, offsets, mode_indices) + radial
        for array in arrays:
            array.flags.writeable = False
        retained = sum(array.nbytes for array in arrays)
        assert retained <= plan['retained_workspace_bound']
        return dict(owner=self, points=points, derivatives=derivatives, fallback=False,
                    xy=xy, powers=powers, radial=radial, mode_indices=mode_indices,
                    retained_bytes=retained, plan=plan)

    def selected_ideal_columns(self, points, indices, derivatives, *, prepared=None):
        points = self.base._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        if prepared is not None and (prepared['owner'] is not self
                or prepared['derivatives'] != derivatives
                or not np.array_equal(prepared['points'], points)):
            raise ValueError('Prepared ideal tables must belong to the same geometry, points, and derivatives')
        if any(sum(d) > 2 for d in derivatives):
            return self._selected_ideal_columns_ridge(points, indices, derivatives)
        if not len(indices) or not derivatives:
            return {d: np.zeros((len(points), len(indices))) for d in derivatives}
        if prepared is None:
            return DiskProfileOperator.selected_ideal_columns(self, points, indices, derivatives)
        # Gram callers can request a large group of output columns. Retain
        # those requested outputs, but keep our additional scratch bounded.
        column_cap = self.ideal_internal_column_cap
        if len(indices) > column_cap:
            result = {d: np.empty((len(points), len(indices))) for d in derivatives}
            for start in range(0, len(indices), column_cap):
                stop = min(start + column_cap, len(indices))
                panel = self.selected_ideal_columns(points, indices[start:stop], derivatives,
                                                    prepared=prepared)
                for d in derivatives:
                    result[d][:, start:stop] = panel[d]
            return result
        xy, powers = prepared['xy'], prepared['powers']
        radial_indices = prepared['mode_indices'][indices]
        radial = prepared['radial'][0][:, radial_indices]
        k = self.harmonics[indices]
        sine = self.is_sine[indices][None, :]

        def component(values, phase=0):
            # Select real/sine parts after multiplication by i**phase. This
            # avoids doing every product-rule arithmetic operation in complex.
            if phase == 0:
                return np.where(sine, values.imag, values.real)
            if phase == 1:
                return np.where(sine, values.real, -values.imag)
            return -np.where(sine, values.imag, values.real)

        solid = component(powers[:, k])
        highest = max(map(sum, derivatives))
        if highest >= 1:
            radial1 = prepared['radial'][1][:, radial_indices]
            powers1 = powers[:, np.maximum(k - 1, 0)]
            solid1 = (k * component(powers1), k * component(powers1, 1))
        if highest >= 2:
            radial2 = prepared['radial'][2][:, radial_indices]
            powers2 = powers[:, np.maximum(k - 2, 0)]
            solid2 = (k * (k - 1) * component(powers2),
                      k * (k - 1) * component(powers2, 1))
        result = {}
        for derivative in derivatives:
            order = sum(derivative)
            if order == 0:
                value = solid * radial
            elif order == 1:
                axis = derivative.index(1)
                value = solid1[axis] * radial + solid * radial1 * (4 * xy[:, axis, None])
            else:
                axes = [axis for axis, count in enumerate(derivative) for _ in range(count)]
                a, b = axes
                radial_a = radial1 * (4 * xy[:, a, None])
                radial_b = radial1 * (4 * xy[:, b, None])
                radial_ab = 16 * xy[:, a, None] * xy[:, b, None] * radial2
                if a == b:
                    radial_ab += 4 * radial1
                second_solid = solid2[a + b] if a + b < 2 else -solid2[0]
                value = (second_solid * radial + solid1[a] * radial_b
                         + solid1[b] * radial_a + solid * radial_ab)
            result[derivative] = (value * self.normalizations[indices][None, :]
                                  * np.prod(self.scale**np.asarray(derivative)))
        return result


class RecurrenceDiskProfileOperator(DiskRecurrenceMixin, DiskProfileOperator):
    """Explicit opt-in disk operator; original class and defaults unchanged."""


class RecurrenceAffineDiskProfileOperator(DiskRecurrenceMixin, AffineDiskProfileOperator):
    """Same recurrence on the existing affine enclosing-disk chart."""


class BatchedDiskRecurrenceMixin:
    """Use the existing batched ideal-product protocol with complete row panels.

    This protocol happens to be named ``ideal_tensor`` in the solver. Here
    it is a dense *bounded row batch*, not a Cartesian tensor-product basis.
    The planner counts all requested jets, their recurrence tables/scratch,
    and coefficient state before choosing a row count under 16 MiB. No
    feature matrix spanning the complete dataset is stored or factorized.
    """

    ideal_internal_column_cap = 256

    def ideal_tensor_plan(self, fields, rows, derivatives):
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        fields, rows = int(fields), int(rows)
        if fields < 1 or rows < 1:
            raise ValueError('fields and rows must be positive integers')
        recurrence = self.ideal_recurrence_plan(1, derivatives)
        if not recurrence['enabled']:
            return dict(enabled=False, reason='disk_recurrence_supports_order_at_most_two')
        fixed = 256*(self.degree+1) + 8*self.size
        fixed += 16*self.size*fields  # input/accumulator plus product temporary
        # The recurrence bound excludes caller outputs. Add every complete
        # jet and both current/previous small panel outputs conservatively.
        per_row = recurrence['construction_workspace_bound'] - (256*(self.degree+1)+8*self.size)
        per_row += 8*len(derivatives)*(self.size + 2*min(self.ideal_internal_column_cap, self.size) + 2*fields)
        maximum_rows = max(0, (WORKSPACE_LIMIT-fixed)//per_row)
        if not maximum_rows:
            return dict(enabled=False, reason='one_complete_disk_jet_row_exceeds_workspace',
                        workspace_limit_bytes=WORKSPACE_LIMIT)
        used_rows = min(rows, maximum_rows, recurrence['maximum_rows'])
        return dict(enabled=True, rows=int(used_rows), requested_rows=rows,
                    workspace_bytes=int(fixed+used_rows*per_row),
                    workspace_limit_bytes=WORKSPACE_LIMIT, coefficient_state_bytes=8*self.size*fields,
                    implementation='bounded_full_disk_jet_row_panels',
                    complete_dataset_feature_cache=False,
                    note='All complete row-batch jets, recurrence tables, temporaries, and coefficient state counted.')

    def prepare_ideal_tensor_forward(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if (coefficients.ndim != 2 or coefficients.shape[0] != self.size
                or not np.all(np.isfinite(coefficients))):
            raise ValueError('coefficients must be a finite coordinates-by-fields matrix')
        return dict(owner=self, coefficients=coefficients)

    def _complete_ideal_row_panels(self, points, derivatives, fields):
        plan = self.ideal_tensor_plan(fields, len(points), derivatives)
        if not plan['enabled'] or plan['rows'] != len(points):
            raise MemoryError('Complete disk jet panels require the row count from ideal_tensor_plan')
        prepared = self.prepare_ideal_columns(points, derivatives)
        return self.selected_ideal_columns(points, np.arange(self.size), derivatives, prepared=prepared)

    def forward_ideal_tensor_jets(self, state, points, derivatives):
        if state['owner'] is not self:
            raise ValueError('Ideal product state belongs to another geometry')
        coefficients = state['coefficients']
        panels = self._complete_ideal_row_panels(points, derivatives, coefficients.shape[1])
        return {d: panel @ coefficients for d, panel in panels.items()}

    def prepare_ideal_tensor_adjoint(self, fields):
        if int(fields) != fields or fields < 1:
            raise ValueError('fields must be a positive integer')
        return dict(owner=self, coefficients=np.zeros((self.size, int(fields))))

    def accumulate_ideal_tensor_jets(self, cotangents, points, state):
        if state['owner'] is not self:
            raise ValueError('Ideal product state belongs to another geometry')
        coefficients = state['coefficients']
        panels = self._complete_ideal_row_panels(points, tuple(cotangents), coefficients.shape[1])
        for d, panel in panels.items():
            cotangent = np.asarray(cotangents[d], float)
            if cotangent.shape != (len(points), coefficients.shape[1]) or not np.all(np.isfinite(cotangent)):
                raise ValueError('cotangents must be finite points-by-fields matrices')
            coefficients += panel.T @ cotangent

    def finish_ideal_tensor_adjoint(self, state):
        if state['owner'] is not self:
            raise ValueError('Ideal product state belongs to another geometry')
        return state['coefficients']


class BatchedRecurrenceDiskProfileOperator(BatchedDiskRecurrenceMixin, RecurrenceDiskProfileOperator):
    """Second opt-in: recurrence plus adaptively bounded complete jet rows."""


class BatchedRecurrenceAffineDiskProfileOperator(BatchedDiskRecurrenceMixin, RecurrenceAffineDiskProfileOperator):
    """Second opt-in with the existing affine enclosing-disk chart."""


def benchmark(output, *, degrees=(16, 32, 64), rows=128, repeats=5, fields=3):
    """Isolated implementation comparison, not a new PDE accuracy result."""
    import json
    import platform
    import time
    import tracemalloc
    from pathlib import Path

    rng = np.random.default_rng(7821)
    angles = rng.uniform(0, 2*np.pi, rows)
    radii = np.sqrt(rng.uniform(0, 1, rows))
    radii[:min(24, rows)] = 1.
    points = np.c_[radii*np.cos(angles), radii*np.sin(angles)]
    points[-1] = 0.
    orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    results = []
    for degree in degrees:
        operator = RecurrenceDiskProfileOperator(degree, centers=33)
        coefficients = rng.normal(size=(operator.size, fields))
        cotangents = {d: rng.normal(size=(rows, fields)) for d in orders}

        def product(fast, transpose):
            prepared = operator.prepare_ideal_columns(points, orders) if fast else None
            result = (np.zeros_like(coefficients) if transpose else
                      {d: np.zeros((rows, fields)) for d in orders})
            for start in range(0, operator.size, 64):
                indices = np.arange(start, min(start+64, operator.size))
                panel = (operator.selected_ideal_columns(points, indices, orders, prepared=prepared)
                         if fast else DiskProfileOperator.selected_ideal_columns(operator, points, indices, orders))
                for d in orders:
                    if transpose:
                        result[indices] += panel[d].T @ cotangents[d]
                    else:
                        result[d] += panel[d] @ coefficients[indices]
            return result

        timing = {(fast, transpose): [] for fast in (False, True) for transpose in (False, True)}
        saved = {}
        for _ in range(repeats):
            for key in timing:
                started = time.perf_counter()
                saved[key] = product(*key)
                timing[key].append(time.perf_counter()-started)
        errors = {}
        prepared = operator.prepare_ideal_columns(points, orders)
        for d in orders:
            error2, reference2, max_error, max_reference = 0., 0., 0., 0.
            for start in range(0, operator.size, 64):
                indices = np.arange(start, min(start+64, operator.size))
                a = operator.selected_ideal_columns(points, indices, orders, prepared=prepared)[d]
                b = DiskProfileOperator.selected_ideal_columns(operator, points, indices, orders)[d]
                error2 += np.sum((a-b)**2)
                reference2 += np.sum(b**2)
                max_error = max(max_error, float(np.max(abs(a-b))))
                max_reference = max(max_reference, float(np.max(abs(b))))
            errors[str(d)] = dict(relative_l2=float(np.sqrt(error2/reference2)),
                                  maximum_absolute_difference=max_error,
                                  maximum_difference_over_reference_max=max_error/max_reference)
        retained = prepared['retained_bytes']
        plan = prepared['plan']
        del prepared
        tracemalloc.start()
        product(True, False)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        actual_forward = saved[True, False]
        actual_adjoint = saved[True, True]
        lhs = sum(np.sum(actual_forward[d]*cotangents[d]) for d in orders)
        rhs = np.sum(coefficients*actual_adjoint)
        old_forward = saved[False, False]
        old_adjoint = saved[False, True]
        seconds = {('recurrence' if fast else 'scipy') + ('_adjoint' if transpose else '_forward'):
                   float(np.median(samples)) for (fast, transpose), samples in timing.items()}
        result = dict(degree=degree, coordinates=operator.size, seconds=seconds,
                      forward_speedup=seconds['scipy_forward']/seconds['recurrence_forward'],
                      adjoint_speedup=seconds['scipy_adjoint']/seconds['recurrence_adjoint'],
                      prepared_retained_bytes=retained, workspace_plan=plan,
                      measured_full_forward_tracemalloc_peak_bytes=peak,
                      jet_agreement=errors,
                      forward_relative_l2=float(np.sqrt(sum(np.sum((actual_forward[d]-old_forward[d])**2) for d in orders)
                                                     /sum(np.sum(old_forward[d]**2) for d in orders))),
                      adjoint_relative_l2=float(np.linalg.norm(actual_adjoint-old_adjoint)/np.linalg.norm(old_adjoint)),
                      adjoint_identity_relative_difference=float(abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1.)))
        complete = BatchedRecurrenceDiskProfileOperator(degree, centers=33)
        complete_plan = complete.ideal_tensor_plan(fields, rows, orders)

        def complete_product(transpose):
            state = (complete.prepare_ideal_tensor_adjoint(fields) if transpose else
                     complete.prepare_ideal_tensor_forward(coefficients))
            result = None if transpose else {d: np.empty((rows, fields)) for d in orders}
            for start in range(0, rows, complete_plan['rows']):
                stop = min(start+complete_plan['rows'], rows)
                if transpose:
                    complete.accumulate_ideal_tensor_jets(
                        {d: v[start:stop] for d, v in cotangents.items()}, points[start:stop], state)
                else:
                    jets = complete.forward_ideal_tensor_jets(state, points[start:stop], orders)
                    for d in orders:
                        result[d][start:stop] = jets[d]
            return complete.finish_ideal_tensor_adjoint(state) if transpose else result

        complete_times = {}
        for transpose in (False, True):
            samples = []
            for _ in range(repeats):
                began = time.perf_counter()
                complete_result = complete_product(transpose)
                samples.append(time.perf_counter()-began)
            label = 'adjoint' if transpose else 'forward'
            complete_times[label] = float(np.median(samples))
            if transpose:
                result['complete_row_adjoint_relative_l2'] = float(
                    np.linalg.norm(complete_result-old_adjoint)/np.linalg.norm(old_adjoint))
            else:
                result['complete_row_forward_relative_l2'] = float(np.sqrt(
                    sum(np.sum((complete_result[d]-old_forward[d])**2) for d in orders)
                    /sum(np.sum(old_forward[d]**2) for d in orders)))
        tracemalloc.start()
        complete_product(False)
        _, complete_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.update(complete_row_seconds=complete_times, complete_row_workspace_plan=complete_plan,
                      complete_row_forward_speedup=seconds['scipy_forward']/complete_times['forward'],
                      complete_row_adjoint_speedup=seconds['scipy_adjoint']/complete_times['adjoint'],
                      complete_row_tracemalloc_peak_bytes=complete_peak)
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(description='Opt-in ideal disk correction implementation benchmark; no PDE solve or accuracy claim.',
                  ordinary_mlp_and_actual_residual='Unchanged inherited implementation; bitwise regression-tested.',
                  rows=rows, fields=fields, repeats=repeats, selected_column_panel=64,
                  derivatives=orders, seed=7821, numpy=np.__version__, platform=platform.platform(),
                  asymptotics='Prepared radial/power construction O(rows*p^2); old separate Jacobi evaluations O(rows*p^3). Both full jet sweeps have O(rows*p^2) output arithmetic.',
                  memory_scope='One prepared batch plus <=64-column scratch; 16 MiB internal workspace cap, caller output panels separate. No full-dataset feature cache.',
                  results=results)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-output', required=True)
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    options = parser.parse_args()
    benchmark(options.benchmark_output, rows=options.rows, repeats=options.repeats)
