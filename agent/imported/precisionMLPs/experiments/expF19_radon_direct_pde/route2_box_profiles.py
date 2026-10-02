"""Streamed box coordinates for a flat tanh network, without a dense map.

Only the coefficient conversion below uses polynomial algebra. Evaluation,
PDE derivatives, and ordinary export use the actual shared tanh bank.

For a homogeneous polynomial f_l of degree l and a unit direction w,
Gaussian integration (Wick's identity) followed by radial normalization gives

  E_B[f_l(y) (w.y)^k] / k!
    = sum_j (Delta^j f_l)(w) /
      [2**(k+2*j) j! ((k-l)/2+j)! (d/2+1)_((l+k)/2)].

Inserting the finite monomial expansion of C_n^(d/2) yields the rational
factors in ``_projection_factors``. This factors the exact old ball-moment
map into univariate Legendre transforms, sparse Laplacians, and angular
polynomial evaluation. No fit, target labels, or matrix inverse is used.
There is no retained M-by-p-by-P map and no hidden P-by-P matrix.

Storage for conversion is O(p*P + M*p + M*d + direction_block*P), with
P=binomial(p+d,d); conservative work is O(d*p**2*P + M*p*P). The prepared
neural state separately stores O(M*H) ordinary readouts. Polynomial
coefficient cancellation can limit accuracy at high degree; this factorized
implementation does not claim arbitrary-degree numerical stability.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial, sqrt
import time

import numpy as np
from scipy.special import eval_jacobi, poch

from route2_directional_operator import DirectionalProfileOperator, _positive_integer
from ridge_frame_dimension_study import sphere_rule
from solver.general_features import _compositions


def _rising(a, n):
    result = Fraction(1)
    for j in range(n):
        result *= a+j
    return result


def _projection_factors(dimension, degree):
    """Exact scalar coefficients, rounded only after finite cancellation.

    result[l][j] contains (n, g_nlj) pairs multiplying Delta**j f_l.
    Orthogonality imposes n<=l and equal parity before any arithmetic.
    """
    lam = Fraction(dimension, 2)
    result = []
    for l in range(degree+1):
        row = []
        for j in range(l//2+1):
            entries = []
            for n in range(l % 2, l+1, 2):
                maximum = (n-l)//2+j
                if maximum < 0:
                    continue
                value = Fraction(0)
                for a in range(min(n//2, maximum)+1):
                    r = (n-l)//2+j-a
                    value += ((-1)**a*_rising(lam, n-a) /
                              (factorial(a)*factorial(r)*
                               _rising(lam+1, (l+n)//2-a)))
                value *= Fraction(2*n+dimension, dimension*4**j*factorial(j))
                if value:
                    entries.append((n, float(value)))
            row.append(tuple(entries))
        result.append(tuple(row))
    return tuple(result)


def _legendre_monomials(degree, dimension):
    """Column n gives coefficients of P_n(sqrt(d)*y), exact rational first."""
    polys = [[Fraction(1)]]
    if degree:
        polys.append([Fraction(0), Fraction(1)])
    for n in range(2, degree+1):
        poly = [Fraction(0)]*(n+1)
        for k, c in enumerate(polys[-1]):
            poly[k+1] += Fraction(2*n-1, n)*c
        for k, c in enumerate(polys[-2]):
            poly[k] -= Fraction(n-1, n)*c
        polys.append(poly)
    result = np.zeros((degree+1, degree+1))
    for n, poly in enumerate(polys):
        for k, c in enumerate(poly):
            # d**(k//2) is exact; only odd powers need sqrt(d).
            result[k, n] = float(c*dimension**(k//2))*(sqrt(dimension) if k % 2 else 1.)
    return result


class BoxProfileOperator:
    """Independent box-Legendre coordinates with streamed actual tanh jets.

    Coordinates have the same degree-first multiindex order as
    ``BallRidgeFeatures``. The box maps into its enclosing unit ball, so
    ``P_alpha`` means product_i P_alpha_i(sqrt(d)*y_i). These coordinates are
    independent but are not orthonormal on that enclosing ball.
    """
    def __init__(self, bounds, degree, centers=257, lam=.2, block_size=128,
                 direction_block_size=16, directions=None, angular_weights=None):
        started = time.perf_counter()
        self.bounds = np.asarray(bounds, float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or len(self.bounds) < 2 or not np.all(np.isfinite(self.bounds))
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError('bounds must have d>=2 finite increasing pairs')
        self.bounds = self.bounds.copy()
        self.dimension = len(self.bounds)
        self.degree = _positive_integer(degree, 'degree', 0)
        self.direction_block_size = _positive_integer(direction_block_size, 'direction_block_size')
        self.midpoint = self.bounds.mean(axis=1)
        self.scale = 2/(self.bounds[:, 1]-self.bounds[:, 0])/sqrt(self.dimension)
        self.multiindices = np.asarray([alpha for n in range(self.degree+1)
                                       for alpha in _compositions(n, self.dimension)], dtype=int)
        self.size = len(self.multiindices)
        self.degrees = self.multiindices.sum(axis=1)
        self.degree_indices = tuple(np.flatnonzero(self.degrees == n) for n in range(self.degree+1))
        self._lookup = {tuple(alpha): k for k, alpha in enumerate(self.multiindices)}
        self._axis_lines = []
        for axis in range(self.dimension):
            lines = []
            for alpha in self.multiindices:
                if alpha[axis] != 0:
                    continue
                top = self.degree-int(alpha.sum())
                line = []
                for n in range(top+1):
                    beta = alpha.copy(); beta[axis] = n
                    line.append(self._lookup[tuple(beta)])
                lines.append(np.asarray(line, dtype=int))
            self._axis_lines.append(tuple(lines))
        self._legendre = _legendre_monomials(self.degree, self.dimension)
        self._factors = _projection_factors(self.dimension, self.degree)
        self._laplacian = []
        for n in range(self.degree+1):
            source, target, factors = [], [], []
            local = ({tuple(self.multiindices[k]): j for j, k in enumerate(self.degree_indices[n-2])}
                     if n >= 2 else {})
            for j, k in enumerate(self.degree_indices[n]):
                alpha = self.multiindices[k]
                for axis in np.flatnonzero(alpha >= 2):
                    beta = alpha.copy(); beta[axis] -= 2
                    source.append(j); target.append(local[tuple(beta)])
                    factors.append(int(alpha[axis])*(int(alpha[axis])-1))
            self._laplacian.append((np.asarray(source, int), np.asarray(target, int), np.asarray(factors, float)))
        if (directions is None) != (angular_weights is None):
            raise ValueError('directions and angular_weights must be supplied together')
        if directions is None:
            directions, angular_weights = sphere_rule(self.dimension, self.degree)
        self.angular_weights = np.asarray(angular_weights, float).copy()
        if (self.angular_weights.shape != (len(directions),)
                or not np.all(np.isfinite(self.angular_weights))
                or not np.isclose(self.angular_weights.sum(), 1., atol=1e-12, rtol=1e-12)):
            raise ValueError('angular weights must be finite and sum to one')
        self.base = DirectionalProfileOperator(directions, self.degree, centers=centers,
            lam=lam, block_size=block_size, midpoint=self.midpoint, scale=self.scale)
        self.directions = self.base.directions
        self.direction_count = self.base.direction_count
        self.tanh_count = self.base.tanh_count
        self.interior_centers = self.base.interior_centers
        self.halo_per_side = self.base.halo_per_side
        self.block_size = self.base.block_size
        self.encoding = self.base.encoding
        coordinate_arrays = [self.bounds, self.midpoint, self.scale, self.multiindices,
                             self.degrees, self._legendre, self.angular_weights]
        coordinate_arrays += list(self.degree_indices)
        coordinate_arrays += [line for axis in self._axis_lines for line in axis]
        coordinate_arrays += [array for triple in self._laplacian for array in triple]
        self.metrics = dict(self.base.metrics, backend='actual_tanh_factored_box_profiles',
            unknowns=self.size, coefficient_count=self.size, degree=self.degree,
            redundant_profile_unknowns=self.base.size, geometry_map_bytes=0,
            coordinate_workspace_bytes=8*(self.degree+1)*self.size,
            angular_panel_bytes=8*min(self.direction_block_size, self.direction_count)*self.size,
            sparse_laplacian_bytes=sum(sum(a.nbytes for a in row) for row in self._laplacian),
            coordinate_state_array_bytes=sum(a.nbytes for a in coordinate_arrays),
            coordinate_python_lookup_entries=len(self._lookup),
            coordinate_scalar_factor_count=sum(len(j) for row in self._factors for j in row),
            angular_weight_total_variation=float(np.abs(self.angular_weights).sum()),
            selected_column_profile_panel_bytes_per_column=8*self.base.size,
            selected_column_neuron_panel_bytes_per_column=8*self.base.centers_per_direction,
            mode_label_bytes=self.multiindices.nbytes+self.degrees.nbytes,
            coordinate_scope='Exact analytical ball-moment factorization; float64 polynomial cancellation limits high-degree precision',
            product_gates=False, polynomials_in_forward=False, construction_uses_target_data=False,
            construction_seconds=time.perf_counter()-started)

    def _coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if coefficients.shape != (self.size,) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f'coefficients must be a finite vector of length {self.size}')
        return coefficients

    def _legendre_forward(self, coefficients):
        result = np.asarray(coefficients, float).copy()
        for lines in self._axis_lines:
            old = result; result = np.empty_like(old)
            for line in lines:
                n = len(line)
                result[line] = self._legendre[:n, :n] @ old[line]
        return result

    def _legendre_transpose(self, cotangent):
        result = np.asarray(cotangent, float).copy()
        for lines in reversed(self._axis_lines):
            old = result; result = np.empty_like(old)
            for line in lines:
                n = len(line)
                result[line] = self._legendre[:n, :n].T @ old[line]
        return result

    def _lap(self, coefficients, degree, transpose=False):
        source, target, factors = self._laplacian[degree]
        size = len(self.degree_indices[degree if transpose else degree-2])
        return np.bincount(source if transpose else target,
            weights=factors*coefficients[target if transpose else source], minlength=size)

    def _profile_polynomials(self, coefficients):
        mono = self._legendre_forward(coefficients)
        result = np.zeros((self.degree+1, self.size))
        for l, indices in enumerate(self.degree_indices):
            current = mono[indices]
            for j, factors in enumerate(self._factors[l]):
                n0 = l-2*j
                for n, value in factors:
                    result[n, self.degree_indices[n0]] += value*current
                if n0 >= 2:
                    current = self._lap(current, n0)
        return result

    def _profile_polynomials_transpose(self, cotangent):
        mono = np.zeros(self.size)
        for l, indices in enumerate(self.degree_indices):
            current = None
            for j in range(l//2, -1, -1):
                n0 = l-2*j
                if current is None:
                    current = np.zeros(len(self.degree_indices[n0]))
                else:
                    current = self._lap(current, n0, transpose=True)
                for n, value in self._factors[l][j]:
                    current += value*cotangent[n, self.degree_indices[n0]]
            mono[indices] = current
        return self._legendre_transpose(mono)

    def _angular_panels(self):
        for start in range(0, self.direction_count, self.direction_block_size):
            rows = slice(start, min(start+self.direction_block_size, self.direction_count))
            monomials = np.ones((len(self.directions[rows]), self.size))
            for axis in range(self.dimension):
                powers = self.directions[rows, axis:axis+1]**np.arange(self.degree+1)[None, :]
                monomials *= powers[:, self.multiindices[:, axis]]
            yield rows, monomials

    def to_directional(self, coefficients):
        polynomials = self._profile_polynomials(self._coefficients(coefficients))
        profiles = np.empty((self.direction_count, self.degree))
        for rows, monomials in self._angular_panels():
            profiles[rows] = (monomials @ polynomials[1:].T)*self.angular_weights[rows, None]
        return self.base.pack(float(polynomials[0, 0])*float(self.angular_weights.sum()), profiles)

    def directional_transpose(self, cotangent):
        bias, profiles = self.base.unpack(cotangent)
        gradient = np.zeros((self.degree+1, self.size))
        gradient[0, 0] = bias*float(self.angular_weights.sum())
        for rows, monomials in self._angular_panels():
            gradient[1:] += (profiles[rows]*self.angular_weights[rows, None]).T @ monomials
        return self._profile_polynomials_transpose(gradient)

    def prepare_forward(self, coefficients):
        return self.base.prepare_forward(self.to_directional(coefficients))

    def prepare_forward_fields(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if (coefficients.ndim != 2 or coefficients.shape[0] != self.size
                or coefficients.shape[1] < 1 or not np.all(np.isfinite(coefficients))):
            raise ValueError(f'coefficients must be a finite {self.size}-by-F matrix, F>=1')
        profiles = np.column_stack([self.to_directional(coefficients[:, field])
                                    for field in range(coefficients.shape[1])])
        return self.base.prepare_forward_fields(profiles)

    def forward_prepared_fields_jets(self, prepared, points, derivatives):
        return self.base.forward_prepared_fields_jets(prepared, points, derivatives)

    def forward_prepared_jets(self, prepared, points, derivatives):
        return self.base.forward_prepared_jets(prepared, points, derivatives)

    def forward_jets(self, coefficients, points, derivatives):
        return self.forward_prepared_jets(self.prepare_forward(coefficients), points, derivatives)

    def forward(self, coefficients, points, derivative=None):
        order = self.base._derivative(derivative)
        return self.forward_jets(coefficients, points, (order,))[order]

    def prepare_adjoint(self):
        return self.base.prepare_adjoint()

    def prepare_adjoint_fields(self, fields):
        return self.base.prepare_adjoint_fields(fields)

    def accumulate_adjoint_fields_jets(self, cotangents, points, accumulator):
        return self.base.accumulate_adjoint_fields_jets(cotangents, points, accumulator)

    def finish_adjoint_fields(self, accumulator):
        profiles = self.base.finish_adjoint_fields(accumulator)
        return np.column_stack([self.directional_transpose(profiles[:, field])
                                for field in range(profiles.shape[1])])

    def accumulate_adjoint_jets(self, cotangents, points, accumulator):
        return self.base.accumulate_adjoint_jets(cotangents, points, accumulator)

    def finish_adjoint(self, accumulator):
        return self.directional_transpose(self.base.finish_adjoint(accumulator))

    def adjoint_jets(self, cotangents, points):
        accumulator = self.prepare_adjoint()
        self.accumulate_adjoint_jets(cotangents, points, accumulator)
        return self.finish_adjoint(accumulator)

    def adjoint(self, cotangent, points, derivative=None):
        return self.adjoint_jets({self.base._derivative(derivative): cotangent}, points)

    def selected_columns(self, points, indices, derivatives):
        """Evaluate only a requested coordinate panel, sharing tanh jets.

        For k requested columns, retain (1+M*p)*k directional coefficients
        and one H*k neuron panel. Tanh bases are evaluated once per direction
        and point batch, independently of k. There is no B*k prepared-neuron
        state, and no complete map unless the caller deliberately requests
        every coordinate (as the dense diagnostic methods below do).
        """
        points = self.base._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        result = {d: np.zeros((len(points), len(indices))) for d in derivatives}
        if not len(indices):
            return result
        directional = np.empty((self.base.size, len(indices)))
        unit = np.zeros(self.size)
        for position, index in enumerate(indices):
            unit[index] = 1.
            directional[:, position] = self.to_directional(unit)
            unit[index] = 0.
        profiles = directional[1:].reshape(self.direction_count, self.degree, len(indices))
        for d in derivatives:
            if sum(d) == 0:
                result[d][:] = directional[0]
        for m in range(self.direction_count):
            readouts = self.base.weights @ profiles[m]
            anchors = self.base.anchor_values @ profiles[m]
            for rows, basis, chain in self.base._blocks(points, m, derivatives):
                for d in derivatives:
                    result[d][rows] += chain[d]*(basis[sum(d)] @ readouts)
                    if sum(d) == 0:
                        result[d][rows] += anchors
        return result

    def prepare_ideal_columns(self, points, derivatives):
        """Univariate tables for one bounded row batch; no coordinate panel.

        The correction operator reuses these tables only while visiting
        coordinate panels at these points, then releases them. For each axis
        retain only derivative orders actually requested in this batch.
        """
        points = self.base._points(points)
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        z = (points-self.midpoint)*(self.scale*sqrt(self.dimension))
        orders = np.arange(self.degree+1)
        axis_tables = {}
        for axis in range(self.dimension):
            for order in {d[axis] for d in derivatives}:
                table = np.zeros((len(points), self.degree+1))
                active = orders >= order
                degrees = orders[active]
                table[:, active] = (eval_jacobi(degrees[None, :]-order, order, order,
                    z[:, axis:axis+1])*poch(degrees+1, order)[None, :]
                    *(self.scale[axis]*sqrt(self.dimension)/2)**order)
                axis_tables[axis, order] = table
        return dict(owner=self, points=points, derivatives=derivatives, axis_tables=axis_tables,
                    retained_bytes=sum(value.nbytes for value in axis_tables.values()))

    def selected_ideal_columns(self, points, indices, derivatives, *, prepared=None):
        """Ideal jets only for explicitly approximate linear correction/panels.

        These are not the encoded tanh features and never replace the primal
        residual, actual Jacobian, or exported model. The caller bounds rows
        and columns: O(Q*d*p*axis_orders + Q*k*number_of_jets), never Q-by-P.
        Optional prepared tables are local to one points/derivatives batch.
        """
        points = self.base._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        if prepared is None:
            prepared = self.prepare_ideal_columns(points, derivatives)
        elif (prepared['owner'] is not self or prepared['derivatives'] != derivatives
              or (prepared['points'] is not points and not np.array_equal(prepared['points'], points))):
            raise ValueError('Prepared ideal tables must belong to the same geometry, points, and derivatives')
        axis_tables = prepared['axis_tables']
        result = {}
        for derivative in derivatives:
            values = np.ones((len(points), len(indices)))
            for axis, order in enumerate(derivative):
                values *= axis_tables[axis, order][:, self.multiindices[indices, axis]]
            result[derivative] = values
        return result

    def evaluate(self, points, derivative=None):
        """Explicit dense diagnostic only; not used by streamed solvers."""
        order = self.base._derivative(derivative)
        return self.selected_columns(points, np.arange(self.size), (order,))[order]

    def ideal_tensor_plan(self, fields, rows, derivatives):
        from route2_box_tensor import tensor_product_plan
        return tensor_product_plan(self.dimension,self.degree,fields,rows,derivatives)

    def prepare_ideal_tensor_forward(self, coefficients):
        coefficients=np.asarray(coefficients,float)
        if coefficients.ndim!=2 or coefficients.shape[0]!=self.size or coefficients.shape[1]<1:
            raise ValueError('Ideal tensor coefficients must have coordinate-by-field shape')
        fields=coefficients.shape[1]
        plan=self.ideal_tensor_plan(fields,1,((0,)*self.dimension,))
        if not plan['enabled']:raise ValueError('Ideal coefficient tensor exceeds bounded allocation cap')
        dense=np.zeros((self.degree+1,)*self.dimension+(fields,))
        dense[tuple(self.multiindices.T)]=coefficients
        return dense

    def forward_ideal_tensor_jets(self, state, points, derivatives):
        from route2_box_tensor import forward_tensor_jets
        plan=self.ideal_tensor_plan(state.shape[-1],len(points),derivatives)
        if not plan['enabled'] or plan['rows']<len(points):
            raise ValueError('Ideal tensor point batch exceeds bounded workspace cap')
        return forward_tensor_jets(state,self.prepare_ideal_columns(points,derivatives))

    def prepare_ideal_tensor_adjoint(self, fields):
        plan=self.ideal_tensor_plan(fields,1,((0,)*self.dimension,))
        if not plan['enabled']:raise ValueError('Ideal coefficient tensor exceeds bounded allocation cap')
        return np.zeros((self.degree+1,)*self.dimension+(fields,))

    def accumulate_ideal_tensor_jets(self, cotangents, points, state):
        from route2_box_tensor import accumulate_tensor_adjoint
        derivatives=tuple(cotangents)
        plan=self.ideal_tensor_plan(state.shape[-1],len(points),derivatives)
        if not plan['enabled'] or plan['rows']<len(points):
            raise ValueError('Ideal tensor point batch exceeds bounded workspace cap')
        accumulate_tensor_adjoint(state,cotangents,self.prepare_ideal_columns(points,derivatives))

    def finish_ideal_tensor_adjoint(self, state):
        return state[tuple(self.multiindices.T)]

    def evaluate_many(self, points, derivatives):
        return self.selected_columns(points, np.arange(self.size), derivatives)

    def compile(self, coefficients):
        return self.base.compile(self.to_directional(coefficients))

    def torch_model(self, coefficients):
        return self.base.torch_model(self.to_directional(coefficients))
