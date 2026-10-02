"""Streamed direct profile coordinates for an ordinary one-hidden-layer tanh MLP.

Unknowns are one global constant and p nonconstant Gegenbauer coefficients
per supplied direction.  No angular-to-coordinate map or Q-by-P feature
matrix is constructed.  This is an operator primitive, not a PDE solver.
Directional coordinates can be redundant; the exact adjoint does not remove
that redundancy or guarantee well-conditioned optimization.
"""
from __future__ import annotations

from math import ceil, sqrt
import numpy as np
from numpy.polynomial import polynomial
from scipy.special import expit

from quill_boundary import encode
from solver.general_features import _tanh_derivative_polynomial


JET_PANEL_BUDGET_BYTES = 16*1024**2


def jet_batch_plan(centers, point_limit, order_count, fields, directions):
    """Conservative temporary-array plan, independent of total query count.

    Allow two sets of jet outputs (a generator's previous panel may still be
    referenced), twelve scalar work arrays, and field multiplication panels.
    One direction/point is the minimum; an exceptionally wide encoding can
    exceed the target even at that minimum, which the returned estimate shows.
    Prepared readouts/adjoints and full query output vectors are separate.
    """
    row_bytes = 8*((2*order_count+12+fields)*centers+fields)
    points = min(point_limit, 64, max(1, JET_PANEL_BUDGET_BYTES//row_bytes))
    direction_count = min(directions, 4, max(1, JET_PANEL_BUDGET_BYTES//(row_bytes*points)))
    return points, direction_count, row_bytes*points*direction_count


def gegenbauer_profiles(z, dimension, degree):
    """C_n^(dimension/2)(z), n=0,...,degree; supports complex contour points."""
    z = np.asarray(z)
    out = np.empty(z.shape + (degree+1,), dtype=np.result_type(z, float))
    out[..., 0] = 1.
    if degree:
        out[..., 1] = dimension*z
    parameter = dimension/2
    for n in range(2, degree+1):
        out[..., n] = (2*(n+parameter-1)*z*out[..., n-1]
                      -(n+2*parameter-2)*out[..., n-2])/n
    return out


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    if int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


class DirectionalProfileOperator:
    """Actual tanh features, streamed in bounded direction and point batches.

    ``directions`` is an arbitrary M-by-d collection of unit vectors, d>=2.
    The encoded scalar argument is ``s_m = omega_m @ ((x-midpoint)*scale)``.
    Every queried point must obey |s_m|<=1 for every supplied direction.
    Default midpoint=0 and scale=1 cover the unit ball.  For a physical box,
    midpoint=(lo+hi)/2 and scale=2/(hi-lo)/sqrt(d) cover the whole box.

    ``theta = [beta, b.ravel()]`` with b.shape=(M,p) stores the coefficients
    of C_1,...,C_p.  The p=0 constant profile is folded into exact bias beta.
    ``centers`` counts interior centers, including interval endpoints; the
    total per direction is L=centers+2*ceil(sqrt(centers)).

    For one field/jet, cost is O(M*L*p + Q*M*(L+d)); prepared readouts or
    adjoints retain O(M*L) storage, in addition to the O(L*p+M*d) model.
    There is no Q-by-M-by-p cache. Several fields share each tanh jet panel;
    direction and point batches obey ``jet_batch_plan`` rather than scaling
    with Q or M. Requested jets also retain their Q-by-field output arrays.
    Arithmetic/conditioning of this redundant parameterization must be
    studied separately from storage.
    """
    def __init__(self, directions, degree, centers=129, lam=.2,
                 midpoint=None, scale=None, block_size=128):
        directions = np.asarray(directions, dtype=float)
        if (directions.ndim != 2 or directions.shape[1] < 2
                or directions.shape[0] < 1 or not np.all(np.isfinite(directions))):
            raise ValueError('directions must be a finite nonempty M-by-d array, d>=2')
        if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-12, rtol=1e-12):
            raise ValueError('directions must be unit vectors; set physical scaling using scale')
        self.directions = directions.copy()
        self.direction_count, self.dimension = directions.shape
        self.degree = _positive_integer(degree, 'degree', 0)
        self.interior_centers = _positive_integer(centers, 'centers', 3)
        self.block_size = _positive_integer(block_size, 'block_size')
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError('lam must be finite and positive')
        self.midpoint = np.zeros(self.dimension) if midpoint is None else np.asarray(midpoint, float)
        self.scale = np.ones(self.dimension) if scale is None else np.asarray(scale, float)
        if self.scale.ndim == 0:
            self.scale = np.full(self.dimension, self.scale)
        if (self.midpoint.shape != (self.dimension,) or self.scale.shape != (self.dimension,)
                or not np.all(np.isfinite(self.midpoint))
                or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0)):
            raise ValueError('midpoint must be a finite d-vector and scale a positive scalar/d-vector')
        self.midpoint, self.scale = self.midpoint.copy(), self.scale.copy()
        self.physical_directions = self.directions*self.scale[None, :]
        self.size = 1+self.direction_count*self.degree
        self.halo_per_side = ceil(sqrt(self.interior_centers))
        self.encoding = encode(lambda z: gegenbauer_profiles(z, self.dimension, self.degree),
                               self.interior_centers-1, lam=lam, halo=self.halo_per_side)
        # Keep one encoding bank. The nonconstant columns are a read-only-by-
        # convention view; duplicating this bank wastes O(N*p) memory.
        self.weights = self.encoding.weights[:, 1:]
        self.anchor_values = gegenbauer_profiles(np.array(-1.), self.dimension, self.degree)[1:]
        self.anchor_z = self.encoding.gamma*(-1.-self.encoding.centers)
        self._anchor_tanh = np.tanh(self.anchor_z)
        anchor_exp = np.exp(-2*np.abs(self.anchor_z))
        self._anchor_tail = anchor_exp/(1+anchor_exp)
        self.centers_per_direction = len(self.encoding.centers)
        self.tanh_count = self.direction_count*self.centers_per_direction
        self.metrics = dict(backend='actual_tanh_directional_profiles',
                            dimension=self.dimension, profile_degree=self.degree,
                            directions=self.direction_count, neurons=self.tanh_count,
                            unknowns=self.size, interior_centers=self.interior_centers,
                            halo_per_side=self.halo_per_side,
                            geometry_map_bytes=0,
                            encoding_bank_bytes=self.encoding.weights.nbytes,
                            jet_anchor_bytes=self._anchor_tanh.nbytes+self._anchor_tail.nbytes,
                            jet_panel_budget_bytes=JET_PANEL_BUDGET_BYTES,
                            maximum_direction_batch=4, maximum_point_batch=64,
                            direction_array_bytes=self.directions.nbytes+self.physical_directions.nbytes)

    def pack(self, beta, b):
        b = np.asarray(b, float)
        if b.shape != (self.direction_count, self.degree):
            raise ValueError('b must have shape (direction_count, degree)')
        return self._theta(np.r_[float(beta), b.ravel()]).copy()

    def unpack(self, theta):
        theta = self._theta(theta)
        return float(theta[0]), theta[1:].reshape(self.direction_count, self.degree)

    def _theta(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.size,) or not np.all(np.isfinite(theta)):
            raise ValueError(f'theta must be a finite vector of length {self.size}')
        return theta

    def _points(self, points):
        points = np.asarray(points, float)
        if (points.ndim != 2 or points.shape[1] != self.dimension
                or not np.all(np.isfinite(points))):
            raise ValueError('points must be a finite Q-by-d array')
        return points

    def _derivative(self, derivative):
        if derivative is None:
            return (0,)*self.dimension
        a = np.asarray(derivative)
        if (a.shape != (self.dimension,) or not np.issubdtype(a.dtype, np.integer)
                or np.any(a < 0)):
            raise ValueError('derivative must contain d nonnegative integers')
        return tuple(int(v) for v in a)

    def _basis(self, z, order):
        """Scalar ridge derivatives, including gamma^order, in actual tanh."""
        t = np.tanh(z)
        if order == 0:
            # Same anchored tanh network, evaluated without subtracting two
            # saturated values.  The export below absorbs anchoring in bias.
            z0 = self.anchor_z[None, :]
            difference = t-np.tanh(z0)
            positive = (z >= 0) & (z0 >= 0)
            negative = (z <= 0) & (z0 <= 0)
            np.copyto(difference, 2*(expit(-2*z0)-expit(-2*z)), where=positive)
            np.copyto(difference, 2*(expit(2*z)-expit(2*z0)), where=negative)
            return difference
        e = np.exp(-2*np.abs(z))
        return ((self.encoding.gamma**order)*4*e/(1+e)**2
                *polynomial.polyval(t, _tanh_derivative_polynomial(order)))

    def _blocks(self, points, m, derivatives):
        direction = self.physical_directions[m]
        chain = {a: np.prod(direction**np.array(a)) for a in derivatives}
        orders = {sum(a) for a in derivatives}
        point_batch, _, _ = jet_batch_plan(self.centers_per_direction, self.block_size,
                                           len(orders), 1, 1)
        for start in range(0, len(points), point_batch):
            stop = min(start+point_batch, len(points))
            s = (points[start:stop]-self.midpoint)@direction
            if np.any(np.abs(s) > 1+2e-12):
                raise ValueError('point outside encoded projection band |omega dot z|<=1')
            z = self.encoding.gamma*(s[:, None]-self.encoding.centers[None, :])
            by_order = self._basis_many(z, orders)
            yield slice(start, stop), by_order, chain

    def _basis_many(self, z, orders):
        """Stable tanh jets sharing tanh and exponential evaluation.

        e/(1+e), e=exp(-2*abs(z)), is the small sigmoid tail on either
        half-line. Differences of these tails retain saturated tanh changes
        without subtracting two numbers rounded to +/-1. The same e also
        supplies sech(z)**2 for every positive derivative order.
        """
        if not orders:
            return {}
        t = np.tanh(z)
        e = np.exp(-2*np.abs(z))
        denominator = 1+e
        result = {}
        if 0 in orders:
            small = e/denominator
            difference = t-self._anchor_tanh
            np.copyto(difference, 2*(self._anchor_tail-small),
                      where=(z >= 0) & (self.anchor_z >= 0))
            np.copyto(difference, 2*(small-self._anchor_tail),
                      where=(z <= 0) & (self.anchor_z <= 0))
            result[0] = difference
        if any(order > 0 for order in orders):
            sech2 = 4*e/(denominator*denominator)
            for order in orders:
                if order:
                    result[order] = ((self.encoding.gamma**order)*sech2*
                        polynomial.polyval(t, _tanh_derivative_polynomial(order)))
        return result

    def _direction_blocks(self, points, derivatives, fields):
        """At most four directions and 64 points per stable jet panel."""
        orders = {sum(a) for a in derivatives}
        point_batch, direction_batch, _ = jet_batch_plan(self.centers_per_direction,
            self.block_size, len(orders), fields, self.direction_count)
        for first in range(0, self.direction_count, direction_batch):
            directions = slice(first, min(first+direction_batch, self.direction_count))
            w = self.physical_directions[directions]
            chain = {a: np.prod(w**np.asarray(a)[None, :], axis=1) for a in derivatives}
            for start in range(0, len(points), point_batch):
                rows = slice(start, min(start+point_batch, len(points)))
                projection = ((points[rows]-self.midpoint) @ w.T).T
                if np.any(np.abs(projection) > 1+2e-12):
                    raise ValueError('point outside encoded projection band |omega dot z|<=1')
                z = self.encoding.gamma*(projection[:, :, None]-self.encoding.centers)
                yield directions, rows, self._basis_many(z, orders), chain

    def forward(self, theta, points, derivative=None):
        a = self._derivative(derivative)
        return self.forward_jets(theta, points, [a])[a]

    def forward_jets(self, theta, points, derivatives):
        return self.forward_prepared_jets(self.prepare_forward(theta), points, derivatives)

    def prepare_forward(self, theta):
        """Compile one coefficient vector once for a complete batched sweep.

        O(M*H*p) work and O(M*H) state, H=centers including halos. The state
        contains ordinary neuron readouts, not point-by-feature evaluations.
        """
        beta, b = self.unpack(theta)
        readouts = np.empty((self.direction_count, self.centers_per_direction))
        anchors = np.empty(self.direction_count)
        for m in range(self.direction_count):
            readouts[m] = self.weights@b[m]
            anchors[m] = self.anchor_values@b[m]
        return dict(beta=beta, readouts=readouts, anchors=anchors)

    def forward_prepared_jets(self, prepared, points, derivatives):
        points = self._points(points)
        derivatives = tuple(dict.fromkeys(self._derivative(a) for a in derivatives))
        out = {a: np.full(len(points), prepared['beta'] if sum(a) == 0 else 0.) for a in derivatives}
        for m in range(self.direction_count):
            readout = prepared['readouts'][m]
            anchor = prepared['anchors'][m]
            for rows, basis, chain in self._blocks(points, m, derivatives):
                for a in derivatives:
                    out[a][rows] += chain[a]*(basis[sum(a)]@readout)
                    if sum(a) == 0:
                        out[a][rows] += anchor
        return out

    def prepare_forward_fields(self, coefficients):
        """Compile F output fields, retaining O(M*H*F) neuron readouts.

        Geometry and the scalar encoding bank are shared. Coefficients are
        an S-by-F matrix; there is no feature-by-point cache.
        """
        coefficients = np.asarray(coefficients, float)
        if (coefficients.ndim != 2 or coefficients.shape[0] != self.size
                or coefficients.shape[1] < 1 or not np.all(np.isfinite(coefficients))):
            raise ValueError(f'coefficients must be a finite {self.size}-by-F matrix, F>=1')
        fields = coefficients.shape[1]
        profiles = coefficients[1:].reshape(self.direction_count, self.degree, fields)
        readouts = np.empty((self.direction_count, self.centers_per_direction, fields))
        anchors = np.empty((self.direction_count, fields))
        for m in range(self.direction_count):
            readouts[m] = self.weights @ profiles[m]
            anchors[m] = self.anchor_values @ profiles[m]
        return dict(beta=coefficients[0].copy(), readouts=readouts, anchors=anchors,
                    fields=fields)

    def forward_prepared_fields_jets(self, prepared, points, derivatives):
        """Evaluate each tanh derivative panel once for all output fields."""
        points = self._points(points)
        derivatives = tuple(dict.fromkeys(self._derivative(a) for a in derivatives))
        fields = prepared['fields']
        out = {a: (np.broadcast_to(prepared['beta'], (len(points), fields)).copy()
                   if sum(a) == 0 else np.zeros((len(points), fields))) for a in derivatives}
        for directions, rows, basis, chain in self._direction_blocks(points, derivatives, fields):
            for a in derivatives:
                local = (basis[sum(a)] @ prepared['readouts'][directions])*chain[a][:, None, None]
                if sum(a) == 0:
                    local += prepared['anchors'][directions, None, :]
                out[a][rows] += np.sum(local, axis=0)
        return out

    def adjoint(self, cotangent, points, derivative=None):
        a = self._derivative(derivative)
        return self.adjoint_jets({a: cotangent}, points)

    def adjoint_jets(self, cotangents, points):
        """Euclidean transpose of stacked requested jets, not an inverse."""
        accumulator = self.prepare_adjoint()
        self.accumulate_adjoint_jets(cotangents, points, accumulator)
        return self.finish_adjoint(accumulator)

    def prepare_adjoint(self):
        """O(M*H) accumulator; apply the shared bank transpose only once."""
        return dict(neuron_gradient=np.zeros((self.direction_count, self.centers_per_direction)),
                    constant_sum=0.)

    def accumulate_adjoint_jets(self, cotangents, points, accumulator):
        points = self._points(points)
        cotangents = {self._derivative(a): np.asarray(y, float) for a, y in cotangents.items()}
        if any(y.shape != (len(points),) or not np.all(np.isfinite(y))
               for y in cotangents.values()):
            raise ValueError('each cotangent must be a finite vector with one entry per point')
        zero = (0,)*self.dimension
        constant_sum = float(np.sum(cotangents[zero])) if zero in cotangents else 0.
        accumulator['constant_sum'] += constant_sum
        for m in range(self.direction_count):
            neuron_gradient = accumulator['neuron_gradient'][m]
            for rows, basis, chain in self._blocks(points, m, tuple(cotangents)):
                for a, y in cotangents.items():
                    neuron_gradient += chain[a]*(basis[sum(a)].T@y[rows])

    def finish_adjoint(self, accumulator):
        out = np.zeros(self.size)
        out[0] = accumulator['constant_sum']
        bgrad = out[1:].reshape(self.direction_count, self.degree)
        for m in range(self.direction_count):
            bgrad[m] = (self.weights.T@accumulator['neuron_gradient'][m]
                        + accumulator['constant_sum']*self.anchor_values)
        return out

    def prepare_adjoint_fields(self, fields):
        """Shared O(M*H*F) cotangent state for F output fields."""
        fields = _positive_integer(fields, 'fields')
        return dict(neuron_gradient=np.zeros((self.direction_count, self.centers_per_direction, fields)),
                    constant_sum=np.zeros(fields), fields=fields)

    def accumulate_adjoint_fields_jets(self, cotangents, points, accumulator):
        """Transpose all Q-by-F jet cotangents using shared basis panels."""
        points = self._points(points)
        fields = accumulator['fields']
        cotangents = {self._derivative(a): np.asarray(y, float) for a, y in cotangents.items()}
        if any(y.shape != (len(points), fields) or not np.all(np.isfinite(y))
               for y in cotangents.values()):
            raise ValueError('each field cotangent must be a finite Q-by-F array')
        zero = (0,)*self.dimension
        if zero in cotangents:
            accumulator['constant_sum'] += np.sum(cotangents[zero], axis=0)
        for directions, rows, basis, chain in self._direction_blocks(points, tuple(cotangents), fields):
            for a, y in cotangents.items():
                accumulator['neuron_gradient'][directions] += (
                    basis[sum(a)].transpose(0, 2, 1) @ y[rows])*chain[a][:, None, None]

    def finish_adjoint_fields(self, accumulator):
        fields = accumulator['fields']
        out = np.empty((self.size, fields))
        out[0] = accumulator['constant_sum']
        profiles = out[1:].reshape(self.direction_count, self.degree, fields)
        for m in range(self.direction_count):
            profiles[m] = (self.weights.T @ accumulator['neuron_gradient'][m]
                           +self.anchor_values[:, None]*accumulator['constant_sum'][None, :])
        return out

    def selected_columns(self, points, indices, derivatives):
        """Only requested coefficient columns, for bounded local preconditioners.

        Callers control both point-batch and column-block sizes.  No complete
        Q-by-S array or global angular coordinate map is used internally.
        """
        points = self._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a vector of valid integer coefficient indices')
        derivatives = tuple(dict.fromkeys(self._derivative(a) for a in derivatives))
        out = {a: np.zeros((len(points), len(indices))) for a in derivatives}
        constant = indices == 0
        for a in derivatives:
            if sum(a) == 0:
                out[a][:, constant] = 1.
        nonconstant = np.flatnonzero(~constant)
        if not len(nonconstant):
            return out
        directions = (indices[nonconstant]-1)//self.degree
        modes = (indices[nonconstant]-1)%self.degree
        for m in np.unique(directions):
            mask = directions == m
            columns = nonconstant[mask]
            bank = self.weights[:, modes[mask]]
            anchor = self.anchor_values[modes[mask]]
            for rows, basis, chain in self._blocks(points, int(m), derivatives):
                for a in derivatives:
                    out[a][rows, columns] = chain[a]*(basis[sum(a)]@bank)
                    if sum(a) == 0:
                        out[a][rows, columns] += anchor
        return out

    def evaluate_many(self, points, derivatives):
        """Explicit dense comparison path; streamed solvers must not call this."""
        return self.selected_columns(points, np.arange(self.size), derivatives)

    def evaluate(self, points, derivative=None):
        """Explicit Q-by-S feature matrix, for small cached-control comparisons."""
        a = self._derivative(derivative)
        return self.evaluate_many(points, [a])[a]

    def compile(self, theta):
        """Return ordinary dense layer arrays, with a single global output bias."""
        beta, b = self.unpack(theta)
        readouts = b@self.weights.T
        # Combine actual readouts before absorbing anchoring into the bias.
        # longdouble is not necessarily wider than float64 on this platform.
        wide = np.longdouble
        bias = (wide(beta)+np.sum(b.astype(wide)*self.anchor_values.astype(wide)[None, :])
                -np.sum(readouts.astype(wide)*np.tanh(self.anchor_z).astype(wide)[None, :]))
        return dict(first_weights=np.repeat(self.encoding.gamma*self.physical_directions,
                                            self.centers_per_direction, axis=0),
                    first_bias=(-self.encoding.gamma*self.encoding.centers[None, :]
                                -self.encoding.gamma*(self.physical_directions@self.midpoint)[:, None]).ravel(),
                    output_weights=readouts.ravel()[None, :], output_bias=np.array([float(bias)]))

    def torch_model(self, theta):
        """An unmodified torch Sequential(Linear,Tanh,Linear), all float64."""
        import torch
        arrays = self.compile(theta)
        model = torch.nn.Sequential(torch.nn.Linear(self.dimension, self.tanh_count, dtype=torch.float64),
                                    torch.nn.Tanh(), torch.nn.Linear(self.tanh_count, 1, dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(arrays['first_weights']))
            model[0].bias.copy_(torch.from_numpy(arrays['first_bias']))
            model[2].weight.copy_(torch.from_numpy(arrays['output_weights']))
            model[2].bias.copy_(torch.from_numpy(arrays['output_bias']))
        return model
