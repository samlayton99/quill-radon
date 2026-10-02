"""Analytical, nonredundant disk coordinates for the actual QUILL tanh MLP.

Degree n has n+1 angular modes. Their canonical directional profile readouts
come from a finite trigonometric identity, never a fit or matrix factorization.
The ideal profiles form an orthonormal basis under normalized disk area.
Actual features remain their tanh encodings: exact orthonormality is not
claimed after encoding, and restricting coefficients removes the additional
near-null directions of the redundant, approximately encoded dictionary.
"""
from __future__ import annotations

from math import sqrt
import numpy as np
from scipy.special import eval_gegenbauer, eval_jacobi, poch
from route2_directional_operator import DirectionalProfileOperator, _positive_integer


class DiskProfileOperator:
    """Canonical disk angular modes with an ordinary tanh forward pass.

    Let theta_m=pi*m/M, M>=p+1. For a mode (n,k), k=n,n-2,...,
    the exact identity is

        mean_m U_n(omega_m dot x)*cos(k*theta_m)
            = R_n^k(r)*cos(k*phi),

    with the analogous sine identity. The normalizing multiplier is
    sqrt(n+1) for k=0 and sqrt(2*(n+1)) otherwise. Degree zero is an exact
    global bias, so there are P=(p+1)*(p+2)/2 unknowns.

    Only the shared one-dimensional encoding bank, an M-by-(p+1) complex
    angular table, and O(P) labels are retained. Degree-wise conversion costs
    O(M*p^2), rather than storing a global M-by-p-by-P map. Prepared forward
    and adjoint states additionally require O(M*N) neuron storage.

    Affine midpoint/scale may be used as in DirectionalProfileOperator, but
    orthonormality refers to normalized area on the reference unit disk.
    """
    def __init__(self, degree, centers=257, lam=.2, direction_count=None,
                 block_size=128, midpoint=None, scale=None):
        self.degree = _positive_integer(degree, 'degree', 0)
        self.direction_count = (self.degree+1 if direction_count is None else
                                _positive_integer(direction_count, 'direction_count'))
        if self.direction_count < self.degree+1:
            raise ValueError('direction_count must be at least degree+1 for the angular identity')
        angles = np.pi*np.arange(self.direction_count)/self.direction_count
        self.angular_table = np.exp(1j*angles[:, None]*np.arange(self.degree+1)[None, :])
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self.base = DirectionalProfileOperator(directions, self.degree, centers=centers,
            lam=lam, block_size=block_size, midpoint=midpoint, scale=scale)
        self.dimension, self.block_size = 2, self.base.block_size
        self.size = (self.degree+1)*(self.degree+2)//2
        self.directions = self.base.directions
        self.tanh_count = self.base.tanh_count
        self.interior_centers = self.base.interior_centers
        self.halo_per_side = self.base.halo_per_side
        self.midpoint, self.scale = self.base.midpoint, self.base.scale
        self.degrees, self.harmonics, self.is_sine, self.normalizations = [], [], [], []
        self.degree_slices = []
        for n in range(self.degree+1):
            start = len(self.degrees)
            for k in range(n, -1, -2):
                for sine in ((False, True) if k else (False,)):
                    self.degrees.append(n)
                    self.harmonics.append(k)
                    self.is_sine.append(sine)
                    self.normalizations.append(sqrt(n+1) if k == 0 else sqrt(2*(n+1)))
            self.degree_slices.append(slice(start, len(self.degrees)))
        self.degrees = np.asarray(self.degrees, int)
        self.harmonics = np.asarray(self.harmonics, int)
        self.is_sine = np.asarray(self.is_sine, bool)
        self.normalizations = np.asarray(self.normalizations, float)
        assert len(self.degrees) == self.size
        self.metrics = dict(self.base.metrics,
            backend='actual_tanh_analytical_disk_modes', unknowns=self.size,
            redundant_profile_unknowns=self.base.size, geometry_map_bytes=0,
            angular_table_bytes=self.angular_table.nbytes,
            mode_label_bytes=sum(a.nbytes for a in (self.degrees, self.harmonics,
                                                    self.is_sine, self.normalizations)),
            coordinate_scope='Analytical disk angular coordinates; orthonormal for ideal U_n profiles, actual tanh encoding retains approximation error; no fitted conversion map')

    def _coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if coefficients.shape != (self.size,) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f'coefficients must be a finite vector of length {self.size}')
        return coefficients

    def _angular_columns(self, indices):
        modes = self.angular_table[:, self.harmonics[indices]]
        return (np.where(self.is_sine[indices][None, :], modes.imag, modes.real)
                *self.normalizations[indices][None, :]/self.direction_count)

    def to_directional(self, coefficients):
        """Analytical canonical readouts, no least squares or stored dense map."""
        coefficients = self._coefficients(coefficients)
        profiles = np.empty((self.direction_count, self.degree))
        for n in range(1, self.degree+1):
            indices = np.arange(self.degree_slices[n].start, self.degree_slices[n].stop)
            profiles[:, n-1] = self._angular_columns(indices) @ coefficients[indices]
        return self.base.pack(coefficients[0], profiles)

    def directional_transpose(self, cotangent):
        """Euclidean transpose of to_directional, not a fitted inverse."""
        bias, profiles = self.base.unpack(cotangent)
        result = np.empty(self.size)
        result[0] = bias
        for n in range(1, self.degree+1):
            indices = np.arange(self.degree_slices[n].start, self.degree_slices[n].stop)
            result[indices] = self._angular_columns(indices).T @ profiles[:, n-1]
        return result

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
        """Only requested columns, sharing each direction's tanh evaluations.

        Temporary readout panels have H-by-k entries for k requested columns;
        the geometry-to-all-coordinate map is never constructed.
        """
        points = self.base._points(points)
        original_indices = np.asarray(indices)
        if (original_indices.ndim != 1 or not np.issubdtype(original_indices.dtype, np.integer)
                or np.any(original_indices < 0) or np.any(original_indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        indices = original_indices.astype(int, copy=False)
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        result = {d: np.zeros((len(points), len(indices))) for d in derivatives}
        positive_positions = np.flatnonzero(self.degrees[indices] > 0)
        constant_positions = np.flatnonzero(self.degrees[indices] == 0)
        for d in derivatives:
            if sum(d) == 0:
                result[d][:, constant_positions] = 1.
        if not len(positive_positions):
            return result
        modes = indices[positive_positions]
        degrees = self.degrees[modes]-1
        # Only a bounded requested-column angular panel, not all P columns.
        angular = self._angular_columns(modes)
        for m in range(self.direction_count):
            readouts = self.base.weights[:, degrees]*angular[m][None, :]
            anchors = self.base.anchor_values[degrees]*angular[m]
            for rows, basis, chain in self.base._blocks(points, m, derivatives):
                for d in derivatives:
                    values = chain[d]*(basis[sum(d)] @ readouts)
                    if sum(d) == 0:
                        values += anchors[None, :]
                    result[d][rows, positive_positions] += values
        return result

    def selected_ideal_columns(self, points, indices, derivatives):
        """Stable Cartesian Zernike jets for analytical correction operators.

        R_n^k(r) exp(ik phi) = (x+iy)^k P_{(n-k)/2}^{(0,k)}(2r^2-1).
        Differentiating this identity avoids polar singularities and angular
        quadrature. No polynomial coefficient expansion is formed. Orders
        above two retain the independent ridge identity below. These panels
        never replace the actual neural state, residual, or exported field.
        """
        points = self.base._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        if any(sum(d)>2 for d in derivatives):
            return self._selected_ideal_columns_ridge(points, indices, derivatives)
        if not len(indices) or not derivatives:
            return {d:np.zeros((len(points),len(indices))) for d in derivatives}
        xy=(points-self.midpoint)*self.scale
        z=(xy[:,0]+1j*xy[:,1])[:,None]
        k=self.harmonics[indices][None,:]
        radial_degree=((self.degrees[indices]-self.harmonics[indices])//2)[None,:]
        t=2*np.sum(xy**2,axis=1)[:,None]-1
        radial=eval_jacobi(radial_degree,0,k,t)
        highest=max(map(sum,derivatives))
        solid=z**k
        if highest>=1:
            radial1=np.where(radial_degree>=1,
                .5*(radial_degree+k+1)*eval_jacobi(np.maximum(radial_degree-1,0),1,k+1,t),0.)
            solid1=k*z**np.maximum(k-1,0)
        if highest>=2:
            radial2=np.where(radial_degree>=2,
                .25*(radial_degree+k+1)*(radial_degree+k+2)*
                eval_jacobi(np.maximum(radial_degree-2,0),2,k+2,t),0.)
            solid2=k*(k-1)*z**np.maximum(k-2,0)
        result={}
        for derivative in derivatives:
            order=sum(derivative)
            if order==0:
                value=solid*radial
            elif order==1:
                axis=derivative.index(1)
                value=(1j**axis)*solid1*radial+solid*radial1*(4*xy[:,axis,None])
            else:
                axes=[axis for axis,count in enumerate(derivative) for _ in range(count)]
                a,b=axes
                radial_a=radial1*(4*xy[:,a,None])
                radial_b=radial1*(4*xy[:,b,None])
                radial_ab=16*xy[:,a,None]*xy[:,b,None]*radial2+(4*radial1 if a==b else 0.)
                value=((1j**(a+b))*solid2*radial+
                       (1j**a)*solid1*radial_b+(1j**b)*solid1*radial_a+solid*radial_ab)
            result[derivative]=(np.where(self.is_sine[indices][None,:],value.imag,value.real)*
                self.normalizations[indices][None,:]*np.prod(self.scale**np.asarray(derivative)))
        return result

    def _selected_ideal_columns_ridge(self, points, indices, derivatives):
        """Independent angular-identity oracle and arbitrary-order fallback.

        Angular identities are evaluated with ideal Gegenbauer profiles.
        Panels are bounded by caller point/column counts; no neuron evaluations
        or complete Q-by-P feature array are required.
        """
        points = self.base._points(points)
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.size)):
            raise ValueError('indices must be a one-dimensional integer array in range')
        derivatives = tuple(dict.fromkeys(self.base._derivative(d) for d in derivatives))
        result = {d: np.zeros((len(points), len(indices))) for d in derivatives}
        constant = self.degrees[indices] == 0
        for derivative in derivatives:
            if sum(derivative) == 0:
                result[derivative][:, constant] = 1.
        positive = np.flatnonzero(~constant)
        if not len(positive):
            return result
        modes = indices[positive]
        degrees, inverse = np.unique(self.degrees[modes], return_inverse=True)
        angular = self._angular_columns(modes)
        for m in range(self.direction_count):
            direction = self.base.physical_directions[m]
            s = (points-self.midpoint) @ direction
            profiles = {}
            for order in {sum(d) for d in derivatives}:
                values = np.zeros((len(points), len(degrees)))
                active = degrees >= order
                values[:, active] = (2**order*poch(1., order)*eval_gegenbauer(
                    degrees[None, active]-order, 1.+order, s[:, None]))
                profiles[order] = values[:, inverse]
            for derivative in derivatives:
                chain = np.prod(direction**np.asarray(derivative))
                result[derivative][:, positive] += chain*profiles[sum(derivative)]*angular[m]
        return result

    def evaluate(self, points, derivative=None):
        """Explicit Q-by-P matrix only when deliberately requested by caller."""
        order = self.base._derivative(derivative)
        return self.selected_columns(points, np.arange(self.size), (order,))[order]

    def evaluate_many(self, points, derivatives):
        return self.selected_columns(points, np.arange(self.size), derivatives)

    def compile(self, coefficients):
        return self.base.compile(self.to_directional(coefficients))

    def torch_model(self, coefficients):
        return self.base.torch_model(self.to_directional(coefficients))
