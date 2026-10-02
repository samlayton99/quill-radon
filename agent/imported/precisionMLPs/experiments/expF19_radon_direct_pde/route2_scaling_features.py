"""Declared active-coordinate prior for an ordinary flat ridge MLP.

This is not learned dimensionality reduction: the user supplies active axes.
The PDE determines all readout coordinates from zero. Export remains a single
Linear/Tanh/Linear network with zeros in inactive input-weight columns.
"""
import time
from itertools import combinations, product
from math import ceil, sqrt
import numpy as np
from solver.ball_ridge_features import BallRidgeFeatures, analytic_profile_map, gegenbauer_profiles
from solver.general_features import _compositions
from quill_boundary import encode


def fourth_order_sphere_rule(dimension):
    """Even-polynomial cubature through degree four with d**2 directions.

    Folded axis weights are (4-d)/(d(d+2)); each folded pair-diagonal
    weight is 2/(d(d+2)). These reproduce E[w_i^2]=1/d,
    E[w_i^4]=3/(d(d+2)), E[w_i^2 w_j^2]=1/(d(d+2)).
    Weights are signed above dimension four, with absolute sum <3.
    Folding does NOT preserve globally odd integrands.
    """
    d = int(dimension)
    if d < 2 or d != dimension:
        raise ValueError('dimension must be an integer >=2')
    directions, weights = [], []
    for i in range(d):
        v = np.zeros(d); v[i] = 1.
        directions.append(v); weights.append((4-d)/(d*(d+2)))
    for i in range(d):
        for j in range(i+1,d):
            for sign in (-1,1):
                v = np.zeros(d); v[i] = 1/sqrt(2); v[j] = sign/sqrt(2)
                directions.append(v); weights.append(2/(d*(d+2)))
    return np.array(directions), np.array(weights)


def sixth_order_sphere_rule(dimension):
    """Even-degree-six rule on axes, pair diagonals, and triple diagonals.

    All points have norm one. Contracting degree-six monomials by sum w_i²=1
    yields the lower moments. The three orbit weights match the three even
    partitions of six, (6), (4,2), (2,2,2). Before antipodal folding they are
    (d²-9d+38)/(4D), 2(5-d)/D, 27/(8D), D=d(d+2)(d+4).
    This is a standard fully symmetric/Smolyak cubature family, not fitted
    to any target or PDE. Signed total variation stays below nine.
    """
    d=int(dimension)
    if d < 2 or d != dimension:
        raise ValueError('dimension must be an integer >=2')
    den=d*(d+2)*(d+4)
    orbit_weights=((d*d-9*d+38)/(4*den),2*(5-d)/den,27/(8*den))
    directions,weights=[],[]
    for count in range(1,min(d,3)+1):
        for axes in combinations(range(d),count):
            for tail in product((-1,1),repeat=count-1):
                v=np.zeros(d);v[list(axes)]=np.array((1,)+tail)/sqrt(count)
                directions.append(v);weights.append(2*orbit_weights[count-1])
    return np.array(directions),np.array(weights)


class LowDegreeSparseRidgeFeatures(BallRidgeFeatures):
    """Same analytic ridge construction with non-tensor degree-4/6 rules."""
    def __init__(self, bounds, degree=2, centers=129, lam=.2):
        if degree not in (0,1,2,3):
            raise ValueError('The sparse angular rule only supports degree <=3')
        started = time.perf_counter()
        self.bounds = np.asarray(bounds,float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or not np.all(np.isfinite(self.bounds)) or np.any(np.diff(self.bounds,axis=1)<=0):
            raise ValueError('bounds must be finite increasing pairs')
        self.degree, self.dimension = int(degree), len(self.bounds)
        self.midpoint = self.bounds.mean(axis=1)
        self.scale = 2/(self.bounds[:,1]-self.bounds[:,0])/sqrt(self.dimension)
        self.multiindices = np.array([v for n in range(degree+1) for v in _compositions(n,self.dimension)])
        self.size = len(self.multiindices)
        rule=sixth_order_sphere_rule if degree==3 else fourth_order_sphere_rule
        self.directions, self.angular_weights = rule(self.dimension)
        self.profile_map = analytic_profile_map(self.dimension,degree,self.directions,self.angular_weights,self.multiindices)
        self.encoding = encode(lambda z:gegenbauer_profiles(z,self.dimension,degree),centers-1,lam=lam,halo=ceil(sqrt(centers)))
        self.encoding.evaluation_mode='anchored'
        self.encoding.anchor_x=-1.
        self.encoding.anchor_value=gegenbauer_profiles(np.array([-1.]),self.dimension,degree)[0]
        self.physical_directions=self.directions*self.scale
        self.metrics=dict(backend='actual_tanh_ball_ridges_sparse_rule', dimension=self.dimension,
            degree=degree,coefficient_count=self.size,directions=len(self.directions),interior_centers=centers,
            halo_per_side=ceil(sqrt(centers)),lam=lam,tanh_count=len(self.directions)*len(self.encoding.centers),
            product_gates=False,polynomials_in_forward=False,construction_uses_target_data=False,
            angular_weights_total_variation=float(abs(self.angular_weights).sum()),
            angular_exactness='Globally even polynomials through degree '+str(6 if degree==3 else 4),
            construction_seconds=time.perf_counter()-started)


class ActiveAxisRidgeFeatures:
    def __init__(self, bounds, active_axes, degree, centers=129, lam=.2):
        self.bounds = np.asarray(bounds, float)
        self.dimension = len(self.bounds)
        self.active_axes = tuple(int(a) for a in active_axes)
        if len(set(self.active_axes)) != len(self.active_axes) or any(a < 0 or a >= self.dimension for a in self.active_axes):
            raise ValueError('active_axes must contain distinct valid axes')
        self.inner = BallRidgeFeatures(self.bounds[list(self.active_axes)], degree, centers, lam)
        self.degree, self.size = self.inner.degree, self.inner.size
        self.multiindices = np.zeros((self.size, self.dimension), dtype=int)
        self.multiindices[:, self.active_axes] = self.inner.multiindices
        self.metrics = dict(self.inner.metrics, ambient_dimension=self.dimension,
                            active_axes=self.active_axes,
                            declared_prior='Solution depends only on supplied active axes; not learned')

    def evaluate(self, points, derivative=None):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError('points must be Q-by-ambient_dimension')
        derivative = (0,)*self.dimension if derivative is None else tuple(derivative)
        if len(derivative) != self.dimension or any(int(a) != a or a < 0 for a in derivative):
            raise ValueError('Invalid derivative')
        if any(v and a not in self.active_axes for a, v in enumerate(derivative)):
            return np.zeros((len(points), self.size))
        return self.inner.evaluate(points[:, self.active_axes], tuple(derivative[a] for a in self.active_axes))

    def compile(self, coefficients):
        result = self.inner.compile(coefficients)
        first = np.zeros((len(result['first_bias']), self.dimension))
        first[:, self.active_axes] = result['first_weights']
        result['first_weights'] = first
        return result

    def torch_model(self, coefficients):
        import torch
        arrays = self.compile(coefficients)
        width, fields = arrays['output_weights'].shape
        model = torch.nn.Sequential(torch.nn.Linear(self.dimension, width, dtype=torch.float64),
                                    torch.nn.Tanh(), torch.nn.Linear(width, fields, dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(arrays['first_weights']))
            model[0].bias.copy_(torch.from_numpy(arrays['first_bias']))
            model[2].weight.copy_(torch.from_numpy(arrays['output_weights'].T))
            model[2].bias.copy_(torch.from_numpy(arrays['output_bias']))
        return model.requires_grad_(False)
