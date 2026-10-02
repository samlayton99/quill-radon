"""Stable disk coordinates restricted to a declared two-dimensional box.

For bounds [a_i,b_i], let m_i=(a_i+b_i)/2 and h_i=(b_i-a_i)/2.
The map z_i=(x_i-m_i)/(sqrt(2)*h_i) sends the complete box into the
reference unit disk. Therefore |omega dot z|<=1 for every unit direction.
The physical enclosing shape is an ellipse with semiaxes sqrt(2)*h_i.

The disk normalization and analytic angular-to-ridge identity are unchanged.
Ideal modes are orthonormal on that enclosing disk, not on the box subset.
No monomial conversion is performed. All deployed fields remain actual tanh
networks; ideal Cartesian/Jacobi jets are only correction/preconditioner tools.
"""
from math import sqrt
import numpy as np
from route2_disk_profiles import DiskProfileOperator


class AffineDiskProfileOperator(DiskProfileOperator):
    def __init__(self,bounds,degree,centers=257,lam=.2,direction_count=None,block_size=128):
        bounds=np.asarray(bounds,float)
        if (bounds.shape!=(2,2) or not np.all(np.isfinite(bounds))
                or np.any(bounds[:,1]<=bounds[:,0])):
            raise ValueError('Affine disk coordinates require finite increasing bounds for exactly two dimensions')
        self.bounds=bounds.copy()
        halfwidth=(bounds[:,1]-bounds[:,0])/2
        super().__init__(degree,centers=centers,lam=lam,direction_count=direction_count,
            block_size=block_size,midpoint=bounds.mean(axis=1),scale=1/(sqrt(2)*halfwidth))
        self.metrics.update(backend='actual_tanh_affine_disk_modes',
            coordinate_chart='z_i=(x_i-midpoint_i)/(sqrt(2)*halfwidth_i)',
            chart_bounds=self.bounds.tolist(),
            reference_box_halfwidth=1/sqrt(2),
            orthogonality_scope='Ideal normalized enclosing disk area, not the restricted physical box',
            coordinate_scope='Stable disk angular coordinates on an affine enclosing ellipse; no monomial conversion and no fitted map')
