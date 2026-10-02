"""Problem declarations: equations and constraints, without sampling boilerplate.

Residual objects expose .coordinates/.fields/.parameters and .block(...), as
provided by the optional DifferentialResidual symbolic frontend. This layer
does not inspect equation identities or choose a named-PDE algorithm.
"""
from dataclasses import dataclass
from math import comb,ceil
import numpy as np
from .general_adaptive import ResidualDeclaration,solve_declared


def trace_sample_count(degree,ambient_dimension,oversampling=2):
    """Samples per flat codimension-one patch for a total-degree basis.

    A degree-p polynomial restricted to a flat boundary lives in dimension
    comb(p+d-1,d-1). This is a scaling rule, not a certificate for arbitrary
    curved surfaces or collocation stability.
    """
    if degree<0 or ambient_dimension<1 or oversampling<=0:
        raise ValueError('Nonnegative degree, positive dimension/oversampling required')
    return max(1,ceil(oversampling*comb(degree+ambient_dimension-1,ambient_dimension-1)))


@dataclass
class Condition:
    name: str
    residual: object
    location: str='boundary'
    points: object=None
    axis: int|None=None
    value: float|None=None
    selector: object=None
    weight: float=1.
    scale: object=1.
    relation: str='eq'

    def sample(self,domain,count,seed):
        if self.points is not None:
            points=np.asarray(self.points,dtype=float)
            if self.selector is not None:points=points[self.selector(points)]
            if not len(points):raise ValueError('Condition has no points after selection')
            return points
        if self.location not in ('boundary','interior','slice'):
            raise ValueError('location must be boundary, interior, or slice')
        if self.location=='slice':
            if self.axis is None or not 0<=self.axis<domain.dimension or self.value is None:
                raise ValueError('Slice requires a coordinate axis and value')
            if not domain.bounds[self.axis,0]<=self.value<=domain.bounds[self.axis,1]:
                raise ValueError('Slice is outside the domain bounds')
        found=[];total=0
        for attempt in range(30):
            n=max(64,2*(count-total))
            z=(domain.boundary(n,seed+104729*attempt).points if self.location=='boundary'
               else domain.interior(n,seed+104729*attempt))
            if self.location=='slice':
                z[:,self.axis]=self.value
                if domain.levelset is not None:z=z[domain._level(z)<=1e-12]
            if self.selector is not None:z=z[np.asarray(self.selector(z),dtype=bool)]
            found.append(z);total+=len(z)
            if total>=count:return np.concatenate(found)[:count]
        raise ValueError('Condition selector/slice has insufficient sampleable points')


class GeneralProblem:
    def __init__(self,domain,equation,conditions,parameter_initial=(),parameter_bounds=None):
        self.domain,self.equation=domain,equation
        self.conditions=tuple(conditions)
        self.parameter_initial=parameter_initial
        self.parameter_bounds=parameter_bounds
        if len(equation.coordinates)!=domain.dimension:
            raise ValueError('Equation coordinates must match the domain dimension')
        for condition in self.conditions:
            r=condition.residual
            if (r.coordinates!=equation.coordinates or r.fields!=equation.fields
                    or r.parameters!=equation.parameters):
                raise ValueError('Equations and conditions must share coordinate, field and parameter declarations')
        if len(parameter_initial)!=len(equation.parameters):
            raise ValueError('Give an initial value for each unknown physical parameter')

    def blocks(self,count,seed):
        blocks=[self.equation.block('equation',self.domain.interior(count,seed))]
        # Infer an approximate total degree from the common interior sampling
        # density, then scale each boundary patch by its own trace dimension.
        # In four inputs sqrt(count) grows as p²; trace dimension grows as p³.
        d=self.domain.dimension;degree=0
        while comb(degree+d,d)<max(1,count/6):degree+=1
        per_patch=trace_sample_count(degree,d)
        for i,condition in enumerate(self.conditions):
            n=max(80,8*int(np.sqrt(count)),per_patch)
            if condition.location=='boundary':n=max(n,2*d*per_patch)
            points=condition.sample(self.domain,n,seed+7919*(i+1))
            blocks.append(condition.residual.block(condition.name,points,
                weight=condition.weight,scale=condition.scale,relation=condition.relation))
        return blocks

    def declaration(self):
        return ResidualDeclaration(self.domain,self.blocks,len(self.equation.fields),
                                   self.parameter_initial,self.parameter_bounds)

    def solve(self,**options):
        return solve_declared(self.declaration(),**options)

    def solve_ridge(self,**options):
        """Solve through an actual flat tanh MLP, with fresh raw-export checks.

        The legacy .solve() product-feature experiments remain reproducible.
        This explicitly named entry point is the ordinary-MLP route2 method.
        """
        from .route2 import solve_route2
        return solve_route2(self.declaration(),**options)
