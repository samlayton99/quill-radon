"""Native tanh corrections that vanish on a prescribed initial plane.

This is a geometry/constraint primitive, independent of the PDE or target.
For an affine-disk polynomial q, first form (z_y-z_initial)q analytically in
disk coordinates. Encode that polynomial into ordinary tanh neurons F_q.
The actual correction is F_q(x,t)-F_q(x,t_initial), implemented by pairing
each neuron with its restriction to the initial plane. No products, custom
activation, polynomial evaluation, or time stepping occur in the export.

Multiplication is a sparse exact basis identity, not a fitted conversion.
The ideal space is all degree<=p polynomials with zero initial trace; actual
features approximate that space and have the paired initial-trace property.
An independently constructed, prescribed initial-data MLP can be added by
concatenating its hidden neurons. It must never contain interior truth.
"""
from __future__ import annotations

from math import sqrt
import numpy as np
from scipy import sparse
import torch
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_disk_profiles import DiskProfileOperator


def multiply_y_minus_trace(source, target, initial_normalized):
    """Sparse map for (y-y0)q in normalized real Zernike coordinates.

    V_n^m=R_n^|m|(r) exp(i*m*theta), m signed. Then
    z V_n^m = (n+m+2)/(2n+2) V_(n+1)^(m+1)
             +(n-m)/(2n+2) V_(n-1)^(m+1).
    Conjugation gives the zbar identity; y=(z-zbar)/(2i).
    Real modes have normalizations sqrt(n+1) for m=0, sqrt(2n+2)
    otherwise. Neither this conversion nor its transpose solves a system.
    """
    lookup={(int(n),int(k),bool(s)):i for i,(n,k,s) in enumerate(
        zip(target.degrees,target.harmonics,target.is_sine))}
    rows=[];columns=[];values=[]
    for j,(n,k,s,norm) in enumerate(zip(source.degrees,source.harmonics,
                                        source.is_sine,source.normalizations)):
        n,k=int(n),int(k)
        terms=({0:complex(norm)} if k==0 else
               {k:norm/(2j),-k:-norm/(2j)} if s else {k:norm/2,-k:norm/2})
        out={}
        for m,value in terms.items():
            for sign in (1,-1):
                factor=sign/(2j)
                for degree,weight in ((n+1,(n+sign*m+2)/(2*(n+1))),
                                      (n-1,(n-sign*m)/(2*(n+1)))):
                    if weight:
                        key=(degree,m+sign)
                        out[key]=out.get(key,0)+factor*weight*value
        for degree,harmonic in sorted({(a,abs(b)) for a,b in out}):
            positive=out.get((degree,harmonic),0)
            negative=out.get((degree,-harmonic),0)
            for sine in ((False,True) if harmonic else (False,)):
                i=lookup[(degree,harmonic,sine)]
                value=(1j*(positive-negative) if sine else
                       positive+negative if harmonic else positive)/target.normalizations[i]
                if abs(np.imag(value))>1e-13:
                    raise ArithmeticError('Real Zernike multiplication produced an imaginary coefficient')
                if np.real(value):rows.append(i);columns.append(j);values.append(float(np.real(value)))
        if initial_normalized:
            rows.append(lookup[(n,k,bool(s))]);columns.append(j);values.append(-float(initial_normalized))
    return sparse.coo_matrix((values,(rows,columns)),shape=(target.size,source.size)).tocsr()


class InitialPlaneProfileOperator(DiskProfileOperator):
    """Two-input (space,time) affine-disk correction, zero at t_initial.

    There are p(p+1)/2 independent ideal coordinates at degree p, rather
    than retaining a redundant zero-trace projection of all disk modes.
    Features and adjoints always use actual tanh sums. Selected ideal panels
    are explicitly named and are only approximate-Jacobian/preconditioner tools.
    """
    def __init__(self,bounds,degree,centers=257,lam=.2,initial_time=None,block_size=64):
        if int(degree)!=degree or degree<1:raise ValueError('degree must be a positive integer')
        self.full=AffineDiskProfileOperator(bounds,degree,centers,lam,block_size=block_size)
        source=AffineDiskProfileOperator(bounds,degree-1,centers,lam,block_size=block_size)
        self.initial_time=float(self.full.bounds[1,0] if initial_time is None else initial_time)
        if not np.isfinite(self.initial_time):raise ValueError('initial_time must be finite')
        initial_z=(self.initial_time-self.full.midpoint[1])*self.full.scale[1]
        self.multiplication=multiply_y_minus_trace(source,self.full,initial_z)
        self.bounds=self.full.bounds.copy()
        self.degree=int(degree);self.size=source.size;self.dimension=2
        self.degrees=source.degrees;self.harmonics=source.harmonics
        self.is_sine=source.is_sine;self.normalizations=source.normalizations
        self.base=self.full.base;self.block_size=block_size
        self.midpoint=self.full.midpoint;self.scale=self.full.scale
        self.directions=self.full.directions;self.direction_count=self.full.direction_count
        self.interior_centers=self.full.interior_centers;self.halo_per_side=self.full.halo_per_side
        self.tanh_count=2*self.full.tanh_count
        self.metrics=dict(self.full.metrics,backend='actual_tanh_initial_plane_difference',
            unknowns=self.size,tanh_count=self.tanh_count,neurons=self.tanh_count,
            trace_time=self.initial_time,sparse_multiplication_entries=self.multiplication.nnz,
            constraint='F_q(x,t)-F_q(x,t_initial); prescribed initial-data model added separately',
            parity_warning='Trace subtraction and affine factor mix time parity; do not infer disk parity blocks',
            coordinate_scope='Independent ideal zero-initial-trace polynomial space; actual paired tanh encoding')

    def _anchor(self,points):
        result=np.array(points,dtype=float,copy=True);result[:,1]=self.initial_time
        return result

    def to_directional(self,coefficients):
        return self.full.to_directional(self.multiplication@self._coefficients(coefficients))

    def directional_transpose(self,cotangent):
        return self.multiplication.T@self.full.directional_transpose(cotangent)

    def prepare_forward_fields(self,coefficients):
        coefficients=np.asarray(coefficients,float)
        if coefficients.ndim!=2 or coefficients.shape[0]!=self.size:raise ValueError('Wrong coefficient shape')
        return self.full.prepare_forward_fields(self.multiplication@coefficients)

    def forward_prepared_jets(self,prepared,points,derivatives):
        output=self.full.forward_prepared_jets(prepared,points,derivatives)
        trace_orders=tuple(order for order in derivatives if order[1]==0)
        if trace_orders:
            traces=self.full.forward_prepared_jets(prepared,self._anchor(points),trace_orders)
            for order in trace_orders:output[order]-=traces[order]
        return output

    def forward_prepared_fields_jets(self,prepared,points,derivatives):
        output=self.full.forward_prepared_fields_jets(prepared,points,derivatives)
        trace_orders=tuple(order for order in derivatives if order[1]==0)
        if trace_orders:
            traces=self.full.forward_prepared_fields_jets(prepared,self._anchor(points),trace_orders)
            for order in trace_orders:output[order]-=traces[order]
        return output

    def accumulate_adjoint_jets(self,cotangents,points,accumulator):
        self.full.accumulate_adjoint_jets(cotangents,points,accumulator)
        traces={order:-values for order,values in cotangents.items() if order[1]==0}
        if traces:self.full.accumulate_adjoint_jets(traces,self._anchor(points),accumulator)

    def accumulate_adjoint_fields_jets(self,cotangents,points,accumulator):
        self.full.accumulate_adjoint_fields_jets(cotangents,points,accumulator)
        traces={order:-values for order,values in cotangents.items() if order[1]==0}
        if traces:self.full.accumulate_adjoint_fields_jets(traces,self._anchor(points),accumulator)

    def _columns(self,points,indices,derivatives,ideal):
        indices=np.asarray(indices)
        if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=self.size):
            raise ValueError('Invalid selected coordinate indices')
        matrix=self.multiplication[:,indices]
        active=np.unique(matrix.nonzero()[0]);small=matrix[active].toarray()
        method=self.full.selected_ideal_columns if ideal else self.full.selected_columns
        values=method(points,active,derivatives)
        trace_orders=tuple(order for order in derivatives if order[1]==0)
        traces=method(self._anchor(points),active,trace_orders) if trace_orders else {}
        return {order:(values[order]-traces[order] if order in traces else values[order])@small
                for order in derivatives}

    def selected_columns(self,points,indices,derivatives):
        return self._columns(points,indices,derivatives,False)

    def selected_ideal_columns(self,points,indices,derivatives):
        return self._columns(points,indices,derivatives,True)

    def compile(self,coefficients):
        original=self.full.compile(self.multiplication@self._coefficients(coefficients))
        weights=np.asarray(original['first_weights'])
        biases=np.asarray(original['first_bias'])
        anchor_weights=weights.copy();anchor_weights[:,1]=0.
        anchor_biases=biases+weights[:,1]*self.initial_time
        output=np.asarray(original['output_weights']).reshape(-1)
        return dict(first_weights=np.concatenate([weights,anchor_weights]),
                    first_bias=np.r_[biases,anchor_biases],output_weights=np.r_[output,-output],
                    output_bias=np.array(0.))

    def torch_model(self,coefficients):
        arrays=self.compile(coefficients)
        model=torch.nn.Sequential(torch.nn.Linear(2,self.tanh_count,dtype=torch.float64),
            torch.nn.Tanh(),torch.nn.Linear(self.tanh_count,1,dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(arrays['first_weights']))
            model[0].bias.copy_(torch.from_numpy(arrays['first_bias']))
            model[2].weight.copy_(torch.from_numpy(arrays['output_weights'][None,:]))
            model[2].bias.zero_()
        return model.requires_grad_(False)


def concatenate_scalar_mlps(*models):
    """Sum ordinary scalar MLPs as one ordinary MLP; no inference wrapper."""
    if not models:raise ValueError('At least one model is required')
    dimension=models[0][0].in_features
    for model in models:
        if ([type(layer) for layer in model]!=[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
                or model[0].in_features!=dimension or model[2].out_features!=1
                or model[0].weight.dtype!=torch.float64):
            raise ValueError('Models must be matching scalar ordinary float64 tanh MLPs')
    width=sum(model[0].out_features for model in models)
    result=torch.nn.Sequential(torch.nn.Linear(dimension,width,dtype=torch.float64),
        torch.nn.Tanh(),torch.nn.Linear(width,1,dtype=torch.float64))
    with torch.no_grad():
        result[0].weight.copy_(torch.cat([model[0].weight for model in models]))
        result[0].bias.copy_(torch.cat([model[0].bias for model in models]))
        result[2].weight.copy_(torch.cat([model[2].weight for model in models],dim=1))
        result[2].bias.copy_(sum(model[2].bias for model in models))
    return result.requires_grad_(False)
