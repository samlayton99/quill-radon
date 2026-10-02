"""Fixed local equation scaling from declared differential sensitivities.

This changes the least-squares weighting, not the exact equation's roots.
It is an optional control, not a substitute for raw residual validation.
No reference values or fitted field are used to determine the scales.
"""
from dataclasses import replace
import numpy as np
import torch
from .general_adaptive import ResidualDeclaration


def local_equation_scales(block,fields,parameters,length_scales,field_scales=None,batch_size=128):
    """At zero jets, ||dF/djet * U/L**order||_2 for each point/equation.

    U are declared field scales and L are domain length scales. The expansion
    point is fixed before fitting; a vanishing differential falls back to one.
    Parameter derivatives are intentionally excluded: this is differential
    operator scaling, not a source/observation-dependent weight.
    """
    if block.aggregation is not None or block.relation!='eq':
        raise ValueError('Local scaling currently requires pointwise equality equations')
    lengths=np.asarray(length_scales,float)
    amplitudes=np.ones(fields) if field_scales is None else np.broadcast_to(np.asarray(field_scales,float),(fields,))
    if (lengths.shape!=(block.points.shape[1],) or np.any(lengths<=0) or not np.all(np.isfinite(lengths))
            or np.any(amplitudes<=0) or not np.all(np.isfinite(amplitudes))):
        raise ValueError('Length and field scales must be positive and finite')
    output=[]
    for start in range(0,len(block.points),batch_size):
        stop=min(start+batch_size,len(block.points));x=torch.tensor(block.points[start:stop],dtype=torch.float64)
        jets={order:torch.zeros((len(x),fields),dtype=x.dtype,requires_grad=True) for order in block.derivatives}
        eta=torch.tensor(np.asarray(parameters,float),dtype=x.dtype).expand(len(x),-1)
        raw=(block.function(x,jets,eta) if block.batch_function is None else
             block.batch_function(x,jets,eta,np.arange(start,stop)))
        if raw.ndim==1:raw=raw[:,None]
        scales=np.zeros(tuple(raw.shape))
        for equation in range(raw.shape[1]):
            gradients=(torch.autograd.grad(raw[:,equation].sum(),list(jets.values()),
                        allow_unused=True,retain_graph=True) if raw.requires_grad else [None]*len(jets))
            for order,gradient in zip(block.derivatives,gradients):
                if gradient is not None:
                    unit=amplitudes/np.prod(lengths**np.asarray(order))
                    scales[:,equation]+=np.sum((gradient.detach().numpy()*unit)**2,axis=1)
        scales=np.sqrt(scales)
        if not np.all(np.isfinite(scales)):raise ValueError('Nonfinite equation sensitivity scale')
        output.append(np.where(scales>np.finfo(float).tiny,scales,1.))
    return np.concatenate(output)


def normalize_equations(declaration,block_names,*,length_scales=None,field_scales=None):
    """Create a declaration with selected equation blocks scaled, physics unchanged.

    This is a finite-residual weighting choice. A normalized stopping check
    cannot be reported as a raw PDE accuracy check. Keep the original
    declaration and validate its equations independently after solving.
    """
    names=frozenset(block_names)
    lengths=(np.diff(declaration.domain.bounds,axis=1).ravel() if length_scales is None
             else np.asarray(length_scales,float))
    def make_blocks(count,seed):
        result=[]
        for block in declaration.make_blocks(count,seed):
            if block.name not in names:
                result.append(block);continue
            scale=local_equation_scales(block,declaration.fields,declaration.parameter_initial,lengths,field_scales)
            def full(x,j,p,original=block.function,scale=scale):
                raw=original(x,j,p)
                if raw.ndim==1:raw=raw[:,None]
                return raw/torch.tensor(scale,dtype=x.dtype,device=x.device)
            def batch(x,j,p,rows,original=block.function,batched=block.batch_function,scale=scale):
                raw=original(x,j,p) if batched is None else batched(x,j,p,rows)
                if raw.ndim==1:raw=raw[:,None]
                return raw/torch.tensor(scale[rows],dtype=x.dtype,device=x.device)
            result.append(replace(block,function=full,batch_function=batch))
        missing=names-{b.name for b in result}
        if missing:raise ValueError(f'Unknown equation blocks: {sorted(missing)}')
        return result
    return ResidualDeclaration(declaration.domain,make_blocks,declaration.fields,
        parameter_initial=np.asarray(declaration.parameter_initial).copy(),
        parameter_bounds=declaration.parameter_bounds)
