"""Equation-independent subsets and growth of constructed product features.

Support selection uses only coefficients from a solved lower-resolution model.
Fresh residual checks must decide whether a restricted/refined solve succeeds;
sparsity or a small discarded coefficient sum is not a PDE error certificate.
"""
from copy import deepcopy
from math import factorial
import numpy as np
from .general_features import ConstructedFeatures


def downward_closure(indices):
    indices=np.asarray(indices,dtype=int)
    if indices.ndim!=2 or len(indices)==0 or np.any(indices<0):
        raise ValueError('Supply nonempty nonnegative multi-indices')
    support={tuple(a) for a in indices};pending=list(support)
    while pending:
        a=pending.pop()
        for j in range(len(a)):
            if a[j]:
                parent=list(a);parent[j]-=1;parent=tuple(parent)
                if parent not in support:support.add(parent);pending.append(parent)
    return np.array(sorted(support,key=lambda a:(sum(a),a)),dtype=int)


def extend_support(indices,layers=1,max_degree=None,mode='total'):
    if int(layers)!=layers or layers<0:raise ValueError('layers must be a nonnegative integer')
    if mode not in ('total','axis'):raise ValueError('mode must be total or axis')
    closed=downward_closure(indices);support={tuple(a) for a in closed}
    if mode=='axis':
        # Extend existing interactions along every coordinate, without
        # enumerating every simultaneous cross-coordinate increase. Subsequent
        # solve/refine rounds can discover additional mixed interactions.
        for a in closed:
            for j in range(len(a)):
                for step in range(1,layers+1):
                    child=list(a);child[j]+=step;child=tuple(child)
                    if max_degree is None or sum(child)<=max_degree:support.add(child)
        return downward_closure(np.array(list(support),dtype=int))
    frontier=list(support)
    for _ in range(layers):
        added=set()
        for a in frontier:
            for j in range(len(a)):
                child=list(a);child[j]+=1;child=tuple(child)
                if (max_degree is None or sum(child)<=max_degree) and child not in support:
                    added.add(child)
        support.update(added);frontier=list(added)
    return downward_closure(np.array(list(support),dtype=int))


def selected_feature_bank(bounds,indices,backend='quill',**options):
    raw=np.asarray(indices)
    if raw.ndim!=2 or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0) or not len(raw):
        raise ValueError('indices must be a nonempty integer array')
    indices=np.array(sorted({tuple(a) for a in raw},key=lambda a:(sum(a),a)),dtype=int)
    if indices.shape[1]!=len(bounds):raise ValueError('Multi-index dimension must match bounds')
    degree=int(indices.sum(axis=1).max())
    bank=ConstructedFeatures(bounds,degree,backend=backend,**options)
    bank.metrics=deepcopy(bank.metrics)
    bank.metrics.update(full_space_coefficient_count=bank.size,
                        coefficient_count=len(indices),selected_multiindices=True)
    bank.multiindices=indices
    bank.size=len(indices)
    return bank


def coefficient_support(features,coefficients,discard_budget,derivatives=None):
    """Discard small coefficient rows, then include all hierarchical parents.

    For exact Legendre products, sum of discarded absolute coefficients bounds
    the componentwise sup norm of the discarded expansion. QUILL bank errors
    perturb that statement. This controls a change of representation, not the
    error of the already learned field against the PDE solution.
    """
    a=np.asarray(coefficients,dtype=float)
    if a.ndim==1:a=a[:,None]
    if a.ndim!=2 or len(a)!=features.size or not np.all(np.isfinite(a)):
        raise ValueError('Coefficient array must have one finite row per feature')
    if not np.isfinite(discard_budget) or discard_budget<0:
        raise ValueError('discard_budget must be nonnegative and finite')
    row_bound=np.max(abs(a),axis=1)
    if derivatives is not None:
        # The exact endpoint maximum of a Legendre derivative supplies a
        # conservative jet-aware score without any named-PDE information.
        bounds=[]
        for derivative in derivatives:
            raw=np.asarray(derivative)
            if (raw.shape!=(features.dimension,) or
                    not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0)):
                raise ValueError('Derivative orders must match the feature dimension')
            bound=np.ones(features.size)
            for axis,r in enumerate(raw):
                bound*=np.array([0. if n<r else factorial(int(n+r))/
                                 (2**int(r)*factorial(int(r))*factorial(int(n-r)))
                                 for n in features.multiindices[:,axis]])*features.scale[axis]**r
            bounds.append(bound)
        if not bounds:raise ValueError('derivatives must be nonempty when supplied')
        row_bound*=np.max(bounds,axis=0)
    order=np.argsort(row_bound,kind='stable')
    remove=order[np.cumsum(row_bound[order])<=discard_budget]
    retain=np.ones(features.size,dtype=bool);retain[remove]=False
    retain[np.all(features.multiindices==0,axis=1)]=True
    indices=downward_closure(features.multiindices[retain])
    chosen={tuple(i) for i in indices}
    removed=np.array([tuple(i) not in chosen for i in features.multiindices])
    return indices,dict(discarded_coefficient_bound=float(row_bound[removed].sum()),
                        derivative_weighted=derivatives is not None,
                        selected_before_closure=int(retain.sum()),
                        retained_after_closure=len(indices))


def embed_coefficients(old_features,old_coefficients,new_features):
    old=np.asarray(old_coefficients)
    if old.ndim==1:old=old[:,None]
    result=np.zeros((new_features.size,old.shape[1]))
    lookup={tuple(a):i for i,a in enumerate(new_features.multiindices)}
    for i,a in enumerate(old_features.multiindices):
        j=lookup.get(tuple(a))
        if j is not None:result[j]=old[i]
    return result
