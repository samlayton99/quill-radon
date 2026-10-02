"""Bounded sum factorization for IDEAL box-coordinate correction products.

No neural residual uses these routines. A total-degree coefficient vector is
zero-padded into a tensor only if a strict small-array plan permits it. Forward
contractions and their transposes share derivative prefixes without any Q-by-P
table. Larger tensor grids retain the selected-column implementation instead.
"""
from math import comb
import numpy as np


def tensor_product_plan(dimension,degree,fields,rows,derivatives,
                        workspace_budget_bytes=16*1024**2,tensor_budget_bytes=4*1024**2):
    side=degree+1
    tensor_bytes=8*fields*side**dimension
    if dimension>8 or tensor_bytes>tensor_budget_bytes:
        return dict(enabled=False,tensor_bytes=tensor_bytes,reason='bounded_tensor_cap')
    axis_orders=sum(len({d[axis] for d in derivatives}) for axis in range(dimension))
    # Two full coefficient tensors cover packing/transposition or gradient +
    # matrix-product output. The reverse tree has at most two largest row
    # tensors plus one smaller tensor at each remaining depth alive together.
    fixed=2*tensor_bytes+8*dimension*comb(degree+dimension,dimension)
    per_row=8*fields*(2*side**(dimension-1)+sum(side**k for k in range(dimension-1)))
    per_row+=8*((axis_orders+4)*side+dimension+3*fields*len(derivatives))
    maximum=max(0,(workspace_budget_bytes-fixed)//max(1,per_row))
    count=min(int(rows),int(maximum))
    return dict(enabled=count>0,rows=count,tensor_bytes=tensor_bytes,
        workspace_bytes=fixed+count*per_row,fixed_bytes=fixed,bytes_per_row=per_row,
        workspace_budget_bytes=workspace_budget_bytes,tensor_budget_bytes=tensor_budget_bytes,
        reason='bounded_sum_factorization' if count else 'bounded_workspace_cap')


def _derivative_tree(derivatives,axes,depth=0):
    if depth==len(axes):return derivatives[0]
    grouped={}
    for derivative in derivatives:
        grouped.setdefault(derivative[axes[depth]],[]).append(derivative)
    return {order:_derivative_tree(group,axes,depth+1) for order,group in grouped.items()}


def forward_tensor_jets(dense,prepared):
    derivatives=prepared['derivatives'];dimension=dense.ndim-1
    axes=tuple(sorted(range(dimension),key=lambda a:len({d[a] for d in derivatives})))
    tree=_derivative_tree(derivatives,axes)
    coefficients=np.transpose(dense,axes+(dimension,))
    result={}
    def visit(current,node,depth):
        for order,child in node.items():
            table=prepared['axis_tables'][axes[depth],order]
            if depth==0:
                value=(table@current.reshape(current.shape[0],-1)).reshape(
                    (len(table),)+current.shape[1:])
            else:
                value=np.einsum('qn,qn...->q...',table,current,optimize=False)
            if depth+1==dimension:result[child]=value
            else:visit(value,child,depth+1)
            del value
    visit(coefficients,tree,0)
    return result


def accumulate_tensor_adjoint(dense_gradient,cotangents,prepared):
    derivatives=prepared['derivatives'];dimension=dense_gradient.ndim-1
    axes=tuple(sorted(range(dimension),key=lambda a:len({d[a] for d in derivatives})))
    tree=_derivative_tree(derivatives,axes)
    gradient=np.transpose(dense_gradient,axes+(dimension,))
    def pull(node,depth):
        if depth==dimension:return cotangents[node]
        accumulated=None
        for order,child in node.items():
            value=pull(child,depth+1)
            table=prepared['axis_tables'][axes[depth],order]
            if depth==0:
                gradient[:]+=(table.T@value.reshape(len(table),-1)).reshape(gradient.shape)
            else:
                expanded=table.reshape(table.shape+(1,)*(value.ndim-1))*value[:,None,...]
                if accumulated is None:accumulated=expanded
                else:accumulated+=expanded
                del expanded
            del value
        return accumulated
    pull(tree,0)
