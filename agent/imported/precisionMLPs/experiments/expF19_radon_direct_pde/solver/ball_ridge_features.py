"""Analytic ball-frame coordinates inside an ordinary multidimensional MLP.

Known Legendre products define only coefficient coordinates at construction.
Exact ball moments and a Gegenbauer reproducing identity convert those known
coordinates to shared directional profiles, then QUILL converts the profiles
to tanh readouts. Neither products nor polynomials occur in neural evaluation.
There is no fit, SVD, PDE solution, or target label in this constructor.

The full cube is mapped into its enclosing unit ball. This simple construction
has expensive angular scaling; it is a correctness prototype, not an optimal
high-dimensional cubature implementation.
"""
from functools import lru_cache
from math import factorial, ceil, sqrt, prod
import time
import numpy as np
from numpy.polynomial import legendre
from quill_boundary import encode
from ridge_frame_dimension_study import sphere_rule
from solver.general_features import _compositions, _evaluate_encoding


def _poch(a,n):
    out=np.longdouble(1)
    for k in range(n):out*=np.longdouble(a)+k
    return out


@lru_cache(None)
def _ball_moment(alpha):
    """Uniform probability measure on B^d; odd moments vanish."""
    if any(a%2 for a in alpha):return np.longdouble(0)
    top=np.longdouble(1)
    for a in alpha:top*=_poch(.5,a//2)
    return top/_poch(len(alpha)/2+1,sum(alpha)//2)


def analytic_profile_map(dimension,degree,directions,angular_weights,indices):
    """Return E[m,n,k]: coefficient of C_n^(d/2)(omega_m.x).

    E=q_m (2n+d)/d * E_{y in ball}[phi_k(y) C_n^(d/2)(omega_m.y)].
    phi_k(y)=product_i P_{k_i}(sqrt(d)*y_i). The integral is evaluated
    algebraically from monomial moments. Higher degrees use arbitrary
    precision for construction only; longdouble is merely float64 on ARM Macs.
    """
    if degree>8:
        return _precise_profile_map(dimension,degree,directions,angular_weights,indices)
    d=dimension;ld=np.longdouble
    powers=[list(_compositions(n,d)) for n in range(degree+1)]
    univariate=[]
    for n in range(degree+1):
        p=legendre.leg2poly(np.eye(degree+1)[n]).astype(ld)
        univariate.append(p*np.sqrt(ld(d))**np.arange(len(p)))
    expansions=[]
    for index in indices:
        terms=[((0,)*d,ld(1))]
        for axis,n in enumerate(index):
            new=[]
            for alpha,c in terms:
                for k,b in enumerate(univariate[n]):
                    if b:
                        beta=list(alpha);beta[axis]=k
                        new.append((tuple(beta),c*b))
            terms=new
        expansions.append(terms)
    integrals=[]
    for k in range(degree+1):
        moments=np.array([[sum(c*_ball_moment(tuple(a+b for a,b in zip(alpha,beta)))
                               for alpha,c in terms) for terms in expansions]
                          for beta in powers[k]],dtype=ld)
        # Accumulate axes without an M-by-number_of_powers-by-d temporary.
        # That temporary alone exceeds a GB for otherwise modest 50D p2.
        exponents=np.asarray(powers[k])
        monomials=np.ones((len(directions),len(exponents)),dtype=ld)
        for axis in range(d):
            monomials*=directions[:,axis:axis+1].astype(ld)**exponents[None,:,axis]
        denominators=np.array([prod(factorial(b) for b in beta) for beta in powers[k]],dtype=ld)
        integrals.append((monomials/denominators)@moments)
    result=np.zeros((len(directions),degree+1,len(indices)),dtype=ld)
    for n in range(degree+1):
        for j in range(n//2+1):
            k=n-2*j
            result[:,n,:]+=(-1)**j*_poch(ld(d)/2,n-j)*2**k/factorial(j)*integrals[k]
        result[:,n,:]*=angular_weights.astype(ld)[:,None]*ld(2*n+d)/d
        degrees=np.sum(indices,axis=1)
        result[:,n,(degrees<n)|((degrees-n)%2!=0)]=0
    return result.astype(float)


def _precise_profile_map(dimension,degree,directions,angular_weights,indices):
    """Same analytic formula, with cancellation resolved BEFORE float export.

    No PDE solution, quadrature fit or regression enters this construction.
    Precision controls basis construction; the exported model stays float64.
    """
    import mpmath as mp
    d=dimension
    with mp.workdps(max(50,3*degree)):
        powers=[list(_compositions(n,d)) for n in range(degree+1)]
        @lru_cache(None)
        def moment(alpha):
            if any(a%2 for a in alpha):return mp.mpf(0)
            return mp.fprod(mp.rf(mp.mpf('.5'),a//2) for a in alpha)/mp.rf(mp.mpf(d)/2+1,sum(alpha)//2)
        univariate=[]
        for n in range(degree+1):
            coefficients=legendre.leg2poly(np.eye(degree+1)[n])
            univariate.append([mp.mpf(float(c))*mp.sqrt(d)**k for k,c in enumerate(coefficients)])
        expansions=[]
        for index in indices:
            terms=[((0,)*d,mp.mpf(1))]
            for axis,n in enumerate(index):
                new=[]
                for alpha,c in terms:
                    for k,b in enumerate(univariate[n]):
                        if b:
                            beta=list(alpha);beta[axis]=k;new.append((tuple(beta),c*b))
                terms=new
            expansions.append(terms)
        integrals=[]
        for k in range(degree+1):
            moments=mp.matrix([[mp.fsum(c*moment(tuple(a+b for a,b in zip(alpha,beta))) for alpha,c in terms)
                                for terms in expansions] for beta in powers[k]])
            monomials=mp.matrix([[mp.fprod(mp.mpf(float(w))**b for w,b in zip(direction,beta))/prod(factorial(b) for b in beta)
                                  for beta in powers[k]] for direction in directions])
            integrals.append(monomials*moments)
        result=np.zeros((len(directions),degree+1,len(indices)))
        degrees=np.sum(indices,axis=1)
        for n in range(degree+1):
            factors=[(-1)**j*mp.rf(mp.mpf(d)/2,n-j)*2**(n-2*j)/factorial(j) for j in range(n//2+1)]
            for feature in np.flatnonzero((degrees>=n)&((degrees-n)%2==0)):
                for m in range(len(directions)):
                    integral=mp.fsum(factor*integrals[n-2*j][m,int(feature)] for j,factor in enumerate(factors))
                    result[m,n,feature]=float(integral*mp.mpf(float(angular_weights[m]))*(2*n+d)/d)
        return result


def gegenbauer_profiles(z,dimension,degree):
    """Complex-compatible recurrence for the explicitly known profiles."""
    z=np.asarray(z);lam=dimension/2
    a=np.empty(z.shape+(degree+1,),dtype=np.result_type(z,float));a[...,0]=1
    if degree:a[...,1]=2*lam*z
    for n in range(2,degree+1):
        a[...,n]=(2*(n+lam-1)*z*a[...,n-1]-(n+2*lam-2)*a[...,n-2])/n
    return a


class BallRidgeFeatures:
    """Flat tanh features with explicit readout map, on a physical box."""
    def __init__(self,bounds,degree,centers=257,lam=.2):
        started=time.perf_counter()
        self.bounds=np.asarray(bounds,float)
        if self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or not np.all(np.isfinite(self.bounds)) or np.any(np.diff(self.bounds,axis=1)<=0):
            raise ValueError('bounds must be finite increasing pairs')
        if self.bounds.shape[0]<2 or int(degree)!=degree or degree<0 or int(centers)!=centers or centers<3 or not np.isfinite(lam) or lam<=0:
            raise ValueError('Require dimension>=2, integer degree>=0, centers>=3, lambda>0')
        self.degree=int(degree);self.dimension=len(self.bounds)
        self.midpoint=self.bounds.mean(axis=1)
        self.scale=2/(self.bounds[:,1]-self.bounds[:,0])/sqrt(self.dimension)
        self.multiindices=np.array([v for n in range(degree+1) for v in _compositions(n,self.dimension)])
        self.size=len(self.multiindices)
        self.directions,self.angular_weights=sphere_rule(self.dimension,self.degree)
        self.profile_map=analytic_profile_map(self.dimension,self.degree,self.directions,self.angular_weights,self.multiindices)
        self.encoding=encode(lambda z:gegenbauer_profiles(z,self.dimension,self.degree),centers-1,lam=lam,halo=ceil(sqrt(centers)))
        self.encoding.evaluation_mode='anchored';self.encoding.anchor_x=-1.
        self.encoding.anchor_value=gegenbauer_profiles(np.array([-1.]),self.dimension,self.degree)[0]
        self.physical_directions=self.directions*self.scale
        self.metrics=dict(backend='actual_tanh_ball_ridges',dimension=self.dimension,degree=self.degree,
            coefficient_count=self.size,directions=len(self.directions),interior_centers=centers,
            halo_per_side=ceil(sqrt(centers)),lam=lam,tanh_count=len(self.directions)*len(self.encoding.centers),
            product_gates=False,polynomials_in_forward=False,construction_uses_target_data=False,
            readout_parameterization='analytic ball-moment and angular map E @ a',
            map_construction_decimal_digits=max(50,3*self.degree) if self.degree>8 else int(np.finfo(np.longdouble).precision),
            construction_seconds=time.perf_counter()-started)

    def evaluate(self,points,derivative=None):
        derivative=(0,)*self.dimension if derivative is None else tuple(derivative)
        return self.evaluate_many(points,[derivative])[derivative]

    def evaluate_many(self,points,derivatives):
        """Share actual ridge profiles across equal total derivative orders.

        Directional chain factors differ across coordinates, but the scalar
        tanh profile derivative does not. No differential operator or named
        PDE is assumed, and no polynomial substitutes for the neural value.
        """
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=self.dimension or not np.all(np.isfinite(points)):
            raise ValueError('points must be a finite Q-by-d array')
        z=(points-self.midpoint)*self.scale
        if np.max(np.linalg.norm(z,axis=1),initial=0)>1+1e-12:
            raise ValueError('Point outside enclosing ball')
        groups={};output={}
        for derivative in derivatives:
            raw=np.array(derivative)
            if raw.shape!=(self.dimension,) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0):
                raise ValueError('Derivative must be d nonnegative integers')
            key=tuple(raw)
            if key in output:continue
            output[key]=np.empty((len(points),self.size))
            groups.setdefault(int(raw.sum()),[]).append((key,np.prod(self.physical_directions**raw,axis=1)))
        block=max(1,50000//len(self.directions))
        for start in range(0,len(points),block):
            projection=z[start:start+block]@self.directions.T
            for order,derivatives_of_order in groups.items():
                profiles=_evaluate_encoding(self.encoding,projection.ravel(),order).reshape(len(projection),len(self.directions),self.degree+1)
                profiles[:,:,0]=1. if order==0 else 0.
                for key,chain in derivatives_of_order:
                    output[key][start:start+block]=np.einsum('qmn,mnk,m->qk',profiles,self.profile_map,chain,optimize=True)
        return output

    def compile(self,coefficients):
        coefficients=np.asarray(coefficients,float)
        if coefficients.ndim==1:coefficients=coefficients[:,None]
        if coefficients.shape[0]!=self.size:raise ValueError('Wrong coefficient count')
        profiles=np.einsum('mnk,kf->mnf',self.profile_map.astype(np.longdouble),coefficients.astype(np.longdouble))
        bias=profiles[:,0,:].sum(axis=0);profiles[:,0,:]=0.
        readout3=np.einsum('jn,mnf->mjf',self.encoding.weights.astype(np.longdouble),profiles)
        # Compile the prescribed anchored representation after combining
        # profiles. Combining already rounded individual biases loses a
        # constant offset. This uses geometry and known profile anchors only,
        # not values of the PDE solution. Export is still ordinary float64.
        z0=self.encoding.gamma*(self.encoding.anchor_x-self.encoding.centers)
        bias+=np.einsum('n,mnf->f',self.encoding.anchor_value.astype(np.longdouble),profiles)
        bias-=np.einsum('j,mjf->f',np.tanh(z0).astype(np.longdouble),readout3)
        readout=readout3.reshape(-1,coefficients.shape[1]).astype(float)
        first=np.repeat(self.encoding.gamma*self.physical_directions,len(self.encoding.centers),axis=0)
        bias1=(-self.encoding.gamma*self.encoding.centers[None,:]-self.encoding.gamma*(self.physical_directions@self.midpoint)[:,None]).ravel()
        return dict(first_weights=first,first_bias=bias1,output_weights=readout,output_bias=bias.astype(float))

    def torch_model(self,coefficients):
        import torch
        arrays=self.compile(coefficients);q=arrays['output_weights'].shape[1];width=len(arrays['first_bias'])
        model=torch.nn.Sequential(torch.nn.Linear(self.dimension,width,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Linear(width,q,dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(arrays['first_weights']));model[0].bias.copy_(torch.from_numpy(arrays['first_bias']))
            model[2].weight.copy_(torch.from_numpy(arrays['output_weights'].T));model[2].bias.copy_(torch.from_numpy(arrays['output_bias']))
        return model.requires_grad_(False)
