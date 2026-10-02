"""Equation-independent domain samples for a residual-based solver.

A box or smooth implicit domain is problem data. No PDE type is inspected.
Implicit boundary samples are projections, not uniform surface quadrature;
use actual quadrature weights for weak-form integrals.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc
import torch


@dataclass
class BoundarySamples:
    points: np.ndarray
    normals: np.ndarray


class ResidualDomain:
    def __init__(self, bounds, levelset=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or not np.all(np.isfinite(self.bounds))
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError('bounds must be finite increasing pairs')
        self.dimension = len(self.bounds)
        self.levelset = levelset

    def _box(self, count, seed):
        if count < 1:
            raise ValueError('sample count must be positive')
        unit = qmc.Sobol(self.dimension, scramble=True, seed=seed).random_base2(
            int(np.ceil(np.log2(count))))[:count]
        return qmc.scale(unit, self.bounds[:, 0], self.bounds[:, 1])

    def _level(self, points, gradient=False):
        x = torch.tensor(points, dtype=torch.float64, requires_grad=gradient)
        y = self.levelset(x).reshape(-1)
        if y.shape != (len(points),) or not torch.all(torch.isfinite(y)):
            raise ValueError('level set must return one finite value per point')
        if gradient:
            g = (torch.autograd.grad(y.sum(), x,allow_unused=True)[0]
                 if y.requires_grad else None)
            if g is None:g=torch.zeros_like(x)
            return y.detach().numpy(), g.detach().numpy()
        return y.detach().numpy()

    def interior(self, count, seed=0):
        if self.levelset is None:
            return self._box(count, seed)
        pieces = []
        found = 0
        for attempt in range(50):
            x = self._box(max(128, 2*(count-found)), seed+104729*attempt)
            x = x[self._level(x) < 0]
            pieces.append(x)
            found += len(x)
            if found >= count:
                return np.concatenate(pieces)[:count]
        raise RuntimeError('Domain sampling budget exhausted; increase or refine domain sampler')

    def boundary(self, count, seed=1):
        if self.levelset is None:
            # Subsampling a Sobol sequence by index modulo face count can fix
            # its leading bits and leave most of a face unsampled.
            counts=np.full(2*self.dimension,count//(2*self.dimension),dtype=int)
            counts[:count % (2*self.dimension)]+=1
            x=np.concatenate([self._box(int(n),seed+104729*face)
                              for face,n in enumerate(counts) if n])
            face=np.repeat(np.arange(2*self.dimension),counts)
            axis, side = face//2, face % 2
            x[np.arange(count), axis] = self.bounds[axis, side]
            normals = np.zeros_like(x)
            normals[np.arange(count), axis] = 2*side-1
            order=np.random.default_rng(seed+1).permutation(count)
            return BoundarySamples(x[order], normals[order])
        pieces, directions = [], []
        found = 0
        span = self.bounds[:, 1]-self.bounds[:, 0]
        # An implicit domain is intersected with its bounding box. Include
        # exposed box-face pieces, not only the zero level surface.
        box_boundary=ResidualDomain(self.bounds).boundary(max(256,4*count),seed+7919)
        inside=self._level(box_boundary.points) < -1e-10
        clipped_points=box_boundary.points[inside]
        clipped_normals=box_boundary.normals[inside]
        reserve=min(count//2,len(clipped_points))
        requested=count-reserve
        for attempt in range(20):
            x = self._box(max(128, 2*(requested-found)), seed+104729*attempt)
            for _ in range(40):
                v, g = self._level(x, gradient=True)
                norm2 = (g*g).sum(axis=1)
                if not np.any(norm2>1e-28):break
                step = v[:, None]*g/np.maximum(norm2[:, None], 1e-28)
                relative = np.max(abs(step)/span, axis=1)
                step /= np.maximum(1., relative/.2)[:, None]
                x = np.clip(x-step, self.bounds[:, 0], self.bounds[:, 1])
            v, g = self._level(x, gradient=True)
            norm = np.linalg.norm(g, axis=1)
            distance = abs(v)/np.maximum(norm, 1e-28)
            good = (distance < 1e-9*np.linalg.norm(span)) & (norm > 1e-12)
            pieces.append(x[good]); directions.append(g[good]/norm[good, None])
            found += int(good.sum())
            if found >= requested:
                points=np.concatenate([np.concatenate(pieces)[:requested],clipped_points[:reserve]])
                normals=np.concatenate([np.concatenate(directions)[:requested],clipped_normals[:reserve]])
                order=np.random.default_rng(seed+2).permutation(count)
                return BoundarySamples(points[order],normals[order])
        if len(clipped_points)>=count and found==0:
            # A level set strictly below zero throughout the box simply
            # leaves the box domain; no interior zero contour is required.
            return BoundarySamples(clipped_points[:count],clipped_normals[:count])
        raise RuntimeError('Implicit boundary projection failed; singular/poorly sampled boundary')
