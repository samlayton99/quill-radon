"""Compile prescribed QUILL polynomial products into ordinary tanh MLPs.

The exported forward graph contains only torch.nn.Linear and torch.nn.Tanh.
Multiplication is APPROXIMATED by a constructed square profile through
    ab = B²[((a+b)/(2B))² - ((a-b)/(2B))²].
No target/PDE solution, product gate, custom activation, or hidden arithmetic
operation appears in the network forward. Consecutive affine bottlenecks may
be fused, with the usual change in floating-point summation order.
"""
from __future__ import annotations

import copy
from math import ceil, log2, sqrt
import time

import numpy as np
from numpy.polynomial import legendre
import torch
from torch import nn

from quill_boundary import encode


def _compositions(total, dimension):
    if dimension == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for tail in _compositions(total-first, dimension-1):
                yield (first,)+tail


def _linear(weight, bias):
    weight, bias = np.asarray(weight, dtype=float), np.asarray(bias, dtype=float)
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=True, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight))
        layer.bias.copy_(torch.as_tensor(bias))
    layer.requires_grad_(False)
    return layer


def fuse_affine_layers(network):
    """Return an equivalent strictly alternating graph; no product nodes."""
    layers = []
    for source in network:
        layer = copy.deepcopy(source)
        if isinstance(layer, nn.Linear) and layers and isinstance(layers[-1], nn.Linear):
            previous = layers.pop()
            with torch.no_grad():
                weight = layer.weight @ previous.weight
                bias = layer.weight @ previous.bias+layer.bias
            layers.append(_linear(weight.numpy(), bias.numpy()))
        else:
            layers.append(layer)
    return nn.Sequential(*layers)


def architecture_metrics(network):
    linear = [layer for layer in network if isinstance(layer, nn.Linear)]
    if any(type(layer) not in (nn.Linear, nn.Tanh) for layer in network):
        raise TypeError('Pure MLP forward permits exactly nn.Linear and nn.Tanh')
    return dict(layer_types=[type(layer).__name__ for layer in network],
                affine_shapes=[[layer.in_features, layer.out_features] for layer in linear],
                tanh_layers=sum(isinstance(layer, nn.Tanh) for layer in network),
                tanh_units=sum(network[i-1].out_features for i,layer in enumerate(network)
                               if isinstance(layer, nn.Tanh)),
                stored_parameters=sum(parameter.numel() for parameter in network.parameters()),
                nonzero_parameters=sum(int(torch.count_nonzero(parameter)) for parameter in network.parameters()),
                parameter_bytes=sum(parameter.numel()*parameter.element_size() for parameter in network.parameters()),
                product_gates=False, custom_activations=False,
                construction_uses_target_data=False)


def analytic_jet(network, points, derivative):
    """Stable analytic chain rule for this exact Linear/Tanh graph, orders0–2.

    These products are differentiation operations, not network forward nodes.
    Ordinary Torch autograd is independently available for comparison; its
    saturated-tanh backward uses a less accurate floating-point subtraction.
    """
    order = sum(derivative)
    if order > 2:
        raise ValueError('This prototype implements derivatives through order2')
    axes = [axis for axis,count in enumerate(derivative) for _ in range(count)]
    value = torch.as_tensor(points, dtype=torch.float64)
    first = torch.zeros_like(value)
    second_first = torch.zeros_like(value)
    mixed = torch.zeros_like(value)
    if order:
        first[:, axes[0]] = 1.
    if order == 2:
        second_first[:, axes[1]] = 1.
    with torch.no_grad():
        for layer in network:
            if isinstance(layer, nn.Linear):
                value = layer(value)
                first = torch.nn.functional.linear(first, layer.weight)
                if order == 2:
                    second_first = torch.nn.functional.linear(second_first, layer.weight)
                    mixed = torch.nn.functional.linear(mixed, layer.weight)
            elif isinstance(layer, nn.Tanh):
                exponent = torch.exp(-2*abs(value))
                slope = 4*exponent/(1+exponent)**2
                activated = layer(value)
                if order == 2:
                    mixed = slope*mixed-2*activated*slope*first*second_first
                    second_first = slope*second_first
                first = slope*first
                value = activated
            else:
                raise TypeError('Unsupported forward layer')
    return value if order == 0 else first if order == 1 else mixed


def autograd_jet(network, points, derivative):
    points = torch.tensor(np.asarray(points), dtype=torch.float64, requires_grad=True)
    result = network(points)
    axes = [axis for axis,count in enumerate(derivative) for _ in range(count)]
    for axis in axes:
        columns = [torch.autograd.grad(result[:, index].sum(), points, create_graph=True,
                                      retain_graph=True)[0][:, axis]
                   for index in range(result.shape[1])]
        result = torch.stack(columns, dim=1)
    return result.detach()


class PureMLPFeatures:
    """Fixed constructed MLP features consumable by the common residual engine.

    The final PDE readout is solved from declared equations, independently of
    this target-free construction. ``export_readout`` returns a complete
    ordinary nn.Sequential network including that readout.
    """
    def __init__(self, bounds, degree, cells=64, lam=.2, square_cells=None,
                 square_lam=None, product_bound=1.05, fuse_affines=False,
                 derivative_method='analytic', multiindices=None):
        started = time.perf_counter()
        self.bounds = np.asarray(bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or not len(self.bounds) or
                not np.all(np.isfinite(self.bounds)) or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError('bounds must contain finite increasing coordinate pairs')
        if isinstance(degree, bool) or int(degree) != degree or degree < 0:
            raise ValueError('degree must be a nonnegative integer')
        if not np.isfinite(product_bound) or product_bound <= 1.:
            raise ValueError('product_bound must exceed1 to allow encoding error')
        if derivative_method not in ('analytic', 'autograd'):
            raise ValueError('derivative_method must be analytic or autograd')
        self.dimension, self.degree = len(self.bounds), int(degree)
        self.derivative_method = derivative_method
        if multiindices is None:
            self.multiindices = np.array([mode for total in range(self.degree+1)
                                          for mode in _compositions(total, self.dimension)], dtype=int)
        else:
            raw = np.asarray(multiindices)
            if (raw.ndim != 2 or raw.shape[1] != self.dimension or len(raw) == 0 or
                    not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or
                    np.any(raw.sum(axis=1) > self.degree) or len(np.unique(raw, axis=0)) != len(raw)):
                raise ValueError('multiindices must be distinct nonnegative integer modes within the degree limit')
            self.multiindices = raw.astype(int)
        self.size = len(self.multiindices)
        leaf_degree = int(np.max(self.multiindices))
        # Public N is the number of interior centers (cells+1); explicitly
        # use its square-root halo instead of encode's cell-count default.
        leaf = encode(lambda z:legendre.legvander(z, leaf_degree), cells, lam=lam,
                      halo=ceil(sqrt(cells+1)))
        square_cells = cells if square_cells is None else square_cells
        square_lam = lam if square_lam is None else square_lam
        square = encode(lambda z:z*z, square_cells, lam=square_lam,
                        halo=ceil(sqrt(square_cells+1)))
        self.leaf_encoding, self.square_encoding = leaf, square
        maximum_factors = max(1, int(np.max(np.count_nonzero(self.multiindices, axis=1))))
        arity = 2**ceil(log2(maximum_factors)) if maximum_factors > 1 else 1
        width = len(leaf.centers)
        physical_scale = 2/(self.bounds[:, 1]-self.bounds[:, 0])
        midpoint = self.bounds.mean(axis=1)
        first_weight = np.zeros((width*self.dimension, self.dimension))
        first_bias = np.empty(width*self.dimension)
        for axis in range(self.dimension):
            section = slice(axis*width, (axis+1)*width)
            first_weight[section, axis] = leaf.gamma*physical_scale[axis]
            first_bias[section] = -leaf.gamma*(midpoint[axis]*physical_scale[axis]+leaf.centers)
        leaf_weight = np.zeros((self.size*arity, width*self.dimension))
        leaf_bias = np.ones(self.size*arity)
        for feature, mode in enumerate(self.multiindices):
            for slot, axis in enumerate(np.flatnonzero(mode)):
                row = feature*arity+slot
                leaf_weight[row, axis*width:(axis+1)*width] = leaf.weights[:, mode[axis]]
                leaf_bias[row] = np.asarray(leaf.bias).reshape(-1)[mode[axis]]
        layers = [_linear(first_weight, first_bias), nn.Tanh(), _linear(leaf_weight, leaf_bias)]
        square_width = len(square.centers)
        product_levels = 0
        while arity > 1:
            next_arity = arity//2
            products = self.size*next_arity
            mix_weight = np.zeros((products*2*square_width, self.size*arity))
            mix_bias = np.tile(-square.gamma*square.centers, 2*products)
            readout_weight = np.zeros((products, products*2*square_width))
            for feature in range(self.size):
                for pair in range(next_arity):
                    output = feature*next_arity+pair
                    left, right = feature*arity+2*pair, feature*arity+2*pair+1
                    for sign_index, sign in enumerate((1., -1.)):
                        section = slice((2*output+sign_index)*square_width,
                                        (2*output+sign_index+1)*square_width)
                        mix_weight[section, left] = square.gamma/(2*product_bound)
                        mix_weight[section, right] = sign*square.gamma/(2*product_bound)
                        readout_weight[output, section] = sign*product_bound**2*square.weights
            layers.extend((_linear(mix_weight, mix_bias), nn.Tanh(),
                           _linear(readout_weight, np.zeros(products))))
            arity = next_arity
            product_levels += 1
        # The prescribed degree-zero feature is exactly the affine constant1;
        # routing a known constant through approximate multipliers is wasteful.
        with torch.no_grad():
            for index, mode in enumerate(self.multiindices):
                if not np.any(mode):
                    layers[-1].weight[index].zero_()
                    layers[-1].bias[index] = 1.
        self.network = nn.Sequential(*layers)
        if fuse_affines:
            self.network = fuse_affine_layers(self.network)
        self.metrics = dict(backend='pure_tanh_mlp', dimension=self.dimension, degree=self.degree,
                            coefficient_count=self.size, product_levels=product_levels,
                            product_bound=product_bound, cells=int(cells), lam=float(lam),
                            interior_centers=int(cells)+1,halo_per_side=leaf.halo_per_side,
                            square_cells=int(square_cells), square_lam=float(square_lam),
                            square_interior_centers=int(square_cells)+1,square_halo_per_side=square.halo_per_side,
                            derivative_method=derivative_method, affine_fusion=bool(fuse_affines),
                            leaf_readout_l1_max=float(np.max(np.sum(abs(leaf.weights), axis=0))),
                            square_readout_l1=float(np.sum(abs(square.weights))),
                            architecture=architecture_metrics(self.network),
                            setup_seconds=time.perf_counter()-started,
                            approximation_scope='Finite tanh square modules approximate multiplication; this is not an exact polynomial identity in floating point or a target-fitted network')

    def evaluate(self, points, derivative=None):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.all(np.isfinite(points)):
            raise ValueError('points must be a finite Q-by-d array')
        derivative = (0,)*self.dimension if derivative is None else tuple(derivative)
        if (len(derivative) != self.dimension or
                any(isinstance(n, bool) or int(n) != n or n < 0 for n in derivative) or sum(derivative) > 2):
            raise ValueError('Derivative multiindex must have nonnegative integer entries, total order≤2')
        if not any(derivative):
            with torch.no_grad():
                return self.network(torch.as_tensor(points, dtype=torch.float64)).numpy()
        function = analytic_jet if self.derivative_method == 'analytic' else autograd_jet
        return function(self.network, points, derivative).numpy()

    def export_readout(self, coefficients, *, fuse_affines=False):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 2 or coefficients.shape[0] != self.size:
            raise ValueError('coefficients must have shape(features, fields)')
        network = nn.Sequential(*[copy.deepcopy(layer) for layer in self.network],
                                _linear(coefficients.T, np.zeros(coefficients.shape[1])))
        return fuse_affine_layers(network) if fuse_affines else network
