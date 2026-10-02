"""Optional SymPy differential-expression frontend for generic residual blocks.

Only symbolic objects are accepted: no expression strings are parsed/evaluated.
SymPy expands differential product/chain rules, then known field derivatives
are replaced by scalar jet arguments and compiled to pointwise Torch algebra.
Import this optional module explicitly; the residual engine does not require it.
"""
from __future__ import annotations

from functools import reduce
import math

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.pycode import PythonCodePrinter
import torch

from .general_residual import ResidualBlock


def _reference(*values):
    return next((value for value in values if isinstance(value, torch.Tensor)), None)


def _tensor(value, reference=None):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64 if reference is None else reference.dtype,
                           device=None if reference is None else reference.device)


def _unary(function):
    return lambda value: function(_tensor(value))


def _binary(function):
    def call(left, right):
        reference = _reference(left, right)
        return function(_tensor(left, reference), _tensor(right, reference))
    return call


def _extremum(function):
    def call(*values):
        reference = _reference(*values)
        return reduce(function, (_tensor(value, reference) for value in values))
    return call


def _where(condition, left, right):
    reference = _reference(left, right)
    # Boolean conditions supply a device, but not the floating result dtype.
    if reference is None and isinstance(condition, torch.Tensor):
        reference = torch.empty((), dtype=torch.float64, device=condition.device)
    left, right = _tensor(left, reference), _tensor(right, reference)
    condition = torch.as_tensor(condition, dtype=torch.bool, device=left.device)
    return torch.where(condition, left, right)


def _heaviside(value, at_zero=.5):
    value = _tensor(value)
    return torch.where(value > 0, torch.ones_like(value),
                       torch.where(value < 0, torch.zeros_like(value), _tensor(at_zero, value)))


_TORCH_FUNCTIONS = {
    'sin': _unary(torch.sin), 'cos': _unary(torch.cos), 'tan': _unary(torch.tan),
    'asin': _unary(torch.asin), 'acos': _unary(torch.acos), 'atan': _unary(torch.atan),
    'atan2': _binary(torch.atan2),
    'sinh': _unary(torch.sinh), 'cosh': _unary(torch.cosh), 'tanh': _unary(torch.tanh),
    'asinh': _unary(torch.asinh), 'acosh': _unary(torch.acosh), 'atanh': _unary(torch.atanh),
    'exp': _unary(torch.exp), 'log': _unary(torch.log), 'sqrt': _unary(torch.sqrt),
    'Abs': _unary(torch.abs), 'abs': _unary(torch.abs), 'sign': _unary(torch.sign),
    'erf': _unary(torch.erf), 'erfc': _unary(torch.erfc),
    'floor': _unary(torch.floor), 'ceiling': _unary(torch.ceil),
    'Min': _extremum(torch.minimum), 'Max': _extremum(torch.maximum),
    'Heaviside': _heaviside, 'where': _where,
    'pi': math.pi, 'E': math.e,
}
_SUPPORTED_FUNCTIONS = {
    sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.atan2,
    sp.sinh, sp.cosh, sp.tanh, sp.asinh, sp.acosh, sp.atanh,
    sp.exp, sp.log, sp.Abs, sp.sign, sp.erf, sp.erfc,
    sp.floor, sp.ceiling, sp.Min, sp.Max, sp.Heaviside, sp.Piecewise,
}


class _TorchPrinter(PythonCodePrinter):
    def _print_sign(self, expression):
        # PythonCodePrinter normally emits a scalar conditional/copy-sign.
        return 'sign('+self._print(expression.args[0])+')'

    def _print_Piecewise(self, expression):
        result = "float('nan')"
        for value, condition in reversed(expression.args):
            if condition == sp.true:
                result = self._print(value)
            else:
                result = f'where({self._print(condition)}, {self._print(value)}, {result})'
        return result

    def _print_And(self, expression):
        return '('+' & '.join('('+self._print(arg)+')' for arg in expression.args)+')'

    def _print_Or(self, expression):
        return '('+' | '.join('('+self._print(arg)+')' for arg in expression.args)+')'

    def _print_Not(self, expression):
        return '(~('+self._print(expression.args[0])+'))'


class DifferentialResidual:
    """Compile declared real-valued scalar equations to a ResidualBlock callback.

    Fields are AppliedUndef objects such as ``u(x,y)`` with exactly the declared
    coordinate arguments. Equations can be scalar SymPy expressions or Equality
    objects. Unknown functions, symbols, axes, unresolved nonlocal operations,
    and distributional derivatives are rejected. Pointwise Piecewise/Abs are
    allowed, with Torch's usual almost-everywhere derivative convention.
    """

    def __init__(self, coordinates, fields, equations, parameters=()):
        self.coordinates, self.fields, self.parameters = tuple(coordinates), tuple(fields), tuple(parameters)
        if not self.coordinates or any(not isinstance(x, sp.Symbol) for x in self.coordinates):
            raise TypeError('coordinates must be a nonempty tuple of SymPy Symbols')
        if len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError('Coordinate symbols must be distinct')
        if not self.fields or any(not isinstance(u, AppliedUndef) for u in self.fields):
            raise TypeError('fields must be a nonempty tuple of applied undefined functions, e.g. u(x,y)')
        if len(set(self.fields)) != len(self.fields) or any(u.args != self.coordinates for u in self.fields):
            raise ValueError('Fields must be distinct and applied to exactly the declared coordinate tuple')
        if (any(not isinstance(p, sp.Symbol) for p in self.parameters) or
                len(set(self.parameters)) != len(self.parameters) or set(self.parameters) & set(self.coordinates)):
            raise ValueError('Parameters must be distinct SymPy Symbols, separate from coordinates')
        if isinstance(equations, (sp.Expr, sp.Equality)):
            equations = (equations,)
        else:
            equations = tuple(equations)
        if not equations or any(not isinstance(e, (sp.Expr, sp.Equality)) for e in equations):
            raise TypeError('equations must contain SymPy scalar expressions/Equalities; strings are not accepted')
        self.equations = tuple(e.lhs-e.rhs if isinstance(e, sp.Equality) else e for e in equations)
        for expression in self.equations:
            self._validate(expression, expanded=False)
        self.expanded_equations = tuple(expression.doit() for expression in self.equations)
        for expression in self.expanded_equations:
            self._validate(expression, expanded=True)

        zero = (0,)*len(self.coordinates)
        replacements, entries = {}, {}
        for expression in self.expanded_equations:
            for derivative in expression.atoms(sp.Derivative):
                if derivative.expr not in self.fields:
                    raise ValueError(f'Unresolved derivative is not a declared field derivative: {derivative}')
                orders = [0]*len(self.coordinates)
                for coordinate, count in derivative.variable_count:
                    orders[self.coordinates.index(coordinate)] += int(count)
                key = (tuple(orders), self.fields.index(derivative.expr))
                entries.setdefault(key, sp.Dummy(f'jet_{len(entries)}', real=True))
                replacements[derivative] = entries[key]
            for field in expression.atoms(AppliedUndef):
                # Including a zero jet that occurs only under a derivative is
                # harmless and keeps substitution deterministic.
                key = (zero, self.fields.index(field))
                entries.setdefault(key, sp.Dummy(f'jet_{len(entries)}', real=True))
                replacements[field] = entries[key]
        compiled_expressions = tuple(expression.xreplace(replacements) for expression in self.expanded_equations)
        used = set().union(*(expression.free_symbols for expression in compiled_expressions))
        entries = {key: symbol for key, symbol in entries.items() if symbol in used}
        self.jet_entries = tuple(sorted(entries, key=lambda key: (sum(key[0]), key[0], key[1])))
        self.derivatives = tuple(sorted({key[0] for key in self.jet_entries}, key=lambda d: (sum(d), d))) or (zero,)
        arguments = self.coordinates+tuple(entries[key] for key in self.jet_entries)+self.parameters
        if any(expression.free_symbols-set(arguments) for expression in compiled_expressions):
            raise ValueError('Internal compilation left an unbound symbol')
        printer = _TorchPrinter(dict(fully_qualified_modules=False,
                                     user_functions={name: name for name in _TORCH_FUNCTIONS}))
        self._compiled = sp.lambdify(arguments, list(compiled_expressions),
                                     modules=[_TORCH_FUNCTIONS], printer=printer,
                                     dummify=True, cse=True)
        self.metrics = dict(coordinates=[str(x) for x in self.coordinates],
                            fields=[str(u) for u in self.fields],
                            parameters=[str(p) for p in self.parameters],
                            equations=[str(e) for e in self.equations],
                            expanded_equations=[str(e) for e in self.expanded_equations],
                            derivatives=[list(d) for d in self.derivatives],
                            maximum_derivative_order=max(map(sum, self.derivatives)),
                            equation_count=len(self.equations),
                            source='SymPy objects; automatic product/chain-rule expansion and jet discovery',
                            scope='Pointwise real Torch algebra; no automatic boundary conditions, gauges, discretization, or continuum guarantees')

    def _validate(self, expression, expanded):
        allowed_symbols = set(self.coordinates+self.parameters)
        unknown_symbols = expression.free_symbols-allowed_symbols
        if unknown_symbols:
            raise ValueError(f'Unbound symbols: {sorted(map(str, unknown_symbols))}')
        if expression.has(sp.I, sp.oo, sp.zoo, sp.nan):
            raise ValueError('Expressions must be finite and real valued')
        for derivative in expression.atoms(sp.Derivative):
            if any(axis not in self.coordinates for axis, _ in derivative.variable_count):
                raise ValueError(f'Derivative uses an undeclared coordinate axis: {derivative}')
        for function in expression.atoms(sp.Function):
            if isinstance(function, AppliedUndef):
                if function not in self.fields:
                    raise ValueError(f'Undeclared function or field application: {function}')
            elif function.func not in _SUPPORTED_FUNCTIONS:
                raise ValueError(f'Unsupported pointwise function: {function.func}')
        if expanded and expression.has(sp.Integral, sp.Sum, sp.Product, sp.Limit, sp.Subs):
            raise ValueError('Unresolved nonlocal operations cannot be represented by pointwise jets')

    def function(self, points, jets, parameters):
        if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != len(self.coordinates):
            raise ValueError('points must be a Torch tensor with shape Q-by-coordinate_count')
        if not points.is_floating_point():
            raise ValueError('points must use a real floating dtype')
        q = len(points)
        if not isinstance(parameters, torch.Tensor) or tuple(parameters.shape) != (q, len(self.parameters)):
            raise ValueError('parameters must have shape Q-by-declared_parameter_count')
        if not parameters.is_floating_point():
            raise ValueError('parameters must use a real floating dtype')
        for derivative in self.derivatives:
            if derivative not in jets or not isinstance(jets[derivative], torch.Tensor) or tuple(jets[derivative].shape) != (q, len(self.fields)):
                raise ValueError(f'Missing or incompatible jet {derivative}; expected Q-by-declared_field_count')
            if not jets[derivative].is_floating_point():
                raise ValueError('Field jets must use a real floating dtype')
        arguments = [points[:, i] for i in range(len(self.coordinates))]
        arguments += [jets[derivative][:, field] for derivative, field in self.jet_entries]
        arguments += [parameters[:, i] for i in range(len(self.parameters))]
        values = self._compiled(*arguments)
        columns = []
        for value in values:
            value = torch.as_tensor(value, dtype=points.dtype, device=points.device)
            if value.ndim == 0:
                value = value.expand(q)
            if value.shape != (q,):
                raise ValueError('Compiled scalar equation did not produce one value per point')
            columns.append(value)
        return torch.stack(columns, dim=1)

    def block(self, name, points, weight=1., scale=1., aggregation=None, relation='eq'):
        return ResidualBlock(name, points, self.function, self.derivatives,
                              weight=weight, scale=scale, aggregation=aggregation, relation=relation)
