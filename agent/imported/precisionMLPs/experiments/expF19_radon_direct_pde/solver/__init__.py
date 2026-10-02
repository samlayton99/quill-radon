"""Constructed neural PDE research solver, with explicitly scoped backends."""
from .base import (Resolution,PeriodicEvolution,ReactionDiffusion1D,Conservation1D,RidgeLinear,
                   solve,solve_periodic,calibrate)
from .tensor_box import BoxProblem
from .elliptic import Annulus,AnnulusProblem
from .verification import solve_verified
from .adaptive_ridges import AdaptiveRidgeProblem, solve_adaptive_ridges
from .ns_extended import PeriodicNS, solve_ns_refined
from .inverse_profile import ProfileForward, ProfileObservations, fit_profile, fit_profile_refined
from .general_domains import ResidualDomain
from .general_features import ConstructedFeatures
from .general_residual import ResidualBlock, ResidualProblem, solve_residual
from .general_adaptive import ResidualDeclaration, solve_declared
from .general_problem import Condition, GeneralProblem

__all__=['Resolution','PeriodicEvolution','ReactionDiffusion1D','Conservation1D','RidgeLinear',
         'BoxProblem','Annulus','AnnulusProblem','solve','solve_periodic','solve_verified','calibrate',
         'AdaptiveRidgeProblem','solve_adaptive_ridges','PeriodicNS','solve_ns_refined',
         'ProfileForward','ProfileObservations','fit_profile','fit_profile_refined',
         'ResidualDomain','ConstructedFeatures','ResidualBlock','ResidualProblem',
         'solve_residual','ResidualDeclaration','solve_declared','Condition','GeneralProblem']
