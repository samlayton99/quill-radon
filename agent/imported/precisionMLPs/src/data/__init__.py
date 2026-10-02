"""Target functions, sampling, and dataset construction."""

from src.data.targets import TargetFn, get_target, get_all_targets, TARGET_REGISTRY
from src.data.sampling import get_sampling_fn
from src.data.dataset import Dataset, build_dataset
