"""Frozen general equation-scaling control on the complete forward family.

The only mathematical change from accuracy_v2 is fixed differential-operator
normalization at zero jets. All raw PDE checks use the original declaration.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import copy
import json
from pathlib import Path
import route2_battletest_forward as benchmark

OUT=benchmark.ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized'


def configure():
    benchmark.OUT=OUT
    benchmark.OPTIONS=copy.deepcopy(benchmark.OPTIONS)
    benchmark.OPTIONS.update(max_seconds=600.,max_iterations=60)
    benchmark.PROTOCOL=copy.deepcopy(benchmark.PROTOCOL)
    benchmark.PROTOCOL.update(version='operator_normalization_v1',seeds=[0],solver_options=benchmark.OPTIONS,
        description='Fixed local differential-operator normalization; no equation-specific settings',
        constraints='PDE divided by its zero-jet differential sensitivity norm using domain widths and unit field amplitude; BC/IC unchanged. Raw conditions separately audited.',
        resource='Single thread;600s soft percase; concurrent work means times are not isolated speed benchmarks.',
        normalization='s(x)=sqrt(sum_alpha (dF/d(u_alpha)|u=0 / L^alpha)^2); positive finite scale, zero fallback1. No target or fitted field used.',
        validation_warning='Driver checks normalized PDE. Raw PDE/condition errors reported separately; no raw-tolerance success inferred from normalized stopping.')
    return benchmark.freeze_protocol(OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=benchmark.CASES)
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();configure()
    if args.case:
        # The common benchmark applies PROTOCOL['normalization'] exactly once
        # to its solve declaration and retains the original for raw audits.
        benchmark.fit(args.case,0)
    if args.evaluate:
        benchmark.evaluate_saved();benchmark.summarize()
