"""Run all declared PDEs through the same refinement/solve/validation policy."""
import argparse
import json
import time
import numpy as np
from general_residual_study import cases,evaluate_case,clean,OUT
from solver.general_adaptive import ResidualDeclaration,solve_declared


def run(case,backend='quill',tolerance=1e-6,max_seconds=180,policy='auto'):
    declaration=ResidualDeclaration(case.domain,case.blocks,case.fields,
                                    case.parameter_initial,case.parameter_bounds)
    solution=solve_declared(declaration,backend=backend,tolerance=tolerance,max_seconds=max_seconds,
                            solver_options={'preconditioner':policy})
    validation=evaluate_case(case,solution)
    result={'case':case.name,'backend':backend,'policy':policy,'metrics':solution.metrics,
            'history':solution.history,'validation':validation,
            'degree_selected':solution.problem.features.degree,
            'solver':solution.solution.metrics,'features':solution.problem.features.metrics,
            'manufactured_forcing':case.manufactured,'note':case.note}
    OUT.mkdir(parents=True,exist_ok=True)
    stem=f'adaptive_{policy}_{case.name}_{backend}'
    path=OUT/(stem+'.json')
    path.write_text(json.dumps(clean(result),indent=2))
    np.savez_compressed(OUT/(stem+'.npz'),
                        coefficients=solution.coefficients,parameters=solution.parameters)
    print(json.dumps({'case':case.name,'backend':backend,'status':solution.status,
                      'seconds':solution.metrics['seconds'],'degree':solution.problem.features.degree,
                      'stages':solution.metrics['stages'],'validation':validation}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['all'])
    parser.add_argument('--backend',default='quill',choices=['quill','polynomial'])
    parser.add_argument('--tolerance',type=float,default=1e-6)
    parser.add_argument('--max-seconds',type=float,default=180)
    parser.add_argument('--policy',choices=['auto','diagonal','block'],default='auto')
    args=parser.parse_args();all_cases=cases()
    names=list(all_cases) if args.cases==['all'] else args.cases
    for name in names:run(all_cases[name],args.backend,args.tolerance,args.max_seconds,args.policy)
