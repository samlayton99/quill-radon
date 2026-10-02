"""Run the bounded maintained-library PINN comparison on the nonlinear square."""
from pathlib import Path
import argparse
from solver.pinn_baseline import train_square

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--adam',type=int,default=15000)
    p.add_argument('--lbfgs',type=int,default=5000)
    p.add_argument('--points',type=int,default=1024)
    a=p.parse_args()
    out=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver'
    train_square(out,a.seed,a.width,a.adam,a.lbfgs,a.points)
