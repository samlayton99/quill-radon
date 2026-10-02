"""The greedy hierarchy: grow a ridge mesh from nothing by choosing, at every round, the
action that buys the most error per unit among

    open_bg       -- add the next batch of nested background directions, each a coarse
                     even block of N0 offsets;
    open_atom     -- add one direction found by projection pursuit on the residual (an
                     atom), with N_atom offsets, then polish all atoms jointly; only offered
                     when that one direction alone removes at least 7% of the residual;
    refine_bg     -- every background block: N -> ceil(1.5 N);
    refine_atoms  -- every atom block: N -> 2 N.

Every candidate is scored by an actual trial fit on the training data and a held-out
validation set inside the region; the score is the drop in log10(validation error) per
unit added. This is the two-floor law in operational form: whichever bottleneck (angular
or longitudinal) is binding shows up as the larger response.

Nothing is ever moved or deleted once installed (the direction sequence is nested, the
blocks are only refined), so the approximation space only grows.
"""

from __future__ import annotations

import time

import numpy as np

from .core import Geometry, make_block, nested_directions, fit_geometry, rel_l2, band_half_width, RCOND
from .atoms import projection_pursuit, varpro_polish


class Grower:
    def __init__(self, d, Ztr, ytr, Zval, yval, budget, N0=8, N_atom=32, bg_batch_frac=0.25,
                 bg_batch_min=4, polish_iters=4, n_pp_sub=6000, rcond=RCOND, seed=0,
                 allow_atoms=True, allow_bg=True, verbose=True, floor=3e-13, stall_rounds=3, atom_gate=0.93):
        self.d, self.Ztr, self.ytr, self.Zval, self.yval = d, Ztr, ytr, Zval, yval
        self.budget, self.N0, self.N_atom = budget, N0, N_atom
        self.bg_batch_frac, self.bg_batch_min = bg_batch_frac, bg_batch_min
        self.polish_iters, self.n_pp_sub, self.rcond = polish_iters, n_pp_sub, rcond
        self.allow_atoms, self.allow_bg, self.verbose, self.floor = allow_atoms, allow_bg, verbose, floor
        self.stall_rounds, self.atom_gate = stall_rounds, atom_gate
        self.rng = np.random.default_rng(seed)
        self.Vseq = nested_directions(d, 2048 if d <= 3 else 4096, seed=seed)
        self.n_bg = 0
        self.geom = Geometry()
        self.history = []

    # -- evaluation --------------------------------------------------------
    def evaluate(self, geom):
        fit = fit_geometry(geom, self.Ztr, self.ytr, rcond=self.rcond)
        pred = fit.predict(geom, self.Zval)
        return fit, rel_l2(pred, self.yval)

    def residual(self, geom, fit):
        return self.ytr - geom.augmented(self.Ztr) @ fit.coef

    # -- candidates --------------------------------------------------------
    def cand_open_bg(self):
        k = max(self.bg_batch_min, int(round(self.bg_batch_frac * self.geom.units / self.N0)))
        room = (self.budget - self.geom.units) // self.N0
        k = min(k, room)
        if k <= 0:
            return None
        g = self.geom.copy()
        for v in self.Vseq[self.n_bg:self.n_bg + k]:
            g.blocks.append(make_block(v, self.Ztr, self.N0, kind="bg"))
        return g, {"n_bg_added": k}

    def cand_open_atom(self, resid):
        if self.geom.units + self.N_atom > self.budget:
            return None
        idx = self.rng.choice(len(self.Ztr), size=min(self.n_pp_sub, len(self.Ztr)), replace=False)
        v, sc, sp = projection_pursuit(self.Ztr[idx], resid[idx], n_off=self.N_atom, rcond=self.rcond)
        if sp > self.atom_gate:                  # no single direction explains the residual: no atom
            return None
        g = self.geom.copy()
        g.blocks.append(make_block(v, self.Ztr, self.N_atom, kind="atom"))
        which = [i for i, b in enumerate(g.blocks) if b.kind == "atom"]
        # polish all atoms jointly, on a row subsample sized to the current width
        n_sub = min(len(self.Ztr), max(6000, 4 * g.units))
        sub = self.rng.choice(len(self.Ztr), size=n_sub, replace=False) if n_sub < len(self.Ztr) else slice(None)
        g, hist = varpro_polish(g, self.Ztr[sub], self.ytr[sub], which=which, iters=self.polish_iters, rcond=self.rcond)
        for b in g.blocks:                       # bands from the full training set
            b.T = band_half_width(b.v, self.Ztr)
        return g, {"pp_coarse": sc, "pp_polished": sp, "joint_polish_rel_res": hist[-1]["rel_residual"],
                   "polish_iters": len(hist) - 1}

    def cand_refine(self, kind, factor):
        """Refine every block of ``kind`` by ``factor``; if that does not fit the budget,
        refine the longest prefix (insertion order) that does."""
        g = self.geom.copy()
        room = self.budget - self.geom.units
        added, n_ref = 0, 0
        for b in g.blocks:
            if b.kind != kind:
                continue
            new_n = int(np.ceil(factor * b.n))
            if added + new_n - b.n > room:
                break
            added += new_n - b.n
            b.n = new_n
            n_ref += 1
        if added == 0:
            return None
        return g, {"units_added": added, "blocks_refined": n_ref}

    # -- the loop ----------------------------------------------------------
    def run(self):
        t_start = time.time()
        fit, err = None, 1.0
        if self.geom.units > 0:
            fit, err = self.evaluate(self.geom)
        self.history.append({"round": 0, "action": "init", "units": self.geom.units, "n_dir": 0,
                             "val_err": err, "seconds": 0.0, **self.geom.describe()})
        rnd, stalled = 0, 0
        while self.geom.units < self.budget and err > self.floor and stalled < self.stall_rounds:
            rnd += 1
            t0 = time.time()
            resid = self.residual(self.geom, fit) if fit is not None else self.ytr.copy()
            cands = {}
            if self.allow_bg:
                cands["open_bg"] = self.cand_open_bg()
            if self.allow_atoms:
                cands["open_atom"] = self.cand_open_atom(resid)
            cands["refine_bg"] = self.cand_refine("bg", 1.5)
            cands["refine_atoms"] = self.cand_refine("atom", 2.0)
            scored = {}
            for name, c in cands.items():
                if c is None:
                    continue
                g, info = c
                f, e = self.evaluate(g)
                added = g.units - self.geom.units
                gain = (np.log10(max(err, 1e-300)) - np.log10(max(e, 1e-300))) / max(added, 1)
                scored[name] = (gain, e, g, f, info, added)
            if not scored:
                break
            best = max(scored, key=lambda k: scored[k][0])
            gain, e, g, f, info, added = scored[best]
            self.geom, fit = g, f
            if best == "open_bg":
                self.n_bg += info["n_bg_added"]
            err_prev, err = err, e
            stalled = stalled + 1 if err > 0.9 * err_prev else 0
            row = {"round": rnd, "action": best, "units": self.geom.units, "units_added": added,
                   "val_err": err, "val_err_prev": err_prev, "gain_per_unit": float(gain),
                   "candidates": {k: {"val_err": v[1], "units_added": v[5], "gain": float(v[0])} for k, v in scored.items()},
                   "info": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()},
                   "seconds": round(time.time() - t0, 1), **self.geom.describe()}
            self.history.append(row)
            if self.verbose:
                cstr = " ".join(f"{k}={v[1]:.1e}(+{v[5]})" for k, v in scored.items())
                print(f"  r{rnd:2d} {best:12s} units={self.geom.units:5d} dirs={self.geom.n_dir:4d} "
                      f"atoms={row['n_atoms']:2d} val={err:.2e} [{row['seconds']:5.1f}s]  {cstr}", flush=True)
        self.total_seconds = time.time() - t_start
        return self.geom, fit, self.history
