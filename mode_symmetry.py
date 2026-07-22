# -*- coding: utf-8 -*-
"""
mode_symmetry.py -- run with ABAQUS PYTHON.
    abaqus python mode_symmetry.py -- <run_id>

For each kept mode, compares the out-of-plane (z) motion of the two wing tips
(x>0 tip vs x<0 tip) to classify symmetric vs antisymmetric wing modes -- the
mechanism behind the near-degenerate FE doublet.
"""
from __future__ import print_function

import os
import sys

import numpy as np
from odbAccess import openOdb

STEP_NAME = "Step-1"
FREQ_ZERO_TOL = 0.1
N_KEEP = 7
Z = 2   # out-of-plane component index (0=x,1=y,2=z)


def main():
    args = [a for a in sys.argv[1:] if a != "--"]
    rid = int(args[0])
    odbp = os.path.join("samples", "runs", "run_%04d" % rid,
                        "run_%04d.odb" % rid)
    odb = openOdb(odbp, readOnly=True)
    try:
        step = odb.steps[STEP_NAME]
        coords = []
        for inst in odb.rootAssembly.instances.values():
            for nd in inst.nodes:
                coords.append(nd.coordinates)
        coords = np.asarray(coords, dtype=np.float64)
        x = coords[:, 0]
        xmax, xmin = x.max(), x.min()
        # tip bands: outer 15% of span on each side
        r_tip = x > (xmax - 0.15 * (xmax - xmin))    # x>0 wing tip
        l_tip = x < (xmin + 0.15 * (xmax - xmin))    # x<0 wing tip
        print("run %04d  nodes=%d  x range [%.0f, %.0f]"
              % (rid, len(x), xmin, xmax))
        print("  tip node counts: right(x>0)=%d  left(x<0)=%d"
              % (r_tip.sum(), l_tip.sum()))
        print("%-6s %8s | %9s %9s | %s"
              % ("mode", "freq", "R_tip_uz", "L_tip_uz", "type"))

        n = 0
        for fr in step.frames:
            if fr.mode == 0:
                continue
            f = float(fr.frequency)
            if f < FREQ_ZERO_TOL:
                continue
            if n >= N_KEEP:
                break
            n += 1
            U = fr.fieldOutputs["U"]
            uz = np.concatenate([b.data for b in U.bulkDataBlocks],
                                axis=0)[:, Z].astype(np.float64)
            scale = np.max(np.abs(uz)) or 1.0
            uz = uz / scale
            R = float(np.mean(uz[r_tip]))
            L = float(np.mean(uz[l_tip]))
            if abs(R) < 0.02 and abs(L) < 0.02:
                kind = "in-plane (low uz)"
            elif R * L > 0:
                kind = "SYMMETRIC (wings in phase)"
            else:
                kind = "ANTISYMMETRIC (out of phase)"
            print("f%-5d %8.1f | %+9.3f %+9.3f | %s" % (n, f, R, L, kind))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
