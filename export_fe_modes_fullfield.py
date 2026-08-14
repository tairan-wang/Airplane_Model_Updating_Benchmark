# -*- coding: utf-8 -*-
"""
export_fe_modes_fullfield.py -- run with ABAQUS PYTHON (odbAccess).
    abaqus python export_fe_modes_fullfield.py -- <run_id> [n_modes] [odb_path]

Exports the full-field mode shapes + node coordinates + eigenfrequencies of the
kept flexible modes from a frequency-analysis ODB, into a plain .npz that the
(non-Abaqus) experiment-vs-FE cross-MAC in mac_analysis can read.

This complements the existing extract_odb.py (which stores only 5 sensor points)
and mac_extract.py (which stores only the MAC matrix, not the shapes/coords).

Writes:  fe_modes_run<ID>.npz  with
    freqs   : (K,)      eigenfrequencies [Hz]
    coords  : (n, 3)    node coordinates (assembly order)
    modes   : (K, n, 3) nodal displacement mode shapes (Ux,Uy,Uz)
    node_ids: (n,)      node labels (for reference)
"""
from __future__ import print_function

import os
import sys
import numpy as np
from odbAccess import openOdb

STEP_NAME = "Step-1"
FREQ_ZERO_TOL = 0.1
N_KEEP_DEFAULT = 13            # cover the experiment out to ~350 Hz (was 7)


def main():
    argv = [a for a in sys.argv[1:] if a != "--"]
    if not argv:
        raise RuntimeError(
            "Usage: abaqus python export_fe_modes_fullfield.py -- <run_id> [n_modes] [odb]")
    rid = int(argv[0])
    n_keep = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else N_KEEP_DEFAULT
    job = "run_%04d" % rid
    rest = [a for a in argv[1:] if not a.isdigit()]
    odb_path = rest[0] if rest else os.path.join(
        "samples", "runs", job, job + ".odb")

    odb = openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps[STEP_NAME]
        coords, node_ids = [], []
        for inst in odb.rootAssembly.instances.values():
            for nd in inst.nodes:
                coords.append(nd.coordinates)
                node_ids.append(nd.label)
        coords = np.asarray(coords, dtype=np.float64)
        node_ids = np.asarray(node_ids, dtype=np.int64)

        freqs, modes = [], []
        for fr in step.frames:
            if fr.mode == 0:
                continue
            f = float(fr.frequency)
            if f < FREQ_ZERO_TOL:
                continue
            if len(freqs) >= n_keep:
                break
            U = fr.fieldOutputs["U"]
            vec = np.concatenate([b.data for b in U.bulkDataBlocks],
                                 axis=0).astype(np.float64)   # (n,3)
            modes.append(vec)
            freqs.append(f)

        freqs = np.asarray(freqs)
        modes = np.asarray(modes)                              # (K,n,3)
        out = "fe_modes_%s.npz" % job
        np.savez(out, freqs=freqs, coords=coords, modes=modes, node_ids=node_ids)
        print("wrote %s  K=%d modes  n=%d nodes  freqs=%s"
              % (out, len(freqs), coords.shape[0],
                 ", ".join("%.2f" % x for x in freqs)))
    finally:
        odb.close()


if __name__ == "__main__":
    main()
