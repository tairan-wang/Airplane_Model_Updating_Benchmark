# -*- coding: utf-8 -*-
"""
mac_extract.py -- run with ABAQUS PYTHON (odbAccess).
    abaqus python mac_extract.py -- <run_id> [run_id2 ...]

Extracts the full-field mode-shape vectors (all node translations) of the 7
kept flexible modes from run_XXXX.odb and computes the auto-MAC matrix. Saves
per run: results dir gets mac_run<ID>.npz with {freqs, mac}. If two runs are
given, also computes the CROSS-MAC between them by matching nodes on their
(possibly different) meshes via nearest-coordinate lookup.
"""
from __future__ import print_function

import os
import sys

import numpy as np
from odbAccess import openOdb

STEP_NAME = "Step-1"
FREQ_ZERO_TOL = 0.1
N_KEEP = 7
OUT_DIR = os.path.join("model_updating", "results")


def mode_field(odb_path):
    """Return (freqs[K], coords[n,3], modes[K, 3n]) for the K kept modes."""
    odb = openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps[STEP_NAME]
        # node coordinates (assembly order == U bulk order for this odb)
        coords = []
        for inst in odb.rootAssembly.instances.values():
            for nd in inst.nodes:
                coords.append(nd.coordinates)
        coords = np.asarray(coords, dtype=np.float64)

        freqs, modes = [], []
        for fr in step.frames:
            if fr.mode == 0:
                continue
            f = float(fr.frequency)
            if f < FREQ_ZERO_TOL:
                continue
            if len(freqs) >= N_KEEP:
                break
            U = fr.fieldOutputs["U"]
            blocks = [b.data for b in U.bulkDataBlocks]
            vec = np.concatenate(blocks, axis=0).astype(np.float64)  # [n,3]
            modes.append(vec.reshape(-1))                            # [3n]
            freqs.append(f)
        return np.asarray(freqs), coords, np.asarray(modes)
    finally:
        odb.close()


def mac_matrix(A, B):
    """MAC between rows of A [KA,D] and rows of B [KB,D]."""
    G = A.dot(B.T)
    da = np.sqrt(np.sum(A * A, axis=1))
    db = np.sqrt(np.sum(B * B, axis=1))
    return (G ** 2) / np.outer(da * da, db * db)


def main():
    args = [a for a in sys.argv[1:] if a != "--"]
    if not args:
        raise RuntimeError("give at least one run_id")
    rids = [int(a) for a in args]
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    data = {}
    for rid in rids:
        odbp = os.path.join("samples", "runs", "run_%04d" % rid,
                            "run_%04d.odb" % rid)
        f, c, M = mode_field(odbp)
        data[rid] = (f, c, M)
        mac = mac_matrix(M, M)
        np.savez(os.path.join(OUT_DIR, "mac_run%04d.npz" % rid),
                 freqs=f, mac=mac)
        print("run %04d  nodes=%d  freqs=%s" % (rid, c.shape[0],
              ", ".join("%.1f" % x for x in f)))
        print("  auto-MAC diag off-check (max off-diagonal): %.3f"
              % np.max(mac - np.eye(len(f))))

    if len(rids) == 2:
        (fa, ca, Ma), (fb, cb, Mb) = data[rids[0]], data[rids[1]]
        # match B's nodes to A's nodes by nearest coordinate, reorder B modes
        # (brute force nearest neighbour; meshes are ~similar size)
        from scipy.spatial import cKDTree
        tree = cKDTree(cb)
        _, idx = tree.query(ca)                      # for each A node -> B node
        # reshape B modes to [K, n, 3], gather idx, flatten
        MbR = Mb.reshape(len(fb), -1, 3)[:, idx, :].reshape(len(fb), -1)
        cross = mac_matrix(Ma, MbR)
        np.savez(os.path.join(OUT_DIR, "mac_cross_%04d_%04d.npz"
                              % (rids[0], rids[1])),
                 freqs_a=fa, freqs_b=fb, mac=cross)
        print("cross-MAC %04d vs %04d saved" % (rids[0], rids[1]))


if __name__ == "__main__":
    main()
