# -*- coding: utf-8 -*-
"""
extract_odb.py
==============
Stage 3 of the INP+solver pipeline: read one frequency-analysis ODB and
write result_XXXX.json (7 non-zero modes + sensor mode shapes).

Runs with Abaqus Python (no CAE GUI):
    abaqus python extract_odb.py -- <run_id>
    abaqus python extract_odb.py -- <run_id> <odb_path>

Expected cwd: the per-run folder containing run_XXXX.odb (and optionally
a stub result_XXXX.json from generate-inp with parameters already filled).
"""

from __future__ import print_function

import json
import math
import os
import sys
import traceback

STEP_NAME = "Step-1"
N_MODES_KEEP = 7
FREQ_ZERO_TOL = 0.1  # Hz

SENSOR_COORDS = [
    (150.0, 60.0, 0.0),
    (250.0, 60.0, 0.0),
    (275.0, 55.0, 0.0),
    (275.0, -55.0, 0.0),
    (-150.0, 60.0, 0.0),
]
SENSOR_DOF = 3

WORK_DIR = os.getcwd()


def nearest_node(instance_nodes, target):
    best, best_d2 = None, 1e30
    for nd in instance_nodes:
        c = nd.coordinates
        d2 = ((c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2
              + (c[2] - target[2]) ** 2)
        if d2 < best_d2:
            best, best_d2 = nd, d2
    return best, math.sqrt(best_d2)


def extract_from_odb(odb_path, result):
    import odbAccess
    odb = odbAccess.openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps[STEP_NAME]
        freqs, shapes = [], {}

        sensor_nodes = []
        all_nodes = []
        for inst in odb.rootAssembly.instances.values():
            all_nodes.extend(inst.nodes)
        xs = [nd.coordinates[0] for nd in all_nodes]
        ys = [nd.coordinates[1] for nd in all_nodes]
        zs = [nd.coordinates[2] for nd in all_nodes]
        result["assembly_bbox"] = {
            "low": [float(min(xs)), float(min(ys)), float(min(zs))],
            "high": [float(max(xs)), float(max(ys)), float(max(zs))],
        }
        for s in SENSOR_COORDS:
            nd, dist = nearest_node(all_nodes, s)
            sensor_nodes.append((nd, dist))

        for fr in step.frames:
            if fr.mode == 0:
                continue
            if len(freqs) >= N_MODES_KEEP:
                break
            f_hz = float(fr.frequency)
            if f_hz < FREQ_ZERO_TOL:
                continue
            freqs.append(f_hz)
            u = fr.fieldOutputs["U"]
            comp = []
            for nd, _dist in sensor_nodes:
                sub = u.getSubset(region=nd)
                if sub.values:
                    comp.append(float(sub.values[0].data[SENSOR_DOF - 1]))
                else:
                    comp.append(float("nan"))
            shapes["mode_{0}".format(len(freqs))] = comp

        result["frequencies_Hz"] = freqs
        result["mode_shapes"] = shapes
        result["sensor_node_distances_mm"] = [
            float(d) for _n, d in sensor_nodes]
    finally:
        odb.close()


def main():
    # Strip optional trailing "--"
    argv = [a for a in sys.argv[1:] if a != "--"]
    if not argv:
        raise RuntimeError(
            "Usage: abaqus python extract_odb.py -- <run_id> [odb_path]")

    run_id = int(argv[0])
    job_name = "run_{0:04d}".format(run_id)
    odb_path = argv[1] if len(argv) > 1 else os.path.join(
        WORK_DIR, job_name + ".odb")
    result_path = os.path.join(WORK_DIR, "result_{0:04d}.json".format(run_id))

    result = {
        "run_id": run_id,
        "status": "started",
        "frequencies_Hz": [],
        "mode_shapes": {},
        "n_distorted_elements": None,
        "message": "",
    }
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                prev = json.load(f)
            for k in ("a", "b", "E1", "E2", "E1_1e11Pa", "E2_1e11Pa"):
                if k in prev:
                    result[k] = prev[k]
        except Exception:
            pass

    try:
        if not os.path.exists(odb_path):
            raise RuntimeError("ODB not found: " + odb_path)

        dat = os.path.splitext(odb_path)[0] + ".dat"
        if os.path.exists(dat):
            with open(dat, "r") as f:
                txt = f.read()
            result["n_distorted_elements"] = txt.lower().count("distorted")

        extract_from_odb(odb_path, result)
        if not result["frequencies_Hz"]:
            raise RuntimeError(
                "ODB contains no non-zero eigenmodes; see "
                + job_name + ".dat")
        result["status"] = "ok"
    except Exception:
        result["status"] = "failed"
        result["message"] = traceback.format_exc()

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("extract_odb run_id={0} status={1} n_modes={2}".format(
        run_id, result["status"], len(result.get("frequencies_Hz", []))))


if __name__ == "__main__":
    main()
