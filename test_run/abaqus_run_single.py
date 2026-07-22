# -*- coding: utf-8 -*-
"""
abaqus_run_single.py
====================
Runs ONE parameterised modal analysis of the airplane toy model.
Executed by Abaqus/CAE in batch mode; called by driver.py.

Usage (normally via driver.py, not by hand):
    abaqus cae noGUI=abaqus_run_single.py -- <run_id> <a> <b> <E1> <E2> [mode]

        run_id : integer tag for this sample
        a      : wingtip x-position in the wing sketch [mm]   (baseline ~296.75)
        b      : wingtip chord [mm]                            (baseline ~26.0)
        E1     : Young's modulus of material 'Al4lianjiequyu1' [MPa]
        E2     : Young's modulus of material 'Al4lianjiequyu2' [MPa]
        mode   : optional; "write_inp" = edit/mesh/write .inp and exit
                 (no solve). Omit or "full" = submit + extract (legacy).

Design principles
-----------------
* The master .cae is NEVER saved -> every run starts from the identical
  pristine state, so sketch behaviour is deterministic.
* Wing-outline lines are found by COORDINATES (any line lying outboard of
  the wing root plane), not by hard-coded repository indices, so the script
  survives changes in sketch history.
* Natural frequencies and mode-shape samples at fixed SENSOR COORDINATES
  are written to a per-run JSON file. Nearest-node lookup makes the sensor
  data invariant to re-meshing/node renumbering (needed for MAC).
* Any failure (regeneration, meshing, job error) is caught and recorded as
  status != "ok" so the outer loop can continue.
"""

from abaqus import *
from abaqusConstants import *
import mesh  # noqa: F401  (registers element types)
import os
import sys
import json
import math
import traceback

# ----------------------------------------------------------------------
# 1. CONFIGURATION -- edit these constants to match your setup
# ----------------------------------------------------------------------
# The driver copies the master .cae into each run folder under this name,
# so we always open a private local copy -- never a shared/synced file.
MASTER_CAE   = "master_model.cae"
MODEL_NAME   = "Model-3Dtest"          # or "Model-a1" for the 2D shell model
PART_NAME    = "Avion_Metal_V6-3"      # wing part
FEATURE_NAME = "Solid extrude-2"       # wing extrusion feature
STEP_NAME    = "Step-1"                # frequency step
N_MODES_KEEP = 7                       # non-zero eigenfrequencies to record
N_EIGEN_SOLVE = 30                     # eigenvalues to extract (incl. rigid-body)
FREQ_ZERO_TOL = 0.1                    # Hz; below this = rigid-body / numerical zero

MAT_E1 = "Al4lianjiequyu1"             # joint material 1
MAT_E2 = "Al4lianjiequyu2"             # joint material 2
POISSON = 0.33

# Fixed wing-sketch anchor coordinates (taken from your journal):
ROOT_X       = 24.0                    # wing root plane (|x| <= ROOT_X is fuselage)
EDGE_Y_TOP   = 70.0                    # unswept edge y (top of root chord)
EDGE_Y_BOT   = -70.0                   # opposite root corner y
MIRROR_TOL   = 1.0                     # tolerance when classifying lines [mm]

# Sensor locations for mode-shape extraction, in GLOBAL assembly coords [mm].
# >>> Replace with your actual accelerometer positions from the modal test. <<<
SENSOR_COORDS = [
    (150.0,  60.0, 0.0),
    (250.0,  60.0, 0.0),
    (275.0,  55.0, 0.0),   # near wingtip -- a now in [266, 286]
    (275.0, -55.0, 0.0),
    (-150.0, 60.0, 0.0),
]
SENSOR_DOF = 3            # component of U to sample (1=x, 2=y, 3=z out-of-plane)

WORK_DIR = os.getcwd()    # driver.py launches each run in its own folder

# ----------------------------------------------------------------------
# 2. PARSE ARGUMENTS
# ----------------------------------------------------------------------
# Abaqus prepends its own argv entries. Ours after '--' are either:
#   <run_id> <a> <b> <E1> <E2>
#   <run_id> <a> <b> <E1> <E2> write_inp|full
MODE = "full"
_argv = list(sys.argv)
if len(_argv) >= 2 and str(_argv[-1]).lower() in ("write_inp", "full"):
    MODE = str(_argv[-1]).lower()
    _argv = _argv[:-1]
try:
    run_id = int(_argv[-5])
    a_par  = float(_argv[-4])
    b_par  = float(_argv[-3])
    E1     = float(_argv[-2])
    E2     = float(_argv[-1])
except (ValueError, IndexError):
    raise RuntimeError(
        "Bad arguments. Call as: abaqus cae noGUI=abaqus_run_single.py -- "
        "<run_id> <a> <b> <E1> <E2> [write_inp|full]. Got: {0}".format(
            sys.argv))

JOB_NAME = "run_{0:04d}".format(run_id)
result = {
    "run_id": run_id, "a": a_par, "b": b_par, "E1": E1, "E2": E2,
    "status": "started", "mode": MODE,
    "frequencies_Hz": [], "mode_shapes": {},
    "n_distorted_elements": None, "message": "", "inp_file": "",
}


def dump_result():
    with open(os.path.join(WORK_DIR, "result_{0:04d}.json".format(run_id)), "w") as f:
        json.dump(result, f, indent=2)


# ----------------------------------------------------------------------
# 3. GEOMETRY EDIT -- coordinate-based wing outline replacement
# ----------------------------------------------------------------------
def edit_wing_sketch(model):
    part = model.parts[PART_NAME]

    # The part may carry a lock flag (Model Tree > Lock) which blocks all
    # geometric edits. Unlock it in-memory; the master file is never saved,
    # so its on-disk state is untouched.
    try:
        part.Unlock(reportWarnings=False)
    except TypeError:
        part.Unlock()
    except Exception:
        pass  # not locked / no Unlock needed

    feat = part.features[FEATURE_NAME]

    sk = model.ConstrainedSketch(name="__edit__", objectToCopy=feat.sketch)
    part.projectReferencesOntoSketch(filter=COPLANAR_EDGES, sketch=sk,
                                     upToFeature=feat)

    # --- identify wing-outline lines: any LINE with a point strictly
    #     outboard of the root plane on either side ---------------------
    to_delete = []
    for gid in sk.geometry.keys():
        g = sk.geometry[gid]
        try:
            if g.curveType != LINE:
                continue
            px = g.pointOn[0]
        except Exception:
            continue
        if abs(px) > ROOT_X + MIRROR_TOL:
            to_delete.append(g)

    # delete associated constraints implicitly by deleting geometry
    if to_delete:
        sk.delete(objectList=tuple(to_delete))

    # --- redraw starboard wing from parameters ------------------------
    y_tip_inner = EDGE_Y_TOP - b_par
    l_edge = sk.Line(point1=(ROOT_X, EDGE_Y_TOP), point2=(a_par, EDGE_Y_TOP))
    l_tip  = sk.Line(point1=(a_par, EDGE_Y_TOP),  point2=(a_par, y_tip_inner))
    sk.VerticalConstraint(addUndoState=False, entity=l_tip)
    l_swp  = sk.Line(point1=(a_par, y_tip_inner), point2=(ROOT_X, EDGE_Y_BOT))

    # --- port wing by mirroring about the vertical centreline ---------
    mirror = sk.ConstructionLine(point1=(0.0, 0.0), point2=(0.0, 100.0))
    sk.copyMirror(mirrorLine=mirror, objectList=(l_edge, l_tip, l_swp))

    feat.setValues(sketch=sk)
    del model.sketches["__edit__"]
    part.regenerate()
    return part


# ----------------------------------------------------------------------
# 3b. SECTION ASSIGNMENTS -- picked cell sets die when the geometry is
#     replaced ("geometry association failure"), leaving every element
#     without properties. Record each assignment's cell bounding box from
#     the PRISTINE model, then rebuild the regions after regeneration.
# ----------------------------------------------------------------------
BBOX_TOL = 2.0   # inflation of recorded bounding boxes [mm]


def record_section_regions(part):
    """Return [(index, sectionName, bbox_dict, n_cells), ...] sorted by
    bbox volume ascending (smallest = most specific region first)."""
    info = []
    for i, sa in enumerate(part.sectionAssignments):
        set_name = sa.region[0]
        reg = None
        for repo in (part.sets, part.allInternalSets):
            try:
                if set_name in repo.keys():
                    reg = repo[set_name]
                    break
            except Exception:
                continue
        if reg is None or len(reg.cells) == 0:
            raise RuntimeError(
                "Cannot resolve cells of section assignment {0} "
                "(set '{1}') in the pristine model.".format(i, set_name))
        bb = reg.cells.getBoundingBox()
        vol = 1.0
        for lo, hi in zip(bb["low"], bb["high"]):
            vol *= max(hi - lo, 1e-6)
        info.append({"index": i, "section": sa.sectionName,
                     "bbox": bb, "volume": vol, "n_cells": len(reg.cells)})
    info.sort(key=lambda d: d["volume"])
    return info


def rebuild_section_regions(part, info):
    """Re-point every section assignment at freshly selected cells."""
    all_set = part.Set(name="__auto_all", cells=part.cells[:])
    claimed = []
    for k, item in enumerate(info):
        i = item["index"]
        if k < len(info) - 1:
            bb = item["bbox"]
            cells = part.cells.getByBoundingBox(
                xMin=bb["low"][0] - BBOX_TOL, yMin=bb["low"][1] - BBOX_TOL,
                zMin=bb["low"][2] - BBOX_TOL, xMax=bb["high"][0] + BBOX_TOL,
                yMax=bb["high"][1] + BBOX_TOL, zMax=bb["high"][2] + BBOX_TOL)
            if len(cells) == 0:
                raise RuntimeError(
                    "Section region rebuild: no cells inside recorded bbox "
                    "for assignment {0} ('{1}').".format(i, item["section"]))
            s = part.Set(name="__auto_sect_{0}".format(i), cells=cells)
        else:
            # largest region (the wing) = everything not yet claimed
            if claimed:
                s = part.SetByBoolean(name="__auto_sect_{0}".format(i),
                                      operation=DIFFERENCE,
                                      sets=tuple([all_set] + claimed))
            else:
                s = all_set
        part.sectionAssignments[i].setValues(region=s)
        claimed.append(s)

    if len(claimed) > 1:
        # sanity: joint + wing must cover every cell exactly once
        n_all = len(part.cells)
        n_last = len(claimed[-1].cells)
        n_rest = 0
        for s in claimed[:-1]:
            n_rest += len(s.cells)
        if n_last + n_rest != n_all:
            raise RuntimeError(
                "Section rebuild mismatch: {0}+{1} != {2} cells"
                .format(n_rest, n_last, n_all))


# ----------------------------------------------------------------------
# 4. RESULT EXTRACTION
# ----------------------------------------------------------------------
def nearest_node(instance_nodes, target):
    best, best_d2 = None, 1e30
    for nd in instance_nodes:
        c = nd.coordinates
        d2 = ((c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2
              + (c[2] - target[2]) ** 2)
        if d2 < best_d2:
            best, best_d2 = nd, d2
    return best, math.sqrt(best_d2)


def extract_from_odb(odb_path):
    import odbAccess
    odb = odbAccess.openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps[STEP_NAME]
        freqs, shapes = [], {}

        # map sensors to nearest nodes (search across all instances once)
        sensor_nodes = []
        all_nodes = []
        for inst in odb.rootAssembly.instances.values():
            all_nodes.extend(inst.nodes)
        # report assembly-frame bounding box to help place SENSOR_COORDS
        xs = [nd.coordinates[0] for nd in all_nodes]
        ys = [nd.coordinates[1] for nd in all_nodes]
        zs = [nd.coordinates[2] for nd in all_nodes]
        result["assembly_bbox"] = {
            "low": [float(min(xs)), float(min(ys)), float(min(zs))],
            "high": [float(max(xs)), float(max(ys)), float(max(zs))]}
        for s in SENSOR_COORDS:
            nd, dist = nearest_node(all_nodes, s)
            sensor_nodes.append((nd, dist))

        for fr in step.frames:
            if fr.mode == 0:          # base state
                continue
            if len(freqs) >= N_MODES_KEEP:
                break
            f_hz = float(fr.frequency)
            if f_hz < FREQ_ZERO_TOL:  # skip rigid-body / near-zero modes
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
            # renumber from 1 = first non-zero mode kept
            shapes["mode_{0}".format(len(freqs))] = comp

        result["frequencies_Hz"] = freqs
        result["mode_shapes"] = shapes
        result["sensor_node_distances_mm"] = [
            float(d) for _n, d in sensor_nodes]
    finally:
        odb.close()


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
try:
    cae_path = os.path.join(WORK_DIR, MASTER_CAE)
    if not os.path.exists(cae_path):
        raise RuntimeError(
            "Local master copy not found: {0}. Run via driver.py, which "
            "copies the master .cae into each run folder.".format(cae_path))
    # remove a stale lock left by a crashed session, if any
    lck = os.path.splitext(cae_path)[0] + ".lck"
    if os.path.exists(lck):
        try:
            os.remove(lck)
        except OSError:
            pass
    openMdb(pathName=cae_path)            # fresh, private copy every run
    model = mdb.models[MODEL_NAME]

    # Models saved by an older Abaqus release open with parts AND the
    # assembly locked in a newer release. Unlock both in-memory.
    ra = model.rootAssembly
    for unlock in (lambda: ra.unlock(),
                   lambda: ra.Unlock(reportWarnings=False),
                   lambda: ra.Unlock()):
        try:
            unlock()
            break
        except AttributeError:
            continue
        except Exception:
            break  # already unlocked or unlock not required

    part = model.parts[PART_NAME]
    sect_info = record_section_regions(part)   # BEFORE the geometry changes

    edit_wing_sketch(model)

    rebuild_section_regions(part, sect_info)   # re-point dead picked sets

    model.materials[MAT_E1].elastic.setValues(table=((E1, POISSON),))
    model.materials[MAT_E2].elastic.setValues(table=((E2, POISSON),))

    # request enough eigenvalues so N_MODES_KEEP flexible modes remain
    # after discarding rigid-body / near-zero frequencies
    model.steps[STEP_NAME].setValues(numEigen=N_EIGEN_SOLVE)

    # Keep the existing mesh (GUI workflow: regenerate geometry, do NOT remesh).
    # Forcing generateMesh() after sketch edits produced zero/negative-volume
    # elements on the updated FEM.
    model.rootAssembly.regenerate()
    n_el = len(part.elements)
    if n_el == 0:
        raise RuntimeError(
            "No mesh left after geometry regenerate (a={0}, b={1}). "
            "Master CAE must already contain a mesh; remeshing is disabled."
            .format(a_par, b_par))
    result["n_elements"] = int(n_el)

    job = mdb.Job(name=JOB_NAME, model=MODEL_NAME, type=ANALYSIS,
                  memory=90, memoryUnits=PERCENTAGE,
                  numCpus=4, numDomains=4, numGPUs=0,
                  nodalOutputPrecision=SINGLE, resultsFormat=ODB)

    if MODE == "write_inp":
        # Stage 1 of the INP+solver pipeline: emit .inp, do not solve.
        job.writeInput(consistencyChecking=OFF)
        inp_path = os.path.join(WORK_DIR, JOB_NAME + ".inp")
        if not os.path.exists(inp_path):
            raise RuntimeError("writeInput did not produce " + inp_path)
        result["inp_file"] = JOB_NAME + ".inp"
        result["status"] = "inp_ready"
    else:
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()

        odb_path = os.path.join(WORK_DIR, JOB_NAME + ".odb")
        if not os.path.exists(odb_path):
            raise RuntimeError("ODB not produced -- job failed; see "
                               + JOB_NAME + ".msg/.dat")

        # count distorted-element warnings from the .dat file, if present
        dat = os.path.join(WORK_DIR, JOB_NAME + ".dat")
        if os.path.exists(dat):
            with open(dat, "r") as f:
                txt = f.read()
            result["n_distorted_elements"] = txt.lower().count("distorted")

        extract_from_odb(odb_path)
        if not result["frequencies_Hz"]:
            raise RuntimeError(
                "ODB contains no eigenmodes -- the solver input was "
                "rejected; see " + JOB_NAME + ".dat for ***ERROR lines.")
        result["status"] = "ok"

except Exception:
    result["status"] = "failed"
    result["message"] = traceback.format_exc()

finally:
    dump_result()
    # IMPORTANT: never mdb.save() -- master file must stay pristine
