# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def joirnal():
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    openMdb(pathName='D:/python_run_package/abaqus_model/fem_model.cae')
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    p = mdb.models['Model-1'].parts['Part-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p1 = mdb.models['Model-3Dtest'].parts['Avion_Metal_V6-3']
    session.viewports['Viewport: 1'].setValues(displayedObject=p1)
    p = mdb.models['Model-3Dtest'].parts['Avion_Metal_V6-3']
    s = p.features['Solid extrude-2'].sketch
    mdb.models['Model-3Dtest'].ConstrainedSketch(name='__edit__', objectToCopy=s)
    s1 = mdb.models['Model-3Dtest'].sketches['__edit__']
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, 
        upToFeature=p.features['Solid extrude-2'], filter=COPLANAR_EDGES)
    d[0].setValues(value=275, )
    d[1].setValues(value=24, )
    s1.unsetPrimaryObject()
    p = mdb.models['Model-3Dtest'].parts['Avion_Metal_V6-3']
    p.features['Solid extrude-2'].setValues(sketch=s1)
    del mdb.models['Model-3Dtest'].sketches['__edit__']
    p = mdb.models['Model-3Dtest'].parts['Avion_Metal_V6-3']
    p.regenerate()
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models['Model-3Dtest'].parts['Avion_Metal_V6-3']
    p.generateMesh()
    a = mdb.models['Model-3Dtest'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    mdb.jobs['Job-test-TW-00'].submit(consistencyChecking=OFF)


