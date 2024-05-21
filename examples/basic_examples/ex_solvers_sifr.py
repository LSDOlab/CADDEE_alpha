import CADDEE_alpha as cd
import csdl_alpha as csdl
import aframe as af
import numpy as np
import time
from VortexAD.core.vlm.vlm_solver import vlm_solver
import lsdo_function_spaces as fs
import aeroelastic_coupling_utils as acu


recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

# Import and plot the geometry
c_172_geometry = cd.import_geometry('pseudo_c172_cambered.stp')

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component 
    aircraft = cd.aircraft.components.Aircraft(geometry=c_172_geometry)

    airframe = aircraft.comps["airframe"] = cd.Component()

    # wing comp
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    wing = cd.aircraft.components.Wing(AR=7.43, S_ref=174, taper_ratio=0.75, geometry=wing_geometry)

    # wing function spaces
    wing.quantities.ind2name = wing_geometry.b_spline_names
    wing.quantities.name2ind = {name: i for i, name in enumerate(wing_geometry.b_spline_names)}

    n = 10
    grid = np.zeros((n**2, 2))
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    for i in range(n):
        for j in range(n):
            index = i * n + j
            grid[index] = [x[i], y[j]]

    force_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, points=grid, order=2, conserve=True)
    wing.quantities.force_space = fs.FunctionSetSpace(2, [force_space] * len(wing.quantities.ind2name))

    displacement_space = fs.BSplineSpace(2, (3,3), (5,5))
    wing.quantities.displacement_space = fs.FunctionSetSpace(2, [displacement_space] * len(wing.quantities.ind2name))

    # wing camber surface
    vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_camber_surface(
        wing, 10, 8, plot=False, spacing_spanwise='linear', 
        spacing_chordwise='linear', grid_search_density=20
    )
    vlm_mesh.discretizations["wing_camber_surface"] = wing_camber_surface
    # c_172_geometry.plot_meshes(wing_camber_surface.nodal_coordinates)

    # wing box beam
    beam_mesh = cd.mesh.BeamMesh()
    num_beam_nodes = 15
    wing_box_beam = cd.mesh.make_1d_box_beam(wing, num_beam_nodes, 0.5, plot=False)
    beam_mesh.discretizations["wing_box_beam"] = wing_box_beam
    wing_box_beam.shear_web_thickness = csdl.Variable(value=np.ones(num_beam_nodes - 1) * 0.01)
    wing_box_beam.top_skin_thickness = csdl.Variable(value=np.ones(num_beam_nodes - 1) * 0.01)
    wing_box_beam.bottom_skin_thickness = csdl.Variable(value=np.ones(num_beam_nodes - 1) * 0.01)

    # add wing to airframe
    airframe.comps["wing"] = wing


    # tail comp
    h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail"])
    h_tail = cd.aircraft.components.Wing(
        S_ref=21.56, span=11.12, AR=None, 
        taper_ratio=0.75, geometry=h_tail_geometry
    )

    # tail camber surface
    tail_camber_surface = cd.mesh.make_vlm_camber_surface(
        h_tail, 8, 3
    )
    vlm_mesh.discretizations["tail_camber_surface"] = tail_camber_surface
    
    # add tail to airframe
    airframe.comps["h_tail"] = h_tail

    # rotor comp
    rotor_geometry = aircraft.create_subgeometry(search_names=["Disk", "PropGeom"])
    rotor = cd.aircraft.components.Rotor(
        radius=1.,
        geometry=rotor_geometry,
    )

    # rotor meshes
    rotor_meshes = cd.mesh.RotorMeshes()
    
    main_rotor_mesh = cd.mesh.make_rotor_mesh(
        rotor_comp=rotor,
        num_radial=30, 
        num_azimuthal=30,
    )
    rotor_meshes.discretizations["main_rotor_mesh"] = main_rotor_mesh

    # add rotor to airframe
    airframe.comps["rotor"] = rotor

    # create the base configuration
    base_config = cd.Configuration(system=aircraft)
    caddee.base_configuration = base_config

    # assign meshes to mesh container
    mesh_container = base_config.mesh_container
    mesh_container["vlm_mesh"] = vlm_mesh
    mesh_container["beam_mesh"] = beam_mesh
    mesh_container["rotor_meshes"] = rotor_meshes

def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=3e3,
        range=60e3,
        mach_number=np.array([0.2]),
    )
    cruise.configuration = base_config
    conditions["cruise"] = cruise


def define_analysis(caddee: cd.CADDEE):
    cruise = caddee.conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    wing = cruise_config.system.comps['airframe'].comps['wing']
    # big fan of it saying what keys exist in the error message
    
    cruise.finalize_meshes()

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_camber_surface = vlm_mesh.discretizations["wing_camber_surface"]
    tail_camber_surface = vlm_mesh.discretizations["tail_camber_surface"]

    camber_surface_coordinates = [wing_camber_surface.nodal_coordinates, tail_camber_surface.nodal_coordinates]
    camber_surface_nodal_velocities = [wing_camber_surface.nodal_velocities, tail_camber_surface.nodal_velocities]

    # run vlm solver
    vlm_outputs = vlm_solver(camber_surface_coordinates, camber_surface_nodal_velocities)
    print("total drag", vlm_outputs.total_drag.value)
    print("total CD", vlm_outputs)
    print("total lift", vlm_outputs.total_lift.value)
    print("total forces", vlm_outputs.total_force.value)
    print("total moments", vlm_outputs.total_moment.value)
    for i in range(len(vlm_outputs.surface_CL)):
        print(f"surface {i} panel forces", vlm_outputs.surface_panel_forces[i].value)
        print(f"surface {i} CL", vlm_outputs.surface_CL[i].value)
        print(f"surface {i} CDi", vlm_outputs.surface_CDi[i].value)
        print(f"surface {i} L", vlm_outputs.surface_lift[i].value)
        print(f"surface {i} Di", vlm_outputs.surface_drag[i].value)

    # VLM to beam
    forces = vlm_outputs.surface_panel_forces[0]
    forces = csdl.reshape(forces, (np.prod(forces.shape[0:-1]), 3))
    force_points = vlm_outputs.surface_panel_force_points[0]
    # This is a bit of a problem if projected points change with the geometry
    force_points_np = force_points.value.reshape((-1, 3))

    projected_force_points = wing.geometry.project(force_points_np, plot=False)
    for i in range(len(projected_force_points)):
        projected_force_points[i] = (wing.quantities.name2ind[projected_force_points[i][0]], projected_force_points[i][1])

    wing.quantities.force_function = force_function = wing.quantities.force_space.fit_function_set(forces, projected_force_points)

    transfer_mesh = wing.quantities.displacement_space.generate_parametric_grid((10, 10))
    beam_nodes = wing_box_beam.nodal_coordinates[0, :, :]
    transfer_forces = force_function.evaluate(transfer_mesh)
    nodal_map = acu.NodalMap()
    weights = nodal_map.evaluate(transfer_mesh, beam_nodes)
    beam_nodal_forces = weights @ transfer_forces

    # Set up beam analysis (with pseudo vlm inputs)
    beam_mesh = mesh_container["beam_mesh"]
    wing_box_beam = beam_mesh.discretizations["wing_box_beam"]
    ttop = wing_box_beam.top_skin_thickness
    tbot = wing_box_beam.bottom_skin_thickness
    tweb = wing_box_beam.shear_web_thickness

    beam_material = af.Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)
    beam_cs = af.CSBox(
        height=wing_box_beam.beam_height, 
        width=wing_box_beam.beam_width, 
        ttop=ttop, tbot=tbot, tweb=tweb
    )

    beam_1 = af.Beam(name='beam_1', mesh=beam_nodes, material=beam_material, cs=beam_cs)
    beam_1.add_boundary_condition(node=7, dof=[1, 1, 1, 1, 1, 1])
    beam_1.add_load(beam_nodal_forces)

    frame = af.Frame()
    frame.add_beam(beam_1)

    solution = frame.evaluate()

    # displacement
    beam_1_displacement = solution.get_displacement(beam_1)
    print("beam displacement", beam_1_displacement.value)

    # stress
    beam_1_stress = solution.get_stress(beam_1)
    print("beam stress", beam_1_stress.value)

    # cg
    cg = solution.cg
    dcg = solution.dcg

    print('cg: ', cg.value)
    print('deformed cg: ', dcg.value)


ts = time.time()
define_base_config(caddee)

define_conditions(caddee)

define_analysis(caddee)
tf = time.time()
print("total time", tf-ts)


