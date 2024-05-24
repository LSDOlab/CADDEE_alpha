import CADDEE_alpha as cd
import csdl_alpha as csdl
import aframe as af
import numpy as np
import time
from VortexAD.core.vlm.vlm_solver import vlm_solver
import matplotlib.pyplot as plt
import lsdo_function_spaces as fs
import aeroelastic_coupling_utils as acu
from ex_utils import plot_vlm

# NOTE: wip


plot = False

recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

# Import and plot the geometry
c_172_geometry = cd.import_geometry('pseudo_c172_cambered.stp')
c_172_geometry.plot()

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component 
    aircraft = cd.aircraft.components.Aircraft(geometry=c_172_geometry)

    airframe = aircraft.comps["airframe"] = cd.Component()

    # wing comp
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing, 0, 2", 
                                                            "Wing, 0, 3",
                                                            "Wing, 1, 8",
                                                            "Wing, 1, 9"])
    wing = cd.aircraft.components.Wing(AR=7.43, S_ref=174, taper_ratio=0.75, geometry=wing_geometry, tight_fit_ffd=False)

    # wing function spaces
    force_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, grid_size=8, order=2, conserve=True)
    wing.quantities.force_space = wing_geometry.create_parallel_space(force_space)
    wing.quantities.force_single_space = force_space

    pressure_space = fs.BSplineSpace(2, (5,5), (7,7))
    wing.quantities.pressure_space = wing_geometry.create_parallel_space(pressure_space)

    displacement_space = fs.BSplineSpace(2, (3,3), (5,5))
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(displacement_space)
    wing.quantities.displacement_single_space = displacement_space

    # wing camber surface
    vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_surface(
        wing, 16, 5, plot=plot, spacing_spanwise='linear', 
        spacing_chordwise='linear', grid_search_density=2
    )
    vlm_mesh.discretizations["wing_camber_surface"] = wing_camber_surface
    # c_172_geometry.plot_meshes(wing_camber_surface.nodal_coordinates)

    # wing box beam
    beam_mesh = cd.mesh.BeamMesh()
    num_beam_nodes = 15
    wing_box_beam = cd.mesh.make_1d_box_beam(wing, num_beam_nodes, 0.5, plot=plot)
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
    tail_camber_surface = cd.mesh.make_vlm_surface(
        h_tail, 10, 8, plot=plot
    )
    vlm_mesh.discretizations["tail_camber_surface"] = tail_camber_surface
    # c_172_geometry.plot_meshes(tail_camber_surface.nodal_coordinates)
    
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
    
    cruise.finalize_meshes()

    force_space = wing.quantities.force_space
    displacement_space = wing.quantities.displacement_space

    disp_coeff_implicit, implicit_disp = displacement_space.generate_implicit_function(num_physical_dimensions=3)
    force_coeff_implicit, implicit_force = force_space.generate_implicit_function(num_physical_dimensions=3)

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_camber_surface = vlm_mesh.discretizations["wing_camber_surface"]
    tail_camber_surface = vlm_mesh.discretizations["tail_camber_surface"]
    wing_camber_mesh = wing_camber_surface.nodal_coordinates

    # deform vlm mesh via displacement function
    # TODO: these 3 lines can probably be abstracted
    transfer_mesh_para = implicit_force.generate_parametric_grid((20, 20))
    transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)
    transfer_mesh_disp = implicit_disp.evaluate(transfer_mesh_para)
    
    # TODO: these 3 lines can probably be abstracted
    map = acu.NodalMap()
    weights = map.evaluate(csdl.reshape(wing_camber_mesh, (np.prod(wing_camber_mesh.shape[0:-1]), 3)), transfer_mesh_phys)
    wing_camber_mesh_displacement = (weights @ transfer_mesh_disp).reshape(wing_camber_mesh.shape)

    camber_surface_coordinates = [wing_camber_mesh + wing_camber_mesh_displacement, tail_camber_surface.nodal_coordinates]
    camber_surface_nodal_velocities = [wing_camber_surface.nodal_velocities, tail_camber_surface.nodal_velocities]

    # run vlm solver
    vlm_outputs = vlm_solver(camber_surface_coordinates, camber_surface_nodal_velocities)
    plot_vlm(vlm_outputs)

    # VLM forces to oml (sifr)

    # VLM to framework
    areas = vlm_outputs.surface_panel_areas[0]
    forces = vlm_outputs.surface_panel_forces[0]
    pressures = csdl.reshape(csdl.norm(forces, axes=(3,)) / areas, (np.prod(forces.shape[0:-1]), 1))
    forces = csdl.reshape(forces, (np.prod(forces.shape[0:-1]), 3))
    camber_force_points = vlm_outputs.surface_panel_force_points[0].value.reshape(-1,3)
    split_fp = np.vstack((camber_force_points + np.array([0, 0, -1]), camber_force_points + np.array([0, 0, 1])))
    oml_fp_para = wing.geometry.project(split_fp, plot=plot)

    force_function = force_space.fit_function_set(csdl.blockmat([[forces/2], [forces/2]]), oml_fp_para)
    wing.quantities.force_function = force_function

    pressure_function = wing.quantities.pressure_space.fit_function_set(csdl.blockmat([[pressures/2], [pressures/2]]), 
                                                                        oml_fp_para, regularization_parameter=1)
    
    # Set up beam analysis
    beam_mesh = mesh_container["beam_mesh"]
    wing_box_beam = beam_mesh.discretizations["wing_box_beam"]
    ttop = wing_box_beam.top_skin_thickness
    tbot = wing_box_beam.bottom_skin_thickness
    tweb = wing_box_beam.shear_web_thickness
    beam_nodes = wing_box_beam.nodal_coordinates[0, :, :]
    num_beam_nodes = beam_nodes.shape[0]

    # implicit framework to beam

    transfer_mesh_forces = implicit_force.evaluate(transfer_mesh_para)
    nodal_map = acu.NodalMap()
    weights = nodal_map.evaluate(transfer_mesh_phys, beam_nodes)
    beam_nodal_forces = weights.T() @ transfer_mesh_forces

    # # show that total force is conserved
    # print("beam nodal forces", np.sum(beam_nodal_forces.value, axis=0))
    # print('vlm wing force', vlm_outputs.surface_force[0].value)

    # beam analysis
    beam_material = af.Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)
    beam_cs = af.CSBox(
        height=wing_box_beam.beam_height, 
        width=wing_box_beam.beam_width, 
        ttop=ttop, tbot=tbot, tweb=tweb
    )

    beam_1 = af.Beam(name='beam_1', mesh=beam_nodes, material=beam_material, cs=beam_cs)
    beam_1.add_boundary_condition(node=7, dof=[1, 1, 1, 1, 1, 1])
    beam_loads = csdl.Variable(shape=(num_beam_nodes, 6), value=0.)
    beam_loads = beam_loads.set(csdl.slice[:, 0:3], beam_nodal_forces)
    beam_1.add_load(beam_loads)

    frame = af.Frame()
    frame.add_beam(beam_1)

    solution = frame.evaluate()

    # displacement
    beam_1_displacement = solution.get_displacement(beam_1)
    print("beam displacement", beam_1_displacement.value)

    # displacement to oml
    nodal_displacement = weights @ beam_1_displacement
    displacement_function = displacement_space.fit_function_set(nodal_displacement, transfer_mesh_para)
    wing.quantities.displacement_function = displacement_function

    # solve implicit system
    solver = csdl.nonlinear_solvers.GaussSeidel(max_iter=10)
    disp_residual = displacement_function.stack_coefficients() - disp_coeff_implicit
    force_residual = force_function.stack_coefficients() - force_coeff_implicit
    solver.add_state(disp_coeff_implicit, disp_residual)
    solver.add_state(force_coeff_implicit, force_residual)
    solver.run()

    # look at results
    plot_vlm(vlm_outputs)
    wing.geometry.plot(color=displacement_function)
    wing.geometry.plot(color=pressure_function)

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

# This takes some time
# recorder.visualize_graph('coupled_sifr')

recorder.stop()
