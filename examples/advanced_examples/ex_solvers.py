'''Example with solvers'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import aframe as af
import numpy as np
import time
from VortexAD.core.vlm.vlm_solver import vlm_solver
import matplotlib.pyplot as plt

plot = False

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
    wing = cd.aircraft.components.Wing(AR=7.43, S_ref=174, taper_ratio=0.75, geometry=wing_geometry, tight_fit_ffd=False)

    # wing camber surface
    vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_surface(
        wing, 10, 8, plot=plot, spacing_spanwise='linear', 
        spacing_chordwise='linear', grid_search_density=20
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
    print("total lift", vlm_outputs.total_lift.value)
    print("total forces", vlm_outputs.total_force.value)
    print("total moments", vlm_outputs.total_moment.value)
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(3, 1)
    # plt.rc('text', usetex=False)
    for i in range(len(vlm_outputs.surface_CL)):
        panel_forces = csdl.sum(vlm_outputs.surface_panel_forces[i][0, :, :, :], axes=(0, ))
        shape = panel_forces.shape
        norm_span = np.linspace(-1, 1, shape[0])

        axs[0].plot(norm_span, panel_forces[:, 0].value)
        axs[0].set_xlabel("norm span")
        axs[0].set_ylabel("Fx")
        axs[1].plot(norm_span, panel_forces[:, 1].value)
        axs[1].set_xlabel("norm span")
        axs[1].set_ylabel("Fy")
        axs[2].plot(norm_span, panel_forces[:, 2].value)
        axs[2].set_xlabel("norm span")
        axs[2].set_ylabel("Fz")

        print(f"surface {i} CL", vlm_outputs.surface_CL[i].value)
        print(f"surface {i} CDi", vlm_outputs.surface_CDi[i].value)
        print(f"surface {i} L", vlm_outputs.surface_lift[i].value)
        print(f"surface {i} Di", vlm_outputs.surface_drag[i].value)

    plt.show()
    # exit()
    # Set up beam analysis (with pseudo vlm inputs)
    beam_mesh = mesh_container["beam_mesh"]
    wing_box_beam = beam_mesh.discretizations["wing_box_beam"]
    ttop = wing_box_beam.top_skin_thickness
    tbot = wing_box_beam.bottom_skin_thickness
    tweb = wing_box_beam.shear_web_thickness
    beam_nodes = wing_box_beam.nodal_coordinates[0, :, :]
    num_beam_nodes = beam_nodes.shape[0]

    # Evenly distribute the VLM forces
    vlm_wing_forces = (vlm_outputs.surface_force[0]/ num_beam_nodes).reshape((3, ))
    vlm_wing_forces_exp = csdl.expand(vlm_wing_forces, (num_beam_nodes, 3), 'j->ij')
    beam_loads = csdl.Variable(shape=(num_beam_nodes, 6), value=0.)
    beam_loads = beam_loads.set(csdl.slice[:, 0:3], vlm_wing_forces_exp)
    beam_material = af.Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)
    beam_cs = af.CSBox(
        height=wing_box_beam.beam_height, 
        width=wing_box_beam.beam_width, 
        ttop=ttop, tbot=tbot, tweb=tweb
    )

    beam_1 = af.Beam(name='beam_1', mesh=beam_nodes, material=beam_material, cs=beam_cs)
    beam_1.add_boundary_condition(node=7, dof=[1, 1, 1, 1, 1, 1])
    beam_1.add_load(beam_loads)

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

recorder.stop()
