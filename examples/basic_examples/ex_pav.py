import CADDEE_alpha as cd
import csdl_alpha as csdl
from VortexAD.core.vlm.vlm_solver import vlm_solver
import numpy as np
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.var_groups import RotorAnalysisInputs


recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

# Import and plot the geometry
pav_geometry = cd.import_geometry('pav.stp')

def define_base_config(caddee: cd.CADDEE):
    aircraft = cd.aircraft.components.Aircraft(geometry=pav_geometry)

    # aircraft.quantities.mass_properties.mass = 815
    # aircraft.quantities.mass_properties.cg_vector = np.array([-1, 1., -1])

    # wing
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    wing = cd.aircraft.components.Wing(
        AR=7, S_ref=15, geometry=wing_geometry, tight_fit_ffd=False,
    )
    aircraft.comps["wing"] = wing

    wing.quantities.mass_properties.mass = 250
    wing.quantities.mass_properties.cg_vector = np.array([-3, 0., -1])

    pav_vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_surface(
        wing, 30, 5, grid_search_density=20, ignore_camber=False,
    )
    # pav_geometry.plot_meshes(wing_camber_surface.nodal_coordinates.value)


    # tail
    tail_geometry = aircraft.create_subgeometry(search_names=["Stabilizer"])
    tail = cd.aircraft.components.Wing(
        AR=5, S_ref=2, geometry=tail_geometry, tight_fit_ffd=False,
    )
    aircraft.comps["tail"] = tail

    tail_camber_surface = cd.mesh.make_vlm_surface(
        tail, 30, 6, grid_search_density=10, ignore_camber=True,
    )
    # pav_geometry.plot_meshes(tail_camber_surface.nodal_coordinates.value)

    tail.quantities.mass_properties.mass = 250
    tail.quantities.mass_properties.cg_vector = np.array([-3, 0., -0])

    pav_vlm_mesh.discretizations["wing_camber_surface"] = wing_camber_surface
    pav_vlm_mesh.discretizations["tail_camber_surface"] = tail_camber_surface


    # pusher prop
    pusher_prop_geom = aircraft.create_subgeometry(search_names=["PropPusher"])
    pusher_prop = cd.aircraft.components.Rotor(radius=0.8, geometry=pusher_prop_geom)


    aircraft.comps["pusher_prop"] = pusher_prop

    num_radial = 30
    num_aziumuthal = 30
    rotor_meshes = cd.mesh.RotorMeshes()
    pusher_prop_discretization = cd.mesh.make_rotor_mesh(
        pusher_prop, num_radial, num_aziumuthal, 3
    )
    pusher_prop_discretization.twist_profile = csdl.Variable(shape=(num_radial, ), value=np.deg2rad(np.linspace(50., 20., num_radial)))
    pusher_prop_discretization.chord_profile = csdl.Variable(shape=(num_radial, ), value=np.linspace(0.24, 0.08, num_radial))
    rotor_meshes.discretizations["pusher_prop"] = pusher_prop_discretization
    

    base_config = cd.Configuration(aircraft)
    mesh_container = base_config.mesh_container
    mesh_container["vlm_mesh"] = pav_vlm_mesh
    mesh_container["rotor_meshes"] = rotor_meshes
    caddee.base_configuration = base_config


def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    pitch_angle = csdl.ImplicitVariable(shape=(1, ), value=np.deg2rad(2.))
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1e3,
        range=60e3,
        speed=50.,
        pitch_angle=pitch_angle,
    )
    cruise.configuration = base_config
    conditions["cruise"] = cruise
    return pitch_angle


def define_analysis(caddee: cd.CADDEE, pitch_angle=None):
    cruise = caddee.conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    
    aircraft = cruise_config.system
    tail = aircraft.comps["tail"]
    elevator = csdl.ImplicitVariable(shape=(1, ), value=0.)
    tail.actuate(elevator)


    cruise.finalize_meshes()

    # VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]

    wing_camber_surface = vlm_mesh.discretizations["wing_camber_surface"]
    tail_camber_surface = vlm_mesh.discretizations["tail_camber_surface"]

    camber_surface_coordinates = [wing_camber_surface.nodal_coordinates, tail_camber_surface.nodal_coordinates]
    camber_surface_nodal_velocities = [wing_camber_surface.nodal_velocities, tail_camber_surface.nodal_velocities]

    vlm_outputs = vlm_solver(camber_surface_coordinates, camber_surface_nodal_velocities, alpha_ML=None)

    # BEM analysis
    rotor_meshes = mesh_container["rotor_meshes"]
    push_prop_discretization = rotor_meshes.discretizations["pusher_prop"]

    ml_model = NACA4412MLAirfoilModel()
    bem_model = BEMModel(num_nodes=cruise._num_nodes, airfoil_model=ml_model)
    rpm=csdl.ImplicitVariable(shape=(1, ), value=2000.)
    bem_inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=push_prop_discretization.nodal_velocities,
        mesh_parameters=push_prop_discretization,
    )

    bem_outputs = bem_model.evaluate(bem_inputs)

    print(bem_outputs.forces.value)
    print(bem_outputs.moments.value)
    print(vlm_outputs.total_force.value)
    print(vlm_outputs.total_moment.value)

    total_forces, total_moments = cruise.assemble_forces_and_moments(
        aero_propulsive_forces=[bem_outputs.forces, vlm_outputs.total_force],
        aero_propulsive_moments=[bem_outputs.moments, vlm_outputs.total_moment],
    )

    cruise_config.assemble_system_mass_properties()
    ac_mps = aircraft.quantities.mass_properties

    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()

    print(total_forces.value)
    print(total_moments.value)

    # ang_to_lin_accel = csdl.cross( total_moments

    acceleartions = eom_model.evaluate(
        total_forces, total_moments,
        cruise.quantities.ac_states, ac_mps
    )

    print(acceleartions.accel_norm.value)
    print(acceleartions.du_dt.value)
    print(acceleartions.dv_dt.value)
    print(acceleartions.dw_dt.value)
    # print(total_forces.value)
    # print(total_moments.value)

    # z_force = total_forces[0, 2]

    # solver = csdl.nonlinear_solvers.BracketedSearch(max_iter=30)
    # solver.add_state(pitch_angle, z_force, bracket=(np.deg2rad(-20), np.deg2rad(20)))
    # solver.run()

    # print(pitch_angle.value * 180/np.pi)



    
    

define_base_config(caddee)


pitch_angle = define_conditions(caddee)


define_analysis(caddee, pitch_angle)
