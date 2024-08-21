'''Handling qualities optimization example'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from VortexAD.core.vlm.vlm_solver import vlm_solver
from CADDEE_alpha.utils.coordinate_transformations import perform_local_to_body_transformation
from modopt import CSDLAlphaProblem, SLSQP


# Start the CSDL recorder
recorder = csdl.Recorder(inline=True, expand_ops=True)
recorder.start()

# import C172 geometry
c172_geom = cd.import_geometry("c172.stp")
plotting_elements = c172_geom.plot(show=False, opacity=0.5, color='#FFCD00')

# make instance of CADDEE class
caddee = cd.CADDEE()

def define_base_config(caddee : cd.CADDEE):
    """Build the system configuration and define meshes."""

    # Make aircraft component and pass in the geometry
    aircraft = cd.aircraft.components.Aircraft(geometry=c172_geom, compute_surface_area=False)

    # instantiation configuration object and pass in system component (aircraft)
    base_config = cd.Configuration(system=aircraft)

    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselag"])
    fuselage_length = csdl.Variable(name="fuselage_length", value=7.5)
    fuselage_length.set_as_design_variable(lower=0.8*7.5, upper=1.2*7.5)
    fuselage = cd.aircraft.components.Fuselage(
        length=fuselage_length, 
        max_height=1.4,
        max_width=1.,
        geometry=fuselage_geometry)

    # Assign fuselage component to aircraft
    aircraft.comps["fuselage"] = fuselage

    # Make wing geometry from aircraft component and instantiate wing component
    wing_geometry = aircraft.create_subgeometry(search_names=["MainWing"])
    aspect_ratio = csdl.Variable(name="wing_aspect_ratio", value=7.72)
    wing_area = csdl.Variable(name="wing_area", value=16.23)
    wing_root_twist = csdl.Variable(name="wing_root_twist", value=0)
    wing_tip_twist = csdl.Variable(name="wing_tip_twist", value=0)
    
    # Set design variables for wing
    aspect_ratio.set_as_design_variable(upper=1.2 * 7.72, lower=0.8 * 7.72, scaler=1/8)
    wing_area.set_as_design_variable(upper=1.2 * 16.23, lower=0.8 * 16.23, scaler=1/16)
    wing_root_twist.set_as_design_variable(upper=np.deg2rad(5), lower=np.deg2rad(-5), scaler=4)
    wing_tip_twist.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=2)
    
    wing = cd.aircraft.components.Wing(
        AR=aspect_ratio, S_ref=wing_area, 
        taper_ratio=0.73, root_twist_delta=wing_root_twist,
        tip_twist_delta=wing_tip_twist, geometry=wing_geometry
    )

    # Assign wing component to aircraft
    aircraft.comps["wing"] = wing

    # Make horizontal tail geometry & component
    h_tail_geometry = aircraft.create_subgeometry(search_names=["HTail"])
    h_tail_AR = csdl.Variable(name="h_tail_AR", value=3.83)
    h_tail_area = csdl.Variable(name="h_tail_area", value=4.04)

    h_tail_AR.set_as_design_variable(lower=0.8 * 3.83, upper=1.5 * 3.83, scaler=1/4)
    h_tail_area.set_as_design_variable(lower=0.8 * 4.04, upper=1.2 * 4.04, scaler=1/4)
    h_tail = cd.aircraft.components.Wing(AR=h_tail_AR, S_ref=h_tail_area, taper_ratio=0.60, geometry=h_tail_geometry)

    # Assign tail component to aircraft
    aircraft.comps["h_tail"] = h_tail

    # Make vertical tail geometry & componen
    v_tail_geometry = aircraft.create_subgeometry(search_names=["VerticalTail"])
    v_tail = cd.aircraft.components.Wing(
        AR=1.26, S_ref=1.94, geometry=v_tail_geometry, 
        skip_ffd=True, orientation="vertical"
    )

    # Assign v-tail component to aircraft
    aircraft.comps["v_tail"] = v_tail

    # propeller
    prop_geom = aircraft.create_subgeometry(search_names=["PropDisk", "Hub", "PropGeom"])
    propeller = cd.aircraft.components.Rotor(
        radius=1.88/2, geometry=prop_geom, compute_surface_area=False, skip_ffd=True,
    )
    aircraft.comps["propeller"] = propeller

    # Connect component geometries
    base_config.connect_component_geometries(fuselage, wing, 0.75 * wing.LE_center + 0.25 * wing.TE_center)
    base_config.connect_component_geometries(fuselage, h_tail, h_tail.TE_center)
    base_config.connect_component_geometries(fuselage, v_tail, v_tail.TE_root)
    base_config.connect_component_geometries(fuselage, propeller, connection_point=fuselage.nose_point)

    # tail moment arem
    wing_qc = 0.75 * wing.LE_center + 0.25 * wing.TE_center
    h_tail_qc = 0.75 * h_tail.LE_center + 0.25 * h_tail.TE_center
    tail_moment_arm = csdl.norm(wing_qc - h_tail_qc)

    # components w/o geometry
    avionics = cd.Component()
    aircraft.comps["avionics"] = avionics

    landing_gear = cd.Component()
    aircraft.comps["landing_gear"] = landing_gear

    payload = cd.Component()
    aircraft.comps["payload"] = payload

    engine = cd.Component()
    aircraft.comps["engine"] = engine

    # Meshing
    mesh_container = base_config.mesh_container

    # Tail 
    tail_chord_surface = cd.mesh.make_vlm_surface(
        wing_comp=h_tail,
        num_chordwise=1, 
        num_spanwise=10,
    )

    # Wing chord surface (lifting line)
    wing_chord_surface = cd.mesh.make_vlm_surface(
        wing_comp=wing,
        num_chordwise=15,
        num_spanwise=30,
    )
    vlm_mesh = cd.mesh.VLMMesh()
    vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
    vlm_mesh.discretizations["h_tail_chord_surface"] = tail_chord_surface

    num_radial = 25
    rotor_meshes = cd.mesh.RotorMeshes()
    propeller_discretization = cd.mesh.make_rotor_mesh(
        propeller, num_radial=num_radial, num_azimuthal=1, num_blades=2
    )
    rotor_meshes.discretizations["propeller_discretization"] = propeller_discretization

    # plot meshes
    # c172_geom.plot_meshes(meshes=[wing_chord_surface.nodal_coordinates.value, tail_chord_surface.nodal_coordinates.value])
    
    # Assign mesh to mesh container
    mesh_container["vlm_mesh"] = vlm_mesh
    mesh_container["rotor_meshes"] = rotor_meshes

    # Set up the geometry: this will run the inner optimization
    base_config.setup_geometry(plot=True)

    # tail moment arem
    wing_qc = 0.75 * wing.LE_center + 0.25 * wing.TE_center
    h_tail_qc = 0.75 * h_tail.LE_center + 0.25 * h_tail.TE_center
    tail_moment_arm = csdl.norm(wing_qc - h_tail_qc)
    print("tail moment arm", tail_moment_arm.value)

    # Assign base configuration to CADDEE instance
    caddee.base_configuration = base_config

def define_conditions(caddee: cd.CADDEE):
    """Define the operating conditions of the aircraft."""
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    pitch_angle = csdl.Variable(name="pitch_angle", value=0.)
    pitch_angle.set_as_design_variable(upper=np.deg2rad(2.5), lower=np.deg2rad(-2), scaler=4)
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=2500,
        range=736.3 * cd.Units.length.mile_to_m,
        pitch_angle=pitch_angle,
        mach_number=0.18,
    )
    cruise.configuration = base_config.copy()
    conditions["cruise"] = cruise

def define_mass_properties(caddee: cd.CADDEE):
    """Define the mass properties of the aircraft."""

    base_config = caddee.base_configuration
    aircraft = base_config.system

    conditions = caddee.conditions
    cruise : cd.aircraft.conditions.CruiseCondition = conditions["cruise"]
    dynamic_pressure = 0.5 * cruise.quantities.atmos_states.density * cruise.parameters.speed**2

    design_gross_weight = csdl.Variable(name="design_gross_weight", value=1100)
    fuel_weight = csdl.Variable(name="fuel_weight", value=250*cd.Units.mass.pound_to_kg)

    ga_aviation_weights = cd.aircraft.models.weights.general_aviation_weights.GeneralAviationWeights(
        design_gross_weight=design_gross_weight,
        dynamic_pressure=dynamic_pressure,
    )

    wing : cd.aircraft.components.Wing = aircraft.comps["wing"]
    wing_center = (wing.LE_center + wing.TE_center) / 2
    wing_qc = 0.75 * wing.LE_center + 0.25 * wing.TE_center
    wing_mass = ga_aviation_weights.evaluate_wing_weight(
        S_ref=wing.parameters.S_ref,
        fuel_weight=fuel_weight,
        AR=wing.parameters.AR,
        sweep=0., taper_ratio=0.72,
        thickness_to_chord=0.13,
    )
    wing.quantities.mass_properties.mass = wing_mass + fuel_weight
    wing.quantities.mass_properties.cg_vector = wing_center

    fuselage : cd.aircraft.components.Fuselage = aircraft.comps["fuselage"]
    fuselage_mass = ga_aviation_weights.evaluate_fuselage_weight(
        S_wet=fuselage.parameters.S_wet,
    )
    fuselage.quantities.mass_properties.mass = fuselage_mass
    fuselage.quantities.mass_properties.cg_vector = wing_center + np.array([0., 0., 0.5])
    
    h_tail : cd.aircraft.components.Wing = aircraft.comps["h_tail"]
    h_tail_center = (h_tail.LE_center + h_tail.TE_center) / 2
    h_tail_mass = ga_aviation_weights.evaluate_horizontal_tail_weight(
        S_ref=h_tail.parameters.S_ref,
    )
    h_tail.quantities.mass_properties.mass = h_tail_mass
    h_tail.quantities.mass_properties.cg_vector = h_tail_center
    
    v_tail : cd.aircraft.components.Wing = aircraft.comps["v_tail"]
    v_tail_mass = ga_aviation_weights.evaluate_vertical_tail_weight(
        S_ref=v_tail.parameters.S_ref,
        AR=v_tail.parameters.AR,
        thickness_to_chord=0.1,
        sweep_c4=np.deg2rad(20),
    )
    v_tail.quantities.mass_properties.mass = v_tail_mass
    v_tail.quantities.mass_properties.cg_vector = h_tail_center - np.array([0., 0., 0.5])

    avionics = aircraft.comps["avionics"]
    avionics_mass = ga_aviation_weights.evaluate_avionics_weight(
        design_range=cruise.parameters.range,
        num_flight_crew=1.,
        fuselage_plan_form_area=None,
        fuselage_length=fuselage.parameters.length,
        fuselage_width=1.1,
        correction_factor=0.5,
    )
    avionics.quantities.mass_properties.mass = avionics_mass
    avionics.quantities.mass_properties.cg_vector = fuselage.nose_point + np.array([-0.3, 0.,0.])

    landing_gear = aircraft.comps["landing_gear"]
    landing_gear_mass = ga_aviation_weights.evaluate_main_landing_gear_weight(
        fuselage_length=fuselage.parameters.length,
        design_range=cruise.parameters.range,
        W_ramp=design_gross_weight*1.02,
        correction_factor=0.4,
    )
    landing_gear.quantities.mass_properties.mass = landing_gear_mass
    landing_gear.quantities.mass_properties.cg_vector = wing_center

    engine = aircraft.comps["engine"]
    engine_mass = csdl.Variable(name="engine_mass", value=120)
    engine.quantities.mass_properties.mass = engine_mass
    engine.quantities.mass_properties.cg_vector = fuselage.nose_point + np.array([-0.1, 0.,0.])

    payload = aircraft.comps["payload"]
    payload_mass = csdl.Variable(name="payload_mass", value=870 * cd.Units.mass.pound_to_kg)
    payload.quantities.mass_properties.mass = payload_mass
    payload.quantities.mass_properties.cg_vector = wing_qc + np.array([0., 0., 0.5])

    weights_solver = cd.aircraft.models.weights.WeightsSolverModel()
    weights_solver.evaluate(
        design_gross_weight, 
        wing_mass, fuselage_mass, h_tail_mass, v_tail_mass, avionics_mass, landing_gear_mass, engine_mass, fuel_weight, payload_mass)


    base_config.assemble_system_mass_properties(update_copies=True)

    total_aircraft_mass = base_config.system.quantities.mass_properties.mass
    total_aircraft_mass.name = "total_aircraft_mass"
    total_aircraft_mass.set_as_constraint(upper=1200, scaler=1e-3)
    # total_aircraft_mass.set_as_objective(scaler=1e-3)

    print(base_config.system.quantities.mass_properties.inertia_tensor.value)

def define_analysis(caddee: cd.CADDEE):
    """Define the analysis of performed on the aircraft."""
    cruise : cd.aircraft.conditions.CruiseCondition = caddee.conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container

    tail = cruise_config.system.comps["h_tail"]
    v_tail = cruise_config.system.comps["v_tail"]
    wing = cruise_config.system.comps["wing"]
    fuselage = cruise_config.system.comps["fuselage"]
    elevator_deflection = csdl.Variable(name="elevator", value=0)
    elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=2)
    tail.actuate(elevator_deflection)

    # Re-evaluate meshes and compute nodal velocities
    cruise.finalize_meshes()

    vlm_mesh = mesh_container["vlm_mesh"]
    wing_chord_surface = vlm_mesh.discretizations["wing_chord_surface"]
    h_tail_chord_surface = vlm_mesh.discretizations["h_tail_chord_surface"]

    lattice_coordinates = [wing_chord_surface.nodal_coordinates, h_tail_chord_surface.nodal_coordinates]
    lattice_nodal_velocities = [wing_chord_surface.nodal_velocities, h_tail_chord_surface.nodal_velocities]

    vlm_outputs = vlm_solver(
        lattice_coordinates, 
        lattice_nodal_velocities, 
        atmos_states=cruise.quantities.atmos_states,
        airfoil_Cd_models=[None, None],
        airfoil_Cl_models=[None, None],
        airfoil_Cp_models=[None, None],
        airfoil_alpha_stall_models=[None, None],
    )

    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment

    # rotor analysis
    thrust_coefficient =  0.0204
    rotor_meshes = mesh_container["rotor_meshes"]
    propeller_discretization = rotor_meshes.discretizations["propeller_discretization"]
    cruise_rpm = csdl.Variable(name="cruise_pusher_rpm", shape=(1, ), value=1500)
    cruise_rpm.set_as_design_variable(scaler=1e-3, lower=500, upper=2000)
    cruise_omega = cruise_rpm / 60 * 2 * np.pi
    radius = propeller_discretization.radius
    thrust_vector = propeller_discretization.thrust_vector
    thrust_origin = propeller_discretization.thrust_origin.reshape((-1, 3))
    rho = cruise.quantities.atmos_states.density
    thrust = thrust_coefficient * rho * np.pi * radius**2 * (cruise_omega * radius)**2
    thrust_forces = thrust * thrust_vector

    # To capture the effect of perturbations in the aircraft states, we need to rotate thrust vector into body-fixed frame
    # NOTE: We only do this for thrust in this example because other solvers automatically do this.
    thrust_forces_body = perform_local_to_body_transformation(
        cruise.quantities.ac_states.phi,
        cruise.quantities.ac_states.theta,
        cruise.quantities.ac_states.psi,
        thrust_forces
    )
    thrust_moments_body = csdl.cross(thrust_origin, thrust_forces_body, axis=1)

    # Parasite drag build up
    drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up
    parasite_drag = drag_build_up_model(cruise.quantities.ac_states, cruise.quantities.atmos_states,
                                        wing.parameters.S_ref, [wing, fuselage, tail, v_tail])
    
    # Summing up the total forces and moments
    total_forces, total_moments = cruise.assemble_forces_and_moments(
        [vlm_forces, thrust_forces_body, parasite_drag], [vlm_moments, thrust_moments_body]
    )

    # Setting force equilibrium constraints
    force_norm = csdl.norm(total_forces)
    moment_norm = csdl.norm(total_moments)

    force_norm.name = "total_forces_norm"
    moment_norm.name = "total_moments_norm"

    force_norm.set_as_constraint(equals=0, scaler=1e-4)
    moment_norm.set_as_constraint(equals=0., scaler=1e-4)

    # Performing linearized stability analysis
    long_stability_results = cruise.perform_linear_stability_analysis(
        total_forces=total_forces,
        total_moments=total_moments,
        ac_states=cruise.quantities.ac_states,
        mass_properties=cruise_config.system.quantities.mass_properties,
    )

    t2d = long_stability_results.time_2_double_phugoid
    t2d.name = "time2double"
    t2d.set_as_objective(scaler=-4e-2)
    print("time to double", t2d.value)

# Run the code (forward evaluation)
define_base_config(caddee=caddee)

define_conditions(caddee=caddee)

define_mass_properties(caddee=caddee)

define_analysis(caddee=caddee)

# Run optimization
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, # Turn off gpu if none available
    gpu=False,
    derivatives_kwargs= {
        "concatenate_ofs" : True # Turn off
    }
)

# Check analytical derivatives against finite difference
jax_sim.check_optimization_derivatives()

# Make CSDLAlphaProblem and initialize optimizer
problem = CSDLAlphaProblem(problem_name="induced_drag_minimization", simulator=jax_sim)
optimizer = SLSQP(problem=problem)


# Solve optimization problem
optimizer.solve()
optimizer.print_results()
recorder.execute()

# Plot geometry after optimization
c172_geom.plot(additional_plotting_elements=plotting_elements, opacity=0.5, color="#00629B")

# Print design variables, constraints, objectives after optimization
for dv in recorder.design_variables.keys():
    print(dv.name, dv.value)

for c in recorder.constraints.keys():
    print(c.name, c.value)

for obj in recorder.objectives.keys():
    print(obj.name, obj.value)