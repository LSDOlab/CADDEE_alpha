import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from VortexAD.core.vlm.vlm_solver import vlm_solver
from ex_utils import plot_vlm
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters


recorder = csdl.Recorder(inline=True, expand_ops=True, debug=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = True
run_ffd = False

# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # Fuselage
    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselage"])
    fuselage = cd.aircraft.components.Fuselage(length=9.144, geometry=fuselage_geometry)
    airframe.comps["fuselage"] = fuselage

    # Main wing
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"])
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2,
                                       geometry=wing_geometry, tight_fit_ffd=True)
    airframe.comps["wing"] = wing

    # Empennage
    empennage = cd.Component()
    airframe.comps["empennage"] = empennage

    # Horizontal tail
    h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_1"])
    h_tail = cd.aircraft.components.Wing(AR=4.3, S_ref=3.7, 
                                         taper_ratio=0.6, geometry=h_tail_geometry)
    empennage.comps["h_tail"] = h_tail

    # Vertical tail
    v_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_2"])
    v_tail = cd.aircraft.components.Wing(AR=1.17, S_ref=2.54, geometry=v_tail_geometry)
    empennage.comps["v_tail"] = v_tail

    # motor group
    power_density = 5000
    motors = cd.Component()
    airframe.comps["motors"] = motors
    pusher_motor = cd.Component(power_density=power_density)
    motors.comps["pusher_motor"] = pusher_motor
    for i in range(8):
        motor = cd.Component(power_density=power_density)
        motors.comps[f"motor_{i}"] = motor

    # Pusher prop 
    pusher_prop_geometry = aircraft.create_subgeometry(
        search_names=["Rotor-9-disk", "Rotor_9_blades", "Rotor_9_Hub"]
    )
    pusher_prop = cd.aircraft.components.Rotor(radius=2.74/2, geometry=pusher_prop_geometry)
    airframe.comps["pusher_prop"] = pusher_prop

    # Lift rotors / motors
    lift_rotors = []
    for i in range(8):
        rotor_geometry = aircraft.create_subgeometry(
            search_names=[f"Rotor_{i+1}_disk", f"Rotor_{i+1}_Hub", f"Rotor_{i+1}_blades"]
        )
        rotor = cd.aircraft.components.Rotor(radius=3.048/2, geometry=rotor_geometry)
        lift_rotors.append(rotor)
        airframe.comps[f"rotor_{i+1}"] = rotor

    # Booms
    booms = cd.Component() # Create a parent component for all the booms
    airframe.comps["booms"] = booms
    for i in range(8):
        boom_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_Support",
        ])
        boom = cd.Component(geometry=boom_geometry)
        booms.comps[f"boom_{i+1}"] = boom

    # battery
    battery = cd.Component(energy_density=380)
    airframe.comps["battery"] = battery

    # payload
    payload = cd.Component()
    airframe.comps["payload"] = payload

    # systems
    systems = cd.Component()
    airframe.comps["systems"] = systems

    # ::::::::::::::::::::::::::: Make meshes :::::::::::::::::::::::::::
    if make_meshes:
    # wing + tail
        vlm_mesh = cd.mesh.VLMMesh()
        wing_chord_surface = cd.mesh.make_vlm_surface(
            wing, 40, 1, LE_interp="ellipse", TE_interp="ellipse", 
            spacing_spanwise="linear", ignore_camber=True, plot=False,
        )
        # Add an airfoil model that shifts the lift curve slope by alpha at Cl0
        wing_chord_surface.embedded_airfoil_model = NACA4412MLAirfoilModel()
        vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
        
        tail_surface = cd.mesh.make_vlm_surface(
            h_tail, 10, 1, ignore_camber=True
        )
        vlm_mesh.discretizations["tail_chord_surface"] = tail_surface

        lpc_geom.plot_meshes([wing_chord_surface.nodal_coordinates, tail_surface.nodal_coordinates])

        # rotors
        num_radial = 30
        num_cp = 4
        blade_parameterization = BsplineParameterization(num_cp=num_cp, num_radial=num_radial)
        rotor_meshes = cd.mesh.RotorMeshes()
        # pusher prop
        pusher_prop_mesh = cd.mesh.make_rotor_mesh(
            pusher_prop, num_radial=num_radial, num_azimuthal=1, num_blades=4, plot=True
        )
        chord_cps = csdl.Variable(shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        twist_cps = csdl.Variable(shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        pusher_prop_mesh.chord_profile = blade_parameterization.evaluate_radial_profile(chord_cps)
        pusher_prop_mesh.twist_profile = blade_parameterization.evaluate_radial_profile(twist_cps)
        rotor_meshes.discretizations["pusher_prop_mesh"] = pusher_prop_mesh

        # lift rotors
        for i in range(8):
            rotor_mesh = cd.mesh.make_rotor_mesh(
                lift_rotors[i], num_radial=30, num_blades=2,
            )
            chord_cps = csdl.Variable(shape=(4, ), value=np.linspace(0.3, 0.1, 4))
            twist_cps = csdl.Variable(shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
            rotor_mesh.chord_profile = blade_parameterization.evaluate_radial_profile(chord_cps)
            rotor_mesh.twist_profile = blade_parameterization.evaluate_radial_profile(twist_cps)
            rotor_meshes.discretizations[f"rotor_{i+1}_mesh"] = rotor_mesh

    # Make base configuration    
    base_config = cd.Configuration(system=aircraft)
    if run_ffd:
        base_config.setup_geometry()
    caddee.base_configuration = base_config

    # Store meshes
    if make_meshes:
        mesh_container = base_config.mesh_container
        mesh_container["vlm_mesh"] = vlm_mesh
        mesh_container["rotor_meshes"] = rotor_meshes


def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # Hover
    hover = cd.aircraft.conditions.HoverCondition(
        altitude=100.,
        time=120.,
    )
    hover.configuration = base_config
    conditions["hover"] = hover

    # Cruise
    pitch_angle = csdl.Variable(shape=(1, ), value=0.)
    pitch_angle.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=10)
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        range=70 * cd.Units.length.kilometer_to_m,
        mach_number=0.18,
        pitch_angle=pitch_angle, # np.linspace(np.deg2rad(-4), np.deg2rad(10), 15),
    )
    cruise.configuration = base_config.copy()
    conditions["cruise"] = cruise

    return pitch_angle


def define_mass_properties(caddee: cd.CADDEE):
    # Get base config and conditions
    base_config = caddee.base_configuration
    conditions = caddee.conditions

    cruise = conditions["cruise"]
    cruise_speed = cruise.parameters.speed[0]

    # Get system component
    aircraft = base_config.system

    # Get airframe and its components
    airframe  = aircraft.comps["airframe"]
    
    # battery
    battery = airframe.comps["battery"]
    battery_mass = csdl.Variable(shape=(1, ), value=800)
    battery_cg = csdl.Variable(shape=(3, ), value=np.array([-3.5, 0., -1.]))
    battery.quantities.mass_properties.mass = battery_mass
    battery.quantities.mass_properties.cg_vector = battery_cg
    
    # Wing
    wing = airframe.comps["wing"]
    wing_area = wing.parameters.S_ref
    wing_AR = wing.parameters.AR

    # Fuselage
    fuselage = airframe.comps["fuselage"]
    fuselage_length = fuselage.parameters.length

    # Empennage
    empennage = airframe.comps["empennage"]
    h_tail = empennage.comps["h_tail"]
    h_tail_area = h_tail.parameters.S_ref
    v_tail = empennage.comps["v_tail"]
    v_tail_area =  v_tail.parameters.S_ref
    
    # Booms
    booms = airframe.comps["booms"]

    # M4-regression mass models (scaled to match better with NDARC)
    nasa_lpc_weights = cd.aircraft.models.weights.nasa_lpc
    scaler = 1.3
    wing_mps = nasa_lpc_weights.compute_wing_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass, 
        cruise_speed=cruise_speed,
    )
    wing_mps.mass = wing_mps.mass * scaler
    wing.quantities.mass_properties.mass = wing_mps.mass
    wing.quantities.mass_properties.cg_vector = wing_mps.cg_vector
    wing.quantities.mass_properties.inertia_tensor = wing_mps.inertia_tensor

    fuselage_mps = nasa_lpc_weights.compute_fuselage_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass,
        cruise_speed=cruise_speed,
    )
    fuselage_mps.mass = fuselage_mps.mass * scaler
    fuselage.quantities.mass_properties.mass = fuselage_mps.mass
    fuselage.quantities.mass_properties.cg_vector = fuselage_mps.cg_vector
    fuselage.quantities.mass_properties.inertia_tensor = fuselage_mps.inertia_tensor

    boom_mps = nasa_lpc_weights.compute_boom_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass,
        cruise_speed=cruise_speed,
    )
    boom_mps.mass = boom_mps.mass * scaler
    booms.quantities.mass_properties.mass = boom_mps.mass
    booms.quantities.mass_properties.cg_vector = boom_mps.cg_vector
    booms.quantities.mass_properties.inertia_tensor = boom_mps.inertia_tensor

    empennage_mps = nasa_lpc_weights.compute_empennage_mps(
        h_tail_area=h_tail_area,
        v_tail_area=v_tail_area,
    )
    empennage_mps.mass = empennage_mps.mass * scaler
    empennage.quantities.mass_properties = empennage_mps
    # empennage.quantities.mass_properties.mass = empennage_mps.mass
    # empennage.quantities.mass_properties.cg_vector = empennage_mps.cg_vector
    # empennage.quantities.mass_properties.inertia_tensor = empennage_mps.inertia_tensor

    # Motors
    motor_group = airframe.comps["motors"]
    motors = list(motor_group.comps.values())
    
        # get rotor meshes to obtain thrust origin (i.e., motor cg)
    mesh_container = base_config.mesh_container
    rotor_meshes = mesh_container["rotor_meshes"]
    
        # Loop over all motors to assign mass properties
    for i, rotor_mesh in enumerate(rotor_meshes.discretizations.values()):
        motor_comp = motors[i]
        motor_mass = csdl.Variable(shape=(1, ), value=25)
        motor_cg = rotor_mesh.thrust_origin
        motor_comp.quantities.mass_properties.mass = motor_mass
        motor_comp.quantities.mass_properties.cg_vector = motor_cg

    # payload
    payload = airframe.comps["payload"]
    payload_mass = csdl.Variable(shape=(1, ), value=540+800)
    payload_cg = csdl.Variable(shape=(3, ), value=np.array([-3.33, 0., -1.5]))
    payload.quantities.mass_properties.mass = payload_mass
    payload.quantities.mass_properties.cg_vector = payload_cg

    # systems
    systems = airframe.comps["systems"]
    systems_mass = csdl.Variable(shape=(1, ), value=244)
    systems_cg = csdl.Variable(shape=(3, ), value=np.array([-3.33, 0., -1.5]))
    systems.quantities.mass_properties.mass = systems_mass
    systems.quantities.mass_properties.cg_vector = systems_cg

    # Assemble system mass properties
    base_config.assemble_system_mass_properties(update_copies=True)


def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions

    # # hover
    # hover = conditions["hover"]
    # hover_config = hover.configuration
    # mesh_container = hover_config.mesh_container
    # rotor_meshes = mesh_container["rotor_meshes"]

    # # Re-evaluate meshes and compute nodal velocities
    # hover.finalize_meshes()

    # # BEM analysis
    # bem_forces = []
    # bem_moments = []
    # rpm_list = []
    # for i in range(8):
    #     rpm = csdl.Variable(shape=(1, ), value=1200)
    #     rpm.set_as_design_variable(upper=1500, lower=500, scaler=1e-3)
    #     rpm_list.append(rpm)
    #     rotor_mesh = rotor_meshes.discretizations[f"rotor_{i+1}_mesh"]
    #     mesh_vel = rotor_mesh.nodal_velocities
        
    #     # Set up BEM model
    #     bem_model = BEMModel(num_nodes=1, airfoil_model=NACA4412MLAirfoilModel())
    #     bem_inputs = RotorAnalysisInputs()
    #     bem_inputs.atmos_states = hover.quantities.atmos_states
    #     bem_inputs.ac_states = hover.quantities.ac_states
    #     bem_inputs.mesh_parameters = rotor_mesh
    #     bem_inputs.rpm = rpm
    #     bem_inputs.mesh_velocity = mesh_vel
        
    #     # Run BEM model and store forces and moment
    #     bem_outputs = bem_model.evaluate(bem_inputs)
    #     bem_forces.append(bem_outputs.forces)
    #     bem_moments.append(bem_outputs.moments)

    # # total forces and moments
    # total_forces_hover, total_moments_hover = hover.assemble_forces_and_moments(
    #     bem_forces, bem_moments
    # )

    # # eom
    # eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    # accel = eom_model.evaluate(
    #     total_forces=total_forces_hover,
    #     total_moments=total_moments_hover,
    #     ac_states=hover.quantities.ac_states,
    #     ac_mass_properties=hover_config.system.quantities.mass_properties
    # )
    # accel_norm = accel.accel_norm
    # accel_norm.set_as_objective()

    # return accel, total_forces_hover, total_moments_hover, rpm_list

    # cruise
    cruise = conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    airframe = cruise_config.system.comps["airframe"]

    # Actuate tail
    tail = airframe.comps["empennage"].comps["h_tail"]
    elevator_deflection = csdl.Variable(shape=(1, ), value=(np.deg2rad(0)))
    elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=10)
    tail.actuate(elevator_deflection)

    # Re-evaluate meshes and compute nodal velocities
    cruise.finalize_meshes()

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]

    # lpc_geom.plot_meshes([wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates])

    # run vlm solver
    lattice_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    lattice_nodal_velocitiies = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]
    alpha_ml_list = [wing_lattice.alpha_ML_mid_panel, tail_lattice.alpha_ML_mid_panel]
    vlm_outputs = vlm_solver(lattice_coordinates, lattice_nodal_velocitiies, alpha_ML=alpha_ml_list)
    
    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment

    # total forces and moments
    total_forces_cruise, total_moments_cruise, inertial_forces, inertial_moments = cruise.assemble_forces_and_moments(
        [vlm_forces], [vlm_moments]
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel = eom_model.evaluate(
        total_forces=total_forces_cruise,
        total_moments=total_moments_cruise,
        ac_states=cruise.quantities.ac_states,
        ac_mass_properties=cruise_config.system.quantities.mass_properties
    )
    accel_norm = accel.accel_norm
    dw_dt = accel.dw_dt
    dp_dt = accel.dp_dt
    dq_dt = accel.dq_dt

    # accel_obj = dw_dt**2 + dp_dt**2 + dq_dt**2
    total_z_forces = csdl.norm(total_forces_cruise[0, 2])
    total_q_moments = csdl.norm(total_moments_cruise[0, 1])
    fm_zq = total_z_forces + total_q_moments
    fm_zq.set_as_objective()


    total_lift = vlm_outputs.total_lift
    total_drag = vlm_outputs.total_drag
    rho = cruise.quantities.atmos_states.density.value
    V = cruise.parameters.speed.value
    CL = total_lift / 0.5 / rho / V**2 / 19.53
    CDi = total_drag / 0.5 / rho / V**2 / 19.53

    plot_vlm(vlm_outputs)
    # print(CL.value* -1)
    # print(np.absolute(CDi.value))

    return accel, total_forces_cruise, total_moments_cruise, elevator_deflection, cruise.parameters.pitch_angle, tail_lattice


define_base_config(caddee)

pitch = define_conditions(caddee)

define_mass_properties(caddee)

accel_norm, total_forces, total_moments, elevator, _, tail_lattice = define_analysis(caddee)
# exit()

pitch.add_name("pitch")
elevator.add_name("elevator")
total_forces.add_name("total_forces")
total_moments.add_name("total_moments")

# from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# # verify_derivatives_inline([accel_norm.accel_norm], [pitch, elevator], 1e-2, raise_on_error=False)
# verify_derivatives_inline([total_forces, total_moments], [pitch], 1e-7, raise_on_error=False)
# recorder.stop()

# exit()
sim = csdl.experimental.PySimulator(recorder)

from modopt import CSDLAlphaProblem
from modopt import SLSQP

prob = CSDLAlphaProblem(problem_name='LPC_trim', simulator=sim)

optimizer = SLSQP(prob, ftol=1e-8, maxiter=25, outputs=['x'])

# Solve your optimization problem
optimizer.solve()
optimizer.print_results()

print(accel_norm.accel_norm.value)
print(accel_norm.du_dt.value)
print(accel_norm.dv_dt.value)
print(accel_norm.dw_dt.value)
print(accel_norm.dp_dt.value)
print(accel_norm.dq_dt.value)
print(accel_norm.dr_dt.value)
print("forces", total_forces.value)
print("moments", total_moments.value)
print("elevator", np.rad2deg(elevator.value))
print("pitch", np.rad2deg(pitch.value))
# for i, rpm in enumerate(rpm_list):
#     print(f"rpm_{i}", rpm.value)
