'''Example lift plus cruise'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from VortexAD.core.vlm.vlm_solver import vlm_solver
from ex_utils import plot_vlm
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.var_groups import RotorAnalysisInputs
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
import lsdo_function_spaces as lfs
import matplotlib.pyplot as plt
from CADDEE_alpha.core.component import VectorizedComponent


recorder = csdl.Recorder(inline=True, expand_ops=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = True
run_ffd = False
run_optimization = True

# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)

    # Make (base) configuration object
    base_config = cd.Configuration(system=aircraft)

    # Airframe container object
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # ---------- Fuselage ----------
    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselage"])
    fuselage = cd.aircraft.components.Fuselage(length=9.144, 
                                               max_height=1.688,
                                               max_width=1.557,
                                               geometry=fuselage_geometry)
    airframe.comps["fuselage"] = fuselage

    # ---------- Main wing ----------
    # ignore_names = ['72', '73', '90', '91', '92', '93', '110', '111'] # rib-like surfaces
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"])#, ignore_names=ignore_names)
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2,
                                       geometry=wing_geometry,thickness_to_chord=0.17,
                                        thickness_to_chord_loc=0.4, tight_fit_ffd=True)
    
    # Make ribs and spars
    top_surface_inds = [75, 79, 83, 87]
    top_geometry = wing.create_subgeometry(search_names=[str(i) for i in top_surface_inds])
    bottom_geometry = wing.create_subgeometry(search_names=[str(i+1) for i in top_surface_inds])
    wing.construct_ribs_and_spars(aircraft.geometry, num_ribs=8, LE_TE_interpolation="ellipse",
                                  top_geometry=top_geometry, bottom_geometry=bottom_geometry)

    # Wing material
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700, nu=0.33)
    
    # Aerodynamic parameters for drag build up
    wing.quantities.drag_parameters.percent_laminar = 70
    wing.quantities.drag_parameters.percent_turbulent = 30
    
    # Function spaces
    # Thickness
    thickness_space = wing_geometry.create_parallel_space(lfs.ConstantSpace(2))
    thickness_var, thickness_function = thickness_space.initialize_function(1, value=0.001)
    wing.quantities.material_properties.set_material(aluminum, thickness_function)

    # Pressure space
    pressure_function_space = lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=6, grid_size=(240, 40), conserve=False)
    indexed_pressue_function_space = wing.geometry.create_parallel_space(pressure_function_space)
    wing.quantities.pressure_space = indexed_pressue_function_space

    # Component hierarchy
    airframe.comps["wing"] = wing

    # Connect wing to fuselage at the quarter chord
    base_config.connect_component_geometries(fuselage, wing, 0.75 * wing.LE_center + 0.25 * wing.TE_center)

    # ---------- Empennage ----------
    empennage = cd.Component()
    airframe.comps["empennage"] = empennage

    # Horizontal tail
    h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_1"])
    h_tail = cd.aircraft.components.Wing(AR=4.3, S_ref=3.7, 
                                         taper_ratio=0.6, geometry=h_tail_geometry)
    empennage.comps["h_tail"] = h_tail
    
    # Connect h-tail to fuselage
    base_config.connect_component_geometries(fuselage, h_tail, h_tail.TE_center)

    # Vertical tail
    v_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_2"])
    # v_tail = cd.Component(AR=1.17, S_ref=2.54, taper_ratio=0.272, geometry=v_tail_geometry, orientation="vertical")
    v_tail = cd.aircraft.components.Wing(AR=1.17, S_ref=2.54, taper_ratio=0.272, geometry=v_tail_geometry, orientation="vertical")
    empennage.comps["v_tail"] = v_tail

    # Connect vtail to fuselage
    base_config.connect_component_geometries(fuselage, v_tail, v_tail.TE_root)

    # ----------motor group ----------
    power_density = 5000
    motors = cd.Component()
    airframe.comps["motors"] = motors
    pusher_motor = cd.Component(power_density=power_density)
    motors.comps["pusher_motor"] = pusher_motor
    for i in range(8):
        motor = cd.Component(power_density=power_density)
        motors.comps[f"motor_{i}"] = motor

    # ---------- Parent component for all rotors ----------
    rotors = cd.Component()
    airframe.comps["rotors"] = rotors
    rotors.quantities.drag_parameters.drag_area = 0.111484
    
    # Pusher prop 
    pusher_prop_geometry = aircraft.create_subgeometry(
        search_names=["Rotor-9-disk", "Rotor_9_blades", "Rotor_9_Hub"]
    )
    pusher_prop = cd.aircraft.components.Rotor(radius=2.74/2, geometry=pusher_prop_geometry)
    rotors.comps["pusher_prop"] = pusher_prop

    # Connect pusher prop to fuselage
    base_config.connect_component_geometries(fuselage, pusher_prop, connection_point=fuselage.tail_point)

    # Lift rotors / motors
    lift_rotors = []
    for i in range(8):
        rotor_geometry = aircraft.create_subgeometry(
            search_names=[f"Rotor_{i+1}_disk", f"Rotor_{i+1}_Hub", f"Rotor_{i+1}_blades"]
        )
        rotor = cd.aircraft.components.Rotor(radius=3.048/2, geometry=rotor_geometry)
        lift_rotors.append(rotor)
        rotors.comps[f"rotor_{i+1}"] = rotor

    # Booms
    booms = cd.Component() # Create a parent component for all the booms
    airframe.comps["booms"] = booms
    for i in range(8):
        boom_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_Support",
        ])
        boom = cd.Component(geometry=boom_geometry)
        boom.quantities.drag_parameters.characteristic_length = 2.4384
        boom.quantities.drag_parameters.form_factor = 1.1
        booms.comps[f"boom_{i+1}"] = boom

        # Connect booms to wing and rotors to fuselage
        rotor = rotors.comps[f"rotor_{i+1}"]
        base_config.connect_component_geometries(rotor, boom)
        base_config.connect_component_geometries(boom, wing, connection_point=boom.ffd_block_face_1)

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
            spacing_spanwise="cosine", ignore_camber=True, plot=False,
        )
        wing_chord_surface.project_airfoil_points()
        vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface


        tail_surface = cd.mesh.make_vlm_surface(
            h_tail, 10, 1, ignore_camber=True
        )
        vlm_mesh.discretizations["tail_chord_surface"] = tail_surface

        # # Beam nodal mesh
        # beam_mesh = cd.mesh.BeamMesh()
        # num_beam_nodes = 21
        # wing_box_beam = cd.mesh.make_1d_box_beam(wing, num_beam_nodes, norm_node_center=0.5, norm_beam_width=0.5, project_spars=True, plot=False)
        # beam_mesh.discretizations["wing_box_beam"] = wing_box_beam

        # lpc_geom.plot_meshes([wing_box_beam.nodal_coordinates])
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

    # Run inner optimization if specified
    if run_ffd:
        base_config.setup_geometry(plot=True)
    caddee.base_configuration = base_config

    # Store meshes
    if make_meshes:
        mesh_container = base_config.mesh_container
        mesh_container["vlm_mesh"] = vlm_mesh
        mesh_container["rotor_meshes"] = rotor_meshes
        

    return


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
    # pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(0))
    pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(2.69268269))
    # pitch_angle.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=10)
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1,
        range=70 * cd.Units.length.kilometer_to_m,
        mach_number=0.18, #np.array([0.18, 0.2, 0.22]),
        pitch_angle=pitch_angle, # np.linspace(np.deg2rad(-4), np.deg2rad(10), 15),
    )
    cruise.configuration = base_config.copy()
    # cruise.vectorized_configuration = base_config.vectorized_copy()
    conditions["cruise"] = cruise

    # +3g 
    pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(7))
    # pitch_angle.set_as_design_variable(upper=np.deg2rad(15), lower=5)
    flight_path_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(5))
    plus_3g = cd.aircraft.conditions.ClimbCondition(
        initial_altitude=1000, 
        final_altitude=2000,
        pitch_angle=pitch_angle,
        fligth_path_angle=flight_path_angle,
        mach_number=0.23,
    )
    plus_3g.configuration = base_config.copy()
    conditions["plus_3g"] = plus_3g


    # -1g 
    pitch_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(-5))
    # pitch_angle.set_as_design_variable(upper=np.deg2rad(0), lower=np.deg2rad(-15))
    flight_path_angle = csdl.Variable(shape=(1, ), value=np.deg2rad(-4))
    minus_1g = cd.aircraft.conditions.ClimbCondition(
        initial_altitude=2000, 
        final_altitude=1000,
        pitch_angle=pitch_angle,
        fligth_path_angle=flight_path_angle,
        mach_number=0.23,
    )
    minus_1g.configuration = base_config.copy()
    conditions["minus_1g"] = minus_1g


    # quasi-steady transition
    transition_mach_numbers = np.array([0.0002941, 0.06489461, 0.11471427, 0.13740796, 0.14708026, 0.15408429, 0.15983874, 0.16485417, 0.16937793, 0.17354959])
    transition_pitch_angles = np.array([-0.0134037, -0.04973228, 0.16195989, 0.10779469, 0.04, 0.06704556, 0.05598293, 0.04712265, 0.03981101, 0.03369678])
    transition_ranges = np.array([0.72, 158., 280., 336., 360., 377., 391., 403., 414., 424.])

    qst = cd.aircraft.conditions.CruiseCondition(
        altitude=300.,
        pitch_angle=transition_pitch_angles,
        mach_number=transition_mach_numbers,
        range=transition_ranges,
    )
    qst.vectorized_configuration = base_config.vectorized_copy(10)
    conditions["qst"] = qst

    return 


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
    base_config = caddee.base_configuration

    # ::::::::::::::::::::::::::: quasi-steady transition :::::::::::::::::::::::::::
    qst = conditions["qst"]
    qst_config = qst.vectorized_configuration
    qst_system = qst_config.system

    airframe = qst_system.comps["airframe"]
    h_tail = airframe.comps["empennage"].comps["h_tail"]
    v_tail = airframe.comps["empennage"].comps["v_tail"]
    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    rotors = airframe.comps["rotors"]
    booms = list(airframe.comps["booms"].comps.values())
    
    qst_mesh_container = qst_config.mesh_container

    tail_actuation_var = csdl.Variable(shape=(10, ), value=0)
    tail_actuation_var.set_as_design_variable(upper=np.deg2rad(20), lower=np.deg2rad(-20))
    h_tail.actuate(angle=tail_actuation_var)

    qst.finalize_meshes()

    # set up VLM analysis
    vlm_mesh = qst_mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]
    
    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
            aoa_range=np.linspace(-12, 16, 50), 
            reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
            mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    Cd_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cd"])
    Cp_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cp"])
    alpha_stall_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["alpha_Cl_min_max"])

    nodal_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    nodal_velocities = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]

    vlm_outputs = vlm_solver(
        mesh_list=nodal_coordinates,
        mesh_velocity_list=nodal_velocities,
        atmos_states=qst.quantities.atmos_states,
        airfoil_alpha_stall_models=[None, None], #[alpha_stall_model, None],
        airfoil_Cd_models=[None, None], #[Cd_model, None],
        airfoil_Cl_models=[Cl_model, None],
        airfoil_Cp_models=[None, None], #[Cp_model, None],
    )

    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment

    # Drag build up
    drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up
    drag_build_up = drag_build_up_model(qst.quantities.ac_states, qst.quantities.atmos_states,
                                        wing.parameters.S_ref, [wing, fuselage, h_tail, v_tail, rotors] + booms)

    # BEM solver
    rotor_meshes = qst_mesh_container["rotor_meshes"]
    pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    mesh_vel = pusher_rotor_mesh.nodal_velocities
    cruise_rpm = csdl.Variable(shape=(10, ), value=2000)
    cruise_rpm.set_as_design_variable(upper=3000, lower=500, scaler=1e-3)
    bem_inputs = RotorAnalysisInputs()
    bem_inputs.ac_states = qst.quantities.ac_states
    bem_inputs.atmos_states =  qst.quantities.atmos_states
    bem_inputs.mesh_parameters = pusher_rotor_mesh
    bem_inputs.mesh_velocity = mesh_vel
    bem_inputs.rpm = cruise_rpm
    bem_model = BEMModel(num_nodes=10, airfoil_model=NACA4412MLAirfoilModel())
    bem_outputs = bem_model.evaluate(bem_inputs)

    # # BEM solvers lift
    # lift_rotor_forces = []
    # lift_rotor_moments = []
    # for i in range(1):
    #     rotor_mesh = rotor_meshes.discretizations[f"rotor_{i+1}_mesh"]
    #     mesh_vel = rotor_mesh.nodal_velocities
    #     rpm = csdl.Variable(shape=(10, ), value=1000)
    #     rpm.set_as_design_variable(lower=10, upper=2000, scaler=np.linspace(1e-3, 1e-1, 10))
    #     lift_rotor_inputs = RotorAnalysisInputs()
    #     lift_rotor_inputs.ac_states = qst.quantities.ac_states
    #     lift_rotor_inputs.atmos_states =  qst.quantities.atmos_states
    #     lift_rotor_inputs.mesh_parameters = rotor_mesh
    #     lift_rotor_inputs.mesh_velocity = mesh_vel
    #     lift_rotor_inputs.rpm = rpm
    #     lift_rotor_model = BEMModel(num_nodes=10, airfoil_model=NACA4412MLAirfoilModel())
    #     lift_rotor_outputs = lift_rotor_model.evaluate(lift_rotor_inputs)
    #     lift_rotor_forces.append(lift_rotor_outputs.forces)
    #     lift_rotor_moments.append(lift_rotor_outputs.moments)


    total_forces_qst, total_moments_qst = qst.assemble_forces_and_moments(
        aero_propulsive_forces=[vlm_forces, bem_outputs.forces, drag_build_up], # + lift_rotor_forces, 
        aero_propulsive_moments=[vlm_moments, bem_outputs.moments], # + lift_rotor_moments,
        ac_mps=base_config.system.quantities.mass_properties,
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_qst = eom_model.evaluate(
        total_forces=total_forces_qst,
        total_moments=total_moments_qst,
        ac_states=qst.quantities.ac_states,
        ac_mass_properties=base_config.system.quantities.mass_properties
    )

    dv_dt = accel_qst.dv_dt
    dw_dt = accel_qst.dw_dt
    dp_dt = accel_qst.dp_dt
    dq_dt = accel_qst.dq_dt
    dr_dt = accel_qst.dr_dt

    zero_accel_norm = csdl.norm(dv_dt + dw_dt + dp_dt + dq_dt + dr_dt)
    zero_accel_norm.name = "residual_norm"
    zero_accel_norm.set_as_objective()
    
    du_dt = accel_qst.du_dt
    du_dt.name = "horizontal acceleration"
    du_dt_constraint = np.array(
            [3.05090108, 1.84555602, 0.67632681, 0.39583939, 0.30159843, 
            0.25379256, 0.22345727, 0.20269499, 0.18808881, 0.17860702]
        )
    du_dt.set_as_constraint(upper=du_dt_constraint, lower=du_dt_constraint)


    # # ::::::::::::::::::::::::::: Hover :::::::::::::::::::::::::::
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
    # accel_hover = eom_model.evaluate(
    #     total_forces=total_forces_hover,
    #     total_moments=total_moments_hover,
    #     ac_states=hover.quantities.ac_states,
    #     ac_mass_properties=hover_config.system.quantities.mass_properties
    # )
    # accel_norm_hover = accel_hover.accel_norm
    # # accel_norm_hover.set_as_constraint(upper=0, lower=0)

    # # ::::::::::::::::::::::::::: Cruise :::::::::::::::::::::::::::
    # cruise = conditions["cruise"]
    # cruise_config = cruise.configuration
    # mesh_container = cruise_config.mesh_container
    # airframe = cruise_config.system.comps["airframe"]
    # wing = airframe.comps["wing"]
    # fuselage = airframe.comps["fuselage"]
    # v_tail = airframe.comps["empennage"].comps["v_tail"]
    # rotors = airframe.comps["rotors"]
    # booms = list(airframe.comps["booms"].comps.values())

    # # Actuate tail
    # tail = airframe.comps["empennage"].comps["h_tail"]
    # elevator_deflection = csdl.Variable(shape=(3, ), value=np.deg2rad(0))
    # elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=10)
    # tail.actuate(elevator_deflection)

    # # Re-evaluate meshes and compute nodal velocities
    # cruise.finalize_meshes()

    # # Set up VLM analysis
    # vlm_mesh = mesh_container["vlm_mesh"]
    # wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    # tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]
    # airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    # airfoil_lower_nodes = wing_lattice._airfoil_lower_para
    # pressure_indexed_space : lfs.FunctionSetSpace = wing.quantities.pressure_space

    # # run vlm solver
    # lattice_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    # lattice_nodal_velocitiies = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]
    # alpha_ml_list = [wing_lattice.alpha_ML_mid_panel, tail_lattice.alpha_ML_mid_panel]
    # # airfoil_Cd_models = [wing_lattice.embedded_airfoil_model_Cd, tail_lattice.embedded_airfoil_model_Cd]
    # airfoil_Cl_models = [wing_lattice.embedded_airfoil_model_Cl, tail_lattice.embedded_airfoil_model_Cl]
    # airfoil_Cp_models = [wing_lattice.embedded_airfoil_model_Cp, tail_lattice.embedded_airfoil_model_Cp]
    # airfoil_alpha_stall_models = [wing_lattice.embedded_airfoil_model_alpha_stall, tail_lattice.embedded_airfoil_model_alpha_stall]
    # reynolds_numbers = [wing_lattice.reynolds_number, tail_lattice.reynolds_number]
    # chord_length_mid_panel = [wing_lattice.mid_panel_chord_length, tail_lattice.mid_panel_chord_length]
    # vlm_outputs = vlm_solver(
    #     lattice_coordinates, 
    #     lattice_nodal_velocitiies, 
    #     alpha_ML=alpha_ml_list,
    #     airfoil_Cd_models=[None, None],#=airfoil_Cd_models,
    #     airfoil_Cl_models=airfoil_Cl_models,
    #     airfoil_Cp_models=airfoil_Cp_models,
    #     airfoil_alpha_stall_models=airfoil_alpha_stall_models,
    #     reynolds_numbers=reynolds_numbers,
    #     chord_length_mid_panel=chord_length_mid_panel,
    # )
    
    # vlm_forces = vlm_outputs.total_force
    # vlm_moments = vlm_outputs.total_moment
    # spanwise_Cp = vlm_outputs.surface_spanwise_Cp[0]
    # spanwise_Cp = csdl.blockmat([[spanwise_Cp[0, :, 0:120].T()], [spanwise_Cp[0, :, 120:].T()]])
    # P_inf = cruise.quantities.atmos_states.pressure
    # V_inf = cruise.parameters.speed
    # rho_inf = cruise.quantities.atmos_states.density
    # spanwise_pressure = spanwise_Cp * 0.5 * rho_inf * V_inf**2 + P_inf


    # pressure_function = pressure_indexed_space.fit_function_set(
    #     values=spanwise_pressure.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
    #     regularization_parameter=1e-4,
    # )

    # Cp_upper = pressure_function.evaluate(airfoil_upper_nodes).reshape((120, 40)).value
    # Cp_lower = pressure_function.evaluate(airfoil_lower_nodes).reshape((120, 40)).value
    # # print(Cp_upper[0, :])
    # # print(Cp_lower[0, :])

    # plt.plot(x_interp, Cp_upper[:, 20], color="r", label="IDW")
    # plt.plot(x_interp, Cp_lower[:, 20], color="r")
    # plt.plot(x_interp, spanwise_pressure[0:120, 20].value, color="b", label="data")
    # plt.plot(x_interp, spanwise_pressure[120:, 20].value, color="b")
    # plt.title("mid panel")
    # plt.legend()
    # # print("\n")
    # # print(spanwise_Cp[0:120, 20].value)
    # # print(spanwise_Cp[120:, 20].value)

    # # wing.geometry.plot_but_good(color=pressure_function)
    

    # # Drag build-up
    # drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up

    # drag_build_up = drag_build_up_model(cruise.quantities.ac_states, cruise.quantities.atmos_states,
    #                                     wing.parameters.S_ref, [wing, fuselage, tail, v_tail, rotors] + booms)
    
    
    # wing.geometry.plot(color=pressure_function)
    # plt.show()
    
    # # BEM solver
    # rotor_meshes = mesh_container["rotor_meshes"]
    # pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    # mesh_vel = pusher_rotor_mesh.nodal_velocities
    # cruise_rpm = csdl.Variable(shape=(1, ), value=2000)
    # cruise_rpm.set_as_design_variable(upper=2500, lower=1200, scaler=1e-3)
    # bem_inputs = RotorAnalysisInputs()
    # bem_inputs.ac_states = cruise.quantities.ac_states
    # bem_inputs.atmos_states =  cruise.quantities.atmos_states
    # bem_inputs.mesh_parameters = pusher_rotor_mesh
    # bem_inputs.mesh_velocity = mesh_vel
    # bem_inputs.rpm = cruise_rpm
    # bem_model = BEMModel(num_nodes=1, airfoil_model=NACA4412MLAirfoilModel())
    # bem_outputs = bem_model.evaluate(bem_inputs)


    # # total forces and moments
    # total_forces_cruise, total_moments_cruise = cruise.assemble_forces_and_moments(
    #     [vlm_forces, drag_build_up, bem_outputs.forces], [vlm_moments, bem_outputs.moments]
    # )

    # # eom
    # eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    # accel_cruise = eom_model.evaluate(
    #     total_forces=total_forces_cruise,
    #     total_moments=total_moments_cruise,
    #     ac_states=cruise.quantities.ac_states,
    #     ac_mass_properties=cruise_config.system.quantities.mass_properties
    # )
    # accel_norm_cruise = accel_cruise.accel_norm
    
    # accel_norm_total = accel_norm_cruise + accel_norm_hover
    
    # accel_norm_total.set_as_objective(scaler=1e-1)


    # total_lift = vlm_outputs.total_lift
    # total_drag = vlm_outputs.total_drag
    # rho = cruise.quantities.atmos_states.density.value
    # V = cruise.parameters.speed.value
    # CL = total_lift / 0.5 / rho / V**2 / 19.53
    # CDi = total_drag / 0.5 / rho / V**2 / 19.53

    # plot_vlm(vlm_outputs)

    return accel_qst #accel_hover, accel_cruise, total_forces_hover, total_forces_cruise


define_base_config(caddee)

define_conditions(caddee)

define_mass_properties(caddee)

accel_qst  = define_analysis(caddee)

recorder.inline = False
recorder.stop()

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    sim = csdl.experimental.JaxSimulator(recorder)
    
    

    # sim = csdl.experimental.PySimulator(
    #     recorder=recorder,
    # )

    sim.check_totals(step_size=1e-3)
    exit()

    prob = CSDLAlphaProblem(problem_name='LPC_trim', simulator=sim)

    optimizer = SLSQP(prob, ftol=1e-8, maxiter=10, outputs=['x'])

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()

print(accel_qst.accel_norm.value)
print(accel_qst.du_dt.value)
print(accel_qst.dv_dt.value)
print(accel_qst.dw_dt.value)
print(accel_qst.dp_dt.value)
print(accel_qst.dq_dt.value)
print(accel_qst.dr_dt.value)

dv_dict = recorder.design_variables

for dv in dv_dict.keys():
    print(dv.value)
