'''Example lift plus cruise'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from VortexAD.core.vlm.vlm_solver import vlm_solver
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.var_groups import RotorAnalysisInputs
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
import lsdo_function_spaces as lfs


# :::::::::::::::::::::::::::::::::: design variables ::::::::::::::::::::::::::::::::::
qst_tail_deflection_dvs = np.array([-1.84169007e-07 , 1.70793925e-03, -8.42541605e-02 ,-7.86878417e-02,
 -3.21922047e-02 , -5.68630747e-02 ,-4.78010192e-02, -3.73583355e-02,
 -3.56211004e-02 , -3.53845675e-02])

qst_pusher_prop_rpm_dvs = np.array([1638.93655147, 1578.22379105, 1501.72488056, 1579.18249055, 1605.73867017,
 1668.23291713, 1708.19493741 ,1747.38205711 ,1783.30303415, 1818.40981192])

qst_front_inner_rotor_rpm_dvs = np.array([1292.88696435 ,1235.80076879 , 780.38393881 , 606.3436678 ,  755.18974247,
  539.92872316 , 521.42052323 , 510.02284784 , 511.87234198 , 498.80056848])

qst_rear_inner_rotor_rpm_dvs = np.array([981.60202437, 1066.48714885,  288.42701985 , 347.14719852,  675.77230288,
  457.21114097 , 472.05340672 , 479.98417784 , 502.27503177,  526.84275169])

qst_front_outer_rotor_rpm_dvs = np.array([1292.60846133 ,1273.6183807 ,  757.58706749 , 604.33476083 , 755.25282312,
  545.87161422 , 529.9578592  , 527.02539216 , 490.63074463 , 466.91126126])

qst_rear_outer_rotor_rpm_dvs = np.array([1043.87469863, 1041.15388279 , 283.42220461,  360.3894776,   682.14046615,
  477.13029284 , 496.56544284  ,495.80428922 , 528.72433065,  558.56842629])

hover_lift_rotor_rpms = np.array([1258.0136741, 1100.80421799, 1194.49090187, 1044.09390726, 1194.50982395, 1044.10857318, 1251.90716776, 1107.93028051])

climb_pitch = 0.16982572
cruise_pitch = 0.05202441
descent_pitch = -0.00239823

climb_tail_deflection = -0.10394263
cruise_tail_deflection = -0.03344455
descent_tail_deflection = -0.00571552

climb_pusher_rpm = 2123.12337628
cruise_pusher_rpm = 1858.81612397
descent_pusher_rpm = 1672.01908959

do_qst = True
do_hover = True
do_cruise = True
do_climb = True
do_descent = True
do_structural_sizing = False

recorder = csdl.Recorder(inline=True, expand_ops=True, debug=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = True
run_ffd = True
run_optimization = True


# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom, compute_surface_area=False)

    # Make (base) configuration object
    base_config = cd.Configuration(system=aircraft)

    # Airframe container object
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # ---------- Fuselage ----------
    fuselage_length = csdl.Variable(name="fuselage_length", shape=(1, ), value=9.144)
    # fuselage_length.set_as_design_variable(lower=0.9 * 9.144, upper=1.1*9.144, scaler=1e-1)
    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselage"])
    fuselage = cd.aircraft.components.Fuselage(
        length=fuselage_length, max_height=1.688,
        max_width=1.557, geometry=fuselage_geometry, skip_ffd=True
    )
    airframe.comps["fuselage"] = fuselage

    # ---------- Main wing ----------
    # ignore_names = ['72', '73', '90', '91', '92', '93', '110', '111'] # rib-like surfaces
    wing_AR = csdl.Variable(name="wing_AR", shape=(1, ), value=12.12)
    wing_AR.set_as_design_variable(lower=0.8 * 12.12, upper=1.2*12.12, scaler=1e-1)
    wing_S_ref = csdl.Variable(name="wing_S_ref", shape=(1, ), value=19.6)
    wing_S_ref.set_as_design_variable(lower=0.8 * 19.6, upper=1.2*19.6, scaler=9e-2)
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"])#, ignore_names=ignore_names)
    wing = cd.aircraft.components.Wing(AR=wing_AR, S_ref=wing_S_ref, taper_ratio=0.2,
                                       geometry=wing_geometry,thickness_to_chord=0.17,
                                        thickness_to_chord_loc=0.4, tight_fit_ffd=True)
    # Make ribs and spars
    # wing.construct_ribs_and_spars(aircraft.geometry, num_ribs=8, LE_TE_interpolation="ellipse")

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
    tail_AR = csdl.Variable(name="tail_AR", shape=(1, ), value=4.3)
    # tail_AR.set_as_design_variable(lower=0.8 * 4.3, upper=1.2*4.3, scaler=5e-1)
    tail_S_ref = csdl.Variable(name="tail_S_ref", shape=(1, ), value=3.7)
    # tail_S_ref.set_as_design_variable(lower=0.8 * 3.7, upper=1.2*3.7, scaler=6e-1)
    h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_1"])
    h_tail = cd.aircraft.components.Wing(
        AR=tail_AR, S_ref=tail_S_ref, 
        taper_ratio=0.6, geometry=h_tail_geometry, skip_ffd=True
    )
    empennage.comps["h_tail"] = h_tail
    
    # Connect h-tail to fuselage
    base_config.connect_component_geometries(fuselage, h_tail, h_tail.TE_center)

    # Vertical tail
    v_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_2"])
    v_tail = cd.aircraft.components.Wing(AR=1.17, S_ref=2.54, taper_ratio=0.272, geometry=v_tail_geometry, skip_ffd=True, orientation="vertical")
    empennage.comps["v_tail"] = v_tail

    # Connect vtail to fuselage
    base_config.connect_component_geometries(fuselage, v_tail, v_tail.TE_root)

    # ----------motor group ----------
    power_density = 5000
    efficiency = 0.95
    motors = cd.Component()
    airframe.comps["motors"] = motors
    pusher_motor = cd.Component(power_density=power_density, efficiency=efficiency)
    motors.comps["pusher_motor"] = pusher_motor
    for i in range(8):
        motor = cd.Component(power_density=power_density, efficiency=efficiency)
        motors.comps[f"motor_{i}"] = motor

    # ---------- Parent component for all rotors ----------
    rotors = cd.Component()
    airframe.comps["rotors"] = rotors
    rotors.quantities.drag_parameters.drag_area = 0.111484
    
    # Pusher prop 
    pusher_prop_geometry = aircraft.create_subgeometry(
        search_names=["Rotor-9-disk", "Rotor_9_blades", "Rotor_9_Hub"]
    )
    pusher_radius = csdl.Variable(name="pusher_radius", shape=(1, ), value=2.74/2)
    pusher_radius.set_as_design_variable(lower=0.8*2.74/2, upper=1.2*2.74/2, scaler=8e-1)
    pusher_prop = cd.aircraft.components.Rotor(radius=pusher_radius, geometry=pusher_prop_geometry, compute_surface_area=False, skip_ffd=True)
    rotors.comps["pusher_prop"] = pusher_prop

    # Connect pusher prop to fuselage
    base_config.connect_component_geometries(fuselage, pusher_prop, connection_point=fuselage.tail_point)

    # Lift rotors / motors
    front_inner_radius = csdl.Variable(name="front_inner_radius",shape=(1, ), value=3.048/2)
    front_inner_radius.set_as_design_variable(upper=1.8, lower=1.2, scaler=1)
    rear_inner_radius = csdl.Variable(name="rear_inner_radius", shape=(1, ), value=3.048/2)
    rear_inner_radius.set_as_design_variable(upper=1.8, lower=1.2, scaler=1)
    front_outer_radius = csdl.Variable(name="front_outer_radius" ,shape=(1, ), value=3.048/2)
    front_outer_radius.set_as_design_variable(upper=1.8, lower=1.2, scaler=1)
    rear_outer_radius = csdl.Variable(name="rear_outer_radius", shape=(1, ), value=3.048/2)
    rear_outer_radius.set_as_design_variable(upper=1.8, lower=1.2, scaler=1)

    r_over_span_radius_1 = (front_inner_radius + front_outer_radius) / wing.parameters.span
    r_over_span_radius_1.name = "front_radii_intersection_constraint"
    r_over_span_radius_1.set_as_constraint(upper=0.2, lower=0.2)

    r_over_span_radius_2 = (rear_inner_radius + rear_outer_radius) / wing.parameters.span
    r_over_span_radius_2.name ="rear_radii_intersection_constraint"
    r_over_span_radius_2.set_as_constraint(upper=0.2, lower=0.2)

    radius_list = [front_outer_radius, rear_outer_radius, front_inner_radius, rear_inner_radius, 
                    front_inner_radius, rear_inner_radius, front_outer_radius, rear_outer_radius]
    
    lift_rotors = []
    for i in range(8):
        rotor_geometry = aircraft.create_subgeometry(
            search_names=[f"Rotor_{i+1}_disk", f"Rotor_{i+1}_Hub", f"Rotor_{i+1}_blades"]
        )
        rotor = cd.aircraft.components.Rotor(radius=radius_list[i], geometry=rotor_geometry, compute_surface_area=False, skip_ffd=True)
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
    battery = cd.Component(energy_density=400)
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
            wing, 40, 2, LE_interp="ellipse", TE_interp="ellipse", 
            spacing_spanwise="linear", ignore_camber=True, plot=False,
        )
        # wing_chord_surface.project_airfoil_points()
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
        chord_cps = csdl.Variable(name="pusher_prop_chord_cps", shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        chord_cps.set_as_design_variable(upper=0.5, lower=0.02, scaler=2)
        twist_cps = csdl.Variable(name="pusher_prop_twist_cps", shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        twist_cps.set_as_design_variable(upper=np.deg2rad(85), lower=np.deg2rad(3), scaler=5)
        pusher_prop_mesh.chord_profile = blade_parameterization.evaluate_radial_profile(chord_cps)
        pusher_prop_mesh.twist_profile = blade_parameterization.evaluate_radial_profile(twist_cps)
        pusher_prop_mesh.radius = pusher_prop.parameters.radius
        rotor_meshes.discretizations["pusher_prop_mesh"] = pusher_prop_mesh

        front_inner_chord = csdl.Variable(name="front_inner_chord",shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        front_inner_chord.set_as_design_variable(upper=0.5, lower=0.02, scaler=2)
        rear_inner_chord = csdl.Variable(name="rear_inner_chord", shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        rear_inner_chord.set_as_design_variable(upper=0.5, lower=0.02, scaler=2)
        front_outer_chord = csdl.Variable(name="front_outer_chord" ,shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        front_outer_chord.set_as_design_variable(upper=0.5, lower=0.02, scaler=2)
        rear_outer_chord = csdl.Variable(name="rear_outer_chord", shape=(4, ), value=np.linspace(0.3, 0.1, 4))
        rear_outer_chord.set_as_design_variable(upper=0.5, lower=0.02, scaler=2)

        chord_cp_list = [front_outer_chord, rear_outer_chord, front_inner_chord, rear_inner_chord, 
                      front_inner_chord, rear_inner_chord, front_outer_chord, rear_outer_chord]
        
        front_inner_twist = csdl.Variable(name="front_inner_twist",shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        front_inner_twist.set_as_design_variable(upper=np.deg2rad(85), lower=np.deg2rad(3), scaler=5)
        rear_inner_twist = csdl.Variable(name="rear_inner_twist", shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        rear_inner_twist.set_as_design_variable(upper=np.deg2rad(85), lower=np.deg2rad(3), scaler=5)
        front_outer_twist = csdl.Variable(name="front_outer_twist" ,shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        front_outer_twist.set_as_design_variable(upper=np.deg2rad(85), lower=np.deg2rad(3), scaler=5)
        rear_outer_twist = csdl.Variable(name="rear_outer_twist", shape=(4, ), value=np.linspace(np.deg2rad(30), np.deg2rad(10), 4))
        rear_outer_twist.set_as_design_variable(upper=np.deg2rad(85), lower=np.deg2rad(3), scaler=5)

        twist_cp_list = [front_outer_twist, rear_outer_twist, front_inner_twist, rear_inner_twist, 
                      front_inner_twist, rear_inner_twist, front_outer_twist, rear_outer_twist]

        # lift rotors
        for i in range(8):
            rotor_mesh = cd.mesh.make_rotor_mesh(
                lift_rotors[i], num_radial=30, num_blades=2,
            )
            chord_cps = chord_cp_list[i] #csdl.Variable(shape=(4, ), value=)
            twist_cps = twist_cp_list[i] #csdl.Variable(shape=(4, ), value=)
            rotor_mesh.chord_profile = blade_parameterization.evaluate_radial_profile(chord_cps)
            rotor_mesh.twist_profile = blade_parameterization.evaluate_radial_profile(twist_cps)
            rotor_mesh.radius = lift_rotors[i].parameters.radius
            rotor_meshes.discretizations[f"rotor_{i+1}_mesh"] = rotor_mesh

    # Run inner optimization if specified
    if run_ffd:
        base_config.setup_geometry(plot=False, recorder=recorder)
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
    if do_cruise:
        pitch_angle = csdl.Variable(name="cruise_pitch", shape=(1, ), value=cruise_pitch)
        pitch_angle.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-10), scaler=10)
        cruise = cd.aircraft.conditions.CruiseCondition(
            altitude=1,
            range=56 * cd.Units.length.kilometer_to_m,
            mach_number=0.18, 
            pitch_angle=pitch_angle,
        )
        cruise.configuration = base_config.copy()
        conditions["cruise"] = cruise

    # Climb
    if do_climb:
        pitch_angle = csdl.Variable(name="climb_pitch", shape=(1, ), value=climb_pitch)
        pitch_angle.set_as_design_variable(upper=np.deg2rad(10), lower=np.deg2rad(-2), scaler=10)
        flight_path_angle = csdl.Variable(name="climb_gamm", shape=(1, ), value=np.deg2rad(6))
        climb = cd.aircraft.conditions.ClimbCondition(
            initial_altitude=300, 
            final_altitude=1000,
            mach_number=0.18,
            pitch_angle=pitch_angle,
            fligth_path_angle=flight_path_angle,
        )
        climb.configuration = base_config.copy()
        conditions["climb"] = climb

    # descent
    if do_descent:
        pitch_angle = csdl.Variable(name="descent_pitch", shape=(1, ), value=descent_pitch)
        pitch_angle.set_as_design_variable(upper=np.deg2rad(2), lower=np.deg2rad(-10), scaler=2)
        flight_path_angle = csdl.Variable(name="descent_gamm", shape=(1, ), value=np.deg2rad(-3))
        descent = cd.aircraft.conditions.ClimbCondition(
            initial_altitude=1000, 
            final_altitude=300,
            mach_number=0.18,
            pitch_angle=pitch_angle,
            fligth_path_angle=flight_path_angle,
        )
        descent.configuration = base_config.copy()
        conditions["descent"] = descent

    if do_structural_sizing:
        # +3g 
        pitch_angle = csdl.Variable(name="3g_pitch", shape=(1, ), value=np.deg2rad(7))
        pitch_angle.set_as_design_variable(upper=np.deg2rad(15), lower=5)
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
        pitch_angle = csdl.Variable(name="m1g_pitch", shape=(1, ), value=np.deg2rad(-5))
        pitch_angle.set_as_design_variable(upper=np.deg2rad(0), lower=np.deg2rad(-15))
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


    if do_qst:
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
        # qst.configuration = base_config.copy()
        conditions["qst"] = qst

    return 

def define_mass_properties(caddee: cd.CADDEE):
    # Get base config and conditions
    base_config = caddee.base_configuration
    conditions = caddee.conditions

    if do_cruise:
        cruise = conditions["cruise"]
        cruise_speed = cruise.parameters.speed[0]
    elif do_climb:
        cruise = conditions["climb"]
        cruise_speed = cruise.parameters.speed[0]
    elif do_descent:
        cruise = conditions["descent"]
        cruise_speed = cruise.parameters.speed[0]
    else:
        cruise_speed = csdl.Variable(shape=(1, ), value=61.24764871)

    # Get system component
    aircraft = base_config.system

    # Get airframe and its components
    airframe  = aircraft.comps["airframe"]
    
    # battery
    battery = airframe.comps["battery"]
    battery_mass = csdl.Variable(name="battery", shape=(1, ), value=800)
    battery_mass.set_as_design_variable(lower=0.8 * 800, upper=1.2 * 800, scaler=8e-2)
    battery_cg = csdl.Variable(shape=(3, ), value=np.array([-2.85, 0., -1.]))
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
    payload_cg = csdl.Variable(shape=(3, ), value=np.array([-3., 0., -1.5]))
    payload.quantities.mass_properties.mass = payload_mass
    payload.quantities.mass_properties.cg_vector = payload_cg

    # systems
    systems = airframe.comps["systems"]
    systems_mass = csdl.Variable(shape=(1, ), value=244)
    systems_cg = csdl.Variable(shape=(3, ), value=np.array([-1., 0., -1.5]))
    systems.quantities.mass_properties.mass = systems_mass
    systems.quantities.mass_properties.cg_vector = systems_cg

    # Assemble system mass properties
    base_config.assemble_system_mass_properties(update_copies=True)

    aircraft_mass = base_config.system.quantities.mass_properties.mass
    aircraft_mass.name = "aircraft_mass"
    aircraft_mass.set_as_objective(scaler=8e-4)

def define_quasi_steady_transition(qst, mass_properties):
    qst_config = qst.vectorized_configuration
    # qst_config = qst.configuration
    qst_system = qst_config.system

    airframe = qst_system.comps["airframe"]
    h_tail = airframe.comps["empennage"].comps["h_tail"]
    v_tail = airframe.comps["empennage"].comps["v_tail"]
    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    rotors = airframe.comps["rotors"]
    booms = list(airframe.comps["booms"].comps.values())
    
    qst_mesh_container = qst_config.mesh_container

    num_nodes = 10

    tail_actuation_var = csdl.Variable(name='qst_tail_actuation', shape=(num_nodes, ), value=qst_tail_deflection_dvs)
    tail_actuation_var.set_as_design_variable(upper=np.deg2rad(20), lower=np.deg2rad(-20), scaler=5)
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

    # BEM solver pusher rotor
    rotor_meshes = qst_mesh_container["rotor_meshes"]
    pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    mesh_vel = pusher_rotor_mesh.nodal_velocities
    qst_rpm = csdl.Variable(name="qst_pusher_rpm", shape=(num_nodes, ), value=qst_pusher_prop_rpm_dvs)
    qst_rpm.set_as_design_variable(upper=3000, lower=400, scaler=1e-3)
    bem_inputs = RotorAnalysisInputs()
    bem_inputs.ac_states = qst.quantities.ac_states
    bem_inputs.atmos_states =  qst.quantities.atmos_states
    bem_inputs.mesh_parameters = pusher_rotor_mesh
    bem_inputs.mesh_velocity = mesh_vel
    bem_inputs.rpm = qst_rpm
    bem_model = BEMModel(num_nodes=num_nodes, airfoil_model=NACA4412MLAirfoilModel())
    bem_outputs = bem_model.evaluate(bem_inputs)
    pusher_prop_power = bem_outputs.total_power
    qst_power = {"pusher_prop" : pusher_prop_power}


    # BEM solvers lift rotors
    lift_rotor_forces = []
    lift_rotor_moments = []

    front_inner_rpm = csdl.Variable(name="qst_front_inner_rpm",shape=(num_nodes, ), value=qst_front_inner_rotor_rpm_dvs)
    front_inner_rpm.set_as_design_variable(lower=100, upper=2000, scaler=1e-3)
    rear_inner_rpm = csdl.Variable(name="qst_rear_inner_rpm", shape=(num_nodes, ), value=qst_rear_inner_rotor_rpm_dvs)
    rear_inner_rpm.set_as_design_variable(lower=100, upper=2000, scaler=1e-3)
    front_outer_rpm = csdl.Variable(name="qst_front_outer_rpm" ,shape=(num_nodes, ), value=qst_front_outer_rotor_rpm_dvs)
    front_outer_rpm.set_as_design_variable(lower=100, upper=2000, scaler=1e-3)
    rear_outer_rpm = csdl.Variable(name="qst_rear_outer_rpm", shape=(num_nodes, ), value=qst_rear_outer_rotor_rpm_dvs)
    rear_outer_rpm.set_as_design_variable(lower=100, upper=2000, scaler=1e-3)

    rpm_list = [front_outer_rpm, rear_outer_rpm, front_inner_rpm, rear_inner_rpm, front_inner_rpm, rear_inner_rpm, front_outer_rpm, rear_outer_rpm]
    for i in range(8):
        rotor_mesh = rotor_meshes.discretizations[f"rotor_{i+1}_mesh"]
        mesh_vel = rotor_mesh.nodal_velocities
        lift_rotor_inputs = RotorAnalysisInputs()
        lift_rotor_inputs.ac_states = qst.quantities.ac_states
        lift_rotor_inputs.atmos_states =  qst.quantities.atmos_states
        lift_rotor_inputs.mesh_parameters = rotor_mesh
        lift_rotor_inputs.mesh_velocity = mesh_vel
        lift_rotor_inputs.rpm = rpm_list[i]
        lift_rotor_model = BEMModel(num_nodes=num_nodes, airfoil_model=NACA4412MLAirfoilModel())
        lift_rotor_outputs = lift_rotor_model.evaluate(lift_rotor_inputs)
        lift_rotor_forces.append(lift_rotor_outputs.forces)
        lift_rotor_moments.append(lift_rotor_outputs.moments)
        qst_power[f"lift_rotor_{i}"] = lift_rotor_outputs.total_power

    qst.quantities.rotor_power_dict = qst_power

    total_forces_qst, total_moments_qst = qst.assemble_forces_and_moments(
        aero_propulsive_forces=[vlm_forces, bem_outputs.forces, drag_build_up] + lift_rotor_forces, 
        aero_propulsive_moments=[vlm_moments, bem_outputs.moments] + lift_rotor_moments,
        ac_mps=mass_properties,
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_qst = eom_model.evaluate(
        total_forces=total_forces_qst,
        total_moments=total_moments_qst,
        ac_states=qst.quantities.ac_states,
        ac_mass_properties=mass_properties
    )

    dv_dt = accel_qst.dv_dt
    dp_dt = accel_qst.dp_dt
    dq_dt = accel_qst.dq_dt
    dr_dt = accel_qst.dr_dt

    zero_accel_norm = csdl.norm(dv_dt + dp_dt + dq_dt + dr_dt) # 
    zero_accel_norm.name = "residual_norm"
    zero_accel_norm.set_as_constraint(lower=0, upper=0, scaler=5)
    
    du_dt = accel_qst.du_dt
    dw_dt = accel_qst.dw_dt
    du_dt.name = "du_dt"
    dw_dt.name = "dw_dt"
    dV_dt_constraint = np.array(
            [3.05090108, 1.84555602, 0.67632681, 0.39583939, 0.30159843, 
            0.25379256, 0.22345727, 0.20269499, 0.18808881, 0.17860702]
        )
    theta = np.array([-0.0134037, -0.04973228, 0.16195989, 0.10779469, 0.04, 0.06704556, 0.05598293, 0.04712265, 0.03981101, 0.03369678])
    
    du_dt_constraint = dV_dt_constraint * np.cos(theta)
    dw_dt_constraint = dV_dt_constraint * np.sin(theta)
    
    du_dt.set_as_constraint(upper=du_dt_constraint, lower=du_dt_constraint, scaler=1)
    dw_dt.set_as_constraint(upper=dw_dt_constraint, lower=dw_dt_constraint, scaler=10)

    return accel_qst, total_forces_qst, total_moments_qst

# def define_hover_but_good(hover):
#     hover_config = hover.configuration
#     mesh_container = hover_config.mesh_container
#     rotor_meshes = mesh_container["rotor_meshes"]

#     # Re-evaluate meshes and compute nodal velocities
#     hover.finalize_meshes()

#     # BEM analysis
#     bem_forces = []
#     bem_moments = []
#     rpm_list = []
#     hover_power = {}

#     rpm_stack = csdl.Variable(shape=(8, ), value=0)
#     radius_stack = csdl.Variable(shape=(8, ), value=0)
#     thrust_vector_stack = csdl.Variable(shape=(8, 3), value=0)
#     thrust_origin_stack = csdl.Variable(shape=(8, 3), value=0)
#     chord_profile_stack = csdl.Variable(shape=(8, 30), value=0)
#     twist_profile_stack = csdl.Variable(shape=(8, 30), value=0)
#     nodal_velocity_stack = csdl.Variable(shape=(8, 3), value=0)

#     for i in range(8):
#         rpm = csdl.Variable(name=f"hover_lift_rotor_{i}_rpm", shape=(1, ), value=hover_lift_rotor_rpms[i])
#         rpm.set_as_design_variable(upper=1500, lower=500, scaler=1e-3)
#         rpm_list.append(rpm)
#         rpm_stack = rpm_stack.set(csdl.slice[i], rpm)

#         rotor_mesh = rotor_meshes.discretizations[f"rotor_{i+1}_mesh"]
#         mesh_vel = rotor_mesh.nodal_velocities
#         nodal_velocity_stack = nodal_velocity_stack.set(
#             slices=csdl.slice[i, :], value=mesh_vel.flatten()
#         )

#         radius_stack = radius_stack.set(csdl.slice[i], rotor_mesh.radius)

#         thrust_vector_stack = thrust_vector_stack.set(
#             csdl.slice[i, :], rotor_mesh.thrust_vector
#         )

#         thrust_origin_stack = thrust_origin_stack.set(
#             csdl.slice[i, :], rotor_mesh.thrust_origin
#         )

#         chord_profile_stack = chord_profile_stack.set(
#             csdl.slice[i, :], rotor_mesh.chord_profile
#         )

#         twist_profile_stack = twist_profile_stack.set(
#             csdl.slice[i, :], rotor_mesh.twist_profile
#         )


#     # Run BEM model and store forces and moment
#     bem_model = BEMModel(
#         num_nodes=1, 
#         airfoil_model=NACA4412MLAirfoilModel(),
#         hover_mode=True,
#     )

#     power_list = []
#     for i in csdl.frange(8):
#         # Set up BEM model
#         bem_inputs = RotorAnalysisInputs()
#         bem_inputs.atmos_states = hover.quantities.atmos_states
#         bem_inputs.ac_states = hover.quantities.ac_states
#         mesh_parameters = RotorMeshParameters(
#             thrust_origin = thrust_origin_stack[i, :],
#             thrust_vector = thrust_vector_stack[i, :],
#             chord_profile = chord_profile_stack[i, :],
#             twist_profile = twist_profile_stack[i, :],
#             num_azimuthal = 1,
#             num_blades = 2,
#             num_radial = 30,
#             radius = radius_stack[i],
#         )
#         bem_inputs.mesh_parameters = mesh_parameters
#         bem_inputs.rpm = rpm_stack[i]
#         bem_inputs.mesh_velocity = nodal_velocity_stack[i, :].reshape((-1, 3))
        
#         bem_outputs = bem_model.evaluate(bem_inputs)
#         bem_forces.append(bem_outputs.forces)
#         bem_moments.append(bem_outputs.moments)
#         power_list.append(bem_outputs.total_power)

#     for power in enumerate(power_list):
#         print(power.value)
#         hover_power[f"lift_rotor_{i+1}"] = power

#     exit()
#     hover.quantities.rotor_power_dict = hover_power

#     # total forces and moments
#     total_forces_hover, total_moments_hover = hover.assemble_forces_and_moments(
#         bem_forces, bem_moments
#     )

#     # eom
#     eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
#     accel_hover = eom_model.evaluate(
#         total_forces=total_forces_hover,
#         total_moments=total_moments_hover,
#         ac_states=hover.quantities.ac_states,
#         ac_mass_properties=hover_config.system.quantities.mass_properties
#     )
#     accel_norm_hover = accel_hover.accel_norm
#     accel_norm_hover.name = "hover_trim_residual"
#     accel_norm_hover.set_as_constraint(upper=0, lower=0)

#     return accel_hover, total_forces_hover, total_moments_hover

def define_hover(hover):
    hover_config = hover.configuration
    mesh_container = hover_config.mesh_container
    rotor_meshes = mesh_container["rotor_meshes"]

    # Re-evaluate meshes and compute nodal velocities
    hover.finalize_meshes()

    # BEM analysis
    bem_forces = []
    bem_moments = []
    rpm_list = []
    hover_power = {}

    for i in range(8):
        rpm = csdl.Variable(name=f"hover_lift_rotor_{i}_rpm", shape=(1, ), value=hover_lift_rotor_rpms[i])
        
        rpm.set_as_design_variable(upper=1500, lower=500, scaler=1e-3)
        rpm_list.append(rpm)
        rotor_mesh = rotor_meshes.discretizations[f"rotor_{i+1}_mesh"]
        mesh_vel = rotor_mesh.nodal_velocities
        
        # Set up BEM model
        bem_inputs = RotorAnalysisInputs()
        bem_inputs.atmos_states = hover.quantities.atmos_states
        bem_inputs.ac_states = hover.quantities.ac_states
        bem_inputs.mesh_parameters = rotor_mesh
        bem_inputs.rpm = rpm
        bem_inputs.mesh_velocity = mesh_vel
        
        # Run BEM model and store forces and moment
        bem_model = BEMModel(
            num_nodes=1, 
            airfoil_model=NACA4412MLAirfoilModel(),
            hover_mode=True,
        )
        bem_outputs = bem_model.evaluate(bem_inputs)
        bem_forces.append(bem_outputs.forces)
        bem_moments.append(bem_outputs.moments)
        hover_power[f"lift_rotor_{i+1}"] = bem_outputs.total_power

    hover.quantities.rotor_power_dict = hover_power

    # total forces and moments
    total_forces_hover, total_moments_hover = hover.assemble_forces_and_moments(
        bem_forces, bem_moments
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_hover = eom_model.evaluate(
        total_forces=total_forces_hover,
        total_moments=total_moments_hover,
        ac_states=hover.quantities.ac_states,
        ac_mass_properties=hover_config.system.quantities.mass_properties
    )
    accel_norm_hover = accel_hover.accel_norm
    accel_norm_hover.name = "hover_trim_residual"
    accel_norm_hover.set_as_constraint(upper=0, lower=0)

    return accel_hover, total_forces_hover, total_moments_hover

def define_cruise(cruise):
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    airframe = cruise_config.system.comps["airframe"]
    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    v_tail = airframe.comps["empennage"].comps["v_tail"]
    rotors = airframe.comps["rotors"]
    booms = list(airframe.comps["booms"].comps.values())

    # Actuate tail
    tail = airframe.comps["empennage"].comps["h_tail"]
    elevator_deflection = csdl.Variable(name="cruise_elevator", shape=(1, ), value=cruise_tail_deflection)
    elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=10)
    tail.actuate(elevator_deflection)

    # Re-evaluate meshes and compute nodal velocities
    cruise.finalize_meshes()

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]
    airfoil_upper_nodes = wing_lattice._airfoil_upper_para
    airfoil_lower_nodes = wing_lattice._airfoil_lower_para
    pressure_indexed_space : lfs.FunctionSetSpace = wing.quantities.pressure_space

    # run vlm solver
    lattice_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    lattice_nodal_velocitiies = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]
    
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
    
    vlm_outputs = vlm_solver(
        lattice_coordinates, 
        lattice_nodal_velocitiies, 
        atmos_states=cruise.quantities.atmos_states,
        airfoil_Cd_models=[None, None],#=airfoil_Cd_models,
        airfoil_Cl_models=[Cl_model, None],
        airfoil_Cp_models=[None, None],
        airfoil_alpha_stall_models=[None, None],
    )
    
    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment
    
    # spanwise_Cp = vlm_outputs.surface_spanwise_Cp[0]
    # spanwise_Cp = csdl.blockmat([[spanwise_Cp[0, :, 0:120].T()], [spanwise_Cp[0, :, 120:].T()]])
    # P_inf = cruise.quantities.atmos_states.pressure
    # V_inf = cruise.parameters.speed
    # rho_inf = cruise.quantities.atmos_states.density
    # spanwise_pressure = spanwise_Cp #* 0.5 * rho_inf * V_inf**2 + P_inf

    # pressure_function = pressure_indexed_space.fit_function_set(
    #     values=spanwise_pressure.reshape((-1, 1)), parametric_coordinates=airfoil_upper_nodes+airfoil_lower_nodes,
    #     regularization_parameter=1e-4,
    # )

    # Cp_upper = pressure_function.evaluate(airfoil_upper_nodes).reshape((120, 40)).value
    # Cp_lower = pressure_function.evaluate(airfoil_lower_nodes).reshape((120, 40)).value
    # # print(Cp_upper[0, :])
    # # print(Cp_lower[0, :])
    # x_interp = 0.5 + 0.5*np.sin(np.pi*(np.linspace(0., 1., 120)-0.5))

    # plt.plot(x_interp, Cp_upper[:, 20], color="r", label="IDW")
    # plt.plot(x_interp, Cp_lower[:, 20], color="r")
    # plt.plot(x_interp, spanwise_pressure[0:120, 20].value, color="b", label="data")
    # plt.plot(x_interp, spanwise_pressure[120:, 20].value, color="b")
    # plt.title("mid panel")
    # plt.legend()

    # wing.geometry.plot_but_good(color=pressure_function)

    # Drag build-up
    drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up

    drag_build_up = drag_build_up_model(cruise.quantities.ac_states, cruise.quantities.atmos_states,
                                        wing.parameters.S_ref, [wing, fuselage, tail, v_tail, rotors] + booms)
    
    
    cruise_power = {}

    # BEM solver
    rotor_meshes = mesh_container["rotor_meshes"]
    pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    mesh_vel = pusher_rotor_mesh.nodal_velocities
    cruise_rpm = csdl.Variable(name="cruise_pusher_rpm", shape=(1, ), value=cruise_pusher_rpm)
    cruise_rpm.set_as_design_variable(upper=2500, lower=1200, scaler=1e-3)
    bem_inputs = RotorAnalysisInputs()
    bem_inputs.ac_states = cruise.quantities.ac_states
    bem_inputs.atmos_states =  cruise.quantities.atmos_states
    bem_inputs.mesh_parameters = pusher_rotor_mesh
    bem_inputs.mesh_velocity = mesh_vel
    bem_inputs.rpm = cruise_rpm
    bem_model = BEMModel(num_nodes=1, airfoil_model=NACA4412MLAirfoilModel())
    bem_outputs = bem_model.evaluate(bem_inputs)
    cruise_power["pusher_prop"] = bem_outputs.total_power
    cruise.quantities.rotor_power_dict = cruise_power

    # total forces and moments
    total_forces_cruise, total_moments_cruise = cruise.assemble_forces_and_moments(
        [vlm_forces, drag_build_up, bem_outputs.forces], [vlm_moments, bem_outputs.moments]
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_cruise = eom_model.evaluate(
        total_forces=total_forces_cruise,
        total_moments=total_moments_cruise,
        ac_states=cruise.quantities.ac_states,
        ac_mass_properties=cruise_config.system.quantities.mass_properties
    )
    accel_norm_cruise = accel_cruise.accel_norm
    accel_norm_cruise.name = "cruise_trim"
    accel_norm_cruise.set_as_constraint(upper=0, lower=0, scaler=4)
    
    return accel_cruise, total_forces_cruise, total_moments_cruise

def define_climb(climb):
    climb_config = climb.configuration
    mesh_container = climb_config.mesh_container
    airframe = climb_config.system.comps["airframe"]
    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    v_tail = airframe.comps["empennage"].comps["v_tail"]
    rotors = airframe.comps["rotors"]
    booms = list(airframe.comps["booms"].comps.values())

    # Actuate tail
    tail = airframe.comps["empennage"].comps["h_tail"]
    elevator_deflection = csdl.Variable(name="climb_elevator", shape=(1, ), value=climb_tail_deflection)
    elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(12), scaler=10)
    tail.actuate(elevator_deflection)

    # Re-evaluate meshes and compute nodal velocities
    climb.finalize_meshes()

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]

    # run vlm solver
    lattice_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    lattice_nodal_velocitiies = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]
    
    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
            aoa_range=np.linspace(-12, 16, 50), 
            reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
            mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    
    vlm_outputs = vlm_solver(
        lattice_coordinates, 
        lattice_nodal_velocitiies, 
        atmos_states=climb.quantities.atmos_states,
        airfoil_Cd_models=[None, None],#=airfoil_Cd_models,
        airfoil_Cl_models=[Cl_model, None],
        airfoil_Cp_models=[None, None],
        airfoil_alpha_stall_models=[None, None],
    )
    
    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment
    
    # Drag build-up
    drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up

    drag_build_up = drag_build_up_model(climb.quantities.ac_states, climb.quantities.atmos_states,
                                        wing.parameters.S_ref, [wing, fuselage, tail, v_tail, rotors] + booms)
    
    
    # BEM solver
    climb_power = {}
    rotor_meshes = mesh_container["rotor_meshes"]
    pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    mesh_vel = pusher_rotor_mesh.nodal_velocities
    climb_rpm = csdl.Variable(name="climb_pusher_rpm", shape=(1, ), value=climb_pusher_rpm)
    climb_rpm.set_as_design_variable(upper=2500, lower=1200, scaler=1e-3)
    bem_inputs = RotorAnalysisInputs()
    bem_inputs.ac_states = climb.quantities.ac_states
    bem_inputs.atmos_states =  climb.quantities.atmos_states
    bem_inputs.mesh_parameters = pusher_rotor_mesh
    bem_inputs.mesh_velocity = mesh_vel
    bem_inputs.rpm = climb_rpm
    bem_model = BEMModel(num_nodes=1, airfoil_model=NACA4412MLAirfoilModel())
    bem_outputs = bem_model.evaluate(bem_inputs)
    climb_power["pusher_prop"] = bem_outputs.total_power
    climb.quantities.rotor_power_dict = climb_power


    # total forces and moments
    total_forces_climb, total_moments_climb = climb.assemble_forces_and_moments(
        [vlm_forces, drag_build_up, bem_outputs.forces], [vlm_moments, bem_outputs.moments]
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_climb = eom_model.evaluate(
        total_forces=total_forces_climb,
        total_moments=total_moments_climb,
        ac_states=climb.quantities.ac_states,
        ac_mass_properties=climb_config.system.quantities.mass_properties
    )
    accel_norm_climb = accel_climb.accel_norm
    # accel_norm_climb.set_as_objective(scaler=1e-1)
    accel_norm_climb.set_as_constraint(upper=0, lower=0, scaler=1e-1)
    
    return accel_climb, total_forces_climb, total_moments_climb

def define_descent(descent):
    descent_config = descent.configuration
    mesh_container = descent_config.mesh_container
    airframe = descent_config.system.comps["airframe"]
    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    v_tail = airframe.comps["empennage"].comps["v_tail"]
    rotors = airframe.comps["rotors"]
    booms = list(airframe.comps["booms"].comps.values())

    # Actuate tail
    tail = airframe.comps["empennage"].comps["h_tail"]
    elevator_deflection = csdl.Variable(name="descent_elevator", shape=(1, ), value=descent_tail_deflection)
    elevator_deflection.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(12), scaler=10)
    tail.actuate(elevator_deflection)

    # Re-evaluate meshes and compute nodal velocities
    descent.finalize_meshes()

    # Set up VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]
    wing_lattice = vlm_mesh.discretizations["wing_chord_surface"]
    tail_lattice = vlm_mesh.discretizations["tail_chord_surface"]

    # run vlm solver
    lattice_coordinates = [wing_lattice.nodal_coordinates, tail_lattice.nodal_coordinates]
    lattice_nodal_velocitiies = [wing_lattice.nodal_velocities, tail_lattice.nodal_velocities]
    
    # Add an airfoil model
    nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="ls417",
            aoa_range=np.linspace(-12, 16, 50), 
            reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
            mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    Cl_model = nasa_langley_airfoil_maker.get_airfoil_model(quantities=["Cl"])
    
    vlm_outputs = vlm_solver(
        lattice_coordinates, 
        lattice_nodal_velocitiies, 
        atmos_states=descent.quantities.atmos_states,
        airfoil_Cd_models=[None, None],#=airfoil_Cd_models,
        airfoil_Cl_models=[Cl_model, None],
        airfoil_Cp_models=[None, None],
        airfoil_alpha_stall_models=[None, None],
    )
    
    vlm_forces = vlm_outputs.total_force
    vlm_moments = vlm_outputs.total_moment
    
    # Drag build-up
    drag_build_up_model = cd.aircraft.models.aero.compute_drag_build_up

    drag_build_up = drag_build_up_model(descent.quantities.ac_states, descent.quantities.atmos_states,
                                        wing.parameters.S_ref, [wing, fuselage, tail, v_tail, rotors] + booms)
    
    
    # BEM solver
    descent_power = {}
    rotor_meshes = mesh_container["rotor_meshes"]
    pusher_rotor_mesh = rotor_meshes.discretizations["pusher_prop_mesh"]
    mesh_vel = pusher_rotor_mesh.nodal_velocities
    descent_rpm = csdl.Variable(name="descent_pusher_rpm", shape=(1, ), value=descent_pusher_rpm)
    descent_rpm.set_as_design_variable(upper=2500, lower=1200, scaler=1e-3)
    bem_inputs = RotorAnalysisInputs()
    bem_inputs.ac_states = descent.quantities.ac_states
    bem_inputs.atmos_states =  descent.quantities.atmos_states
    bem_inputs.mesh_parameters = pusher_rotor_mesh
    bem_inputs.mesh_velocity = mesh_vel
    bem_inputs.rpm = descent_rpm
    bem_model = BEMModel(num_nodes=1, airfoil_model=NACA4412MLAirfoilModel())
    bem_outputs = bem_model.evaluate(bem_inputs)
    descent_power["pusher_prop"] = bem_outputs.total_power
    descent.quantities.rotor_power_dict = descent_power


    # total forces and moments
    total_forces_descent, total_moments_descent = descent.assemble_forces_and_moments(
        [vlm_forces, drag_build_up, bem_outputs.forces], [vlm_moments, bem_outputs.moments]
    )

    # eom
    eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel()
    accel_descent = eom_model.evaluate(
        total_forces=total_forces_descent,
        total_moments=total_moments_descent,
        ac_states=descent.quantities.ac_states,
        ac_mass_properties=descent_config.system.quantities.mass_properties
    )
    accel_norm_descent = accel_descent.accel_norm
    accel_norm_descent.set_as_constraint(upper=0, lower=0, scaler=1e-1)
    
    return accel_descent, total_forces_descent, total_moments_descent

def define_analysis(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration
    base_mps = base_config.system.quantities.mass_properties

    if do_hover:
        hover = conditions["hover"]
        accel_hover, total_forces_hover, total_moments_hover = define_hover(hover)
    
    if do_qst:
        qst = conditions["qst"]
        accel_qst, total_forces_qst, total_moments_qst = define_quasi_steady_transition(qst, base_mps)

    if do_climb:
        climb = conditions["climb"]
        accel_climb, total_forces_climb, total_moments_climb = define_climb(climb)

    if do_cruise:
        cruise = conditions["cruise"]
        accel_cruise, total_forces_cruise, total_moments_cruise = define_cruise(cruise)

    if do_descent:
        descent = conditions["descent"]
        accel_descent, total_forces_descent, total_moments_descent = define_descent(descent)


    # accel_norm = accel_hover.accel_norm + accel_climb.accel_norm + accel_cruise.accel_norm
    accel_norm = accel_hover.accel_norm 
    # accel_norm = accel_descent.accel_norm #(accel_climb.accel_norm**2 + accel_cruise.accel_norm**2)**0.5
    return accel_norm

def define_post_proecss(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration
    airframe = base_config.system.comps["airframe"]
    rotors = airframe.comps["rotors"]
    battery = airframe.comps["battery"]
    battery_mass = battery.quantities.mass_properties.mass
    energy_density = battery.parameters.energy_density
    total_energy_available = battery_mass * energy_density * 3600

    # get rotor power for each conditions
    # Hover
    hover = conditions["hover"]
    hover_time = hover.parameters.time.reshape((-1, 1))
    hover_power_dict = hover.quantities.rotor_power_dict
    total_hover_power = 0
    for lift_rotor_power in hover_power_dict.values():
        total_hover_power = total_hover_power +  lift_rotor_power
    total_hover_power = total_hover_power.reshape((-1, 1))

    # qst
    if do_qst:
        qst = conditions["qst"]
        qst_time = qst.parameters.time.reshape((-1, 1))
        qst_power_dict = qst.quantities.rotor_power_dict
        qst_pusher_rotor_power = qst_power_dict["pusher_prop"]
        total_qst_power = qst_pusher_rotor_power
        for lift_rotor_power in qst_power_dict.values():
            total_qst_power = total_qst_power +  lift_rotor_power

        total_qst_power = total_qst_power.reshape((-1, 1))
    else:
        qst_time = csdl.Variable(shape=(1, 1), value=0)
        total_qst_power = csdl.Variable(shape=(1, 1), value=0)

    # climb
    climb = conditions["climb"]
    climb_time = climb.parameters.time.reshape((-1, 1))
    climb_power_dict = climb.quantities.rotor_power_dict
    climb_pusher_rotor_power = climb_power_dict["pusher_prop"].reshape((-1, 1))

    # cruise
    cruise = conditions["cruise"]
    cruise_time = cruise.parameters.time.reshape((-1, 1))
    cruise_power_dict = cruise.quantities.rotor_power_dict
    cruise_pusher_rotor_power = cruise_power_dict["pusher_prop"].reshape((-1, 1))

    # descent
    descent = conditions["descent"]
    descent_time = descent.parameters.time.reshape((-1, 1))
    descent_power_dict = descent.quantities.rotor_power_dict
    descent_pusher_rotor_power = descent_power_dict["pusher_prop"].reshape((-1, 1))

    total_power = csdl.vstack((total_hover_power, total_qst_power, climb_pusher_rotor_power, cruise_pusher_rotor_power, descent_pusher_rotor_power)) / 0.95
    mission_time_vec = csdl.vstack((hover_time, qst_time, climb_time, cruise_time, descent_time))
    num_nodes = total_power.shape[0]
    time_vec = csdl.Variable(shape=(num_nodes, ), value=0)
    cum_sum = 0
    for i in range(num_nodes):
        cum_sum = cum_sum + mission_time_vec[i]
        time_vec = time_vec.set(
            slices=csdl.slice[i],
            value=cum_sum 
        )

    mission_energy = csdl.sum(total_power * mission_time_vec)
    soc : csdl.Variable = (total_energy_available - 2.2 * mission_energy) / total_energy_available
    soc.name = "final_soc"

    soc.set_as_constraint(lower=0.2, scaler=2)

define_base_config(caddee)

define_conditions(caddee)

define_mass_properties(caddee)

accel_norm = define_analysis(caddee)

define_post_proecss(caddee)

recorder.count_operations()
# recorder.count_origins(n=20, mode="line")


if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    from modopt import IPOPT
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder, gpu=False,
        # additional_outputs=[accel.du_dt, accel.dv_dt, accel.dw_dt, accel.dp_dt, accel.dq_dt, accel.dr_dt, total_forces, total_moments]
        # additional_outputs=[accel_norm],
    )
    # jax_sim.run()


    py_sim = csdl.experimental.PySimulator(
        recorder=recorder,
    )

    py_sim.check_totals()

    # jax_sim.check_totals(step_size=1e-5)
    # py_sim.check_totals(step_size=1e-5)
    # exit()

    prob = CSDLAlphaProblem(problem_name='LPC_trim', simulator=jax_sim)

    optimizer = SLSQP(prob, ftol=1e-8, maxiter=50, outputs=['x'])
    # optimizer = IPOPT(prob, solver_options={'max_iter': 200, 'tol': 1e-7})

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()

recorder.execute()

import pickle

dv_save_dict = {}
constraints_save_dict = {}

dv_dict = recorder.design_variables
constraint_dict = recorder.constraints

csdl.inline_export("trim_opt_descent")

for dv in dv_dict.keys():
    dv_save_dict[dv.name] = dv.value
    print(dv.value)

for c in constraint_dict.keys():
    constraints_save_dict[c.name] = c.value
    print(c.value)

with open("lpc_dv_dict.pickle", "wb") as handle:
    pickle.dump(dv_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("lpc_dv_dict.pickle", "wb") as handle:
    pickle.dump(constraints_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("total_forces", jax_sim[total_forces])
# print("total_moments", jax_sim[total_moments])

# print("\n")
# print("accel norm sum", jax_sim[accel_norm])
# print("\n")

dv_dict = recorder.design_variables

for dv in dv_dict.keys():
    dv.save()


for c in constraint_dict.keys():
    constraints_save_dict[c.name] = c.value
    c.save()

exit()


# ::::::::::::::::::::::::::::::::::::PySim::::::::::::::::::::::::::::::::::::
# total_forces [[-4.97112674  0.02406187 -9.51770693]]
# total_moments [[  0.51460276 -24.35518937   0.66755083]]


# du_dt [-0.00112339]
# dv_dt [0.00021442]
# dw_dt [-0.00317855]
# dp_dt [3.82166697e-05]
# dq_dt [0.00016413]
# dr_dt [7.97913106e-05]


# [0.05170884]
# [-0.03332447]
# [1858.4147944]
    


# ::::::::::::::::::::::::::::::::::::JAXSim GPU (torch cpu) run 1::::::::::::::::::::::::::::::::::::
# total_forces [[ 1.10272070e+02  2.49237931e-02 -6.00345138e+02]]
# total_moments [[ 5.33619742e-01 -2.08330948e+03  6.90579867e-01]]


# du_dt [-0.07829857]
# dv_dt [0.00022122]
# dw_dt [0.0715057]
# dp_dt [3.97731993e-05]
# dq_dt [-0.07118617]
# dr_dt [8.24702433e-05]


# [0.05461365]
# [-0.03775286]
# [1866.82700542]
    
# ::::::::::::::::::::::::::::::::::::JAXSim GPU (torch cpu) run 2::::::::::::::::::::::::::::::::::::
# total time 4.849106550216675
# total_forces [[ 1.91832639e+02  2.44923702e-02 -2.74584511e+02]]
# total_moments [[ 5.24149543e-01 -1.17848826e+03  6.78327571e-01]]


# du_dt [0.01511043]
# dv_dt [0.00021732]
# dw_dt [0.00686962]
# dp_dt [3.90614933e-05]
# dq_dt [-0.02481406]
# dr_dt [8.10108873e-05]


# [0.05317487]
# [-0.0362718]


# ::::::::::::::::::::::::::::::::::::JAXSim CPU run 1::::::::::::::::::::::::::::::::::::
# total_forces [[-8.98941318e+00  2.40175509e-02  2.41861932e+01]]
# total_moments [[  0.51363152 103.36185926   0.66627966]]


# du_dt [-0.01356258]
# dv_dt [0.00021401]
# dw_dt [0.03086753]
# dp_dt [3.81444251e-05]
# dq_dt [-0.0072491]
# dr_dt [7.96403213e-05]


# [0.0515592]
# [-0.03318248]
# [1858.11727248]


# ::::::::::::::::::::::::::::::::::::JAXSim CPU run 2::::::::::::::::::::::::::::::::::::
# total_forces [[-8.98941318e+00  2.40175509e-02  2.41861932e+01]]
# total_moments [[  0.51363152 103.36185926   0.66627966]]


# du_dt [-0.01356258]
# dv_dt [0.00021401]
# dw_dt [0.03086753]
# dp_dt [3.81444251e-05]
# dq_dt [-0.0072491]
# dr_dt [7.96403213e-05]


# [0.0515592]
# [-0.03318248]
# [1858.11727248]



#          ==============
#          Scipy summary:
#          ==============
#          Problem                    : LPC_trim
#          Solver                     : scipy_slsqp
#          Success                    : True
#          Message                    : Optimization terminated successfully
#          Objective                  : 0.0002790552862106368
#          Gradient norm              : 0.7409795232014156
#          Total time                 : 244.7714831829071
#          Major iterations           : 17
#          Total function evals       : 52
#          Total gradient evals       : 17
#          ==================================================
# /home/marius/Desktop/packages/lsdo_lab/CSDL_alpha/csdl_alpha/src/operations/division.py:19: RuntimeWarning: divide by zero encountered in divide
#   return x/y
# nonlinear solver: bracketed_search converged in 42 iterations.
# nonlinear solver: bracketed_search converged in 34 iterations.
# total_forces [[-3.22678579  0.02405088 -0.97092618]]
# total_moments [[0.51436201 1.37619778 0.66722884]]


# du_dt [-0.00050072]
# dv_dt [0.00021431]
# dw_dt [-0.00112254]
# dp_dt [3.81993796e-05]
# dq_dt [0.00025605]
# dr_dt [7.97526367e-05]


# [0.05167176]
# [-0.03329087]
# [1858.53207306]


#   ==============
#          Scipy summary:
#          ==============
#          Problem                    : LPC_trim
#          Solver                     : scipy_slsqp
#          Success                    : True
#          Message                    : Optimization terminated successfully
#          Objective                  : 0.0002790552862106368
#          Gradient norm              : 0.7409795232014156
#          Total time                 : 247.31665658950806
#          Major iterations           : 17
#          Total function evals       : 52
#          Total gradient evals       : 17
#          ==================================================
# /home/marius/Desktop/packages/lsdo_lab/CSDL_alpha/csdl_alpha/src/operations/division.py:19: RuntimeWarning: divide by zero encountered in divide
#   return x/y
# nonlinear solver: bracketed_search converged in 42 iterations.
# nonlinear solver: bracketed_search converged in 34 iterations.
# total_forces [[-3.22678579  0.02405088 -0.97092618]]
# total_moments [[0.51436201 1.37619778 0.66722884]]


# du_dt [-0.00050072]
# dv_dt [0.00021431]
# dw_dt [-0.00112254]
# dp_dt [3.81993796e-05]
# dq_dt [0.00025605]
# dr_dt [7.97526367e-05]


# [0.05167176]
# [-0.03329087]
# 1858.53207306




# total_forces [[-1.13285697e+02  2.40383106e-02  4.95744642e+00]]
# total_moments [[  0.51409049 313.98250329   0.66681587]]


# du_dt [-0.18529727]
# dv_dt [0.0002142]
# dw_dt [0.33750011]
# dp_dt [3.81802194e-05]
# dq_dt [-0.10076421]
# dr_dt [7.97095103e-05]


# [0.05163007]
# [-0.03331046]
# [1849.54752017]
    

# total_forces [[-1.34462168e+02  2.39919663e-02  3.38712928e+01]]
# total_moments [[  0.51306434 440.0387343    0.66563708]]


# du_dt [-0.18773904]
# dv_dt [0.00021387]
# dw_dt [0.33804439]
# dp_dt [3.80910506e-05]
# dq_dt [-0.0985313]
# dr_dt [7.9575203e-05]


# [0.05147152]
# [-0.03299989]
# [1847.84743352]
    


# :::::::::::::::::::::::::::::::::::::Operations count (w/ tail actuation) :::::::::::::::::::::::::::::::::::::
# Reshape : 11713
# BroadcastMult : 6979
# GetVarIndex : 6911
# Add : 5655
# SetVarIndex : 4187
# Neg : 3389
# Mult : 2125
# Loop : 2047
# BroadcastSetIndex : 1921
# RightBroadcastPower : 1562

    


# :::::::::::::::::::::::::::::::::::::Operations count (w/o tail actuation) :::::::::::::::::::::::::::::::::::::
# Reshape : 6043
# GetVarIndex : 2881
# SetVarIndex : 2587
# Loop : 2027
# BroadcastMult : 1859
# Mult : 1645
# Add : 1455
# RightBroadcastPower : 1392
# Neg : 809
# Sum : 663


Reshape : 12353
BroadcastMult : 6979
GetVarIndex : 6911
Add : 5655
SetVarIndex : 3547
Neg : 3389
Mult : 2125
Loop : 2047
BroadcastSetIndex : 1921
RightBroadcastPower : 1562


Reshape : 11713
BroadcastMult : 6979
GetVarIndex : 6911
Add : 5655
SetVarIndex : 4187
Neg : 3389
Mult : 2125
Loop : 2047
BroadcastSetIndex : 1921
RightBroadcastPower : 1562


Reshape : 10429
GetVarIndex : 6055
BroadcastMult : 5695
Add : 5655
Neg : 3389
Mult : 2125
SetVarIndex : 2051
Loop : 2047
BroadcastSetIndex : 1921

Reshape : 5479
GetVarIndex : 2465
Loop : 2047
Add : 1755
SetVarIndex : 1751
Mult : 1675
RightBroadcastPower : 1412
Neg : 989
BroadcastMult : 895
Sum : 683

Reshape : 4939
GetVarIndex : 2465
Loop : 2047
Add : 1755
SetVarIndex : 1751
Mult : 1567
Neg : 989
RightBroadcastPower : 980
BroadcastMult : 787
Sum : 683

