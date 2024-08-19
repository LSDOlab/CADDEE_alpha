import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from lsdo_acoustics.core.models.broadband.GL.GL_model import GL_model, GLVariableGroup
from lsdo_acoustics.core.models.total_noise_model import total_noise_model
from lsdo_acoustics.core.models.tonal.Lowson.Lowson_model import Lowson_model, LowsonVariableGroup
from lsdo_acoustics import Acoustics
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.utils.var_groups import RotorAnalysisInputs
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.utils.plot import make_polarplot


recorder = csdl.Recorder(inline=True)
recorder.start()


rotor_geom = cd.import_geometry("NASA_rotor_geom.stp")
caddee = cd.CADDEE()

def define_base_configuration(caddee : cd.CADDEE):
    rotor = cd.aircraft.components.Rotor(
        radius=0.860552, geometry=rotor_geom
    )
    
    # Tilt the rotor by 3.04 degree for current test case
    rotor.actuate(y_tilt_angle=np.deg2rad(-3.04))
    # rotor.plot()

    num_radial = 35
    num_azimuthal = 70

    rotor_discretization = cd.mesh.make_rotor_mesh(
        rotor_comp=rotor,
        num_radial=num_radial, 
        num_azimuthal=num_azimuthal,
        num_blades=4
    )

    chord_cps_best_acoustics = np.array([0.02, 0.02585203, 0.05309617, 0.04029735])
    twist_cps_best_acoustics = np.array([0.42892635, 0.37543559, 0.16945265, 0.16945265])
    # Acoustic optimal SPL:  55.97 (dBA)
    # Torque : 163 (N-m)
    
    chord_cps_best_aero = np.array([0.0858 , 0.08140782 ,0.0462  ,   0.0462    ])
    twist_cps_best_aero = np.array([ 0.29189949  ,0.09637374, -0.05890706 ,-0.00593083])
    # Torque : 55.41461249 (N-m) 
    # SPL: 65.20195215 (dB-A)

    chord_cps_hybrid = np.array([0.0858 ,    0.08140782, 0.0462   ,  0.0462    ])
    twist_cps_hybrid = np.array([ 0.29189949 , 0.09637374, -0.05890706 ,-0.00593083])
    # Torque: 56.49745151
    # SPL: 62.42214467

    chord_cps = csdl.Variable(name="chord_cps",shape=(4, ), value=0.066 * np.ones(shape=(4, )))
    chord_cps.set_as_design_variable(upper=1.3 * 0.066, lower=0.7*0.066, scaler=4)
    twict_cps = csdl.Variable(name="twist_cps",shape=(4, ), value=np.linspace(np.deg2rad(5.53), np.deg2rad(5.53-8), 4))
    twict_cps.set_as_design_variable(upper=np.deg2rad(70), lower=np.deg2rad(-5))
    chord_profile = BsplineParameterization(num_cp=4, num_radial=num_radial, order=3).evaluate_radial_profile(chord_cps)
    rotor_discretization.chord_profile = chord_profile
    twist_profile = BsplineParameterization(num_cp=4, num_radial=num_radial, order=3).evaluate_radial_profile(twict_cps)
    rotor_discretization.twist_profile = twist_profile

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    
    norm_radius = np.linspace(0.2, 1, 35)
    R = 0.860552
    # Plot chord/radius on the left y-axis
    ax1.plot(norm_radius, chord_profile.value/R, 'b-', label='Chord/Radius')
    ax1.set_xlabel('r/R')
    ax1.set_ylabel('c/R', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for blade twist
    ax2 = ax1.twinx()
    ax2.plot(norm_radius, twist_profile.value * 180/np.pi, 'r--', label='Blade Twist')
    ax2.set_ylabel('Blade Twist (degrees)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.grid(True)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()
    # exit()

    twist_0 = twict_cps[0]
    twist_1 = twict_cps[1]
    twist_2 = twict_cps[2]
    twist_3 = twict_cps[3]

    # twist_constr_1 = twist_0 - twist_1
    # twist_constr_1.set_as_constraint(lower=0)
    # twist_constr_2 = twist_1 - twist_2
    # twist_constr_2.set_as_constraint(lower=0)
    # twist_constr_3 = twist_2 - twist_3
    # twist_constr_3.set_as_constraint(lower=0)

    rotor_meshes = cd.mesh.RotorMeshes()
    rotor_meshes.discretizations["rotor_discretization"] = rotor_discretization

    base_config = cd.Configuration(system=rotor)
    mesh_container = base_config.mesh_container
    mesh_container["rotor_meshes"] = rotor_meshes

    caddee.base_configuration = base_config

def define_conditions(caddee : cd.CADDEE):
    base_config = caddee.base_configuration

    cruise = cd.aircraft.conditions.CruiseCondition(
        speed=43.86,
        range=1.,  
        altitude=1.,
    )
    cruise.configuration = base_config.copy()
    caddee.conditions["cruise"] = cruise

    print("cruise mach number-----", cruise.parameters.mach_number.value)
    print("cruise time------------", cruise.parameters.time.value)
    print("air density------------", cruise.quantities.atmos_states.density.value)
    print("speed of sound---------", cruise.quantities.atmos_states.speed_of_sound.value)

def define_analysis(caddee : cd.CADDEE):
    cruise : cd.aircraft.conditions.CruiseCondition = caddee.conditions["cruise"]
    cruise_config = cruise.configuration


    cruise.finalize_meshes()

    mesh_container = cruise_config.mesh_container
    rotors_meshes = mesh_container["rotor_meshes"]
    rotor_discretization = rotors_meshes.discretizations["rotor_discretization"]
    mesh_vel = rotor_discretization.nodal_velocities
    rpm = csdl.Variable(name="rpm", value=2113)

    # theta_0 = csdl.Variable(name="theta_0", value=np.deg2rad(8.16))
    # theta_0.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(20), scaler=3)
    # theta_1_c = csdl.Variable(name="theta_1_c", value=np.deg2rad(1.52))
    # theta_1_c.set_as_design_variable(lower=np.deg2rad(-5), upper=np.deg2rad(5), scaler=10)
    # theta_1_s = csdl.Variable(name="theta_1_s", value=np.deg2rad(-4.13))
    # theta_1_s.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(5), scaler=7)

    rotor_inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=mesh_vel,
        mesh_parameters=rotor_discretization,
        atmos_states=cruise.quantities.atmos_states,
        ac_states=cruise.quantities.ac_states,
        theta_0=np.deg2rad(8.16), #,  9.2, 9.37)
        theta_1_c=np.deg2rad(1.52), #  0.3, 1.11),
        theta_1_s=np.deg2rad(-4.13),#  -6.8 -3.23),
        xi_0=np.deg2rad(-0.9), #-0.95, # -1.3
    )

    airfoil_model_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="naca_0012",
        aoa_range=np.linspace(-12, 16, 50),
        reynolds_range=[2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 4e6],
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6]
    )
 
    airfoil_model = airfoil_model_maker.get_airfoil_model(
        quantities=["Cl", "Cd"],
    )
    
    peters_he_model = PetersHeModel(
        num_nodes=1, 
        airfoil_model=airfoil_model,
        tip_loss=True,
        Q=5,
        M=5,
    )

    rotor_aero_outputs = peters_he_model.evaluate(rotor_inputs)

    radius = 80
    angle = np.deg2rad(45)
    broadband_acoustics = Acoustics(aircraft_position=np.array([radius * np.sin(angle) ,0., -radius * np.cos(angle)]))
    broadband_acoustics.add_observer('obs', np.array([0., 0., 0.,]), time_vector=np.array([0.]))
    observer_data = broadband_acoustics.assemble_observers()

    gl_vg = GLVariableGroup(
        thrust_vector=rotor_discretization.thrust_vector,
        thrust_origin=rotor_discretization.thrust_origin,
        rotor_radius=rotor_discretization.radius,
        CT=rotor_aero_outputs.thrust_coefficient,
        chord_profile=rotor_discretization.chord_profile,
        mach_number=cruise.parameters.mach_number,
        rpm=rpm,
        num_radial=35,
        num_tangential=70,
        speed_of_sound=cruise.quantities.atmos_states.speed_of_sound,
    )

    gl_spl, gl_spl_A_weighted = GL_model(
        GLVariableGroup=gl_vg,
        observer_data=observer_data,
        num_blades=4,
        num_nodes=1,
        A_weighting=True,
    )

    r_nondim = csdl.linear_combination(0.2*rotor_discretization.radius+0.01, rotor_discretization.radius- 0.01, 35)/ rotor_discretization.radius

    lowson_vg = LowsonVariableGroup(
        thrust_vector=rotor_discretization.thrust_vector,
        thrust_origin=rotor_discretization.thrust_origin,
        RPM=rpm,
        speed_of_sound=cruise.quantities.atmos_states.speed_of_sound,
        rotor_radius=rotor_discretization.radius,
        mach_number=cruise.parameters.mach_number, 
        density=cruise.quantities.atmos_states.density,
        num_radial=35,
        num_tangential=70,
        dD=rotor_aero_outputs.sectional_drag,
        dT=rotor_aero_outputs.sectional_thrust,
        phi=rotor_aero_outputs.sectional_inflow_angle,
        chord_profile=rotor_discretization.chord_profile,
        nondim_sectional_radius=r_nondim.flatten(),
        thickness_to_chord_ratio=0.12,
    )

    lowson_spl, lowson_spl_A_weighted  = Lowson_model(
        LowsonVariableGroup=lowson_vg,
        observer_data=observer_data,
        num_blades=4,
        num_nodes=1,
        modes=[1, 2, 3],
        A_weighting=True,
        toggle_thickness_noise=True,
    )

    total_spl = total_noise_model(SPL_list=[lowson_spl_A_weighted, gl_spl_A_weighted])
    total_spl.name = "hover_total_noise"
    total_spl.set_as_objective(scaler=1/50)

    C_T : csdl.Variable = rotor_aero_outputs.thrust_coefficient
    C_T.name = "thrust_coefficient"
    C_T.set_as_constraint(equals=0.0064, scaler=1/0.0064)

    total_power = rotor_aero_outputs.total_power
    total_power.name = "total_power"
    total_torque = rotor_aero_outputs.total_torque

    # objective = 0.5 * total_spl + 0.5 * total_torque
    # total_torque.set_as_objective(scaler=1/50)

    # print(C_T.value)
    # print(total_torque.value)
    # print(rotor_aero_outputs.total_torque.value)
    # print(total_spl.value)
    # print(lowson_spl_A_weighted.value)
    # print(gl_spl_A_weighted.value)
    # make_polarplot(data=rotor_aero_outputs.sectional_thrust, plot_contours=True, plot_min_max=False)
    # exit()

    
define_base_configuration(caddee=caddee)

define_conditions(caddee=caddee)

define_analysis(caddee=caddee)


from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, SNOPT, PySLSQP
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=False, derivatives_kwargs= {
        "concatenate_ofs" : True
    }
)

prob = CSDLAlphaProblem(problem_name='full_optimization_iteration_1', simulator=jax_sim)

optimizer = SNOPT(
    prob, 
    solver_options = {
        'append2file' : False,
        'continue_on_failure': True,
        'Major iterations':300, 
        'Major optimality':1e-3, 
        'Major feasibility':2.4e-3,
        'Major step limit':1.5,
        'Linesearch tolerance':0.6,
    }
)

optimizer.solve()
optimizer.print_results()

recorder.execute()

for dv in recorder.design_variables.keys():
    print(dv.name, dv.value)

for c in recorder.constraints.keys():
    print(c.name, c.value)

for o in recorder.objectives.keys():
    print(o.name, o.value)

