'''Aeroacoustic Optimization'''
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
from modopt import CSDLAlphaProblem
from modopt import SLSQP


recorder = csdl.Recorder(inline=True)
recorder.start()

rotor_geom = cd.import_geometry("NASA_rotor_geom.stp")
caddee = cd.CADDEE()

def define_base_configuration(caddee : cd.CADDEE):
    rotor = cd.aircraft.components.Rotor(
        radius=0.860552, geometry=rotor_geom
    )
    
    # Tilt the rotor by 3.04 degrees (NASA test case)
    rotor.actuate(y_tilt_angle=np.deg2rad(-3.04))

    num_radial = 35
    num_azimuthal = 70

    rotor_discretization = cd.mesh.make_rotor_mesh(
        rotor_comp=rotor,
        num_radial=num_radial, 
        num_azimuthal=num_azimuthal,
        num_blades=4
    )

    chord_cps = csdl.Variable(name="chord_cps",shape=(4, ), value=0.066 * np.ones(shape=(4, )))
    chord_cps.set_as_design_variable(upper=1.3 * 0.066, lower=0.7*0.066, scaler=4)
    twict_cps = csdl.Variable(name="twist_cps",shape=(4, ), value=np.linspace(np.deg2rad(5.53), np.deg2rad(5.53-8), 4))
    twict_cps.set_as_design_variable(upper=np.deg2rad(70), lower=np.deg2rad(-5))
    chord_profile = BsplineParameterization(num_cp=4, num_radial=num_radial, order=3).evaluate_radial_profile(chord_cps)
    rotor_discretization.chord_profile = chord_profile
    twist_profile = BsplineParameterization(num_cp=4, num_radial=num_radial, order=3).evaluate_radial_profile(twict_cps)
    rotor_discretization.twist_profile = twist_profile

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

def define_analysis(caddee : cd.CADDEE):
    cruise : cd.aircraft.conditions.CruiseCondition = caddee.conditions["cruise"]
    cruise_config = cruise.configuration

    cruise.finalize_meshes()

    mesh_container = cruise_config.mesh_container
    rotors_meshes = mesh_container["rotor_meshes"]
    rotor_discretization = rotors_meshes.discretizations["rotor_discretization"]
    mesh_vel = rotor_discretization.nodal_velocities
    rpm = csdl.Variable(name="rpm", value=2113)

    # Setting up rotor analysis inputs
    rotor_inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=mesh_vel,
        mesh_parameters=rotor_discretization,
        atmos_states=cruise.quantities.atmos_states,
        ac_states=cruise.quantities.ac_states,
        theta_0=np.deg2rad(8.16), # Collective
        theta_1_c=np.deg2rad(1.52), # Cyclic cosine
        theta_1_s=np.deg2rad(-4.13), # Cyclic sine
        xi_0=np.deg2rad(-0.9), # lag
    )

    # 3D airfoil model 
    airfoil_model_maker = ThreeDAirfoilMLModelMaker(
        airfoil_name="naca_0012",
        aoa_range=np.linspace(-12, 16, 50),
        reynolds_range=[2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 4e6],
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6]
    )
 
    airfoil_model = airfoil_model_maker.get_airfoil_model(
        quantities=["Cl", "Cd"],
    )
    
    # Peters--He dynamic inflow model
    peters_he_model = PetersHeModel(
        num_nodes=1, 
        airfoil_model=airfoil_model,
        tip_loss=True,
        Q=5, # highest power of r/R
        M=5, # highest harmonic number
    )
    rotor_aero_outputs = peters_he_model.evaluate(rotor_inputs)
    
    # Constrain thrust coefficient
    C_T = rotor_aero_outputs.thrust_coefficient
    C_T.name = "thrust_coefficient"
    C_T.set_as_constraint(equals=0.0064, scaler=1/0.0064)
    
    total_torque = rotor_aero_outputs.total_torque

    # Setting up the acoustic observer location
    radius = 80
    angle = np.deg2rad(45)
    broadband_acoustics = Acoustics(aircraft_position=np.array([radius * np.sin(angle) ,0., -radius * np.cos(angle)]))
    broadband_acoustics.add_observer('obs', np.array([0., 0., 0.,]), time_vector=np.array([0.]))
    observer_data = broadband_acoustics.assemble_observers()

    # Broadband noise (Gill--Lee model)
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

    # Tonal noise model (Lowson)
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

    # Total noise model
    total_spl = total_noise_model(SPL_list=[lowson_spl_A_weighted, gl_spl_A_weighted])
    total_spl.name = "hover_total_noise"


    # Weighted objective between aerodynamic efficiency (torque) and noise
    objective = 0.5 * total_spl + (1 - 0.5) * total_torque
    objective.name = "weighted_objective"
    objective.set_as_objective(scaler=1/50)

define_base_configuration(caddee=caddee)

define_conditions(caddee=caddee)

define_analysis(caddee=caddee)

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder, gpu=False, derivatives_kwargs= {
        "concatenate_ofs" : True
    }
)

problem = CSDLAlphaProblem(problem_name='full_optimization_iteration_1', simulator=jax_sim)
optimizer = SLSQP(problem=problem)

optimizer.solve()
optimizer.print_results()

recorder.execute()

# Print design variables, constraints and objectives after optimization
for dv in recorder.design_variables.keys():
    print(dv.name, dv.value)

for c in recorder.constraints.keys():
    print(c.name, c.value)

for o in recorder.objectives.keys():
    print(o.name, o.value)

