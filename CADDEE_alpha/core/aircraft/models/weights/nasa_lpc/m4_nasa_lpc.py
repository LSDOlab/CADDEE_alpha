import pickle
from pathlib import Path
from CADDEE_alpha.utils.var_groups import MassProperties
import csdl_alpha as csdl
from typing import Union
import numpy as np

# _REPO_ROOT_FOLDER = Path(__file__).parents[0]

# with open(_REPO_ROOT_FOLDER / "regression_parameters_empennage.pickle", "rb") as handle:
#     reg_dict = pickle.load(handle)

# print(reg_dict)
# exit()

booms_left_inner = {
    'BoomInPort_struct_cg_X': [ 1.54766201e-02, -2.51307016e-02, 2.54893639e-01, 5.95809033e-07, 9.06333588e-05, 1.192134517922292],
    'BoomInPort_struct_cg_Y': [-6.54867899e-02, -1.04372294e-01,  3.33676357e-04,  2.51533678e-06, 5.32777466e-06, -0.004467713325373435],
    'BoomInPort_struct_cg_Z': [-4.25994565e-03, -3.50829424e-04,  4.38885378e-02, -2.12479091e-07, -4.10232551e-06, 2.094509667974993],
    'BoomInPort_struct_Ixx_local': [ 2.00307540e-02,  3.28455906e-02, -9.10844806e-04, -7.58407666e-07, 3.36158412e-05, -0.23310214145473873],
    'BoomInPort_struct_Iyy_local' : [ 4.03821027e+00,  6.70386373e+00, -2.30515994e-01, -1.11048525e-03, 8.54351064e-03, -108.5033896107725],
    'BoomInPort_struct_Izz_local': [ 4.11060799e+00,  6.72815462e+00, -8.88410917e-02, -1.13160542e-03, 5.11464168e-03, -111.22144622176114],
}

booms_left_outer = {
    'BoomOutPort_struct_cg_X': [1.71740520e-02, -2.64682836e-02,  2.55016900e-01, -2.20001873e-06, 3.76109198e-05, 1.2448800989812634],
    'BoomOutPort_struct_cg_Y': [-1.46141876e-01, -2.36064399e-01,  1.38218292e-03,  2.28332783e-07, -1.91351060e-04, 0.009347243817250828],
    'BoomOutPort_struct_cg_Z': [-6.73398944e-03, -5.60040037e-03,  4.39705011e-02, -2.08455188e-07, 1.07404833e-06, 2.118779892551238],
    'BoomOutPort_struct_Ixx_local': [ 1.95996708e-02,  3.19524773e-02, -5.09250983e-04,  2.38451956e-08, 5.44519956e-05, -0.22753600490712644],
    'BoomOutPort_struct_Iyy_local': [ 4.05773296e+00,  6.58058369e+00, -1.45798024e-01,  2.05256744e-04, 2.42420393e-02, -110.5773080356552],
    'BoomOutPort_struct_Izz_local': [ 3.98741571e+00,  6.60197432e+00, -1.10279278e-01,  1.48800712e-03, 3.46399450e-02, -111.85288355825745]
}

booms_right_inner = {
    'BoomInStar_struct_cg_X' : [ 1.55778075e-02, -2.51876469e-02,  2.55285631e-01, -1.81606610e-06, 4.18790037e-05, 1.1937198128350257],
    'BoomInStar_struct_cg_Y': [ 6.51911677e-02,  1.05037697e-01,  2.33781220e-04, -2.35564742e-06, 5.52120104e-06, -0.004195430257126986] ,
    'BoomInStar_struct_cg_Z': [-4.26486454e-03, -3.46887349e-04,  4.39100885e-02, -3.81544510e-09, -5.16555074e-06, 2.094163322618808],
    'BoomInStar_struct_Ixx_local': [ 2.00568872e-02,  3.47702450e-02, -6.53516757e-04, -5.49041083e-08, 9.41543277e-06, -0.2597957633687349],
    'BoomInStar_struct_Iyy_local': [ 4.13199296e+00,  6.42457619e+00, -7.96728116e-02,  7.14952065e-04, 2.93436436e-02, -111.636159920105],
    'BoomInStar_struct_Izz_local': [ 4.13020593e+00,  6.56518882e+00, -1.59321588e-01, -8.72214063e-04, 2.01824536e-02, -110.11891067917249],
}

booms_right_outer = {
    'BoomOutStar_struct_cg_X': [ 1.72288347e-02, -2.67051196e-02,  2.54852116e-01,  9.89334401e-08, 6.06107340e-05, 1.2439033549932343],
    'BoomOutStar_struct_cg_Y': [ 1.45026108e-01,  2.38143740e-01, -3.96379307e-03, -3.78568586e-06, -1.43419836e-04, 0.034243949994238854],
    'BoomOutStar_struct_cg_Z': [-6.73517998e-03, -5.58305290e-03,  4.39857737e-02, -3.39119130e-08, -3.91117429e-06, 2.118557350528714],
    'BoomOutStar_struct_Ixx_local': [ 1.93522526e-02,  3.13365694e-02, -7.44741716e-04,  3.48710373e-06, 4.49451312e-05, -0.21811004207456708],
    'BoomOutStar_struct_Iyy_local': [ 4.14104096e+00,  6.61201524e+00, -2.84299032e-01, -9.62709519e-04, 4.99756052e-03, -108.78895143861884],
    'BoomOutStar_struct_Izz_local': [ 4.07718041e+00,  6.62642379e+00, -9.00930186e-02, -1.38145329e-04,1.65079702e-02, -111.1214387032658],
}

boom_mass_coeffs = [ 3.33883108e+00,  5.40016118e+00, -9.17180256e-02,  1.28646937e-04, -6.25697773e-03, -1.17008833e+01]

booms_reg = [booms_left_inner, booms_left_outer, booms_right_inner, booms_right_outer]

fuselage_reg = {
    'fuselage_mass': [ 1.01472161e+00, -4.06758251e-01,  4.25974124e+01,  3.10575276e-02, 6.87345416e-02, -1.16727769e+02],
    'fuselage_struct_cg_X': [ 1.67449602e-02, -1.23207371e-02,  4.90804477e-01, -2.85732761e-04, 1.29581238e-03, -0.6211621578156898],
    'fuselage_struct_cg_Z': [ 2.04733876e-03,  9.80617344e-06,  5.74947432e-02, -6.70040225e-07, -1.56022401e-05, 1.1209786221663782],
    'fuselage_struct_Ixx_local': [ 2.25127273e+00, -1.85707470e+00,  5.11677033e+01, -3.62084046e-02, -4.91167934e-02, -201.94348875650056],
    'fuselage_struct_Iyy_local': [-4.85003613e+00,  2.03987586e+00,  2.89049283e+02,  9.51160015e-03, -4.42001564e+00, -1032.685663411943],
    'fuselage_struct_Izz_local': [-1.09272490e+01, -7.99435250e+00,  2.62136153e+02, -1.30566173e-01, -3.74401378e+00, -444.6749130358837],
    'fuselage_struct_Ixz_local': [-1.60504311e+00, -6.56730156e-01,  3.38025631e+01, -3.04377534e-03, 1.66851198e-01, -123.8341477054152],
}

wing_reg = {
    'wing_mass': [1.11379136e+01,  3.14761829e+01,  7.89132288e-01, -2.14257921e-02, 2.40041303e-01, -3.20236992e+02],
    'wing_struct_cg_X': [ 9.94396743e-03, -3.05611792e-02,  2.56921181e-01, -7.39025396e-06,-1.03626829e-04, 1.4057829024098356],
    'wing_struct_cg_Z': [-9.55284281e-04,  3.96588505e-04,  4.31125644e-02, -4.73824146e-06, -1.07005070e-04, 2.156881222891959],
    'wing_struct_Ixx_local': [2.76071395e+02, 3.59794413e+02, 9.66210889e+01, 1.64281071e-01, 8.94049058e+00, -7780.772684440635],
    'wing_struct_Iyy_local': [ 4.70146714e+00,  9.12620835e-01,  1.17140499e+00,  6.30138605e-03, -3.29329535e-03, -71.05960014473001],
    'wing_struct_Izz_local': [2.83046432e+02, 3.61938062e+02, 7.56255523e+01, 1.59788925e-01, 2.65561862e+00, -7353.906535279631],
    'wing_struct_Ixz_local': [-3.01635151e-01, -1.19201026e-01, -6.01160477e-02, -3.71504542e-05, -3.12613363e-03, 5.2272209068023425]
}

h_tail_reg = {
    'htail_struct_cg_X': [7.06883576e-02, 3.84929919e-04, 8.61381597e+00], 
    'htail_struct_cg_Z': [-1.16697306e-03, -2.91660614e-04,  2.43639119e+00], 
    'htail_struct_Ixx_local': [ 18.92005127,  -0.35697627, -33.45142861], 
    'htail_struct_Iyy_local': [ 1.19654233, -0.00797203, -2.17981443], 
    'htail_struct_Izz_local': [ 20.54893615,  -0.20019707, -37.82299028], 
    'htail_struct_Iyz_local': [ 0.00561562,  0.01458515, -0.05720038]
}

v_tail_reg = {
    'vtail_struct_cg_X': [1.22303656e-03, 4.29673559e-01, 7.44086060e+00], 
    'vtail_struct_cg_Z': [3.14155013e-05, 1.15201331e-01, 2.80560953e+00], 
    'vtail_struct_Ixx_local': [ 0.01142087,  5.03615053, -6.3960446 ], 
    'vtail_struct_Iyy_local': [ -0.02605902,  16.15112795, -20.49196436], 
    'vtail_struct_Izz_local': [ -0.02455155,  11.32820227, -14.38350579], 
    'vtail_struct_Ixz_local': [-0.01761771,  4.91034131, -6.19919581], 
}

empennage_reg = [h_tail_reg, v_tail_reg]

empennage_mass_coeff = [8.43266623, 10.05410839, -0.19944806479469435]

def compute_boom_mps(
    wing_area: Union[csdl.Variable, float, int],
    wing_AR: Union[csdl.Variable, float, int],
    fuselage_length: Union[csdl.Variable, float, int],
    battery_mass: Union[csdl.Variable, float, int],
    cruise_speed: Union[csdl.Variable, float, int],
    reference_frame: str = "flight_dynamics"
) -> MassProperties:
    """Compute the mass properties of the booms of NASA's 
    lift-plus-cruise air taxi.

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    reference_frame : str, optional
        the reference w.r.t. which the mass properties are computed;
        options are "geometric" (x back, y right z up) and "flight_dynamics"
        (x forward, y right, z down), by default "flight_dynamics"

    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """
    csdl.check_parameter(reference_frame, "reference_frame",
                         values=("flight_dynamics", "geometric"))
    
    cg_list = []
    it_list = []
    for boom_reg in booms_reg:
        cg_vec = csdl.Variable(shape=(3, ), value=0.)
        i_mat = csdl.Variable(shape=(3, 3), value=0.)
        for name, coeffs in boom_reg.items():
            qty = evaluate_regression(
                wing_area, wing_AR, fuselage_length,
                battery_mass, cruise_speed, coeffs
            )
            if 'cg_X' in name:
                cg_vec =  cg_vec.set(csdl.slice[0], qty)
            elif 'cg_Y' in name:
                cg_vec =  cg_vec.set(csdl.slice[1], qty)
            elif 'cg_Z' in name:
                cg_vec =  cg_vec.set(csdl.slice[2], qty)
            elif 'Ixx' in name:
                i_mat = i_mat.set(csdl.slice[0, 0], qty)
            elif 'Iyy' in name:
                i_mat = i_mat.set(csdl.slice[1, 1], qty)
            elif 'Izz' in name:
                i_mat = i_mat.set(csdl.slice[2, 2], qty)
        cg_list.append(cg_vec)
        it_list.append(i_mat)

    mass_coeffs = boom_mass_coeffs
    total_boom_mass = evaluate_regression(
        wing_area, wing_AR, fuselage_length,
        battery_mass, cruise_speed, mass_coeffs,
    )

    mass_per_boom_pair = total_boom_mass / 4 # left/right + inner/outer
    total_boom_cg = csdl.Variable(shape=(3, ), value=0.)

    # compute total boom cg
    for cg in cg_list:
        total_boom_cg = total_boom_cg + cg * mass_per_boom_pair

    total_boom_cg = total_boom_cg / total_boom_mass

    # zero out cg-y and flip x,z depending on reference frame
    if reference_frame == "flight_dynamics":
        total_boom_cg = total_boom_cg * np.array([-1, 0, -1])

    else:
        total_boom_cg = total_boom_cg * np.array([1, 0, 1])

    # compute total boom inertia tensor (about total cg)
    total_boom_I = csdl.Variable(shape=(3, 3), value=0.)
    
    # parallel axis theorem (parallel axis is total boom cg)
    x =  total_boom_cg[0]
    y =  total_boom_cg[1]
    z =  total_boom_cg[2]
    
    transl_mat = csdl.Variable(shape=(3, 3), value=0.)
    transl_mat = transl_mat.set(csdl.slice[0, 0], y**2 + z**2)
    transl_mat = transl_mat.set(csdl.slice[0, 1], -x * y)
    transl_mat = transl_mat.set(csdl.slice[0, 2], -x * z)
    transl_mat = transl_mat.set(csdl.slice[1, 0], -y * x)
    transl_mat = transl_mat.set(csdl.slice[1, 1], x**2 + z**2)
    transl_mat = transl_mat.set(csdl.slice[1, 2], -y * z)
    transl_mat = transl_mat.set(csdl.slice[2, 0], -z * x)
    transl_mat = transl_mat.set(csdl.slice[2, 1], -z * y)
    transl_mat = transl_mat.set(csdl.slice[2, 2], x**2 + y**2)
    transl_mat = mass_per_boom_pair * transl_mat

    for it in it_list:
        it_boom_cg = it + transl_mat
        total_boom_I = total_boom_I + it_boom_cg

    # assemble boom mps data class
    boom_mps = MassProperties(
        mass=total_boom_mass,
        cg_vector=total_boom_cg,
        inertia_tensor=total_boom_I,
    )

    return boom_mps

def compute_empennage_mps(
    h_tail_area: Union[csdl.Variable, float, int],
    v_tail_area: Union[csdl.Variable, float, int],
    reference_frame: str = "flight_dynamics"
):
    csdl.check_parameter(reference_frame, "reference_frame",
                         values=("flight_dynamics", "geometric"))
    cg_list = []
    it_list = []
    for reg in empennage_reg:
        cg_vec = csdl.Variable(shape=(3, ), value=0.)
        i_mat = csdl.Variable(shape=(3, 3), value=0.)
        for name, coeffs in reg.items():
            qty = evaluate_empennage_regression(
                h_tail_area, v_tail_area, coeffs
            )
            if 'cg_X' in name:
                cg_vec =  cg_vec.set(csdl.slice[0], qty)
            elif 'cg_Y' in name:
                cg_vec =  cg_vec.set(csdl.slice[1], qty)
            elif 'cg_Z' in name:
                cg_vec =  cg_vec.set(csdl.slice[2], qty)
            elif 'Ixx' in name:
                i_mat = i_mat.set(csdl.slice[0, 0], qty)
            elif 'Iyy' in name:
                i_mat = i_mat.set(csdl.slice[1, 1], qty)
            elif 'Izz' in name:
                i_mat = i_mat.set(csdl.slice[2, 2], qty)
            elif 'Ixz' in name:
                i_mat = i_mat.set(csdl.slice[0, 2], qty)
                i_mat = i_mat.set(csdl.slice[2, 0], qty)
            elif 'Iyz' in name:
                i_mat = i_mat.set(csdl.slice[1, 2], qty)
                i_mat = i_mat.set(csdl.slice[2, 1], qty)

        cg_list.append(cg_vec)
        it_list.append(i_mat)

    mass_coeffs = empennage_mass_coeff
    total_empennage_mass = evaluate_empennage_regression(
        htail_area=h_tail_area, vtail_area=v_tail_area, coeffs=mass_coeffs
    )

    total_empennage_cg = csdl.Variable(shape=(3, ), value=0.)

    # compute total boom cg
    for i, cg in enumerate(cg_list):
        if i == 0:
            # weigh h tail a more
            total_empennage_cg = total_empennage_cg + cg * 0.65 * total_empennage_mass
        else:
            # weigh v tail a less
            total_empennage_cg = total_empennage_cg + cg * 0.35 * total_empennage_mass

    total_empennage_cg = total_empennage_cg / total_empennage_mass

    # zero out cg-y and flip x,z depending on reference frame
    if reference_frame == "flight_dynamics":
        total_empennage_cg = total_empennage_cg * np.array([-1, 0, -1])

    else:
        total_empennage_cg = total_empennage_cg * np.array([1, 0, 1])

    # compute total empennage inertia tensor (about total cg)
    total_empennage_I = csdl.Variable(shape=(3, 3), value=0.)
    
    # parallel axis theorem (parallel axis is total empennage cg)
    x =  total_empennage_cg[0]
    y =  total_empennage_cg[1]
    z =  total_empennage_cg[2]
    
    transl_mat = csdl.Variable(shape=(3, 3), value=0.)
    transl_mat = transl_mat.set(csdl.slice[0, 0], y**2 + z**2)
    transl_mat = transl_mat.set(csdl.slice[0, 1], -x * y)
    transl_mat = transl_mat.set(csdl.slice[0, 2], -x * z)
    transl_mat = transl_mat.set(csdl.slice[1, 0], -y * x)
    transl_mat = transl_mat.set(csdl.slice[1, 1], x**2 + z**2)
    transl_mat = transl_mat.set(csdl.slice[1, 2], -y * z)
    transl_mat = transl_mat.set(csdl.slice[2, 0], -z * x)
    transl_mat = transl_mat.set(csdl.slice[2, 1], -z * y)
    transl_mat = transl_mat.set(csdl.slice[2, 2], x**2 + y**2)
    transl_mat = (total_empennage_mass / 2) * transl_mat

    for it in it_list:
        it_empennage_cg = it + transl_mat
        total_empennage_I = total_empennage_I + it_empennage_cg

    empennage_mps = MassProperties(
        mass=total_empennage_mass,
        cg_vector=total_empennage_cg,
        inertia_tensor=total_empennage_I,
    )

    return empennage_mps

def compute_wing_mps(
    wing_area: Union[csdl.Variable, float, int],
    wing_AR: Union[csdl.Variable, float, int],
    fuselage_length: Union[csdl.Variable, float, int],
    battery_mass: Union[csdl.Variable, float, int],
    cruise_speed: Union[csdl.Variable, float, int],
    reference_frame: str = "flight_dynamics"
) -> MassProperties:
    """Compute the mass properties of the wing of NASA's 
    lift-plus-cruise air taxi.

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    reference_frame : str, optional
        the reference w.r.t. which the mass properties are computed;
        options are "geometric" (x back, y right z up) and "flight_dynamics"
        (x forward, y right, z down), by default "flight_dynamics"

    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """
    csdl.check_parameter(wing_area, "wing_area", types=(csdl.Variable, float, int))
    csdl.check_parameter(wing_AR, "wing_AR", types=(csdl.Variable, float, int))
    csdl.check_parameter(fuselage_length, "fuselage_length", types=(csdl.Variable, float, int))
    csdl.check_parameter(battery_mass, "battery_mass", types=(csdl.Variable, float, int))
    csdl.check_parameter(cruise_speed, "cruise_speed", types=(csdl.Variable, float, int))

    if not isinstance(wing_area, csdl.Variable):
        wing_area = csdl.Variable(shape=(1, ), value=wing_area)

    if not isinstance(wing_AR, csdl.Variable):
        wing_AR = csdl.Variable(shape=(1, ), value=wing_AR)

    if not isinstance(fuselage_length, csdl.Variable):
        fuselage_length = csdl.Variable(shape=(1, ), value=fuselage_length)

    if not isinstance(battery_mass, csdl.Variable):
        battery_mass = csdl.Variable(shape=(1, ), value=battery_mass)

    if not isinstance(cruise_speed, csdl.Variable):
        cruise_speed = csdl.Variable(shape=(1, ), value=cruise_speed)

    cg_vec = csdl.Variable(shape=(3, ), value=0.)
    i_mat = csdl.Variable(shape=(3, 3), value=0.)

    for name, coeffs in wing_reg.items():
        qty = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, coeffs
        )
        if "mass" in name:
            m = qty
        elif 'cg_X' in name:
            cg_vec =  cg_vec.set(csdl.slice[0], qty)
        elif 'cg_Z' in name:
            cg_vec =  cg_vec.set(csdl.slice[2], qty)
        elif 'Ixx' in name:
            i_mat = i_mat.set(csdl.slice[0, 0], qty)
        elif 'Iyy' in name:
            i_mat = i_mat.set(csdl.slice[1, 1], qty)
        elif 'Izz' in name:
            i_mat = i_mat.set(csdl.slice[2, 2], qty)
        elif 'Ixz' in name:
            i_mat = i_mat.set(csdl.slice[0, 2], qty)
            i_mat = i_mat.set(csdl.slice[2, 0], qty)

    if reference_frame == "flight_dynamics":
        cg_vec = cg_vec * np.array([-0.9, 0, -1])

    wing_mps = MassProperties(
        mass=m, cg_vector=cg_vec, inertia_tensor=i_mat
    )

    return wing_mps

def compute_fuselage_mps(
    wing_area: Union[csdl.Variable, float, int],
    wing_AR: Union[csdl.Variable, float, int],
    fuselage_length: Union[csdl.Variable, float, int],
    battery_mass: Union[csdl.Variable, float, int],
    cruise_speed: Union[csdl.Variable, float, int],
    reference_frame: str = "flight_dynamics"
) -> MassProperties:
    """Compute the mass properties of the fuselage of NASA's 
    lift-plus-cruise air taxi.

    Parameters
    ----------
    wing_area : Union[csdl.Variable, float, int]
        _description_
    wing_AR : Union[csdl.Variable, float, int]
        _description_
    fuselage_length : Union[csdl.Variable, float, int]
        _description_
    battery_mass : Union[csdl.Variable, float, int]
        _description_
    cruise_speed : Union[csdl.Variable, float, int]
        _description_
    reference_frame : str, optional
        the reference w.r.t. which the mass properties are computed;
        options are "geometric" (x back, y right z up) and "flight_dynamics"
        (x forward, y right, z down), by default "flight_dynamics"

    Returns
    -------
    MassProperties
        instance of MassProperties data class
    """

    cg_vec = csdl.Variable(shape=(3, ), value=0.)
    i_mat = csdl.Variable(shape=(3, 3), value=0.)

    for name, coeffs in fuselage_reg.items():
        qty = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, coeffs
        )
        if "mass" in name:
            m = qty
        elif 'cg_X' in name:
            cg_vec =  cg_vec.set(csdl.slice[0], qty)
        elif 'cg_Z' in name:
            cg_vec =  cg_vec.set(csdl.slice[2], qty)
        elif 'Ixx' in name:
            i_mat = i_mat.set(csdl.slice[0, 0], qty)
        elif 'Iyy' in name:
            i_mat = i_mat.set(csdl.slice[1, 1], qty)
        elif 'Izz' in name:
            i_mat = i_mat.set(csdl.slice[2, 2], qty)
        elif 'Ixz' in name:
            i_mat = i_mat.set(csdl.slice[0, 2], qty)
            i_mat = i_mat.set(csdl.slice[2, 0], qty)

    if reference_frame == "flight_dynamics":
        cg_vec = cg_vec * np.array([-0.88, 0, -1])

    wing_mps = MassProperties(
        mass=m, cg_vector=cg_vec, inertia_tensor=i_mat
    )

    return wing_mps

__all__ = [compute_boom_mps, compute_empennage_mps, compute_fuselage_mps, compute_wing_mps]

def evaluate_regression(wing_area, wing_AR, fuselage_length, battery_mass, cruise_speed, coeffs):
    qty = coeffs[0] * wing_area + coeffs[1] * wing_AR + coeffs[2] * fuselage_length \
          + coeffs[3] * battery_mass + coeffs[4] * cruise_speed + coeffs[5]
    
    return qty

def evaluate_empennage_regression(htail_area, vtail_area, coeffs):
    qty = coeffs[0] * htail_area + coeffs[1] * vtail_area + coeffs[2]
    
    return qty

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    wing_AR = 12
    wing_area = 19.5
    fuselage_length = 9
    battery_mass = 800
    cruise_speed = 67

    boom_mps = compute_boom_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass,
        cruise_speed=cruise_speed,
    )

    print("Boom MPs")
    print(boom_mps.mass)
    print(boom_mps.cg_vector.value)
    print(boom_mps.inertia_tensor.value)
    print("\n")

    wing_mps = compute_wing_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass,
        cruise_speed=cruise_speed,
    )

    print("wing MPs")
    print(wing_mps.mass)
    print(wing_mps.cg_vector.value)
    print(wing_mps.inertia_tensor.value)
    print("\n")

    fuselage_mps = compute_fuselage_mps(
        wing_area=wing_area,
        wing_AR=wing_AR,
        fuselage_length=fuselage_length,
        battery_mass=battery_mass,
        cruise_speed=cruise_speed,
    )

    print("fuselage MPs")
    print(fuselage_mps.mass)
    print(fuselage_mps.cg_vector.value)
    print(fuselage_mps.inertia_tensor.value)
    print("\n")

    empennage_mps = compute_empennage_mps(
        h_tail_area=3.5,
        v_tail_area=2.5,
    )

    print("empennage MPs")
    print(empennage_mps.mass)
    print(empennage_mps.cg_vector.value)
    print(empennage_mps.inertia_tensor.value)
    print("\n")

