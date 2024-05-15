import pickle
from pathlib import Path
from CADDEE_alpha.utils.var_groups import MassProperties
import csdl_alpha as csdl


_REPO_ROOT_FOLDER = Path(__file__).parents[0]

with open(_REPO_ROOT_FOLDER / "regression_parameters.pickle", "rb") as handle:
    reg_dict = pickle.load(handle)

print(reg_dict.keys())


booms_left_inner = [
    'BoomInPort_struct_cg_X',
    'BoomInPort_struct_cg_Y',
    'BoomInPort_struct_cg_Z',
    'BoomInPort_struct_Ixx_local',
    'BoomInPort_struct_Iyy_local',
    'BoomInPort_struct_Izz_local',
]

booms_left_outer = [
    'BoomOutPort_struct_cg_X', 
    'BoomOutPort_struct_cg_Y', 
    'BoomOutPort_struct_cg_Z', 
    'BoomOutPort_struct_Ixx_local', 
    'BoomOutPort_struct_Iyy_local', 
    'BoomOutPort_struct_Izz_local',
]

booms_right_inner = [
    'BoomInStar_struct_cg_X', 
    'BoomInStar_struct_cg_Y', 
    'BoomInStar_struct_cg_Z', 
    'BoomInStar_struct_Ixx_local', 
    'BoomInStar_struct_Iyy_local', 
    'BoomInStar_struct_Izz_local',
]

booms_right_outer = [
    'BoomOutStar_struct_cg_X', 
    'BoomOutStar_struct_cg_Y', 
    'BoomOutStar_struct_cg_Z', 
    'BoomOutStar_struct_Ixx_local', 
    'BoomOutStar_struct_Iyy_local', 
    'BoomOutStar_struct_Izz_local',
]

fuselage = [
    'fuselage_struct_cg_X', 
    'fuselage_struct_cg_Z', 
    'fuselage_struct_Ixx_local', 
    'fuselage_struct_Iyy_local', 
    'fuselage_struct_Izz_local', 
    'fuselage_struct_Ixz_local', 
]

wing = [
    'wing_struct_cg_X', 
    'wing_struct_cg_Z', 
    'wing_struct_Ixx_local', 
    'wing_struct_Iyy_local', 
    'wing_struct_Izz_local', 
    'wing_struct_Ixz_local'
]


def evaluate_regression(wing_area, wing_AR, fuselage_length, battery_mass, cruise_speed, coeffs):
    qty = coeffs[0] * wing_area + coeffs[1] * wing_AR + coeffs[2] * fuselage_length \
          + coeffs[3] * battery_mass + coeffs[4] * cruise_speed + coeffs[5]
    
    return qty

class M4RegLPCBooms: 
    def evaluate(
        self,
        wing_area,
        wing_AR,
        fuselage_length,
        battery_mass,
        cruise_speed,
    ): 
        
        # left inner
        boom_li_cgx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_cg_X"]
        )

        boom_li_cgy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_cg_Y"]
        )

        boom_li_cgz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_cg_Z"]
        )

        boom_li_cg_vec = csdl.Variable(shape=(3, ), value=0.)
        boom_li_cg_vec = boom_li_cg_vec.set(
            csdl.slice[0], boom_li_cgx
        )

        boom_li_cg_vec = boom_li_cg_vec.set(
            csdl.slice[1], boom_li_cgx
        )

        boom_li_cg_vec = boom_li_cg_vec.set(
            csdl.slice[2], boom_li_cgx
        )

        boom_li_ixx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_Ixx_local"]
        )

        boom_li_iyy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_yy_local"]
        )

        boom_li_izz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInPort_struct_Izz_local"]
        )

        # booms_li_mps = MassProperties(
        #     cg_vector=csdl.Variable()
        # )


        # left outer
        boom_lo_cgx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_cg_X"]
        )

        boom_lo_cgy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_cg_Y"]
        )

        boom_lo_cgz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_cg_Z"]
        )

        boom_lo_ixx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_Ixx_local"]
        )

        boom_lo_iyy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_yy_local"]
        )

        boom_lo_izz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutPort_struct_Izz_local"]
        )

        # right inner
        boom_ri_cgx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_cg_X"]
        )

        boom_ri_cgy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_cg_Y"]
        )

        boom_ri_cgz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_cg_Z"]
        )

        boom_ri_ixx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_Ixx_local"]
        )

        boom_ri_iyy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_yy_local"]
        )

        boom_ri_izz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomInStar_struct_Izz_local"]
        )


        # right outer
        boom_ro_cgx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_cg_X"]
        )

        boom_ro_cgy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_cg_Y"]
        )

        boom_ro_cgz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_cg_Z"]
        )

        boom_ro_ixx = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_Ixx_local"]
        )

        boom_ro_iyy = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_yy_local"]
        )

        boom_ro_izz = evaluate_regression(
            wing_area, wing_AR, fuselage_length,
            battery_mass, cruise_speed, 
            reg_dict["BoomOutStar_struct_Izz_local"]
        )



class M4RegLPCWing: pass

class M4RegLPCFuselage: pass
