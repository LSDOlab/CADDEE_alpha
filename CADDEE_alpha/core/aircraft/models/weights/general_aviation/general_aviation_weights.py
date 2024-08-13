import numpy as np 
import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
from CADDEE_alpha.utils.units import Units



class GeneralAviationWeights:
    def __init__(
        self,
        design_gross_weight : Union[csdl.Variable, float, int],
        dynamic_pressure : Union[csdl.Variable, float, int], # = rho * V^2 = [mass / length^3 * length^2/time^2] = [mass / length / time^2]
    ) -> None:
        self.design_gross_weight = design_gross_weight * (1/Units.mass.pound_to_kg)
        self.dynamic_pressure = dynamic_pressure * (1/Units.mass.pound_to_kg) / (1/Units.length.foot_to_m)

    def evaluate_wing_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        fuel_weight : Union[float, int, csdl.Variable],
        AR : Union[float, int, csdl.Variable],
        sweep : Union[float, int, csdl.Variable],
        taper_ratio : Union[float, int, csdl.Variable],
        thickness_to_chord : Union[float, int, csdl.Variable]=0.12,
        nz : Union[float, int, csdl.Variable] = 3.75,
        correction_factor : Union[float, int] = 1
    ):
        S_ref = S_ref * 1 / Units.area.sq_ft_to_sq_m
        W_fuel = fuel_weight * 1 / Units.mass.pound_to_kg

        W_wing = 0.036 * S_ref**0.758 * W_fuel**0.035 * (AR/np.cos(sweep)**2)**0.6 * self.dynamic_pressure**0.006 \
        * taper_ratio**0.04 * (100 * thickness_to_chord / np.cos(sweep))**-0.3 * (nz * self.design_gross_weight)**0.49 

        return W_wing * correction_factor * Units.mass.pound_to_kg
    
    def evaluate_fuselage_weight(
        self,
        S_wet : Union[float, int, csdl.Variable],
        ulf : Union[float, int, csdl.Variable] = 3.75,
        fuselage_length : Union[float, int, csdl.Variable] = None,
        avergae_fuselage_diameter : Union[float, int, csdl.Variable] = None,
        correction_factor : Union[float, int, csdl.Variable] = 1.,
    ):
        
        if S_wet is not None:
            S_wet = S_wet / Units.area.sq_ft_to_sq_m
            pass

        elif fuselage_length is not None and avergae_fuselage_diameter is not None:
            xl = fuselage_length * (1 / Units.length.foot_to_m)
            d_av = avergae_fuselage_diameter * (1 / Units.length.foot_to_m)

            S_wet = 3.14159 * (xl / d_av - 1.7) * d_av**2

        else:
            raise Exception("Insufficient inputs defined")

        W_fuse = 0.052 * S_wet**1.086 * (ulf * self.design_gross_weight) ** 0.177 * self.dynamic_pressure**0.241 

        return W_fuse * correction_factor * Units.mass.pound_to_kg
    
    def evaluate_horizontal_tail_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        ulf : Union[float, int, csdl.Variable] = 4.5,
        correction_factor : Union[float, int, csdl.Variable] = 1.,
    ):
        S_ref = S_ref / Units.area.sq_ft_to_sq_m

        W_h_tail = 0.016 * S_ref**0.873 * (ulf * self.design_gross_weight)**0.414 * self.dynamic_pressure**0.122

        return W_h_tail * Units.mass.pound_to_kg * correction_factor
    
    def evaluate_vertical_tail_weight(
        self,
        S_ref : Union[float, int, csdl.Variable],
        AR : Union[float, int, csdl.Variable],
        thickness_to_chord : Union[float, int, csdl.Variable],
        sweep_c4 : Union[float, int, csdl.Variable],
        ulf : Union[float, int, csdl.Variable] = 4.5,
        hht : Union[float, int, csdl.Variable] = 0.,
        correction_factor : Union[float, int, csdl.Variable] = 1.,
    ):
        S_ref = S_ref / Units.area.sq_ft_to_sq_m
        
        W_v_tail = 0.073 * (1.0 + 0.2 * hht) * (ulf * self.design_gross_weight)**0.376 * self.dynamic_pressure**0.122 * \
            S_ref**0.873 * (AR / np.cos(sweep_c4)**2)**0.357 / ((100 * thickness_to_chord) / np.cos(sweep_c4))**0.49
        
        return W_v_tail * Units.mass.pound_to_kg * correction_factor
    
    def evaluate_avionics_weight(
        self,
        design_range : Union[float, int, csdl.Variable],
        num_flight_crew : Union[int, csdl.Variable],
        fuselage_plan_form_area : Union[float, int, csdl.Variable, None],
        fuselage_length : Union[float, int, csdl.Variable] = None,
        fuselage_width : Union[float, int, csdl.Variable] = None,
        correction_factor : Union[float, csdl.Variable] = 1.,
    ):
        design_range = design_range / Units.length.nautical_mile_to_m

        if fuselage_plan_form_area is not None:
            fuselage_plan_form_area = fuselage_plan_form_area / Units.area.sq_ft_to_sq_m

        elif fuselage_length is not None and fuselage_width is not None:
            fuselage_plan_form_area = fuselage_width / Units.length.foot_to_m * fuselage_length / Units.length.foot_to_m

        W_avionics = 15.8 * design_range**0.1 * num_flight_crew**0.7 * fuselage_plan_form_area**0.43

        return W_avionics * Units.mass.pound_to_kg * correction_factor
    
    def evaluate_main_landing_gear_weight(
        self,
        fuselage_length : Union[float, int, csdl.Variable],
        design_range : Union[float, int, csdl.Variable],
        W_ramp : Union[float, int, csdl.Variable],
        correction_factor : Union[float, int] = 1.
    ):
        design_range = design_range / Units.length.nautical_mile_to_m
        W_ramp = W_ramp / Units.mass.pound_to_kg
        fuselage_length = fuselage_length / Units.length.foot_to_m

        wldg = W_ramp * (1 - 0.00004 * design_range)
        xmlg = 0.75 * fuselage_length * 12 

        w_lg = 0.0117 * wldg**0.95 * xmlg**0.43

        return w_lg * Units.mass.pound_to_kg * correction_factor
        


@dataclass
class GAInstrumentsWeightInputs:
    """Parameters (in English units!)"""
    mach_max : Union[float, int, csdl.Variable]
    num_flight_crew : Union[float, int, csdl.Variable]
    num_wing_mounted_engines : Union[float, int, csdl.Variable]
    num_fuselage_mounted_engines : Union[float, int, csdl.Variable]
    S_fuse_planform : Union[float, int, csdl.Variable, None] 
    
    xl : Union[float, int, csdl.Variable] = None
    xlw : Union[float, int, csdl.Variable] = None

class GAInstrumentsWeightModel():
    def evaluate(self, inputs: GAInstrumentsWeightInputs):
        mach_max = inputs.mach_max
        nfc = inputs.num_flight_crew
        nwme = inputs.num_wing_mounted_engines
        nfmw = inputs.num_fuselage_mounted_engines
        S_fuse_planform = inputs.S_fuse_planform
        xl = inputs.xl
        xlw = inputs.xlw

        if xl is not None and xlw is not None:
            S_fuse_planform = xlw * xl

        W_instruments = 0.48 * S_fuse_planform**0.57 * mach_max**0.5* (10 + 2.5 *nfc + nwme  + 1.5 * nfmw) 

        return W_instruments



if __name__ == "__main__":
    from scipy.optimize import newton, root_scalar
    # Example usage:
    W_wing_est = 200  # Initial estimate for wing weight
    W_htail_est = 60
    W_vtail_est = 60
    W_fuse_est = 600   # Initial estimate for fuselage weight
    W_mlg_est = 50
    W_avionics_est = 50
    W_instruments_est = 50

    # Assuming you have these parameters defined
    S_ref = 174
    S_ref_tail = 22
    S_ref_v_tail = 11.3
    W_fuel = 240
    AR = 7.32
    AR_vtail = 2
    sweep_c4 = np.deg2rad(0)
    taper_ratio = 0.75
    thickness_to_chord = 0.12
    dynamic_pressure = 1481.35 * 0.9 * 0.19**2
    nz = 4.5
    S_wet = 200
    q_cruise = 1481.35 * 0.9 * 0.19**2
    

    ulf = 3.75
    xl = 28
    l_fuse = 28
    d_av = 4

    design_range = 600


