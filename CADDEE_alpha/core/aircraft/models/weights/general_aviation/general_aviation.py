import numpy as np 
import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union


@dataclass
class GAWingWeightInputs:
    S_ref : Union[float, int, csdl.Variable]
    W_fuel : Union[float, int, csdl.Variable]
    AR : Union[float, int, csdl.Variable]
    sweep_c4 : Union[float, int, csdl.Variable]
    taper_ratio : Union[float, int, csdl.Variable]
    thickness_to_chord : Union[float, int, csdl.Variable]
    dynamic_pressure : Union[float, int, csdl.Variable]
    W_gross_design : Union[float, int, csdl.Variable]
    nz : Union[float, int, csdl.Variable] = 3.75
    correction_factor : Union[float, int] = 1

class GAWingWeightModel:
    def evaluate(self, inputs :  GAWingWeightInputs):
        """Evaluate an estimate for the wing weight."""
        S_w = inputs.S_ref
        W_fw = inputs.W_fuel
        AR = inputs.AR
        sweep = inputs.sweep_c4
        taper_ratio = inputs.taper_ratio
        t_o_c = inputs.thickness_to_chord
        q = inputs.dynamic_pressure
        nz = inputs.nz
        Wdg = inputs.W_gross_design
        cf = inputs.correction_factor

        # print(S_w)
        # print(W_fw)
        # print(AR)
        # print(sweep)
        # print(taper_ratio)
        # print(q)

        W_wing = 0.036 * S_w**0.758 * W_fw**0.035 * (AR/np.cos(sweep)**2)**0.6 * q**0.006 \
        * taper_ratio**0.04 * (100 * t_o_c / np.cos(sweep))**-0.3 * (nz * Wdg)**0.49

        # print(W_wing)
        # exit()


        return W_wing * cf


@dataclass
class GAFuselageWeightInputs:
    """
    Parameters (in English units!)
    ----------
    - S_wet : wetted area

    - q_cruise : cruise dynamic pressure

    - ulf : structural ultimate load factor (default = 3.75)

    - xl : total fuselage length; used to compute S_wet (default = None)
    
    - d_av : average fuselage diameter; used to compute S_wet (default = None)

    - W_gross_design : design gross weight 
    """
    S_wet : Union[float, int, csdl.Variable]
    q_cruise : Union[float, int, csdl.Variable]
    W_gross_design : Union[float, int, csdl.Variable]
    ulf : Union[float, int, csdl.Variable] = 3.75
    xl : Union[float, int, csdl.Variable] = None
    d_av : Union[float, int, csdl.Variable] = None
    correction_factor : Union[float, int, csdl.Variable] = 1.


class GAFuselageWeigthModel():
    def evaluate(self, inputs : GAFuselageWeightInputs):
        """Evaluate an estimate for the fuselage weight."""
        S_wet = inputs.S_wet
        q_cruise = inputs.q_cruise
        ulf = inputs.ulf
        xl = inputs.xl
        d_av = inputs.d_av
        dg = inputs.W_gross_design
        correction_factor = inputs.correction_factor

        if S_wet is not None:
            pass

        elif xl is not None and d_av is not None:
            S_wet = 3.14159 * (xl / d_av - 1.7) * d_av**2

        W_fuse = 0.052 * S_wet**1.086 * (ulf * dg) ** 0.177 * q_cruise**0.241 * correction_factor

        return W_fuse


@dataclass
class GAHorizontalTailInputs:
    """Parameters (in English units!)"""
    S_ref : Union[float, int, csdl.Variable]
    W_gross_design : Union[float, int, csdl.Variable]
    q_cruise : Union[float, int, csdl.Variable]
    ulf : Union[float, int, csdl.Variable] = 4.5

class GAHorizontalTailWeigthModel():
    def evaluate(self, inputs : GAHorizontalTailInputs):
        """Evaluate an estimate for the vertical tail weight."""
        S_ref = inputs.S_ref
        q_cruise = inputs.q_cruise
        ulf = inputs.ulf
        dg = inputs.W_gross_design

        W_h_tail = 0.016 * S_ref**0.873 * (ulf * dg)**0.414 * q_cruise**0.122

        return W_h_tail

@dataclass
class GAVerticalTailInputs:
    """Parameters (in English units!)"""
    S_ref : Union[float, int, csdl.Variable]
    AR : Union[float, int, csdl.Variable]
    W_gross_design : Union[float, int, csdl.Variable]
    q_cruise : Union[float, int, csdl.Variable]
    t_o_c : Union[float, int, csdl.Variable]
    sweep_c4 : Union[float, int, csdl.Variable]
    ulf : Union[float, int, csdl.Variable] = 4.5
    hht : Union[float, int, csdl.Variable] = 0.


class GAVerticalTailWeigthModel():
    def evaluate(self, inputs : GAVerticalTailInputs):
        """Evaluate an estimate for the Vertical tail weight."""
        S_ref = inputs.S_ref
        q_cruise = inputs.q_cruise
        ulf = inputs.ulf
        dg = inputs.W_gross_design
        hht = inputs.hht
        AR = inputs.AR
        sweep = inputs.sweep_c4
        toc = inputs.t_o_c

        W_v_tail = 0.073 * (1.0 + 0.2 * hht) * (ulf * dg)**0.376 * q_cruise**0.122 * \
        S_ref**0.873 * (AR / np.cos(sweep)**2)**0.357 / ((100 * toc) / np.cos(sweep))**0.49

        return W_v_tail
    

@dataclass
class GAMainLandingGearWeightInputs:
    """Parameters (in English units!)"""
    fuselage_length : Union[float, int, csdl.Variable]
    design_range : Union[float, int, csdl.Variable]
    W_ramp : Union[float, int, csdl.Variable]
    correction_factor : Union[float, int] = 1.

class GAMainLandingGearWeightModel():
    def evaluate(self, inputs: GAMainLandingGearWeightInputs):
        fl = inputs.fuselage_length
        dr = inputs.design_range
        drw = inputs.W_ramp
        cf = inputs.correction_factor

        wldg = drw * (1 - 0.00004 * dr)
        xmlg = 0.75 * fl * 12 * cf

        w_lg = 0.0117 * wldg**0.95 * xmlg**0.43

        return w_lg

@dataclass
class GAAvionicsWeightInputs:
    """Parameters (in English units!)"""
    design_range : Union[float, int, csdl.Variable]
    num_flight_crew : Union[int, csdl.Variable]
    S_fuse_planform : Union[float, int, csdl.Variable, None] 
    xl : Union[float, int, csdl.Variable] = None
    xlw : Union[float, int, csdl.Variable] = None
    correction_factor : Union[float, csdl.Variable] = 1.

class GAAvionicsWeightModel:
    def evaluate(self, inputs: GAAvionicsWeightInputs):
        dr = inputs.design_range
        nfc = inputs.num_flight_crew
        S_fuse_planform = inputs.S_fuse_planform
        xl = inputs.xl
        xlw = inputs.xlw
        cf = inputs.correction_factor

        if xl is not None and xlw is not None:
            S_fuse_planform = xlw * xl

        W_avionics = 15.8 * dr**0.1 * nfc**0.7 * S_fuse_planform**0.43

        return W_avionics * cf


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


def solve_design_gross_weight(W_wing_est, W_htail_est, W_vtail_est, W_fuse_est, W_mlg_est, W_avionics_est, W_instruments_est):
    """Solves for the design gross weight by iterating over the estimates of wing and fuselage weights."""
    Wdg = 2200 # W_wing_est + W_fuse_est + W_htail_est + W_vtail_est + W_mlg_est + W_avionics_est + W_instruments_est
    W_wing = W_wing_est
    W_fuse = W_fuse_est
    W_htail = W_htail_est
    W_vtail = W_vtail_est
    W_mlg = W_mlg_est
    W_avionics = W_avionics_est
    W_instruments = W_instruments_est
    max_iterations = 100
    tolerance = 1e-6
    iteration = 0

    while True:
        Wdg_prev = Wdg
        W_wing = GAWingWeightModel().evaluate(GAWingWeightInputs(S_ref=S_ref, W_fuel=W_fuel, AR=AR, sweep_c4=sweep_c4,
                                                          taper_ratio=taper_ratio, thickness_to_chord=thickness_to_chord,
                                                          dynamic_pressure=dynamic_pressure, nz=nz, W_gross_design=Wdg))
        W_fuse = GAFuselageWeigthModel().evaluate(GAFuselageWeightInputs(S_wet=310, q_cruise=q_cruise, ulf=4.5, xl=None, d_av=None,
                                                           W_gross_design=Wdg, correction_factor=2.3))
        
        W_htail = GAHorizontalTailWeigthModel().evaluate(GAHorizontalTailInputs(S_ref=S_ref_tail, W_gross_design=Wdg, q_cruise=q_cruise))

        W_vtail = GAVerticalTailWeigthModel().evaluate(GAVerticalTailInputs(S_ref=S_ref_v_tail, W_gross_design=Wdg, q_cruise=q_cruise, 
                                                                                         t_o_c=0.12, sweep_c4=np.deg2rad(15), AR=AR_vtail))
        
        W_mlg = GAMainLandingGearWeightModel().evaluate(GAMainLandingGearWeightInputs(l_fuse, design_range, Wdg * 1.05))

        W_avionics = GAAvionicsWeightModel().evaluate(GAAvionicsWeightInputs(design_range, 1, 21))

        W_instruments = GAInstrumentsWeightModel().evaluate(GAInstrumentsWeightInputs(0.19, 1, 0, 1, None, xl, d_av))
        
        print(W_wing, W_htail, W_vtail, W_fuse, W_mlg, W_avionics, W_instruments)
        Wdg = W_wing + W_fuse + W_htail + W_vtail + W_mlg + W_avionics + W_instruments
        iteration += 1

        if abs(Wdg - Wdg_prev) < tolerance or iteration >= max_iterations:
            break

    if iteration >= max_iterations:
        raise ValueError("Solution did not converge within the maximum number of iterations.")

    return Wdg, iteration



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

    Wdg, iteration = solve_design_gross_weight(W_wing_est, W_htail_est, W_vtail_est, W_fuse_est, W_mlg_est, W_avionics_est, W_instruments_est)
    print("Design Gross Weight:", Wdg, "iterations", iteration)


