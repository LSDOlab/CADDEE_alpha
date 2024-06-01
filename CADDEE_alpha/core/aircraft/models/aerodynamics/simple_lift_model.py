import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
import numpy as np
from CADDEE_alpha.core.aircraft.conditions.aircraft_condition import AircaftStates
from CADDEE_alpha.core.aircraft.models.atmosphere.simple_atmosphere_model import AtmosphericStates


@dataclass
class SimpleLiftModelInputs:
    ac_states : AircaftStates
    atmos_state : AtmosphericStates
    AR : Union[float, int, csdl.Variable]
    Cl_alpha : Union[float, int, csdl.Variable]
    Cl_0 : Union[float, int, csdl.Variable]
    max_sweep : Union[float, int, csdl.Variable]
    S_wet : Union[float, int, csdl.Variable]
    S_ref : Union[float, int, csdl.Variable]
    d_fuse : Union[float, int, csdl.Variable]
    quarter_chord : Union[float, int, csdl.Variable, None] = None
    wing_incidence : Union[float, int, csdl.Variable, None] = None


class SimpleLiftModel(csdl.Model):
    """Implements semi-empirical formula for 3D lift curve slope.
    
    We then compute the lift based on the angle of attack. 
    """
    def initialize(self):
        self.parameters.declare("num_nodes", default=1, types=int)

    def evaluate(self, inputs : SimpleLiftModelInputs):
        num_nodes = self.parameters["num_nodes"]

        AR = inputs.AR
        Cl_alpha = inputs.Cl_alpha
        Cl_0 = inputs.Cl_0
        max_sweep = inputs.max_sweep
        S_wet = inputs.S_wet
        S_ref = inputs.S_ref
        d_fuse = inputs.d_fuse
        inc_angle = inputs.wing_incidence

        atmos_states = inputs.atmos_state
        ac_states = inputs.ac_states

        a = atmos_states.speed_of_sound
        rho = atmos_states.density

        u = ac_states.u
        v = ac_states.v
        w = ac_states.w
        V = (u**2 + v**2 + w**2)**0.5
        theta = ac_states.theta

        if inc_angle is not None:
            alpha = theta + inc_angle
        else:
            alpha = theta

        M = V / a
        
        beta = 1 - M**2 
        eta = Cl_alpha / (2 * np.pi / beta)
        b = (S_ref * AR)**0.5
        F = 1.07 * (1 + d_fuse / b)

        CL_alpha = 2 * np.pi * AR / (2 + (4 + AR**2 * beta**2 / eta**2 * (1 + np.tan(max_sweep)**2 / beta**2))**0.5) * (S_wet / S_ref) * F

        C_L = Cl_0 + CL_alpha * alpha

        L = C_L * 0.5 * rho * V**2 * S_ref

        F = np.zeros((num_nodes, 3))
        F[:, 2] = -L * np.cos(theta)
        F[:, 0] = L * np.sin(theta)

        return F
    

if __name__ == "__main__":

    theta = np.deg2rad(30)

    M_theta = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0., 1., 0.],
        [np.sin(theta), 0., np.cos(theta)]
    ])

    F = np.array([0, 0, -10])
    r = np.array([0., 4., 0.5])

    moment_1 = M_theta @ np.cross(r, F)
    moment_2 = np.cross(M_theta @ r, M_theta @ F)

    print("moment", moment_1)
    print("moment", moment_2)
    print("force local", F)
    print("force body", M_theta @ F)
    exit()

    wing_area = 16.17
    wetted_area = 2.1 *  wing_area
    AR = 7.32
    Cl_alpha = 5.73
    Cl_0 = 0.25
    d_fuse = 1.
    
    ac_states = AircaftStates(
        u=64.,
        theta=np.array([np.deg2rad(-1), 0, np.deg2rad(1), np.deg2rad(2), np.deg2rad(3)])
    )
     
    atmos = AtmosphericStates()

    inputs = SimpleLiftModelInputs(
        ac_states=ac_states,
        atmos_state=atmos,
        AR=AR,
        Cl_alpha=Cl_alpha,
        Cl_0=Cl_0,
        max_sweep=0.,
        S_wet=wetted_area,
        S_ref=wing_area,
        d_fuse=d_fuse,
    )

    lift_model = SimpleLiftModel(num_nodes=5)

    F = lift_model.evaluate(inputs=inputs)

    print("Forces", F)






