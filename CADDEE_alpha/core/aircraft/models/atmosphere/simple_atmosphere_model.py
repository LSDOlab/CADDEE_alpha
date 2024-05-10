import csdl_alpha as csdl
from typing import Union
import numpy as np
from dataclasses import dataclass


@dataclass
class AtmosphericStates(csdl.VariableGroup):
    density : Union[float, int, csdl.Variable] = 1.225
    speed_of_sound : Union[float, int, csdl.Variable] = 343
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5

class SimpleAtmosphereModel:
    """Model class for simple atmosphere model."""
    def evaluate(self, altitude : Union[float, int, np.ndarray, csdl.Variable]) -> AtmosphericStates:
        """Evaluate the atmospheric states at a given altitude"""
        h = altitude * 1e-3

        # Constants
        L = 6.5 # K/km
        R = 287
        T0 = 288.16
        P0 = 101325
        g0 = 9.81
        mu0 = 1.735e-5
        S1 = 110.4
        gamma = 1.4

        # Temperature 
        T = - h * L + T0

        # Pressure 
        P = P0 * (T/T0)**(g0/(L * 1e-3)/R)
        
        # Density
        rho = P/R/T
        # self.print_var(rho)
        
        # Dynamic viscosity (using Sutherland's law)  
        mu = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

        # speed of sound 
        a = (gamma * R * T)**0.5

        atmos_states = AtmosphericStates(
            density=rho, speed_of_sound=a, temperature=T, pressure=P,
            dynamic_viscosity=mu
        )

        return atmos_states