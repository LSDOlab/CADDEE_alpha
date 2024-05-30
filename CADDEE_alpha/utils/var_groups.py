import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np


@dataclass
class AircaftStates(csdl.VariableGroup):
    u : Union[float, int, csdl.Variable, np.ndarray] = 0
    v : Union[float, int, csdl.Variable, np.ndarray] = 0
    w : Union[float, int, csdl.Variable, np.ndarray] = 0
    p : Union[float, int, csdl.Variable, np.ndarray] = 0
    q : Union[float, int, csdl.Variable, np.ndarray] = 0
    r : Union[float, int, csdl.Variable, np.ndarray] = 0
    phi : Union[float, int, csdl.Variable, np.ndarray] = 0
    theta : Union[float, int, csdl.Variable, np.ndarray] = 0
    psi : Union[float, int, csdl.Variable, np.ndarray] = 0
    x : Union[float, int, csdl.Variable, np.ndarray] = 0
    y : Union[float, int, csdl.Variable, np.ndarray] = 0
    z : Union[float, int, csdl.Variable, np.ndarray] = 0

@dataclass
class AtmosphericStates(csdl.VariableGroup):
    density : Union[float, int, csdl.Variable] = 1.225
    speed_of_sound : Union[float, int, csdl.Variable] = 343
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5


@dataclass
class MassProperties:
    mass :  Union[float, int, csdl.Variable, None] = None 
    cg_vector : Union[np.ndarray, csdl.Variable, None] = None
    inertia_tensor : Union[np.ndarray, csdl.Variable, None] = None

    # mass: Union[float, int, csdl.Variable, None] = None
    # cg_vector: Union[np.ndarray, csdl.Variable, None] = None
    # inertia_tensor: Union[np.ndarray, csdl.Variable, None] = None

    # @property
    # def mass(self):
    #     return self._mass

    # @mass.setter
    # def mass(self, value):
    #     if not isinstance(value, (float, int, csdl.Variable, type(None))):
    #         raise TypeError("Mass must be a float, int, csdl.Variable, or None")
    #     if isinstance(value, csdl.Variable):
    #         if value.shape != (1, ):
    #             if value.shape == (1, 1):
    #                 value.reshape(1, )
    #             else:
    #                 raise ValueError(f"Shape of mass csdl Variable can only be (1, ), received {value.shape}")
    #     self._mass = value

    # @property
    # def cg_vector(self):
    #     return self._cg_vector

    # @cg_vector.setter
    # def cg_vector(self, value):
    #     if not isinstance(value, (np.ndarray, csdl.Variable, type(None))):
    #         raise TypeError("CG vector must be a numpy array, csdl.Variable, or None")
    #     if value is not None and value.ndim != 1:
    #         raise ValueError("CG vector must be a 1-dimensional numpy array")
    #     self._cg_vector = value

    # @property
    # def inertia_tensor(self):
    #     return self._inertia_tensor

    # @inertia_tensor.setter
    # def inertia_tensor(self, value):
    #     if not isinstance(value, (np.ndarray, csdl.Variable, type(None))):
    #         raise TypeError("Inertia tensor must be a numpy array, csdl.Variable, or None")
    #     if value is not None and value.ndim != 2:
    #         raise ValueError("Inertia tensor must be a 2-dimensional numpy array")
    #     self._inertia_tensor = value


if __name__ == "__main__":
    mps = MassProperties()
    
    
    # try:
    #     mps.mass = np.zeros((3, ))
    # except:
    #     print("Catches shape exception")

    # mps.mass = csdl.Variable(name='mass', shape=(1, 1))
    # mps.mass = csdl.Variable(name='mass', shape=(2, 1))


