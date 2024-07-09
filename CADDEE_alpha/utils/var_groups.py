import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import numpy as np
import lsdo_function_spaces as fs
from caddee_materials import Material
from csdl_alpha.utils.typing import VariableLike

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

@staticmethod
def compute_cf_laminar(Re):
    return 1.328 / Re**0.5

@staticmethod
def compute_cf_turbulent(Re, M):
    Cf = 0.455 / (csdl.log(Re, 10)**2.58 * (1 + 0.144 * M**2)**0.65)
    return Cf

@dataclass
class DragBuildUpQuantities:
    characteristic_length: Union[csdl.Variable, float, int, None] = None
    form_factor: Union[csdl.Variable, float, int, None] = None
    interference_factor: Union[csdl.Variable, float, int, None] = 1.1
    cf_laminar_fun = compute_cf_laminar
    cf_turbulent_fun = compute_cf_turbulent
    percent_laminar : Union[csdl.Variable, float, int] = 10
    percent_turbulent : Union[csdl.Variable, float, int] = 90
    drag_area: Union[csdl.Variable, float, None] = None

class MaterialProperties:
    def __init__(self, component, direction: str = 'centered'):
        """Initializes the MaterialProperties class.

        Parameters
        ----------
        component : str
            Component to which the material properties are applied.
        direction : str, optional
            Direction in which the materials are stacked, by default 'centered'.
            Options are 'centered', 'out', and 'in'.
        """
        self.material = None
        self.thickness = None
        self.component = component
        self.direction = direction
        self.material_stack = [] # {material, surface_indices, bounding_function, thickness, orientation}
        # TODO: think about using a dict for things like orientation and thickness

    def set_material(self, material:Material, thickness:Union[VariableLike, fs.FunctionSet]):
        """Sets the material and thickness for the component. 

        Parameters
        ----------
        material : Material
            Material to be set.

        thickness : Union[VariableLike, fs.FunctionSet]
            Thickness of the material. Can be a constant, variable, or function set.
        """
        self.material = material
        self.thickness = thickness

    def add_material(self, material: Material, thickness:Union[VariableLike, fs.FunctionSet], 
                     surface_indices:list=None, bounding_function:fs.FunctionSet=None,
                     orientation:Union[VariableLike, fs.FunctionSet]=None, insert_index=None):
        """Adds a material to the material stack.

        Parameters
        ----------
        material : Material
            Material to be added.

        thickness : Union[VariableLike, fs.FunctionSet]
            Thickness of the material. Can be a constant, variable, or function set.

        surface_indices : list, optional
            surface_indices to which the material is applied. 
            Inferred from thickness or bounding function if given as function set, otherwise defaults to all surface_indices.

        bounding_function : fs.FunctionSet, optional
            Function set that bounds the material, by default None.

        orientation : Union[VariableLike, fs.FunctionSet], optional
            Orientation of the material, by default None. Angle of the material with respect to local u axis.

        insert_index : int, optional
            Index at which to insert the material in the material stack, by default appended to end of stack.
        """
        if surface_indices is None:
            if isinstance(thickness, fs.FunctionSet):
                surface_indices = list(thickness.functions.keys())
            elif isinstance(bounding_function, fs.FunctionSet):
                surface_indices = bounding_function.functions.keys()
            else:
                surface_indices = self.component.geometry.functions.keys()
        else:
            if isinstance(thickness, fs.FunctionSet):
                if set(surface_indices) != set(thickness.functions.keys()):
                    raise ValueError("surface_indices must match thickness functions")
            if isinstance(bounding_function, fs.FunctionSet):
                if set(surface_indices) != set(bounding_function.functions.keys()):
                    raise ValueError("surface_indices must match bounding function functions")

        if insert_index is None or insert_index >= len(self.material_stack):
            self.material_stack.append({'material':material, 
                                        'surface_indices':surface_indices, 
                                        'bounding_function':bounding_function, 
                                        'thickness':thickness, 
                                        'orientation':orientation})
        else:
            self.material_stack.insert(insert_index, {'material':material, 
                                                      'surface_indices':surface_indices, 
                                                      'bounding_function':bounding_function, 
                                                      'thickness':thickness, 
                                                      'orientation':orientation})
        
    def remove_material(self, index):
        """Removes a material from the material stack.

        Parameters
        ----------
        index : int
            Index of the material to be removed.
        """
        self.material_stack.pop(index)

    def set_direction(self, direction):
        """Sets the direction of the material stack.

        Parameters
        ----------
        direction : str
            Direction in which the materials are stacked.
            Options are 'centered', 'out', and 'in'.
        """
        self.direction = direction

    def clear_materials(self):
        """Clears the material stack."""
        self.material_stack = []

    def get_material_stack(self, surface_index:int) -> list:
        """Returns the material stack for a given surface index.

        Parameters
        ----------
        surface_index : int
            Surface index for which the material stack is returned.

        Returns
        -------
        list
            Material stack for the given surface index.
            Dictionary containing ('material', 'bounding_function', 'thickness', 'orientation')
        """
        material_stack = []
        for material_info in self.material_stack:
            surface_indices = material_info['surface_indices']
            if surface_index in surface_indices:
                material_stack.append({'material':material_info['material'], 
                                       'bounding_function':material_info['bounding_function'], 
                                       'thickness':material_info['thickness'], 
                                       'orientation':material_info['orientation']})
        return material_stack 
    
    def evaluate_thickness(self, parametric_coordinates):
        """Evaluates the thickness of the material or material stack at the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : list
            Parametric coordinates at which to evaluate the material stack.

        Returns
        -------
        list
            List of thicknesses evaluated at the given parametric coordinates.
        """
        # bypass slow loop if thickness is not none
        if self.thickness is not None:
            if isinstance(self.thickness, fs.FunctionSet):
                return self.thickness.evaluate(parametric_coordinates)
            else:
                return np.ones(len(parametric_coordinates)) * self.thickness

        # TODO: group inds together
        index_group = {}
        for i, parametric_coordinate in enumerate(parametric_coordinates):
            ind = parametric_coordinate[0]
            if ind not in index_group:
                index_group[ind] = []
            index_group[ind].append(i)

        out = csdl.Variable(shape=(len(parametric_coordinates),), value=0)
        parametric_coordinates = np.array(parametric_coordinates, dtype='O,O')
        for ind, inds in index_group.items():
            inds = np.array(inds)
            material_stack = self.get_material_stack(ind)
            if len(material_stack) == 0:
                if isinstance(self.thickness, fs.FunctionSet):
                    evaluated_thickness = self.thickness.evaluate(parametric_coordinates=[parametric_coordinates[inds[0]]])
                elif self.thickness is not None:
                    evaluated_thickness = self.thickness
                else:
                    evaluated_thickness = np.zeros(len(inds))
            else:
                evaluated_thickness = np.zeros(len(inds))
                for material_info in material_stack:
                    thickness = material_info['thickness']
                    bounding_function = material_info['bounding_function']
                    if isinstance(thickness, fs.FunctionSet):
                        thickness = thickness.evaluate(parametric_coordinates[inds])
                    if isinstance(bounding_function, fs.FunctionSet):
                        bounding_function = bounding_function.evaluate(parametric_coordinates[inds])
                        thickness = thickness * bounding_function
                    evaluated_thickness = evaluated_thickness + thickness
            out = out.set(csdl.slice[[(ind,) for ind in inds]], evaluated_thickness.reshape((-1, 1)))



        # out = csdl.Variable(shape=(len(parametric_coordinates),), value=0)
        # for i, parametric_coordinate in enumerate(parametric_coordinates):
        #     ind = parametric_coordinate[0]
        #     material_stack = self.get_material_stack(ind)
        #     if len(material_stack) == 0:
        #         if isinstance(self.thickness, fs.FunctionSet):
        #             evaluated_thickness = self.thickness.evaluate(parametric_coordinates=[parametric_coordinate])
        #         elif self.thickness is not None:
        #             evaluated_thickness = self.thickness
        #         else:
        #             evaluated_thickness = 0
        #     else:
        #         evaluated_thickness = 0
        #         for material_info in material_stack:
        #             thickness = material_info['thickness']
        #             bounding_function = material_info['bounding_function']
        #             if isinstance(thickness, fs.FunctionSet):
        #                 thickness = thickness.evaluate([parametric_coordinate])
        #             if isinstance(bounding_function, fs.FunctionSet):
        #                 bounding_function = bounding_function.evaluate([parametric_coordinate])
        #                 thickness = thickness * bounding_function
        #             evaluated_thickness = evaluated_thickness + thickness
        #     out = out.set(csdl.slice[i], evaluated_thickness)
        return out

    def evaluate_stack(self, parametric_coordinates):
        """Evaluates the material stack at the given parametric coordinates.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            Parametric coordinates at which to evaluate the material stack.

        Returns
        -------
        list
            List of list of dicts containing the evaluated material properties.
            {'material', 'thickness', 'orientation'}
        """
        # TODO: addd computation of normal vectors somewhere
        out = []
        for parametric_coordinate in parametric_coordinates:
            ind = parametric_coordinate[0]
            material_stack = self.get_material_stack(ind)
            evaluated_stack = []
            for material_info in material_stack:
                material = material_info['material']
                thickness = material_info['thickness']
                bounding_function = material_info['bounding_function']
                orientation = material_info['orientation']
                if isinstance(thickness, fs.FunctionSet):
                    thickness = thickness.evaluate(parametric_coordinate)
                if isinstance(orientation, fs.FunctionSet):
                    orientation = orientation.evaluate(parametric_coordinate)
                if isinstance(bounding_function, fs.FunctionSet):
                    bounding_function = bounding_function.evaluate(parametric_coordinate)
                    thickness = thickness * bounding_function

                evaluated_stack.append({'material':material,
                                        'thickness':thickness, 
                                        'orientation':orientation})
            out.append(evaluated_stack)
        return out        

class MassProperties:
    def __init__(
        self,
        mass :  Union[float, int, csdl.Variable, None] = None,
        cg_vector : Union[np.ndarray, csdl.Variable, None] = None,
        inertia_tensor : Union[np.ndarray, csdl.Variable, None] = None,
    ):
        self._mass = mass
        self._cg_vector = cg_vector
        self._inertia_tensor = inertia_tensor

        self.mass = self._mass
        self.cg_vector = self._cg_vector
        self.inertia_tensor = self._inertia_tensor


    @property
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self, value):
        if not isinstance(value, (csdl.Variable, float, int, type(None))):
            raise ValueError(f"mass must be of type csdl.Variable, float or int, None; received {type(value)}")
        if isinstance(value, csdl.Variable):
            try:
                value = value.reshape((1, ))
            except:
                raise ValueError(f"'mass' must be a scaler. Received variable of shape {value.shape}")
        
        self._mass = value

    @ property
    def cg_vector(self):
        return self._cg_vector
    
    @cg_vector.setter
    def cg_vector(self, value):
        if not isinstance(value, (csdl.Variable, np.ndarray, type(None))):
            raise ValueError(f"cg_vector must be of type csdl.Variable, np.ndarray, or None; received {type(value)}")
        if isinstance(value, csdl.Variable):
            try:
                value = value.reshape((3, ))
            except:
                raise ValueError(f"'cg_vecor' must be a vector of size 3. Received variable of shape {value.shape}")
        
        self._cg_vector = value

    @property
    def inertia_tensor(self):
        return self._inertia_tensor
    
    @inertia_tensor.setter
    def inertia_tensor(self, value):
        if not isinstance(value, (csdl.Variable, np.ndarray, type(None))):
            raise ValueError(f"inertia_tensor must be of type csdl.Variable, np.ndarray, or None; received {type(value)}")
        if isinstance(value, csdl.Variable):
            try:
                value = value.reshape((3, 3))
            except:
                raise ValueError(f"'inertia_tensor' must be a matrix of size (3, 3). Received variable of shape {value.shape}")
        
        self._inertia_tensor = value



if __name__ == "__main__":
    mps = MassProperties()
    
    print(mps.__dict__)

    mps.mass = 1# np.array([1, 2, 3])
    print(getattr(mps, "mass"))

    # try:
    #     mps.mass = np.zeros((3, ))
    # except:
    #     print("Catches shape exception")

    # mps.mass = csdl.Variable(name='mass', shape=(1, 1))
    # mps.mass = csdl.Variable(name='mass', shape=(2, 1))


