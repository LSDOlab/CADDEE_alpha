from CADDEE_alpha.utils.var_groups import MassProperties, MaterialProperties, DragBuildUpQuantities
from lsdo_geo import Geometry
from lsdo_function_spaces import FunctionSet
from CADDEE_alpha.utils.caddee_dict import CADDEEDict
from CADDEE_alpha.core.mesh.mesh import DiscretizationsDict
from typing import Union, List
import numpy as np
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from dataclasses import dataclass
import time


class VectorizedAttributes:
    def __init__(self, attribute_list, num_nodes) -> None:
        self.attribute_list = attribute_list
        self.num_nodes = num_nodes

    def __getattr__(self, name):
        child_attribute_list = []
        if hasattr(self.attribute_list[0], name):
            if callable(getattr(self.attribute_list[0], name)):
                def method(*args, **kwargs):
                    return_list = []
                    for comp in self.attribute_list:
                        output = getattr(comp, name)(*args, **kwargs)
                        if output:
                            return_list.append(output)
                    return return_list
                return method
            else:
                for i in range(self.num_nodes):
                    attr = getattr(self.attribute_list[i], name)
                    child_attribute_list.append(attr)
                
                if isinstance(child_attribute_list[0], (list, dict, set)) or hasattr(child_attribute_list[0], '__dict__'):
                    return VectorizedAttributes(child_attribute_list, self.num_nodes)
                else:
                    return child_attribute_list
        else:
            existing_attrs = [attr for attr in dir(self.attribute_list[0]) if not attr.startswith(("__", "_"))]
            raise AttributeError(f"Attribute {name} does not exist. Existing attributes are {existing_attrs}")
        
    def __setattr__(self, name: str, value) -> None:
        if name in {"attribute_list", "num_nodes"}:
            # Directly set the instance attributes
            super().__setattr__(name, value)
        else:
            # Set the attribute on each component in the attribute list
            for comp in self.attribute_list:
                setattr(comp, name, value)

# class VectorizedComponent:
#     def __init__(self, component, num_nodes) -> None:
#         from CADDEE_alpha.utils.copy_comps import copy_comps
#         self.num_nodes = num_nodes
#         self.comps = {}

#         for comp_name, comp in component.comps.items():
#             self.comps[comp_name] = VectorizedComponent(comp, num_nodes)

#         self.comp_list = []
#         for i in range(num_nodes):
#             self.comp_list.append(copy_comps(component))

class VectorizedComponent:
    def __init__(self, component, num_nodes, comp_list=None) -> None:
        from CADDEE_alpha.utils.copy_comps import copy_comps
        self.num_nodes = num_nodes
        self.comps = {}
        
        if comp_list is None:
            self.comp_list = []
            for i in range(num_nodes):
                self.comp_list.append(copy_comps(component))
        else:
            self.comp_list = comp_list
 
        for comp_name, comp in component.comps.items():
            child_comp_list = [child_comp.comps[comp_name] for child_comp in self.comp_list]
            self.comps[comp_name] = VectorizedComponent(comp, num_nodes, child_comp_list)

    def __getattr__(self, name):
        attr_list = []
        if hasattr(self.comp_list[0], name):
            if callable(getattr(self.comp_list[0], name)):
                def method(*args, **kwargs):
                    return_list = []
                    for i, comp in enumerate(self.comp_list):
                        args_i = [arg[i] for arg in args]
                        kwargs_i = {key: arg[i] for key, arg in kwargs.items()}
                        output = getattr(comp, name)(*args_i, **kwargs_i)
                        if output:
                            return_list.append(output)
                    return return_list
                return method
            else:
                for i in range(self.num_nodes):
                    attr = getattr(self.comp_list[i], name)
                    attr_list.append(attr)
                
                if isinstance(attr_list[0], (list, dict, set)) or hasattr(attr_list[0], '__dict__'):
                    vectorized_attributes = VectorizedAttributes(attr_list, self.num_nodes)
                    return vectorized_attributes
                else:
                    return attr_list
        else:
            existing_attrs = [attr for attr in dir(self.comp_list[0]) if not attr.startswith("__")]
            raise AttributeError(f"Attribute {name} does not exist. Existing attributes are {existing_attrs}")
        
# class VectorizedAttributes:
#     def __init__(self, attribute_list, num_nodes) -> None:
#         self.attribute_list = attribute_list
#         self.num_nodes = num_nodes

#     def __getattr__(self, name):
#         child_attribute_list = []
#         if hasattr(self.attribute_list[0], name):
#             if callable(self.attribute_list[0].__getattribute__(name)):
#                 def method(*args, **kwargs):
#                     return_list = []
#                     for comp in self.attribute_list:
#                         output = comp.__getattribute__(name)(*args, **kwargs)
#                         if output:
#                             return_list.append(output)
                    
#                     return return_list
#                 return method
#             else:
#                 for i in range(self.num_nodes):
#                     attr = self.attribute_list[i].__getattribute__(name)
#                     child_attribute_list.append(attr)
                
#                 return child_attribute_list
#         else:
#             raise AttributeError(f"Attribute {name} does not exist.")


# class VectorizedComponent:
#     def __init__(self, component, num_nodes) -> None:
#         from CADDEE_alpha.utils.copy_comps import copy_comps
#         self.num_nodes = num_nodes
#         self.comps = {}

#         for comp_name, comp in component.comps.items():
#             self.comps[comp_name] = VectorizedComponent(comp, num_nodes)

#         self.comp_list : List[Component] = []
#         for i in range(num_nodes):
#             self.comp_list.append(copy_comps(component))

#     def __getattr__(self, name):
#         attr_list = []
#         if hasattr(self.comp_list[0], name):
#             if callable(self.comp_list[0].__getattribute__(name)):
#                 def method(*args, **kwargs):
#                     return_list = []
#                     for i, comp in enumerate(self.comp_list):
#                         args_i = [arg[i] for arg in args]
#                         kwargs_i = {key:arg[i] for key, arg in kwargs.items()}
#                         output = comp.__getattribute__(name)(*args_i, **kwargs_i)
#                         if output:
#                             return_list.append(output)
                    
#                     return return_list
#                 return method
             
#             else:
#                 if name in ["quantities", "parameters"]:
#                     quant_list = [self.comp_list[i].__getattribute__(name) for i in range(self.num_nodes)]
#                     return VectorizedAttributes(quant_list, self.num_nodes)
#                 for i in range(self.num_nodes):                    
#                     attr = self.comp_list[i].__getattribute__(name)
#                     attr_list.append(attr)
                
#                 return attr_list
#         else:
#             raise AttributeError(f"Attribute {name} does not exist.")
        
        

class ComponentQuantities:
    def __init__(
        self, 
        mass_properties: MassProperties = None,
        material_properties: MaterialProperties = None,
        drag_parameters: DragBuildUpQuantities = None
    ) -> None:
        
        self._mass_properties = mass_properties
        self._material_properties = material_properties
        self._drag_parameters = drag_parameters
    
        self.surface_mesh = []
        self.surface_area = None

        if mass_properties is None:
            self.mass_properties = MassProperties()

        if material_properties is None:
            self.material_properties = MaterialProperties(self)

        if drag_parameters is None:
            self.drag_parameters =  DragBuildUpQuantities()
        

    @property
    def mass_properties(self):
        return self._mass_properties
    
    @mass_properties.setter
    def mass_properties(self, value):
        if not isinstance(value, MassProperties):
            raise ValueError(f"'mass_properties' must be of type {MassProperties}, received {value}")
        
        self._mass_properties = value

    @property
    def material_properties(self):
        return self._material_properties
    
    @material_properties.setter
    def material_properties(self, value):
        if not isinstance(value, MaterialProperties):
            raise ValueError(f"'material_properties' must be of type {MassProperties}, received {type(value)}")
        self._material_properties = value

    @property
    def drag_parameters(self):
        return self._drag_parameters
    
    @drag_parameters.setter
    def drag_parameters(self, value):
        if not isinstance(value, DragBuildUpQuantities):
            raise ValueError(f"'drag_parameters' must be of type {DragBuildUpQuantities}, received {type(value)}")
        self._drag_parameters = value

    # def __post_init__(self):
    #     self.mass_properties = MassProperties()
    #     if self.material_properties is None:
    #         self.material_properties = MaterialProperties(self)
    #     self.surface_mesh = []
    #     self.surface_area = None
    #     self.drag_parameters = DragBuildUpQuantities()

@dataclass
class ComponentParameters:
    pass

class Component:
    """The base component class.
    
    Attributes
    ---------
    comps : dictionary
        Dictionary of the sub/children components

    geometry : Union[Geometry]

    quantities : dictionary
        General container data; by default contains
        - mass_properties
        - function_spaces
        - meshes
    """
    # Default function spaces for components 
    _constant_b_spline_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
    _linear_b_spline_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    _linear_b_spline_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
    _quadratic_b_spline_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(3,))
    _cubic_b_spline_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))
    
    # Instance counter for naming components under the hood
    _instance_count = 0

    # Boolean attribute to keep track of whether component is a copy
    _is_copy = False

    # Private attrbute to allow certain components to be excluded from FFD    
    _skip_ffd = False

    parent = None

    def __init__(self, geometry : Union[FunctionSet, None]=None, 
                 **kwargs) -> None: 
        csdl.check_parameter(geometry, "geometry", types=(FunctionSet), allow_none=True)
        
        # Increment instance count and set private component name (will be obsolete in the future)
        Component._instance_count += 1
        self._name = f"component_{self._instance_count}"
        self._discretizations: DiscretizationsDict =  DiscretizationsDict()

        # set class attributes
        self.geometry : Union[FunctionSet, Geometry, None] = geometry
        self.comps : ComponentDict = ComponentDict(parent=self)
        self.quantities : ComponentQuantities = ComponentQuantities()
        self.parameters : ComponentParameters = ComponentParameters()

        # Set any keyword arguments on parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)
        
        if geometry is not None and isinstance(geometry, FunctionSet):
            self.quantities.surface_area = self._compute_surface_area(geometry=geometry)
            if "do_not_remake_ffd_block" in kwargs.keys():
                pass
            else:
                t1 = time.time()
                self._ffd_block = self._make_ffd_block(self.geometry)
                t2 = time.time()

                self.ffd_block_face_1 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.]))
                self.ffd_block_face_2 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.]))
                self.ffd_block_face_3 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0., 0.5]))
                self.ffd_block_face_4 = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.5]))
                self.ffd_block_face_5 = self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5]))
                self.ffd_block_face_6 = self._ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5]))
                
                print("time for making ffd block", t2-t1)
    
    def create_subgeometry(self, search_names:list[str], ignore_names:list[str]=[]) -> FunctionSet:
        """Create a sub-geometry by providing the search names of the e.g., OpenVSP component.
        
        This method can be overwritten by subcomponents to be tailored toward specific needs, 
        e.g., to make a new sub-component from the OML, like a spar from the wing OML
        """
        # Check if component is already the FunctionSet
        if isinstance(self.geometry, FunctionSet):
            component_geometry = self.geometry.declare_component(name=self._name, function_search_names=search_names, ignore_names=ignore_names)
        
        # Find the top-level component that is the FunctionSet
        else:
            system_component = self._find_system_component(self)
            system_geometry = system_component.geometry
            component_geometry =  system_geometry.declare_component(name=self._name, function_search_names=search_names, ignore_names=ignore_names)

        return component_geometry

    def plot(self):
        """Plot a component's geometry."""
        if self.geometry is None:
            raise ValueError(f"Cannot plot component {self} since its geometry is None.")
        else:
            self.geometry.plot()

    def actuate(self):
        raise NotImplementedError(f"'actuate' has not been implemented for component of type {type(self)}")

    def _make_ffd_block(self, entities, 
                        num_coefficients : tuple=(2, 2, 2), 
                        order: tuple=(1, 1, 1), 
                        num_physical_dimensions : int=3):
        """
        Call 'construct_ffd_block_around_entities' function.

        This method constructs a Cartesian FFD block with linear B-splines
        and 2 degrees of freedom in all dimensions.
        """
        ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities, 
                                                   num_coefficients=num_coefficients, degree=order)
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        return ffd_block 
    
    def _setup_ffd_block(self):
        raise NotImplementedError(f"'_setup_ffd_block' has not been implemented for {type(self)}")
    
    def _extract_geometric_quantities_from_ffd_block(self):
        raise NotImplementedError(f"'_extract_geometric_quantities_from_ffd_block' has not been implemented for {type(self)}")
    
    def _setup_ffd_parameterization(self):
        raise NotImplementedError(f"'_setup_ffd_parameterization' has not been implemented for {type(self)}")

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False):
        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients +  csdl.expand(rigid_body_translation, shape, action="k->ijk")
        
        # action = 'l->ijkl'
        # shape = self._ffd_block.coefficients.shape
        # self._ffd_block.coefficients = self._ffd_block.coefficients + csdl.expand(rigid_body_translation, shape, action=action)

        parameterization_solver.add_parameter(rigid_body_translation, cost=0.1)


    def _find_system_component(self, parent) -> FunctionSet:
        """Find the top-level system component by traversing the component hiearchy"""
        if parent is None:
            return self
        else:
            parent = self.parent
            self._find_system_component(parent)

    def _compute_surface_area(self, geometry:Geometry):
        """Compute the surface area of a component."""
        parametric_mesh_grid_num = 10

        surfaces = geometry.functions
        surface_area = csdl.Variable(shape=(1, ), value=1)

        surface_mesh = self.quantities.surface_mesh

        for i in surfaces.keys():
            oml_para_mesh = []
            for u in np.linspace(0, 1, parametric_mesh_grid_num):
                for v in np.linspace(0, 1, parametric_mesh_grid_num):
                    oml_para_mesh.append((i, np.array([u,v]).reshape((1,2))))
            
            coords_vec = geometry.evaluate(oml_para_mesh).reshape((parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
            surface_mesh.append(coords_vec)
            
            coords_u_end = coords_vec[1:, :, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))
            coords_u_start = coords_vec[:-1, :, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))

            coords_v_end = coords_vec[:, 1:, :].reshape((parametric_mesh_grid_num, parametric_mesh_grid_num-1, 3))
            coords_v_start = coords_vec[:, :-1, :].reshape((parametric_mesh_grid_num, parametric_mesh_grid_num-1, 3))

            u_vectors = coords_u_end - coords_u_start
            u_vectors_start = u_vectors # .reshape((-1, ))
            u_vectors_1 = u_vectors_start[:, :-1, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
            u_vectors_2 = u_vectors_start[:, 1:, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))


            v_vectors = coords_v_end - coords_v_start
            v_vectors_start = v_vectors # .reshape((-1, ))
            v_vectors_1 = v_vectors_start[:-1, :, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
            v_vectors_2 = v_vectors_start[1:, :, :].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))

            area_vectors_left_lower = csdl.cross(u_vectors_1, v_vectors_2, axis=2)
            area_vectors_right_upper = csdl.cross(v_vectors_1, u_vectors_2, axis=2)
            area_magnitudes_left_lower = csdl.norm(area_vectors_left_lower, ord=2, axes=(2, ))
            area_magnitudes_right_upper = csdl.norm(area_vectors_right_upper, ord=2, axes=(2, ))
            area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper)/2
            wireframe_area = csdl.sum(area_magnitudes)
            surface_area =  surface_area + wireframe_area 

        return surface_area


class ComponentDict(dict):
    def __init__(self, parent: Component, *args, **kwargs):
        super().__init__(*args, **kwargs)
        csdl.check_parameter(parent, "parent", types=(Component))
        self.parent = parent
    
    def __getitem__(self, key) -> Component:
        # Check if key exists
        if key not in self:
            raise KeyError(f"The component '{key}' does not exist. Existing components: {list(self.keys())}")
        else:
            return super().__getitem__(key)
    
    def __setitem__(self, key, value : Component, allow_overwrite=False):
        # Check type
        if not isinstance(value, Component):
            raise TypeError(f"Components must be of type(s) {Component}; received {type(value)}")
        
        # Check if key is already specified
        elif key in self:
            if allow_overwrite is False:
                raise Exception(f"Component {key} has already been set and cannot be re-set.")
            else:
                super().__setitem__(key, value)
                value.parent = self.parent

        # Set item otherwise
        else:
            super().__setitem__(key, value)
            value.parent = self.parent


if __name__ == "__main__":
    def unpack_attributes(obj, _depth=0, _visited=None):
        if _visited is None:
            _visited = set()
            
        obj_id = id(obj)
        if obj_id in _visited:
            return {}
        
        _visited.add(obj_id)
        
        attributes = {}
        for attr_name in dir(obj):
            # Ignore private and protected attributes
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
            except AttributeError:
                continue
            
            # Check if the attribute is itself an object that we should unpack
            if hasattr(attr_value, '__dict__'):
                attributes[attr_name] = unpack_attributes(attr_value, _depth + 1, _visited)
            else:
                attributes[attr_name] = attr_value
        
        return attributes

    test_comp = Component(
        geometry=None,
        AR=13, S_ref=50
    )

    print(unpack_attributes(test_comp))

    vectorized_comp = VectorizedComponent(test_comp, 5)


    # vectorized_comp.parameters.sweep = 

    # masses = vectorized_comp.quantities.mass_properties.mass
    # masses.value

    # vectorized_comp.quantities.mass_properties.mass.value

    print(vectorized_comp.parameters.AR)

        
