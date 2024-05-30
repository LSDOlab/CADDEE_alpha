from CADDEE_alpha.utils.var_groups import MassProperties, MaterialProperties
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


@dataclass
class ComponentQuantities:
    mass_properties : MassProperties = None
    material_properties : MaterialProperties = None
    def __post_init__(self):
        self.mass_properties = MassProperties()
        if self.material_properties is None:
            self.material_properties = MaterialProperties(self)
        self.surface_area = None
        self.characteristic_length = None
        self.form_factor = None
        self.interference_factor = 1.1
        self.cf_laminar_fun = compute_cf_laminar
        self.cf_turbulent_fun = compute_cf_turbulent
        self.percent_laminar = 20
        self.percent_turbulent = 80

def compute_cf_laminar(Re):
    return 1.328 / Re**0.5

def compute_cf_turbulent(Re, M):
    Cf = 0.455 / (csdl.log(Re, 10)**2.58 * (1 + 0.144 * M**2)**0.65)
    return Cf

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
        self.comps : ComponentDict = ComponentDict(types=Component)
        self.quantities : ComponentQuantities = ComponentQuantities()
        self.parameters : ComponentParameters = ComponentParameters()

        # Set any keyword arguments on parameters
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)

        if geometry is not None and isinstance(geometry, FunctionSet):
            self.quantities.surface_area = self._compute_surface_area(geometry=geometry)

        # if isinstance(geometry, FunctionSet):
        #     system_component_geometry = self.create_subgeometry(search_names=[""])
        #     self.geometry = system_component_geometry

    
    def create_subgeometry(self, search_names : List[str]) -> FunctionSet:
        """Create a sub-geometry by providing the search names of the e.g., OpenVSP component.
        
        This method can be overwritten by subcomponents to be tailored toward specific needs, 
        e.g., to make a new sub-component from the OML, like a spar from the wing OML
        """
        # Check if component is already the FunctionSet
        if isinstance(self.geometry, FunctionSet):
            component_geometry = self.geometry.declare_component(name=self._name, function_search_names=search_names)
        
        # Find the top-level component that is the FunctionSet
        else:
            system_component = self._find_system_component(self)
            system_geometry = system_component.geometry
            component_geometry =  system_geometry.declare_component(name=self._name, function_search_names=search_names)

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
        
    def _update_coefficients_after_inner_optimization(self, parameterization_solver_states, geometry : FunctionSet, update_dict : dict, plot : bool=False):
        """Re-evaluate and re-assign the geomtry coefficients after the inner optimization has run"""
        ffd_block = update_dict[f'{self._name}_ffd_block']
        component = update_dict[f'{self._name}_component']
        ffd_sectional_parameterization = update_dict[f'{self._name}_ffd_sectional_parameterization']
        coefficient_names = update_dict[f'{self._name}_coefficient_names']
        b_splines = update_dict[f'{self._name}_b_splines']
        section_parametric_coordinates = update_dict[f'{self._name}_section_parametric_coordinates']
        sectional_parameter_names = update_dict[f'{self._name}_sectional_parameter_names']
        sectional_parameters = {}
        for i in range(len(coefficient_names)):
            # Update B-spline coefficients
            b_splines[i].coefficients = parameterization_solver_states[coefficient_names[i]]
 
            # Evaluate and store all B-splines
            b_spline_evaluation = b_splines[i].evaluate(section_parametric_coordinates)
            sectional_parameters[sectional_parameter_names[i]] = b_spline_evaluation
 
        # Evaluate all sections
        ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=plot)
 
        # Evaluate ffd block
        component_coefficients = ffd_block.evaluate(ffd_block_coefficients, plot=plot)
 
        # Assign coefficients
        if isinstance(self.geometry, FunctionSet):
            geometry.assign_coefficients(coefficients=component_coefficients)
        else:
            geometry.assign_coefficients(coefficients=component_coefficients, b_spline_names=component.b_spline_names)
 
        return geometry

    # list of things that need to happen (for FFD)
    # Figure out copies
    # Ex wing:
    # - Make ffd block
    #   
    # - Do the parameterization
    #   - Instantiation of parameterization obejct 
    #       - Make parameterization solver object
    #       - Make ffd block
    #       - Sectional parameterization
    #       - B-spline parameterization of the sectional parameters
    #       - define parameters in terms of geometry
    #   
    #   - Evaluation of parameterion object
    #       - evaluate parameterization solver
    #       - evaluate b-splines
    #       - evaluate sectional parameterization
    #       - evaluate ffd blcok 
    #   - Assign coefficient
    # - Evaluate actuations NOTE: actuations have to happen before the quantities are re-evaluated
    # - Evaluate geometric quantities (for us, dependent on type of component)
    #   - Extract the max dimensions from block (for projections to make meshes also span etc)
    #   
    def _setup_geometry(self, parameterization_solver, system_geometry, plot : bool=False):
        raise NotImplementedError(f"'_setup_geometry' has not been implemented for component {type(self)}")

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

        for i in surfaces.keys():
            oml_para_mesh = []
            for u in np.linspace(0, 1, parametric_mesh_grid_num):
                for v in np.linspace(0, 1, parametric_mesh_grid_num):
                    oml_para_mesh.append((i, np.array([u,v]).reshape((1,2))))
            
            coords_vec = geometry.evaluate(oml_para_mesh).reshape((parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
            
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


class ComponentDict(CADDEEDict):
    def __getitem__(self, key) -> Component:
        return super().__getitem__(key)
