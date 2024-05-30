from CADDEE_alpha.core.component import Component
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
from lsdo_geo import FFDBlock
from typing import Union
import numpy as np
import csdl 
import m3l
import lsdo_geo.splines.b_splines as bsp
from dataclasses import dataclass
from lsdo_function_spaces import FunctionSet


@dataclass
class FuselageParameters:
    length : Union[float, int, csdl.Variable]
    max_width : Union[float, int, csdl.Variable]
    max_height : Union[float, int, csdl.Variable]
    cabin_depth : Union[float, int, csdl.Variable]
    S_wet : Union[float, int, csdl.Variable]

class Fuselage(Component):
    """The fuslage component class.
    
    Parameters
    ----------
    - length
    - max_width
    - max_height

    Note that parameters may be design variables for optimizaiton.
    If a geometry is provided, the geometry parameterization sovler
    will manipulate the geometry through free-form deformation such 
    that the wing geometry satisfies these parameters.

    Attributes
    ----------
    - parameters : data class storing the above parameters
    - geometry : b-spline set or subset containing the wing geometry
    - comps : dictionary for children components
    - quantities : dictionary for storing (solver) data (e.g., field data)
    """
    def __init__(self, 
                 length : Union[int, float, csdl.Variable], 
                 max_width : Union[int, float, csdl.Variable], 
                 max_height : Union[int, float, csdl.Variable], 
                 cabin_depth : Union[int, float, csdl.Variable], 
                 S_wet : Union[int, float, csdl.Variable, None] = None,
                 geometry : Union[FunctionSet, None] = None,
                 **kwargs
                 ) -> None:
        super().__init__(geometry, **kwargs)
        self._name = f"fuselag_{self._instance_count}"
        self.parameters : FuselageParameters = FuselageParameters(
            length=length,
            max_height=max_height,
            max_width=max_width,
            cabin_depth=cabin_depth,
            S_wet=S_wet,
        )

        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (FunctionSet)):
                raise TypeError(f"wing gometry must be of type {FunctionSet}")
            else:
                # Automatically make the FFD block upon instantiation 
                self._ffd_block = self._make_ffd_block(self.geometry)

    def _setup_ffd_block(self, ffd_block, parameterization_solver : ParameterizationSolver, 
                         system_geometry, plot : bool=False):
        """Set up the wing ffd block."""
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            parameterized_points_shape=ffd_block.coefficients_shape,
            principal_parametric_dimension=1
        )

        ffd_block_sectional_parameterization.add_sectional_translation(name=f'{self._name}_stretch_x', axis=0)
        ffd_block_sectional_parameterization.add_sectional_translation(name=f'{self._name}_stretch_y', axis=1)
        ffd_block_sectional_parameterization.add_sectional_translation(name=f'{self._name}_stretch_z', axis=2)

        fuselage_stretch_coefficients_x = m3l.Variable(name=f'{self._name}_stretch_x_coefficients', shape=(2,), value=np.array([-0., 0.]))
        fuselage_stretch_b_spline_x = bsp.BSpline(name=f'{self._name}_stretch_x_b_spline', space=self._linear_b_spline_curve_2_dof_space, 
                                                coefficients=fuselage_stretch_coefficients_x, num_physical_dimensions=1)

        fuselage_stretch_coefficients_y = m3l.Variable(name=f'{self._name}_stretch_y_coefficients', shape=(2,), value=np.array([-0., 0.]))
        fuselage_stretch_b_spline_y = bsp.BSpline(name=f'{self._name}_stretch_y_b_spline', space=self._linear_b_spline_curve_2_dof_space, 
                                                coefficients=fuselage_stretch_coefficients_y, num_physical_dimensions=1)
        
        fuselage_stretch_coefficients_z = m3l.Variable(name=f'{self._name}_stretch_z_coefficients', shape=(2,), value=np.array([-0., 0.]))
        fuselage_stretch_b_spline_z = bsp.BSpline(name=f'{self._name}_stretch_z_b_spline', space=self._linear_b_spline_curve_2_dof_space, 
                                                coefficients=fuselage_stretch_coefficients_z, num_physical_dimensions=1)
        
        parameterization_solver.declare_state(name=f'{self._name}_stretch_x_coefficients', state=fuselage_stretch_coefficients_x)
        parameterization_solver.declare_state(name=f'{self._name}_stretch_y_coefficients', state=fuselage_stretch_coefficients_y)
        parameterization_solver.declare_state(name=f'{self._name}_stretch_z_coefficients', state=fuselage_stretch_coefficients_z)

        # Evaluate the B-splines to obtain sectional parameters
        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
        fuselage_stretch_x = fuselage_stretch_b_spline_x.evaluate(section_parametric_coordinates)
        fuselage_stretch_y = fuselage_stretch_b_spline_y.evaluate(section_parametric_coordinates)
        fuselage_stretch_z = fuselage_stretch_b_spline_z.evaluate(section_parametric_coordinates)

         # Store sectional parameters
        sectional_parameters = {
            f'{self._name}_stretch_x' : fuselage_stretch_x,
            f'{self._name}_stretch_y' : fuselage_stretch_y,
            f'{self._name}_stretch_z' : fuselage_stretch_z, 
        }

        # Evaluate entire FFD block based on sectional parameters
        fuselage_ffd_block_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=True)
        fuselage_coefficients = ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=True)

        # Assign the coefficients to the top-level parent (i.e., system) geometry
        if isinstance(self.geometry, FunctionSet):
            system_geometry.assign_coefficients(coefficients=fuselage_coefficients)
        else:
            system_geometry.assign_coefficients(coefficients=fuselage_coefficients, b_spline_names=self.geometry.b_spline_names)

    def _extract_geometric_quantities_from_ffd_block(self, ffd_block : FFDBlock, system_geometry, plot : bool=False):
        """Extract the following quantities from the FFD block:
            - fuselage length
            - fuselage max width
            - fuselage max heigtht
        """
        ffd_block_coefficients = ffd_block.coefficients.value

        B_matrix_nose = ffd_block.compute_evaluation_map(np.array([0., 0.5, 0.5]))
        B_matrix_tail = ffd_block.compute_evaluation_map(np.array([1., 0.5, 0.5]))
        nose_point = B_matrix_nose @ ffd_block_coefficients
        tail_point = B_matrix_tail @ ffd_block_coefficients

        nose = self.geometry.project(nose_point, plot=plot)
        tail = self.geometry.project(tail_point, plot=plot)

        B_matrix_left = ffd_block.compute_evaluation_map(np.array([0.5, 0., 0.5]))
        B_matrix_right = ffd_block.compute_evaluation_map(np.array([0.5, 1.0, 0.5]))
        left_point = B_matrix_left @ ffd_block_coefficients
        right_point = B_matrix_right @ ffd_block_coefficients

        left = self.geometry.project(left_point, plot=plot)
        right = self.geometry.project(right_point, plot=plot)

        B_matrix_top = ffd_block.compute_evaluation_map(np.array([0.5, 0.5, 1.]))
        B_matrix_bottom = ffd_block.compute_evaluation_map(np.array([0.5, 0.5, 0]))
        top_point = B_matrix_top @ ffd_block_coefficients
        bottom_point = B_matrix_bottom @ ffd_block_coefficients

        top = self.geometry.project(top_point, plot=plot)
        bottom = self.geometry.project(bottom_point, plot=plot)

        fuselage_length = tail - nose
        fuselage_width = right - left
        fuselage_height = top - bottom

        return m3l.norm(fuselage_length), m3l.norm(fuselage_width), m3l.norm(fuselage_height)
    
    def _setup_ffd_parameterization(self, parameterization_solver : ParameterizationSolver, length, 
                                    width, height):
        """Set up the fuselage parameterization."""

        # Declare quantities that the inner optimization will aim to enforce
        parameterization_solver.declare_input(name=f'{self._name}_length', input=length)
        parameterization_solver.declare_input(name=f'{self._name}_height', input=width)
        parameterization_solver.declare_input(name=f'{self._name}_width', input=height)

        # Store the parameterization inputs in a dictionary 
        parameterization_inputs = {}
        parameterization_inputs[f'{self._name}_length'] = self.parameters.length
        parameterization_inputs[f'{self._name}_height'] = self.parameters.max_height
        parameterization_inputs[f'{self._name}_width'] = self.parameters.max_width

        return parameterization_inputs
    
    def _setup_geometry(self, parameterization_solver, system_geometry, plot: bool = False):
        """Set up the fuselage geometry (mainly for FFD)"""

        # Get ffd block
        fuselage_ffd_block = self._ffd_block

        # Set up the ffd block
        self._setup_ffd_block(fuselage_ffd_block, parameterization_solver, system_geometry)

        # Get fuselage geometric quantities
        length, width, height = self._extract_geometric_quantities_from_ffd_block(
            fuselage_ffd_block, system_geometry, plot=plot
        )

        # Get parameterization inputs dictionary
        parameterization_inputs = self._setup_ffd_parameterization(
            parameterization_solver, length, width, height
        )
        
        return parameterization_inputs