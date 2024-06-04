from CADDEE_alpha.core.component import Component
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import lsdo_function_spaces as lfs
from typing import Union
import numpy as np
import csdl_alpha as csdl 
from dataclasses import dataclass
from lsdo_function_spaces import FunctionSet


@dataclass
class FuselageParameters:
    length : Union[float, int, csdl.Variable]
    max_width : Union[float, int, csdl.Variable]
    max_height : Union[float, int, csdl.Variable]
    cabin_depth : Union[float, int, csdl.Variable]
    S_wet : Union[float, int, csdl.Variable]

@dataclass
class FuselageGeometricQuantities:
    length: csdl.Variable
    width: csdl.Variable
    height: csdl.Variable

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

        # # compute form factor (according to Raymer)
        # f = length / csdl.maximum(max_height, max_width)
        # FF = 1 + 60 / f**3 + f / 400
        # self.quantities.drag_parameters.form_factor = FF
        # self.quantities.drag_parameters.characteristic_length = length

        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (FunctionSet)):
                raise TypeError(f"wing gometry must be of type {FunctionSet}")
            else:
                # Automatically make the FFD block upon instantiation 
                self._ffd_block = self._make_ffd_block(self.geometry)

                # Extract dimensions (height, width, length) from the FFD block
                self._nose_point = geometry.project(self._ffd_block.evaluate(np.array([0., 0.5, 0.5])))
                self._tail_point = geometry.project(self._ffd_block.evaluate(np.array([1., 0.5, 0.5])))

                self._left_point = geometry.project(self._ffd_block.evaluate(np.array([0.5, 0., 0.5])))
                self._right_point = geometry.project(self._ffd_block.evaluate(np.array([0.5, 1., 0.5])))

                self._top_point = geometry.project(self._ffd_block.evaluate(np.array([0.5, 0.5, 1.])))
                self._bottom_point = geometry.project(self._ffd_block.evaluate(np.array([0.5, 0.5, 0.])))


    def _setup_ffd_block(self, ffd_block, parameterization_solver, plot : bool=False):
        """Set up the fuselage ffd block."""
        
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=0
        )
        if plot:
            ffd_block_sectional_parameterization.plot()

        # Make B-spline functions for changing geometric quantities
        length_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_length_stretch_b_sp_coeffs",
        )

        height_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_height_stretch_b_sp_coeffs",
        )

        width_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_width_stretch_b_sp_coeffs",
        )

        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))

        length_stretch_sectional_parameters = length_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        height_stretch_sectional_parameters = height_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        width_stretch_sectional_parameters = width_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        sectional_parameters = VolumeSectionalParameterizationInputs()
        sectional_parameters.add_sectional_translation(axis=0, translation=length_stretch_sectional_parameters)
        sectional_parameters.add_sectional_stretch(axis=1, stretch=height_stretch_sectional_parameters)
        sectional_parameters.add_sectional_stretch(axis=2, stretch=width_stretch_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

        # set the coefficient in the geometry
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
                    
        self.geometry.set_coefficients(geometry_coefficients)

        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action='j->ij')

        # Add B-spline coefficients to parameterization solver
        parameterization_solver.add_parameter(length_stretch_b_spline.coefficients, cost=100)
        parameterization_solver.add_parameter(height_stretch_b_spline.coefficients, cost=100)
        parameterization_solver.add_parameter(width_stretch_b_spline.coefficients, cost=100)
        parameterization_solver.add_parameter(rigid_body_translation, cost=1000)

        return


    def _extract_geometric_quantities_from_ffd_block(self):
        """Extract the following quantities from the FFD block:
            - fuselage length
            - fuselage max width
            - fuselage max heigtht
        """

        # Re-evaluate dimensions from FFD block
        nose = self.geometry.evaluate(self._nose_point)
        tail = self.geometry.evaluate(self._tail_point)

        left = self.geometry.evaluate(self._left_point)
        right = self.geometry.evaluate(self._right_point)

        top = self.geometry.evaluate(self._top_point)
        bottom = self.geometry.evaluate(self._bottom_point)

        fuselage_length = csdl.norm(tail - nose)
        fuselage_width = csdl.norm(right - left)
        fuselage_height = csdl.norm(top - bottom)

        fuselage_geometric_qts = FuselageGeometricQuantities(
            length=fuselage_length,
            width=fuselage_width,
            height=fuselage_height,
        )

        return fuselage_geometric_qts
    
    def _setup_ffd_parameterization(self, fuselage_geometric_qts : FuselageGeometricQuantities, ffd_geometric_variables):
        """Set up the fuselage parameterization."""
        # If user doesn't specify length, use initial geometry
        if self.parameters.length is None:
            pass
        else:
            length_input = self.parameters.length
            ffd_geometric_variables.add_geometric_variable(fuselage_geometric_qts.length, length_input)

        # If user doesn't specify height, use initial geometry
        if self.parameters.max_height is None:
            pass
        else:
            height_input = self.parameters.max_height
            ffd_geometric_variables.add_geometric_variable(fuselage_geometric_qts.height, height_input)

        # If user doesn't specify width, use initial geometry
        if self.parameters.max_width is None:
            pass
        else:
            width_input = self.parameters.max_width
            ffd_geometric_variables.add_geometric_variable(fuselage_geometric_qts.width, width_input)


        return 
    
    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot: bool = False):
        """Set up the fuselage geometry (mainly for FFD)"""

        # Get ffd block
        fuselage_ffd_block = self._ffd_block

        # Set up the ffd block
        self._setup_ffd_block(fuselage_ffd_block, parameterization_solver)

        # Get fuselage geometric quantities
        fuselage_geom_qts = self._extract_geometric_quantities_from_ffd_block()

        # Define geometric constraints
        self._setup_ffd_parameterization(fuselage_geom_qts, ffd_geometric_variables)
        
        return