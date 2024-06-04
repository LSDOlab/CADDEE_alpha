from CADDEE_alpha.core.component import Component
from lsdo_geo import construct_ffd_block_around_entities, construct_tight_fit_ffd_block
import lsdo_function_spaces as lfs
from typing import Union, List
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass


@dataclass
class WingParameters:
    AR : Union[float, int, csdl.Variable]
    S_ref : Union[float, int, csdl.Variable]
    span : Union[float, int, csdl.Variable]
    sweep : Union[float, int, csdl.Variable]
    incidence : Union[float, int, csdl.Variable]
    taper_ratio : Union[float, int, csdl.Variable]
    dihedral : Union[float, int, csdl.Variable]
    thickness_to_chord : Union[float, int, csdl.Variable]
    S_wet : Union[float, int, csdl.Variable, None]=None

@dataclass
class WingGeometricQuantities:
    span: csdl.Variable
    center_chord: csdl.Variable
    left_tip_chord: csdl.Variable
    right_tip_chord: csdl.Variable
    sweep_angle_left: csdl.Variable
    sweep_angle_right: csdl.Variable
    dihedral_angle_left: csdl.Variable
    dihedral_angle_right: csdl.Variable


class Wing(Component):
    """The wing component class.
    
    Parameters
    ----------
    - AR : aspect ratio
    - S_ref : reference area
    - S_wet : wetted area (None default)
    - span (None default)
    - dihedral (deg) (None default)
    - sweep (deg) (None default)
    - taper_ratio (None default)

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
        AR : Union[int, float, csdl.Variable], 
        S_ref : Union[int, float, csdl.Variable], 
        S_wet : Union[int, float, csdl.Variable] = None,
        span : Union[int, float, csdl.Variable, None] = None, 
        dihedral : Union[int, float, csdl.Variable, None] = None, 
        sweep : Union[int, float, csdl.Variable, None] = None, 
        incidence : Union[int, float, csdl.Variable, None] = None, 
        taper_ratio : Union[int, float, csdl.Variable, None] = None, 
        thickness_to_chord_ratio : Union[int, float, csdl.Variable, None] = None, 
        geometry : Union[lfs.FunctionSet, None]=None,
        optimize_wing_twist : bool=False,
        tight_fit_ffd: bool = True,
        **kwargs
    ) -> None:
        super().__init__(geometry=geometry, kwargs=kwargs)
        
        # Do type checking 
        csdl.check_parameter(AR, "AR", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(S_ref, "S_ref", types=(int, float, csdl.Variable))
        csdl.check_parameter(S_wet, "S_wet", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(span, "span", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(dihedral, "dihedral", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(sweep, "sweep", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(incidence, "incidence", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(taper_ratio, "taper_ratio", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(thickness_to_chord_ratio, "thickness_to_chord_ratio", types=(int, float, csdl.Variable), allow_none=True)

        if all(arg is not None for arg in [AR, S_ref, span]):
            raise Exception("Cannot specifiy AR, S_ref, and span at the same time.")
        # if any(arg is not None for arg in [dihedral, sweep, incidence]) and geometry is not None:
        #     raise NotImplementedError("'sweep' and 'dihedral' and 'incidence' not implemented if geometry is not None")
        
        self._name = f"wing_{self._instance_count}"
        self.parameters : WingParameters =  WingParameters(
            AR=AR,
            S_ref=S_ref,
            span=span,
            sweep=sweep,
            incidence=incidence,
            dihedral=dihedral,
            taper_ratio=taper_ratio, 
            thickness_to_chord=thickness_to_chord_ratio,
        )

        # Set the wetted area if a geometry is provided
        if geometry is not None:
            self.parameters.S_wet = self.quantities.surface_area

        self._optimize_wing_twist = optimize_wing_twist

        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (lfs.FunctionSet)):
                raise TypeError(f"wing gometry must be of type {lfs.FunctionSet}")
            else:
                # Automatically make the FFD block upon instantiation 
                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd)
                # self._ffd_block.plot()

                # Compute the corner points of the wing 
                self._LE_left_point = geometry.project(self._ffd_block.evaluate(np.array([1., 0., 0.62])), extrema=True)
                self._LE_mid_point = geometry.project(self._ffd_block.evaluate(np.array([1., 0.5, 0.62])), extrema=True)
                self._LE_right_point = geometry.project(self._ffd_block.evaluate(np.array([1., 1.0, 0.62])), extrema=True)

                self._TE_left_point = geometry.project(self._ffd_block.evaluate(np.array([0., 0., 0.5])), extrema=True)
                self._TE_mid_point = geometry.project(self._ffd_block.evaluate(np.array([0., 0.5, 0.5])), extrema=True)
                self._TE_right_point = geometry.project(self._ffd_block.evaluate(np.array([0., 1.0, 0.5])), extrema=True)


    def actuate(self, angle : Union[float, int, csdl.Variable], axis_location : float=0.25):
        """Actuate (i.e., rotate) the wing about an axis location at or behind the leading edge.
        
        Parameters
        ----------
        angle : float, int, or csdl.Variable
            rotation angle (deg)

        axis_location : float (default is 0.25)
            location of actuation axis with respect to the leading edge;
            0.0 corresponds the leading and 1.0 corresponds to the trailing edge
        """
        wing_geometry = self.geometry
        # check if wing_geometry is not None
        if wing_geometry is None:
            raise ValueError("wing component cannot be actuated since it does not have a geometry (i.e., geometry=None)")

        # Check if if actuation axis is between 0 and 1
        if axis_location < 0.0 or axis_location > 1.0:
            raise ValueError("axis_loaction should be between 0 and 1")
        
        LE_center = wing_geometry.evaluate(self._LE_mid_point)
        TE_center = wing_geometry.evaluate(self._TE_mid_point)

        # Add the user_specified axis location
        actuation_center = csdl.linear_combination(
            LE_center, TE_center, 1, np.array([1 -axis_location]), np.array([axis_location])
        ).flatten()

        var = csdl.Variable(shape=(3, ), value=np.array([0., 1., 0.]))

        # Compute the actuation axis vector
        axis_origin = actuation_center - var
        axis_vector = actuation_center + var - axis_origin

        # Rotate the component about the axis
        wing_geometry.rotate(axis_origin=axis_origin, axis_vector=axis_vector / csdl.norm(axis_vector), angles=angle)

        # Re-evaluate all the discretizations associated with the wing
        for discretization_name, discretization in self._discretizations.items():
            try:
                discretization = discretization._update()
                self._discretizations[discretization_name] = discretization
            except AttributeError:
                raise Exception(f"The discretization {discretization_name} does not have an '_update' method, which is neded to" + \
                                " re-evaluate the geometry/meshes after the geometry coefficients have been changes")


    def _make_ffd_block(self, 
            entities : List[lfs.Function], 
            num_coefficients : tuple=(2, 2, 2), 
            degree: tuple=(1, 1, 1), 
            num_physical_dimensions : int=3,
            tight_fit: bool = True,
        ):
        """
        Call 'construct_ffd_block_around_entities' function. 

        Note that we overwrite the Component class's method to 
        - make a "tight-fit" ffd block instead of a cartesian one
        - to provide higher degree B-splines or more degrees of freedom
        if needed (via num_coefficients)
        """
        if tight_fit:
            ffd_block = construct_tight_fit_ffd_block(name=self._name, entities=entities, 
                                                    num_coefficients=num_coefficients, degree=degree)
        else:
            num_coefficients = (2, 3, 2) # NOTE: hard coding here might be limiting
            ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities,
                                                            num_coefficients=num_coefficients, degree=degree)
        
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        return ffd_block 
    
    def _setup_ffd_block(self, ffd_block, parameterization_solver, plot : bool=False):
        """Set up the wing ffd block."""
        
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=1
        )
        if plot:
            ffd_block_sectional_parameterization.plot()
        
        # Make B-spline functions for changing geometric quantities
        chord_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space, 
            coefficients=csdl.ImplicitVariable(
                shape=(3, ), 
                value=np.array([-0, 0, 0])
            ),
            name=f"{self._name}_chord_stretch_b_sp_coeffs"
        )

        span_stretch_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_span_stretch_b_sp_coeffs",
        )

        sweep_translation_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(3, ),
                value=np.array([0., 0., 0.,]),
            ),
            name=f"{self._name}_sweep_transl_b_sp_coeffs"
        )

        dihedral_translation_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(3, ),
                value=np.array([0., 0., 0.,]),
            ),
            name=f"{self._name}_dihedral_transl_b_sp_coeffs"
        )

        # twist_b_spline = lfs.Function(
        #     space=self._linear_b_spline_3_dof_space,
        #     coefficients=csdl.ImplicitVariable(
        #         shape=(3, ),
        #         value=np.array([0., 0., 0.,]),
        #     ),
        #     name=f"{self._name}_twist_b_sp_coeffs"
        # )

        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
        
        chord_stretch_sectional_parameters = chord_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        span_stretch_sectional_parameters = span_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        dihedral_translation_sectional_parameters = dihedral_translation_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        # twist_sectional_parameters = twist_b_spline.evaluate(
        #     parametric_b_spline_inputs
        # )

        sectional_parameters = VolumeSectionalParameterizationInputs()
        sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
        sectional_parameters.add_sectional_translation(axis=1, translation=span_stretch_sectional_parameters)
        sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
        sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translation_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False) 

        # set the coefficients
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
        self.geometry.set_coefficients(geometry_coefficients)

        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action='j->ij')

        # Add the coefficients of all B-splines to the parameterization solver
        parameterization_solver.add_parameter(chord_stretch_b_spline.coefficients)
        parameterization_solver.add_parameter(span_stretch_b_spline.coefficients)
        parameterization_solver.add_parameter(sweep_translation_b_spline.coefficients)
        parameterization_solver.add_parameter(dihedral_translation_b_spline.coefficients)
        parameterization_solver.add_parameter(rigid_body_translation, cost=0.1)


        return 

    def _extract_geometric_quantities_from_ffd_block(self) -> WingGeometricQuantities:
        """Extract the following quantities from the FFD block:
            - Span
            - root chord length
            - tip chord lengths
            - sweep/dihedral angles

        Note that this helper function will not work well in all cases (e.g.,
        in cases with high sweep or taper)
        """
        # Re-evaluate the corner points of the FFD block (plus center)
        # Center
        LE_center = self.geometry.evaluate(self._LE_mid_point)
        TE_center = self.geometry.evaluate(self._TE_mid_point)

        qc_center = 0.75 * LE_center + 0.25 * TE_center

        # Left side
        LE_left = self.geometry.evaluate(self._LE_left_point)
        TE_left = self.geometry.evaluate(self._TE_left_point)

        qc_left = 0.75 * LE_left + 0.25 * TE_left

        # Right side 
        LE_right = self.geometry.evaluate(self._LE_right_point)
        TE_right = self.geometry.evaluate(self._TE_right_point)

        qc_right = 0.75 * LE_right + 0.25 * TE_right

        # Compute span, root/tip chords, sweep, and dihedral
        span = LE_left - LE_right
        center_chord = TE_center - LE_center
        left_tip_chord = TE_left - LE_left
        right_tip_chord = TE_right - LE_right

        qc_spanwise_left = qc_left - qc_center
        qc_spanwise_right = qc_right - qc_center

        sweep_angle_left = csdl.arcsin(qc_spanwise_left[0] / csdl.norm(qc_spanwise_left))
        sweep_angle_right = csdl.arcsin(qc_spanwise_right[0] / csdl.norm(qc_spanwise_right))

        dihedral_angle_left = csdl.arcsin(qc_spanwise_left[2] / csdl.norm(qc_spanwise_left))
        dihedral_angle_right = csdl.arcsin(qc_spanwise_right[2] / csdl.norm(qc_spanwise_right))

        wing_geometric_qts = WingGeometricQuantities(
            span=csdl.norm(span),
            center_chord=csdl.norm(center_chord),
            left_tip_chord=csdl.norm(left_tip_chord),
            right_tip_chord=csdl.norm(right_tip_chord),
            sweep_angle_left=sweep_angle_left,
            sweep_angle_right=sweep_angle_right,
            dihedral_angle_left=dihedral_angle_left,
            dihedral_angle_right=dihedral_angle_right
        )

        return wing_geometric_qts

    def _setup_ffd_parameterization(self, wing_geom_qts: WingGeometricQuantities, ffd_geometric_variables):
        """Set up the wing parameterization."""

        # Set or compute the values for those quantities
        # AR = b**2/S_ref

        if self.parameters.AR is not None and self.parameters.S_ref is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio
            
            if not isinstance(self.parameters.AR , csdl.Variable):
                self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

            if not isinstance(self.parameters.S_ref , csdl.Variable):
                self.parameters.S_ref = csdl.Variable(shape=(1, ), value=self.parameters.S_ref)
                
            span_input = (self.parameters.AR * self.parameters.S_ref)**0.5
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1

        elif self.parameters.S_ref is not None and self.parameters.span is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio

            if not isinstance(self.parameters.span , csdl.Variable):
                self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

            if not isinstance(self.parameters.S_ref , csdl.Variable):
                self.parameters.S_ref = csdl.Variable(shape=(1, ), value=self.parameters.S_ref)

            span_input = self.parameters.span
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1

        else:
            raise NotImplementedError

        if self.parameters.sweep is not None:
            sweep_input = self.parameters.sweep
        else:
            sweep_input = csdl.Variable(shape=(1, ), value=0.)

        if self.parameters.dihedral is not None:
            dihedral_input = self.parameters.dihedral
        else:
            dihedral_input = csdl.Variable(shape=(1, ), value=0.)

        # Compute constraints: user input - geometric qty equivalent
        # span_constraint = wing_geom_qts.span - span_input
        # center_chord_constraint = wing_geom_qts.center_chord - root_chord_input
        # tip_chord_left_constraint = wing_geom_qts.left_tip_chord - tip_chord_left_input
        # tip_chord_right_constraint = wing_geom_qts.right_tip_chord - tip_chord_right_input

        # sweep_constraint_1 = wing_geom_qts.sweep_angle_left - sweep_input
        # sweep_constraint_2 = wing_geom_qts.sweep_angle_right - sweep_input

        # dihedral_constraint_1 = wing_geom_qts.dihedral_angle_left - dihedral_input
        # dihedral_constraint_2 = wing_geom_qts.dihedral_angle_right - dihedral_input

        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.span, span_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.center_chord, root_chord_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.left_tip_chord, tip_chord_left_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.right_tip_chord, tip_chord_right_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.sweep_angle_left, sweep_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.sweep_angle_right, sweep_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.dihedral_angle_left, dihedral_input)
        ffd_geometric_variables.add_geometric_variable(wing_geom_qts.dihedral_angle_right, dihedral_input)

        return

        # return [span_constraint, center_chord_constraint, tip_chord_left_constraint, 
        #         tip_chord_right_constraint, sweep_constraint_1, sweep_constraint_2, 
        #         dihedral_constraint_1, dihedral_constraint_2]

    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False):
        """Set up the wing geometry (mainly the FFD)"""
        # Get the ffd block
        wing_ffd_block = self._ffd_block

        # Set up the ffd block
        self._setup_ffd_block(wing_ffd_block, parameterization_solver, plot=plot)

        # Get wing geometric quantities (as csdl variable)
        wing_geom_qts = self._extract_geometric_quantities_from_ffd_block()

        # Define the geometric constraints
        self._setup_ffd_parameterization(wing_geom_qts, ffd_geometric_variables)
        
        return 
        
         
