from CADDEE_alpha.core.component import Component
from typing import List
from lsdo_geo import construct_ffd_block_around_entities, construct_tight_fit_ffd_block
from lsdo_function_spaces import FunctionSet
import lsdo_geo.splines.b_splines as bsp
from typing import Union
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
from lsdo_geo import FFDBlock
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
                 geometry : Union[FunctionSet, None]=None,
                 optimize_wing_twist : bool=False,
                 tight_fit_ffd: bool = True,
                 **kwargs) -> None:
        super().__init__(geometry=geometry, kwargs=kwargs)
        
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
        if any(arg is not None for arg in [dihedral, sweep, incidence]) and geometry is not None:
            raise NotADirectoryError("'sweep' and 'dihedral' and 'incidence' not implemented if geometry is not None")
        
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
            if not isinstance(self.geometry, (FunctionSet)):
                raise TypeError(f"wing gometry must be of type {FunctionSet}")
            else:
                # Automatically make the FFD block upon instantiation 
                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd)
                # self._ffd_block.plot()

                # Compute the corner points of the wing  
                self._LE_left_point = geometry.project(self._ffd_block.evaluate(np.array([1., 0., 0.62])))
                self._LE_mid_point = geometry.project(self._ffd_block.evaluate(np.array([1., 0.5, 0.62])))
                self._LE_right_point = geometry.project(self._ffd_block.evaluate(np.array([1., 1.0, 0.62])))

                self._TE_left_point = geometry.project(self._ffd_block.evaluate(np.array([0., 0., 0.5])))
                self._TE_mid_point = geometry.project(self._ffd_block.evaluate(np.array([0., 0.5, 0.5])))
                self._TE_right_point = geometry.project(self._ffd_block.evaluate(np.array([0., 1.0, 0.5])))


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
                        entities : List[bsp.BSpline], 
                        num_coefficients : tuple=(2, 2, 2), 
                        order: tuple=(1, 1, 1), 
                        num_physical_dimensions : int=3,
                        tight_fit: bool = True,
                    ):
        """
        Call 'construct_ffd_block_around_entities' function. 

        Note that we overwrite the Component class's method to 
        - make a "tight-fit" ffd block instead of a cartesian one
        - to provide higher order B-splines or more degrees of freedom
        if needed (via num_coefficients)
        """
        if tight_fit:
            ffd_block = construct_tight_fit_ffd_block(name=self._name, entities=entities, 
                                                    num_coefficients=num_coefficients, degree=order)
        else:
            ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities,
                                                            num_coefficients=num_coefficients, degree=order)
        
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        return ffd_block 
    
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

        # Add sectional degrees of freedom via B-splines; initialize their coefficients
        ffd_block_sectional_parameterization.add_sectional_stretch(name=f'sectional_{self._name}_chord_stretch', axis=0)
        ffd_block_sectional_parameterization.add_sectional_translation(name=f'sectional_{self._name}_span_stretch', axis=1)
        ffd_block_sectional_parameterization.add_sectional_translation(name=f'sectional_{self._name}_translation_x', axis=0)
        ffd_block_sectional_parameterization.add_sectional_translation(name=f'sectional_{self._name}_translation_z', axis=2)
        if self._optimize_wing_twist:
            ffd_block_sectional_parameterization.add_sectional_rotation(name='sectional_wing_twist', axis=1)


        chord_stretch_coefficients = csdl.Variable(name=f'{self._name}_chord_stretch_coefficients', shape=(3,), value=np.array([0., 0., 0.]))
        chord_stretch_b_spline = bsp.BSpline(name=f'{self._name}_chord_stretch_b_spline', space=self._linear_b_spline_curve_3_dof_space, 
                                                coefficients=chord_stretch_coefficients, num_physical_dimensions=1)

        span_stretch_coefficients = csdl.Variable(name=f'{self._name}_span_stretch_coefficients', shape=(2,), value=np.array([-0., 0.]))
        span_stretch_b_spline = bsp.BSpline(name=f'{self._name}_span_stretch_b_spline', space=self._linear_b_spline_curve_2_dof_space, 
                                                coefficients=span_stretch_coefficients, num_physical_dimensions=1)

        # TODO: make wing twist parameterization more general (currently, arbitrarily choose space + DOFs)
        if self._optimize_wing_twist:
            twist_coefficients = csdl.Variable(name=f'{self._name}_twist_coefficients', shape=(5,), value=np.array([0., 0., 0., 0., 0.]))
            twist_b_spline = bsp.BSpline(name='twist_b_spline', space=self._cubic_b_spline_curve_5_dof_space,
                                                    coefficients=twist_coefficients, num_physical_dimensions=1)

        # TODO: sweep considerations
        translation_x_coefficients = csdl.Variable(name=f'{self._name}_translation_x_coefficients', shape=(1,), value=np.array([0.]))
        translation_x_b_spline = bsp.BSpline(name=f'{self._name}_translation_x_b_spline', space=self._constant_b_spline_curve_1_dof_space,
                                                coefficients=translation_x_coefficients, num_physical_dimensions=1)

        # TODO: Dihedral needs linear or higher order
        translation_z_coefficients = csdl.Variable(name=f'{self._name}_translation_z_coefficients', shape=(1,), value=np.array([0.]))
        translation_z_b_spline = bsp.BSpline(name=f'{self._name}_translation_z_b_spline', space=self._constant_b_spline_curve_1_dof_space,
                                                coefficients=translation_z_coefficients, num_physical_dimensions=1)
        

        # Declaring the states (i.e., the coefficients) to the parameterization solver
        parameterization_solver.declare_state(name=f'{self._name}_chord_stretch_coefficients', state=chord_stretch_coefficients)
        parameterization_solver.declare_state(name=f'{self._name}_span_stretch_coefficients', state=span_stretch_coefficients, penalty_factor=1.e3)
        parameterization_solver.declare_state(name=f'{self._name}_translation_x_coefficients', state=translation_x_coefficients)
        parameterization_solver.declare_state(name=f'{self._name}_translation_z_coefficients', state=translation_z_coefficients)
        if self._optimize_wing_twist:
            parameterization_solver.declare_state(name=f'{self._name}_twist_coefficients', state=twist_coefficients)

        # Evaluate the B-splines to obtain sectional parameters
        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
        sectional_chord_stretch = chord_stretch_b_spline.evaluate(section_parametric_coordinates)
        sectional_span_stretch = span_stretch_b_spline.evaluate(section_parametric_coordinates)
        sectional_translation_x = translation_x_b_spline.evaluate(section_parametric_coordinates)
        sectional_translation_z = translation_z_b_spline.evaluate(section_parametric_coordinates)
        if self._optimize_wing_twist:
            sectional_wing_twist = twist_b_spline.evaluate(section_parametric_coordinates)

        # Store sectional parameters
        sectional_parameters = {
            f'sectional_{self._name}_chord_stretch':sectional_chord_stretch,
            f'sectional_{self._name}_span_stretch':sectional_span_stretch,
            f'sectional_{self._name}_translation_x' : sectional_translation_x, 
            f'sectional_{self._name}_translation_z' : sectional_translation_z,
        }

        if self._optimize_wing_twist:
            sectional_parameters[f'sectional_{self._name}_twist'] = sectional_wing_twist,

        # Evaluate entire FFD block based on sectional parameters
        wing_ffd_block_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=True)
        wing_coefficients = ffd_block.evaluate(wing_ffd_block_coefficients, plot=True)

        # Assign the coefficients to the top-level parent (i.e., system) geometry
        if isinstance(self.geometry, FunctionSet):
            system_geometry.assign_coefficients(coefficients=wing_coefficients)
        else:
            system_geometry.assign_coefficients(coefficients=wing_coefficients, b_spline_names=self.geometry.b_spline_names)

        # Store information for update call after the inner optimization has been evaluated
        self._update_dict = {}
        self._update_dict[f'{self._name}_ffd_block'] = ffd_block
        self._update_dict[f'{self._name}_component'] = self.geometry
        self._update_dict[f'{self._name}_ffd_sectional_parameterization'] = ffd_block_sectional_parameterization
        self._update_dict[f'{self._name}_coefficient_names'] = [
            f'{self._name}_chord_stretch_coefficients',
            f'{self._name}_span_stretch_coefficients',
            f'{self._name}_translation_x_coefficients',
            f'{self._name}_translation_z_coefficients',
        ]
        self._update_dict[f'{self._name}_b_splines'] = [
            chord_stretch_b_spline,
            span_stretch_b_spline,
            translation_x_b_spline,
            translation_z_b_spline,
        ]
        self._update_dict[f'{self._name}_section_parametric_coordinates'] = section_parametric_coordinates
        self._update_dict[f'{self._name}_sectional_parameter_names'] = [
            f'sectional_{self._name}_chord_stretch',
            f'sectional_{self._name}_span_stretch',
            f'sectional_{self._name}_translation_x',
            f'sectional_{self._name}_translation_z'
        ]

    def _extract_geometric_quantities_from_ffd_block(self, ffd_block : FFDBlock, system_geometry, plot : bool=False):
        """Extract the following quantities from the FFD block:
            - Span
            - root chord length
            - tip chord length

        Note that this helper function will not work well in all cases (e.g.,
        in cases with high sweep or taper)
        """
        ffd_block_coefficients = ffd_block.coefficients.value

        B_matrix_LE_center = ffd_block.compute_evaluation_map(np.array([0., 0.5, 0.5]))
        B_matrix_TE_center = ffd_block.compute_evaluation_map(np.array([1., 0.5, 0.5]))
        LE_center_point = B_matrix_LE_center @ ffd_block_coefficients
        TE_center_point = B_matrix_TE_center @ ffd_block_coefficients

        LE_center = self.geometry.project(LE_center_point, plot=plot)
        TE_center = self.geometry.project(TE_center_point, plot=plot)


        B_matrix_LE_left = ffd_block.compute_evaluation_map(np.array([0., 0., 0.5]))
        B_matrix_TE_left = ffd_block.compute_evaluation_map(np.array([1., 0., 0.5]))
        LE_left_point = B_matrix_LE_left @ ffd_block_coefficients
        TE_left_point = B_matrix_TE_left @ ffd_block_coefficients

        LE_left = self.geometry.project(LE_left_point, plot=plot)
        TE_left = self.geometry.project(TE_left_point, plot=plot)


        B_matrix_LE_right = ffd_block.compute_evaluation_map(np.array([0., 1., 0.5]))
        B_matrix_TE_right = ffd_block.compute_evaluation_map(np.array([1., 1., 0.5]))
        LE_right_point = B_matrix_LE_right @ ffd_block_coefficients
        TE_right_point = B_matrix_TE_right @ ffd_block_coefficients

        LE_right = self.geometry.project(LE_right_point, plot=plot)
        TE_right = self.geometry.project(TE_right_point, plot=plot)


        B_matrix_span_right = ffd_block.compute_evaluation_map(np.array([0.5, 1., 1.]))
        B_matrix_span_left = ffd_block.compute_evaluation_map(np.array([0.5, 0., 1.]))
        span_right_point = B_matrix_span_right @ ffd_block_coefficients
        span_left_point = B_matrix_span_left @ ffd_block_coefficients

        span_right = self.geometry.project(span_right_point, plot=plot)
        span_left = self.geometry.project(span_left_point, plot=plot)


        span = system_geometry.evaluate(span_right) - system_geometry.evaluate(span_left)
        center_chord = system_geometry.evaluate(TE_center) -system_geometry.evaluate(LE_center)
        left_tip_chord = system_geometry.evaluate(TE_left) - system_geometry.evaluate(LE_left) 
        right_tip_chord = system_geometry.evaluate(TE_right) - system_geometry.evaluate(LE_right)

        return csdl.norm(span), csdl.norm(center_chord), csdl.norm(left_tip_chord), csdl.norm(right_tip_chord)

    def _setup_ffd_parameterization(self, parameterization_solver : ParameterizationSolver, span, 
                                    center_chord, tip_chord_right, tip_chord_left):
        """Set up the wing parameterization."""

        # Declare quantities that the inner optimization will aim to enforce
        parameterization_solver.declare_input(name=f'{self._name}_span', input=span)
        parameterization_solver.declare_input(name=f'{self._name}_root_chord', input=center_chord)
        parameterization_solver.declare_input(name=f'{self._name}_tip_chord_right', input=tip_chord_right)
        parameterization_solver.declare_input(name=f'{self._name}_tip_chord_left', input=tip_chord_left)
        
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
                
            wing_span_input = (self.parameters.AR * self.parameters.S_ref)**0.5
            wing_root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * wing_span_input)
            wing_tip_chord_left_input = wing_root_chord_input * taper_ratio 
            wing_tip_chord_right_input = wing_tip_chord_left_input * 1

        elif self.parameters.S_ref is not None and self.parameters.span is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio

            if not isinstance(self.parameters.span , csdl.Variable):
                self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

            if not isinstance(self.parameters.S_ref , csdl.Variable):
                self.parameters.S_ref = csdl.Variable(shape=(1, ), value=self.parameters.S_ref)

            wing_span_input = self.parameters.span
            wing_root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * wing_span_input)
            wing_tip_chord_left_input = wing_root_chord_input * taper_ratio 
            wing_tip_chord_right_input = wing_tip_chord_left_input * 1


        # Rename variable and operation names to match the declare_input name
        wing_span_input.name = f'{self._name}_span'
        if wing_span_input.operation is not None:
            wing_span_input.operation.output_name = f'{self._name}_span'
        
        wing_root_chord_input.name = f'{self._name}_root_chord'
        wing_root_chord_input.operation.output_name = f'{self._name}_root_chord'
        
        wing_tip_chord_left_input.name = f'{self._name}_tip_chord_left'
        wing_tip_chord_left_input.operation.output_name = f'{self._name}_tip_chord_left'
        
        wing_tip_chord_right_input.name = f'{self._name}_tip_chord_right'
        wing_tip_chord_right_input.operation.output_name = f'{self._name}_tip_chord_right'
    
        # Store the parameterization inputs in a dictionary 
        parameterization_inputs = {}
        parameterization_inputs[f'{self._name}_span'] = wing_span_input
        parameterization_inputs[f'{self._name}_root_chord'] = wing_root_chord_input
        parameterization_inputs[f'{self._name}_tip_chord_right'] = wing_tip_chord_right_input
        parameterization_inputs[f'{self._name}_tip_chord_left'] = wing_tip_chord_left_input

        return parameterization_inputs

    def _setup_geometry(self, parameterization_solver, system_geometry, plot=False):
        """Set up the wing geometry (mainly the FFD)"""
        # Get/ Make the ffd block
        wing_ffd_block = self._ffd_block #self._make_ffd_block(self.geometry)

        # Set up the ffd block
        self._setup_ffd_block(wing_ffd_block, parameterization_solver, system_geometry)

        # Get wing geometric quantities (as m3l/csdl variable)
        span, center_chord, left_tip_chord, right_tip_chord = \
            self._extract_geometric_quantities_from_ffd_block(wing_ffd_block, system_geometry, plot=plot)

        # Get the parameterization inputs dictionary
        parameterization_inputs = self._setup_ffd_parameterization(parameterization_solver, span, 
                                                                   center_chord, right_tip_chord, left_tip_chord)
        
        return parameterization_inputs
        
         
