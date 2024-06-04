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
import lsdo_geo as lg


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
                 **kwargs) -> None:
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
        
    def construct_ribs_and_spars(self, geometry:lg.Geometry, 
                             num_ribs:int=None, spar_locations:np.ndarray=None, 
                             rib_locations:np.ndarray=None, surf_index:int=1000, 
                             offset:np.ndarray=np.array([0., 0., .23]), 
                             num_rib_pts:int=20, plot_projections:bool=False,
                             spar_function_space:lfs.FunctionSpace=None,
                             rib_function_space:lfs.FunctionSpace=None):
        """
        Construct ribs and spars for the given wing geometry.

        Parameters
        ----------
        geometry : lg.Geometry
            The geometry object to which the ribs and spars will be added.
        num_ribs : int, optional
            The number of ribs to be constructed. If not provided, it will be determined based on the rib_locations array.
        spar_locations : np.ndarray, optional
            The locations of the spars along the wing span. If not provided, default values of [0.25, 0.75] will be used.
        rib_locations : np.ndarray, optional
            The locations of the ribs along the wing span. If not provided, evenly spaced values between 0 and 1 will be used.
        surf_index : int, optional
            The starting index for the surface functions in the geometry object. Default value is 1000.
        offset : np.ndarray, optional
            The offset vector to be applied for projection. Default value is [0., 0., .23].
        num_rib_pts : int, optional
            The number of interpolation points to be used for each rib. Default value is 20.
        plot_projections : bool, optional
            Whether to plot the projection results. Default value is False.
        spar_function_space : fs.FunctionSpace, optional
            The function space to be used for the spars. Defaults to a linear BSplineSpace.
        rib_function_space : fs.FunctionSpace, optional
            The function space to be used for the ribs. Defaults to a linear BSplineSpace.
        """

        # TODO: reconstruct wing with ribs and spars (eg, reconstruct the ffd block?)
        #       right now it just adds the ribs and spars to the geometry object, not the wing component (self)

        # TODO: add interpolation for spars (between ribs)
        # TODO: add surface panel creation for water-tightness
        # TODO: add export for meshing
        # TODO: add option to extend ribs to the root and tip
        
        wing = self

        if spar_locations is None:
            spar_locations = np.array([0.25, 0.75])
        if rib_locations is None:
            rib_locations = np.linspace(0, 1, num_ribs)
        if num_ribs is None:
            num_ribs = rib_locations.shape[0]
        num_spars = spar_locations.shape[0]
        if spar_function_space is None:
            spar_function_space = lfs.BSplineSpace(2, 1, (num_ribs, 2))
        if rib_function_space is None:
            rib_function_space = lfs.BSplineSpace(2, 1, (num_rib_pts*(num_spars-1)+1, 2))

        # gather important points (right only)
        root_te = wing.geometry.evaluate(wing._TE_mid_point).value
        root_le = wing.geometry.evaluate(wing._LE_mid_point).value
        r_tip_te = wing.geometry.evaluate(wing._TE_right_point).value
        r_tip_le = wing.geometry.evaluate(wing._LE_right_point).value

        # get spar start/end points (root and tip)
        root_tip_pts = np.zeros((num_spars, 2, 3))
        for i in range(num_spars):
            root_tip_pts[i,0] = (1-spar_locations[i]) * root_le + spar_locations[i] * root_te
            root_tip_pts[i,1] = (1-spar_locations[i]) * r_tip_le + spar_locations[i] * r_tip_te

        # get intersections between ribs and spars
        # we'll use these points directly to make the spars
        # TODO: vectorize across spars?
        spar_projection_points = np.zeros((num_spars, num_ribs, 3))
        for i in range(num_spars):
            for j in range(num_ribs):
                spar_projection_points[i,j] = root_tip_pts[i,0] * (1-rib_locations[j]) + root_tip_pts[i,1] * rib_locations[j]
            
        # make projection points for ribs
        # TODO: vectorize across ribs?
        rib_projection_points = np.zeros((num_ribs, num_rib_pts*(num_spars-1)+1, 3))
        for i in range(num_ribs):
            rib_projection_points[i,0] = spar_projection_points[0,i]
            for j in range(num_spars-1):
                rib_projection_points[i,j*num_rib_pts+1:(j+1)*num_rib_pts+1] = np.linspace(spar_projection_points[j,i], spar_projection_points[j+1,i], num_rib_pts+1)[1:]

        direction = np.array([0., 0., 1.])
        ribs_top = wing.geometry.project(rib_projection_points+offset, direction=-direction, grid_search_density_parameter=10, plot=plot_projections)
        ribs_bottom = wing.geometry.project(rib_projection_points-offset, direction=direction, grid_search_density_parameter=10, plot=plot_projections)
        ribs_top_array = np.array(ribs_top, dtype='O,O').reshape((num_ribs, num_rib_pts*(num_spars-1)+1))   # it's easier to keep track of this way
        ribs_bottom_array = np.array(ribs_bottom, dtype='O,O').reshape((num_ribs, num_rib_pts*(num_spars-1)+1))

        # create spars
        coeff_flip = np.eye(3)
        coeff_flip[1,1] = -1
        for i in range(num_spars):
            parametric_points = ribs_top_array[:,i*num_rib_pts].tolist() + ribs_bottom_array[:,i*num_rib_pts].tolist()
            fitting_values = wing.geometry.evaluate(parametric_points).value    # TODO: .value or no .value? - at least convert to the coefficients thing
            u_coords = np.linspace(0, 1, num_ribs)
            fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
            spar_coeffs = spar_function_space.fit(fitting_values, fitting_coords)
            spar = lfs.Function(spar_function_space, spar_coeffs)
            right_spar_coeffs = spar_coeffs @ coeff_flip
            right_spar = lfs.Function(spar_function_space, right_spar_coeffs)
            geometry.functions[surf_index] = spar
            geometry.function_names[surf_index] = "Wing_l_spar_"+str(i)
            surf_index += 1
            geometry.functions[surf_index] = right_spar
            geometry.function_names[surf_index] = "Wing_r_spar_"+str(i)
            surf_index += 1

        for i in range(num_ribs):
            parameteric_points = ribs_top_array[i].tolist() + ribs_bottom_array[i].tolist()
            fitting_values = wing.geometry.evaluate(parameteric_points).value
            u_coords = np.linspace(0, 1, ribs_top_array.shape[1])
            fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
            rib_coeffs = rib_function_space.fit(fitting_values, fitting_coords)
            rib = lfs.Function(rib_function_space, rib_coeffs)
            geometry.functions[surf_index] = rib
            geometry.function_names[surf_index] = "Wing_rib_"+str(i)
            surf_index += 1
            if i > 0:
                right_rib_coeffs = rib_coeffs @ coeff_flip
                right_rib = lfs.Function(rib_function_space, right_rib_coeffs)
                geometry.functions[surf_index] = right_rib
                geometry.function_names[surf_index] = "Wing_rib_"+str(-i)
                surf_index += 1
         
