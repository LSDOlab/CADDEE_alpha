from CADDEE_alpha.core.component import Component
from CADDEE_alpha.core.mesh.mesh import MeshContainer
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
import time


@dataclass
class WingParameters:
    AR : Union[float, int, csdl.Variable]
    S_ref : Union[float, int, csdl.Variable]
    span : Union[float, int, csdl.Variable]
    sweep : Union[float, int, csdl.Variable]
    incidence : Union[float, int, csdl.Variable]
    taper_ratio : Union[float, int, csdl.Variable]
    dihedral : Union[float, int, csdl.Variable]
    root_twist_delta : Union[int, float, csdl.Variable, None]
    tip_twist_delta : Union[int, float, csdl.Variable, None]
    thickness_to_chord : Union[float, int, csdl.Variable] = 0.15
    thickness_to_chord_loc : float = 0.3
    MAC: Union[float, None] = None
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
    def __init__(
        self, 
        AR : Union[int, float, csdl.Variable, None], 
        S_ref : Union[int, float, csdl.Variable, None],
        span : Union[int, float, csdl.Variable, None] = None, 
        dihedral : Union[int, float, csdl.Variable, None] = None, 
        sweep : Union[int, float, csdl.Variable, None] = None, 
        taper_ratio : Union[int, float, csdl.Variable, None] = None,
        incidence : Union[int, float, csdl.Variable] = 0, 
        root_twist_delta : Union[int, float, csdl.Variable] = 0,
        tip_twist_delta : Union[int, float, csdl.Variable] = 0,
        thickness_to_chord: float = 0.15,
        thickness_to_chord_loc: float = 0.3,
        geometry : Union[lfs.FunctionSet, None]=None,
        tight_fit_ffd: bool = False,
        orientation: str = "horizontal",
        **kwargs
    ) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry=geometry, **kwargs)
        
        # Do type checking 
        csdl.check_parameter(AR, "AR", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(S_ref, "S_ref", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(span, "span", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(dihedral, "dihedral", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(sweep, "sweep", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(incidence, "incidence", types=(int, float, csdl.Variable))
        csdl.check_parameter(taper_ratio, "taper_ratio", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(root_twist_delta, "root_twist_delta", types=(int, float, csdl.Variable))
        csdl.check_parameter(tip_twist_delta, "tip_twist_delta", types=(int, float, csdl.Variable))
        csdl.check_parameter(orientation, "orientation", values=["horizontal", "vertical"])

        # Check if wing is over-parameterized
        if all(arg is not None for arg in [AR, S_ref, span]):
            raise Exception("Wing comp over-parameterized: Cannot specifiy AR, S_ref, and span at the same time.")
        # Check if wing is under-parameterized
        if sum(1 for arg in [AR, S_ref, span] if arg is None) >= 2:
            raise Exception("Wing comp under-parameterized: Must specify two out of three: AR, S_ref, and span.")
        
        if orientation == "vertical" and dihedral is not None:
            raise ValueError("Cannot specify dihedral for vertical wing.")
        
        if incidence is not None:
            if incidence != 0.:
                raise NotImplementedError("incidence has not yet been implemented")

        self._name = f"wing_{self._instance_count}"
        self._tight_fit_ffd = tight_fit_ffd
        self._orientation = orientation
        
        # Assign parameters
        self.parameters : WingParameters =  WingParameters(
            AR=AR,
            S_ref=S_ref,
            span=span,
            sweep=sweep,
            incidence=incidence,
            dihedral=dihedral,
            taper_ratio=taper_ratio,
            root_twist_delta=root_twist_delta,
            tip_twist_delta=tip_twist_delta,
            thickness_to_chord=thickness_to_chord,
            thickness_to_chord_loc=thickness_to_chord_loc,
        )

        # Compute MAC (i.e., characteristic length)
        if taper_ratio is None:
            taper_ratio = 1
        if AR is not None and S_ref is not None:
            lam = taper_ratio
            span = (AR * S_ref)**0.5
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif S_ref is not None and span is not None:
            lam = taper_ratio
            span = self.parameters.span
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif span is not None and AR is not None:
            lam = taper_ratio
            S_ref = span**2 / AR
            self.parameters.S_ref = S_ref
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC

        # Compute form factor according to Raymer 
        # (ignoring Mach number; include in drag build up model)
        x_c_m = self.parameters.thickness_to_chord_loc
        t_o_c = self.parameters.thickness_to_chord

        if t_o_c is None:
            t_o_c = 0.15
        if sweep is None:
            sweep = 0.

        FF = (1 + 0.6 / x_c_m + 100 * (t_o_c) ** 4) * csdl.cos(sweep) ** 0.28
        self.quantities.drag_parameters.form_factor = FF

        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (lfs.FunctionSet)):
                raise TypeError(f"wing gometry must be of type {lfs.FunctionSet}")
            else:
                # Set the wetted area
                self.parameters.S_wet = self.quantities.surface_area

                t3 = time.time()
                # Make the FFD block upon instantiation
                ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd, degree=(1, 2, 1), num_coefficients=(2, 2, 2))
                t4 = time.time()
                print("time for making ffd_block", t4-t3)
                # ffd_block.plot()

                # Compute the corner points of the wing 
                if self._orientation == "horizontal":
                    self._LE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0., 0.5])), plot=False, extrema=True)
                    self._LE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])), plot=False, extrema=True)
                    self._LE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 1.0, 0.5])), plot=False, extrema=True)

                    self._TE_left_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.5])), plot=False, extrema=True)
                    self._TE_mid_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])),  plot=False, extrema=True)
                    self._TE_right_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 1.0, 0.5])), plot=False, extrema=True)

                    self.LE_left_tip = geometry.evaluate(self._LE_left_point)
                    self.LE_right_tip = geometry.evaluate(self._LE_right_point)

                    self.TE_left_tip = geometry.evaluate(self._TE_left_point)
                    self.TE_right_tip = geometry.evaluate(self._TE_right_point)


                    self.LE_center = geometry.evaluate(self._LE_mid_point)
                    self.TE_center = geometry.evaluate(self._TE_mid_point)

                else:
                    self._LE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)
                    self._LE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 1.])), direction=np.array([-1., 0., 0.]), plot=False, extrema=False)

                    self._TE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.])), plot=False, extrema=True)
                    self._TE_root_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 1.])), plot=False, extrema=True)

                    self.LE_root = geometry.evaluate(self._LE_root_point)
                    self.TE_root = geometry.evaluate(self._TE_root_point)

                    # print(self.geometry.evaluate(self._LE_tip_point).value)
                    # print(self.geometry.evaluate(self._LE_root_point).value)
                    # print(self.geometry.evaluate(self._TE_tip_point).value)
                    # print(self.geometry.evaluate(self._TE_root_point).value)
                    # exit()

                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=False)

                # print("time for computing corner points", t6-t5)
            # internal geometry projection info
            self._dependent_geometry_points = [] # {'parametric_points', 'function_space', 'fitting_coords', 'mirror'}
            self._base_geometry = self.geometry.copy()

    def actuate(self, angle : Union[float, int, csdl.Variable], axis_location : float=0.25, mesh_container : MeshContainer = None):
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

        # # Re-evaluate all the discretizations associated with the wing
        # for discretization_name, discretization in self._discretizations.items():
        #     discretization._geom = wing_geometry
        #     try:
        #         discretization = discretization._update()
        #         self._discretizations[discretization_name] = discretization
        #     except AttributeError:
        #         raise Exception(f"The discretization {discretization_name} does not have an '_update' method, which is neded to" + \
        #                         " re-evaluate the geometry/meshes after the geometry coefficients have been changed")
            

        #     # Update the meshes in the mesh container
        #     if mesh_container is not None:
        #         print("mesh_container", mesh_container.items())
        #         for mesh_name, mesh in mesh_container.items():
        #             for discretization_name_mesh, discretization_mesh in mesh.discretizations.items():
        #                 if discretization_mesh.identifier == discretization.identifier:
        #                     mesh.discretizations[discretization_name_mesh] = discretization


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
            if self._orientation == "horizontal":
                num_coefficients = (2, 3, 2) # NOTE: hard coding here might be limiting
            else:
                degree = (1, 1, 1)
                num_coefficients = (2, 2, 2)
            ffd_block = construct_ffd_block_around_entities(name=self._name, entities=entities,
                                                            num_coefficients=num_coefficients, degree=degree)
        
        ffd_block.coefficients.name = f'{self._name}_coefficients'

        return ffd_block 
    
    def _setup_ffd_block(self, ffd_block, parameterization_solver, plot : bool=False):
        """Set up the wing ffd block."""

        if self._orientation == "horizontal":
            principal_parametric_dimension = 1
        else:
            principal_parametric_dimension = 2

        
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=principal_parametric_dimension,
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

        if self.parameters.sweep is not None:
            sweep_translation_b_spline = lfs.Function(
                space=self._linear_b_spline_3_dof_space,
                coefficients=csdl.ImplicitVariable(
                    shape=(3, ),
                    value=np.array([0., 0., 0.,]),
                ),
                name=f"{self._name}_sweep_transl_b_sp_coeffs"
            )

        if self.parameters.dihedral is not None:
            dihedral_translation_b_spline = lfs.Function(
                space=self._linear_b_spline_3_dof_space,
                coefficients=csdl.ImplicitVariable(
                    shape=(3, ),
                    value=np.array([0., 0., 0.,]),
                ),
                name=f"{self._name}_dihedral_transl_b_sp_coeffs"
            )

        coefficients=csdl.Variable(
                shape=(3, ),
                value=np.array([0., 0., 0.,]),
        )
        coefficients = coefficients.set(csdl.slice[0], self.parameters.tip_twist_delta)
        coefficients = coefficients.set(csdl.slice[1], self.parameters.root_twist_delta)
        coefficients = coefficients.set(csdl.slice[2], self.parameters.tip_twist_delta)
        twist_b_spline = lfs.Function(
            space=self._linear_b_spline_3_dof_space,
            coefficients=coefficients,
            name=f"{self._name}_twist_b_sp_coeffs"
        )

        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
        
        chord_stretch_sectional_parameters = chord_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        span_stretch_sectional_parameters = span_stretch_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        if self.parameters.sweep is not None:
            sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
                parametric_b_spline_inputs
            )
        if self.parameters.dihedral is not None:
            dihedral_translation_sectional_parameters = dihedral_translation_b_spline.evaluate(
                parametric_b_spline_inputs
            )
        twist_sectional_parameters = twist_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        sectional_parameters = VolumeSectionalParameterizationInputs()
        if self._orientation == "horizontal":
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=1, translation=span_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
            if self.parameters.dihedral is not None:
                sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translation_sectional_parameters)
            sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)
        else:
            sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
            sectional_parameters.add_sectional_translation(axis=2, translation=span_stretch_sectional_parameters)
            if self.parameters.sweep is not None:
                sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False) 

        # set the base coefficients
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
        self._base_geometry.set_coefficients(geometry_coefficients)
        
        # re-fit the dependent geometry points
        coeff_flip = np.eye(3)
        coeff_flip[1,1] = -1
        for item in self._dependent_geometry_points:
            fitting_points = self._base_geometry.evaluate(item['parametric_points'])
            coefficients = item['function_space'].fit(fitting_points, item['fitting_coords'])
            geometry_coefficients.append(coefficients)
            if item['mirror']:
                geometry_coefficients.append(coefficients @ coeff_flip)

        # set full geometry coefficients
        self.geometry.set_coefficients(geometry_coefficients)

        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action='j->ij')

        # Add the coefficients of all B-splines to the parameterization solver
        parameterization_solver.add_parameter(chord_stretch_b_spline.coefficients)
        parameterization_solver.add_parameter(span_stretch_b_spline.coefficients)
        if self.parameters.sweep is not None:
            parameterization_solver.add_parameter(sweep_translation_b_spline.coefficients)
        if self.parameters.dihedral is not None:
            parameterization_solver.add_parameter(dihedral_translation_b_spline.coefficients)
        parameterization_solver.add_parameter(rigid_body_translation, cost=10)

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
        if self._orientation == "horizontal":
            # Re-evaluate the corner points of the FFD block (plus center)
            # Root
            LE_center = self.geometry.evaluate(self._LE_mid_point)
            TE_center = self.geometry.evaluate(self._TE_mid_point)

            qc_center = 0.75 * LE_center + 0.25 * TE_center

            # Tip
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

        else:
            # Re-evaluate the corner points of the FFD block (plus center)
            # Root
            LE_root = self.geometry.evaluate(self._LE_root_point)
            TE_root = self.geometry.evaluate(self._TE_root_point)

            qc_root = 0.75 * LE_root + 0.25 * TE_root

            # Tip 
            LE_tip = self.geometry.evaluate(self._LE_tip_point)
            TE_tip = self.geometry.evaluate(self._TE_tip_point)

            qc_tip = 0.75 * LE_tip + 0.25 * TE_tip

            # Compute span, root/tip chords, sweep, and dihedral
            span = TE_tip - TE_root
            root_chord = TE_root - LE_root
            tip_chord = TE_tip - LE_tip
            qc_tip_root = qc_tip - qc_root
            sweep_angle = csdl.arcsin(qc_tip_root[0] / csdl.norm(qc_tip_root))


            wing_geometric_qts = WingGeometricQuantities(
                span=csdl.norm(span),
                center_chord=csdl.norm(root_chord),
                left_tip_chord=csdl.norm(tip_chord),
                right_tip_chord=None,
                sweep_angle_left=sweep_angle,
                sweep_angle_right=None,
                dihedral_angle_left=None,
                dihedral_angle_right=None,
            )

        return wing_geometric_qts

    def _setup_ffd_parameterization(self, wing_geom_qts: WingGeometricQuantities, ffd_geometric_variables):
        """Set up the wing parameterization."""
        # TODO: set up parameters as constraints

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
        
        elif self.parameters.span is not None and self.parameters.AR is not None:
            if self.parameters.taper_ratio is None:
                taper_ratio = 1.
            else:
                taper_ratio = self.parameters.taper_ratio

            if not isinstance(self.parameters.AR , csdl.Variable):
                self.parameters.AR = csdl.Variable(shape=(1, ), value=self.parameters.AR)

            if not isinstance(self.parameters.span , csdl.Variable):
                self.parameters.span = csdl.Variable(shape=(1, ), value=self.parameters.span)

            span_input = self.parameters.span
            root_chord_input = 2 * self.parameters.S_ref/((1 + taper_ratio) * span_input)
            tip_chord_left_input = root_chord_input * taper_ratio 
            tip_chord_right_input = tip_chord_left_input * 1

        else:
            raise NotImplementedError

        # Set constraints: user input - geometric qty equivalent
        if self._orientation == "horizontal":
            ffd_geometric_variables.add_variable(wing_geom_qts.span, span_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.center_chord, root_chord_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.left_tip_chord, tip_chord_left_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.right_tip_chord, tip_chord_right_input)
        else:
            ffd_geometric_variables.add_variable(wing_geom_qts.span, span_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.center_chord, root_chord_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.left_tip_chord, tip_chord_left_input)

        if self.parameters.sweep is not None:
            sweep_input = self.parameters.sweep
            ffd_geometric_variables.add_variable(wing_geom_qts.sweep_angle_left, sweep_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.sweep_angle_right, sweep_input)

        if self.parameters.dihedral is not None:
            dihedral_input = self.parameters.dihedral
            ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_left, dihedral_input)
            ffd_geometric_variables.add_variable(wing_geom_qts.dihedral_angle_right, dihedral_input)

        # print(wing_geom_qts.center_chord.value, root_chord_input.value)
        # print(wing_geom_qts.left_tip_chord.value, tip_chord_left_input.value)
        # print(wing_geom_qts.right_tip_chord.value, tip_chord_right_input.value)

        return

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
        
    def construct_ribs_and_spars(
            self, 
            geometry:lg.Geometry,
            top_geometry:lg.Geometry=None,
            bottom_geometry:lg.Geometry=None,
            num_ribs:int=None, 
            spar_locations:np.ndarray=None, 
            rib_locations:np.ndarray=None,
            LE_TE_interpolation=None,
            surf_index:int=1000,
            offset:np.ndarray=np.array([0., 0., .23]), 
            num_rib_pts:int=20, 
            plot_projections:bool=False,
            spar_function_space:lfs.FunctionSpace=None,
            rib_function_space:lfs.FunctionSpace=None,
            full_length_ribs:bool=False,
            finite_te:bool=True,
            export_wing_box:bool=False,
            export_half_wing:bool=False,
            spanwise_multiplicity:int=1,
            exclute_te:bool=False,
        ):
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
        LE_TE_interpolation : str, optional
            The method of interpolating the leading and trailing edge; This might be useful for swept and curved wing geometries. Default is None
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
        full_length_ribs : bool, optional
            If true, the ribs will be extended to the root and tip of the wing. Default value is False.
        finite_te : bool, optional
            If true, the trailing edge will have a finite thickness. Default value is True.
        export_wing_box : bool, optional
            If true, a water-tight wing box geometry will be exported to a .igs file. Default value is False.
        """
        csdl.check_parameter(num_ribs, "num_ribs", types=int, allow_none=True)
        csdl.check_parameter(spar_locations, "spar_locations", types=np.ndarray, allow_none=True)
        csdl.check_parameter(rib_locations, "rib_locations", types=np.ndarray, allow_none=True)
        csdl.check_parameter(LE_TE_interpolation, "LE_TE_interpolation", values=("ellipse", None))
        csdl.check_parameter(surf_index, "surf_index", types=int)

        # Check if if spar and rib locations are between 0 and 1
        if spar_locations is not None:
            if not np.all((spar_locations > 0) & (spar_locations < 1)):
                raise ValueError("all spar locations must be between 0 and 1 (excluding the endpoints)")
        
        if rib_locations is not None:
            if not np.all((rib_locations >= 0) & (rib_locations <= 1)):
                raise ValueError("all rib locations must be between 0 and 1 (including the endpoints)")
            if spanwise_multiplicity > 1:
                raise ValueError("spanwise_multiplicity is not yet supported if rib_locations are provided")
        
        # TODO: add interpolation for spars (between ribs)
        # TODO: add surface panel creation for water-tightness
        # TODO: add export for meshing
        # TODO: add option to extend ribs to the root and tip
        
        wing = self
        if export_wing_box:
            wing_box_geometry = lg.Geometry(functions=[])
            wing_box_surf_index = 0

        if spar_locations is None:
            spar_locations = np.array([0.25, 0.75])
        if rib_locations is None:
            rib_locations = np.linspace(0, 1, num_ribs)
        if num_ribs is None:
            num_ribs = rib_locations.shape[0]
        num_spars = spar_locations.shape[0]
        num_ribs = (num_ribs-1)*spanwise_multiplicity + 1
        if spanwise_multiplicity > 1:
            rib_locations = np.linspace(0, 1, num_ribs)
        if num_spars == 1:
            raise Exception("Cannot have single spar. Provide at least two normalized spar locations.")
        if spar_function_space is None:
            spar_function_space = lfs.BSplineSpace(2, (1, 1), (num_ribs, 2))
        if rib_function_space is None:
            if full_length_ribs:
                if exclute_te:
                    rib_function_space = lfs.BSplineSpace(2, 1, (num_rib_pts*(num_spars)+1, 2))
                else:
                    rib_function_space = lfs.BSplineSpace(2, 1, (num_rib_pts*(num_spars+1)+1, 2))
            else:
                rib_function_space = lfs.BSplineSpace(2, 1, (num_rib_pts*(num_spars-1)+1, 2))

        # gather important points (right only)
        root_te = wing.geometry.evaluate(wing._TE_mid_point, non_csdl=True)
        root_le = wing.geometry.evaluate(wing._LE_mid_point, non_csdl=True)
        r_tip_te = wing.geometry.evaluate(wing._TE_right_point, non_csdl=True)
        r_tip_le = wing.geometry.evaluate(wing._LE_right_point, non_csdl=True)

        # figure out what the top and bottom surfaces are via projections
        if top_geometry is None or bottom_geometry is None:
            eps = 1e-1
            num_pts = 20
            root_center = (root_te + root_le) / 2 + [0, eps, 0]
            r_tip_center = (r_tip_te + r_tip_le) / 2 - [0, eps, 0]
            center_line = np.linspace(root_center, r_tip_center, num_pts)
            top_projection_points = center_line+offset
            bottom_projection_points = center_line-offset
            top_points_parametric = wing.geometry.project(top_projection_points)
            bottom_points_parametric = wing.geometry.project(bottom_projection_points)

            top_surfaces = []
            bottom_surfaces = []
            for i in range(num_pts):
                top_ind, point = top_points_parametric[i]
                bottom_ind, point = bottom_points_parametric[i]
                if not top_ind in top_surfaces:
                    top_surfaces.append(top_ind)
                if not bottom_ind in bottom_surfaces:
                    bottom_surfaces.append(bottom_ind)
            top_surfaces = [wing.geometry.function_names[i] for i in top_surfaces]
            bottom_surfaces = [wing.geometry.function_names[i] for i in bottom_surfaces]
            if top_geometry is None:
                top_geometry = wing.create_subgeometry(top_surfaces)
            if bottom_geometry is None:
                bottom_geometry = wing.create_subgeometry(bottom_surfaces)






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
        
        if LE_TE_interpolation is not None or full_length_ribs:
            # need leading and trailing edge points for each rib in eiter case
            from scipy.interpolate import interp1d
            LE_TE_points = np.zeros((2, num_ribs, 3))
            interp_y = spar_projection_points[0, :, 1] 
            
            for i in range(2):
                if i == 0:
                    left_point = self.geometry.evaluate(self._LE_left_point, non_csdl=True)
                    mid_point = self.geometry.evaluate(self._LE_mid_point, non_csdl=True)
                    right_point = self.geometry.evaluate(self._LE_right_point, non_csdl=True)
                else:
                    left_point = self.geometry.evaluate(self._TE_left_point, non_csdl=True)
                    mid_point = self.geometry.evaluate(self._TE_mid_point, non_csdl=True)
                    right_point = self.geometry.evaluate(self._TE_right_point, non_csdl=True)

                array_to_project = np.zeros((num_ribs, 3))
                y = np.array([left_point[1], mid_point[1], right_point[1]])
                z = np.array([left_point[2], mid_point[2], right_point[2]])
                fz = interp1d(y, z, kind="linear")

                # Set up equation for an ellipse
                h = left_point[0]
                b = 2 * (h - mid_point[0]) # Semi-minor axis
                a = right_point[1] # semi-major axis
                if i == 0:
                    array_to_project[:, 0] = (b**2 * (1 - interp_y**2/a**2))**0.5 + h
                else:
                    array_to_project[:, 0] = -(b**2 * (1 - interp_y**2/a**2))**0.5 + h
                array_to_project[:, 1] = interp_y
                array_to_project[:, 2] = fz(interp_y)

                LE_TE_points[i, :, :] = array_to_project
            LE_TE_points_parametric = self.geometry.project(LE_TE_points)
            LE_TE_points_eval = self.geometry.evaluate(LE_TE_points_parametric, non_csdl=True).reshape((2, num_ribs, 3))
            LE_TE_points_parametric = np.array(LE_TE_points_parametric, dtype='O,O').reshape((2, num_ribs))

        if LE_TE_interpolation is not None:
            root_tip_pts = np.zeros((num_spars, num_ribs, 3))
            for i in range(num_spars):
                root_tip_pts[i, :, 0] = (1-spar_locations[i]) * LE_TE_points_eval[0, :, 0] + spar_locations[i] * LE_TE_points_eval[-1, :, 0]
                root_tip_pts[i, :, 1] = (LE_TE_points_eval[0, :, 1] + LE_TE_points_eval[-1, :, 1]) / 2 # Take the average between LE and TE for y
                root_tip_pts[i, :, 2] = (LE_TE_points_eval[0, :, 2] + LE_TE_points_eval[-1, :, 2]) / 2 # Take the average between LE and TE for z

            root_tip_pts[:, :, 1] = np.mean(root_tip_pts[:, :, 1], axis=0)

            spar_projection_points = root_tip_pts


        # make projection points for ribs
        # TODO: vectorize across ribs?
        if full_length_ribs:
            # add the leading and trailing edge points as if they're spar locations
            rib_projection_base = np.zeros((num_spars+2, num_ribs, 3))
            rib_projection_base[1:-1] = spar_projection_points
            le_points = LE_TE_points_eval[0, :, :]
            te_points = LE_TE_points_eval[-1, :, :]
            le_points[:, 1] = te_points[:, 1] = (le_points[:, 1] + te_points[:, 1]) / 2
            rib_projection_base[0] = le_points
            rib_projection_base[-1] = te_points
        else:
            rib_projection_base = spar_projection_points

        chord_n = rib_projection_base.shape[0]
        # rib_projection_points = np.zeros((num_ribs, (chord_n-1)+1, 3))
        # for i in range(num_ribs):
        #     rib_projection_points[i,0] = rib_projection_base[0,i]
        #     for j in range(chord_n-1):
        #         if j == 0 and full_length_ribs:
        #             # cosine spacing near the leading edge
        #             spacing = (1-np.cos(np.linspace(0, np.pi/2, num_rib_pts+1)))**0.7
        #             for k in range(num_rib_pts):
        #                 rib_projection_points[i,j*num_rib_pts+1+k] = rib_projection_base[j,i] + spacing[k+1] * (rib_projection_base[j+1,i] - rib_projection_base[j,i])
        #         else:
        #             rib_projection_points[i,j*num_rib_pts+1:(j+1)*num_rib_pts+1] = np.linspace(rib_projection_base[j,i], rib_projection_base[j+1,i], num_rib_pts+1)[1:]

        direction = np.array([0., 0., 1.])
        ribs_top = top_geometry.project(rib_projection_base+offset, direction=direction, grid_search_density_parameter=10, plot=plot_projections)
        ribs_bottom = bottom_geometry.project(rib_projection_base-offset, direction=direction, grid_search_density_parameter=10, plot=plot_projections)



        ribs_top_base_array = np.array(ribs_top, dtype='O,O').reshape((chord_n, num_ribs))   # it's easier to keep track of this way
        ribs_bottom_base_array = np.array(ribs_bottom, dtype='O,O').reshape((chord_n, num_ribs))
        ribs_top_array = np.empty((num_ribs, (chord_n-1)*num_rib_pts+1), dtype='O,O')
        ribs_bottom_array = np.empty((num_ribs, (chord_n-1)*num_rib_pts+1), dtype='O,O')
        for i in range(chord_n-1):
            for j in range(num_ribs):
                top_ind = ribs_top_base_array[i,j][0]
                bottom_ind = ribs_bottom_base_array[i,j][0]
                top_linspace = np.linspace(ribs_top_base_array[i,j][1], ribs_top_base_array[i+1,j][1], num_rib_pts)
                bottom_linspace = np.linspace(ribs_bottom_base_array[i,j][1], ribs_bottom_base_array[i+1,j][1], num_rib_pts)
                ribs_top_array[j,i*num_rib_pts:(i+1)*num_rib_pts] = np.array([(top_ind, x) for x in top_linspace], dtype='O,O')
                ribs_bottom_array[j,i*num_rib_pts:(i+1)*num_rib_pts] = np.array([(bottom_ind, x) for x in bottom_linspace], dtype='O,O')
        ribs_top_array[:,-1] = ribs_top_base_array[-1,:]
        ribs_bottom_array[:,-1] = ribs_bottom_base_array[-1,:]

        if plot_projections:
            wing.geometry.evaluate(ribs_top_array.flatten().tolist(), plot=True, non_csdl=True)
            wing.geometry.evaluate(ribs_bottom_array.flatten().tolist(), plot=True, non_csdl=True)


        
        if full_length_ribs:
            # Make top and bottom leading edge point the same
            ribs_top_array[:,0] = ribs_bottom_array[:,0]
            if not finite_te:
                # Make top and bottom trailing edge point the same
                ribs_top_array[:,-1] = ribs_bottom_array[:,-1]
        if exclute_te:
            ribs_top_array = ribs_top_array[:,:-num_rib_pts]
            ribs_bottom_array = ribs_bottom_array[:,:-num_rib_pts]
        
        # create spars
        coeff_flip = np.eye(3)
        coeff_flip[1,1] = -1
        if full_length_ribs:
            if False:
                spar_range = range(1, num_spars)
            else:
                spar_range = range(1, num_spars+1)
        else:
            spar_range = range(num_spars)

        for i in spar_range:
            parametric_points = ribs_top_array[:,i*num_rib_pts].tolist() + ribs_bottom_array[:,i*num_rib_pts].tolist()
            u_coords = np.linspace(0, 1, num_ribs)
            fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
            spar, right_spar = self._fit_surface(parametric_points, fitting_coords, spar_function_space, True, True)
            self._add_geometry(surf_index, spar, "Wing_l_spar_", i)
            surf_index = self._add_geometry(surf_index, spar, "Wing_l_spar_", i, geometry)
            self._add_geometry(surf_index, right_spar, "Wing_r_spar_", i)
            surf_index = self._add_geometry(surf_index, right_spar, "Wing_r_spar_", i, geometry)

        if exclute_te:
            chord_n -= 1

        # create ribs
        for i in range(0, num_ribs, spanwise_multiplicity):
            parameteric_points = ribs_top_array[i].tolist() + ribs_bottom_array[i].tolist()
            u_coords = np.linspace(0, 1, ribs_top_array.shape[1])
            fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
            out = self._fit_surface(parameteric_points, fitting_coords, rib_function_space, i>0, True)
            self._add_geometry(surf_index, out, "Wing_rib_", i)
            surf_index = self._add_geometry(surf_index, out, "Wing_rib_", i, geometry)

            if export_wing_box:
                # Make different surfaces for each rib segment between the spars
                u_coords = np.linspace(0, 1, num_rib_pts+1)
                fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
                rib_panel_function_space = lfs.BSplineSpace(2, 1, (num_rib_pts+1, 2))
                for j in range(chord_n-1):
                    top_parametric_points = ribs_top_array[i,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist()
                    bottom_parametric_points = ribs_bottom_array[i,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist()
                    surfs = self._fit_surface(top_parametric_points+bottom_parametric_points, fitting_coords, rib_panel_function_space, i>0 and not export_half_wing, False)
                    wing_box_surf_index = self._add_geometry(wing_box_surf_index, surfs, "Wing_rib_panel_", i, wing_box_geometry)

                if i > 0:
                    # create surface panels
                    panel_function_space = lfs.BSplineSpace(2, (1, 1), (num_rib_pts+1, spanwise_multiplicity+1))
                    u_coords = np.linspace(0, 1, num_rib_pts+1)
                    fitting_coords = []
                    for j in range(spanwise_multiplicity+1):
                        fitting_coords += [[u, j/spanwise_multiplicity] for u in u_coords]

                    fitting_coords = np.array(fitting_coords)
                    for j in range(chord_n-1):
                        top_parametric_points = []
                        bottom_parametric_points = []
                        for k in range(spanwise_multiplicity+1):
                            top_parametric_points += ribs_top_array[i-k,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist()
                            bottom_parametric_points += ribs_bottom_array[i-k,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist()
                        # top_parametric_points = (ribs_top_array[i-1,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist() +
                        #                          ribs_top_array[i,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist())
                        # bottom_parametric_points = (ribs_bottom_array[i-1,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist() +
                        #                             ribs_bottom_array[i,j*num_rib_pts:(j+1)*num_rib_pts+1].tolist())
                        top_surfs = self._fit_surface(top_parametric_points, fitting_coords, panel_function_space, not export_half_wing, False)
                        wing_box_surf_index = self._add_geometry(wing_box_surf_index, top_surfs, "Wing_top_panel_", i, wing_box_geometry)
                        bottom_surfs = self._fit_surface(bottom_parametric_points, fitting_coords, panel_function_space, not export_half_wing, False)
                        wing_box_surf_index = self._add_geometry(wing_box_surf_index, bottom_surfs, "Wing_bottom_panel_", i, wing_box_geometry)

                    # create spar segments
                    spar_segment_function_space = lfs.BSplineSpace(2, (1, 1), (spanwise_multiplicity+1, 2))
                    # spar_segment_function_space = lfs.BSplineSpace(2, 1, (2, 2))
                    if full_length_ribs:
                        if finite_te and not exclute_te:
                            spar_range = range(1, num_spars+2)
                        else:
                            spar_range = range(1, num_spars+1)
                    else:
                        spar_range = range(0, num_spars)
                    fitting_coords = []
                    for j in range(spanwise_multiplicity+1):
                        fitting_coords += [[0, j/spanwise_multiplicity], [1, j/spanwise_multiplicity]]
                    for j in spar_range:
                        top_parametric_points = []
                        bottom_parametric_points = []
                        top_fitting_coords = []
                        bottom_fitting_coords = []
                        for k in range(spanwise_multiplicity+1):
                            top_parametric_points.append(ribs_top_array[i-k,j*num_rib_pts])
                            bottom_parametric_points.append(ribs_bottom_array[i-k,j*num_rib_pts])
                            top_fitting_coords += [k/spanwise_multiplicity, 0.]
                            bottom_fitting_coords += [k/spanwise_multiplicity, 1.]
                        # top_parametric_points = [ribs_top_array[i-2,j*num_rib_pts], ribs_top_array[i-1,j*num_rib_pts], ribs_top_array[i,j*num_rib_pts]]
                        # bottom_parametric_points = [ribs_bottom_array[i-2,j*num_rib_pts], ribs_bottom_array[i-1,j*num_rib_pts], ribs_bottom_array[i,j*num_rib_pts]]
                        # fitting_coords = np.array([[0., 0.], [0.5, 0], [1., 0.], [0., 1.], [0.5, 1.], [1., 1.]])
                        fitting_coords = np.array(top_fitting_coords + bottom_fitting_coords)
                        surfs = self._fit_surface(top_parametric_points+bottom_parametric_points, fitting_coords, spar_segment_function_space, not export_half_wing, False)
                        wing_box_surf_index = self._add_geometry(wing_box_surf_index, surfs, "Wing_spar_segment_", i, wing_box_geometry)

        if export_wing_box:
            wing_box_geometry.plot(opacity=0.5)
            wing_box_geometry.export_iges("wing_box.igs")
         
    def _fit_surface(self, parametric_points:list, fitting_coords:list, function_space:lfs.FunctionSpace, mirror:bool, dependent:bool):
        """Fit a surface to the given parametric points."""
        if dependent:
            self._dependent_geometry_points.append({'parametric_points':parametric_points, 
                                                    'fitting_coords':fitting_coords,
                                                    'function_space':function_space,
                                                    'mirror':mirror})    
        fitting_values = self.geometry.evaluate(parametric_points)
        coefficients = function_space.fit(fitting_values, fitting_coords)
        function = lfs.Function(function_space, coefficients)
        if mirror:
            coeff_flip = np.eye(3)
            coeff_flip[1,1] = -1
            coefficients = coefficients @ coeff_flip
            mirror_function = lfs.Function(function_space, coefficients)
            return function, mirror_function
        return function

    def _add_geometry(self, surf_index, function, name, append=None, geometry=None):
        """Add a function to the geometry object."""
        if geometry is None:
            geometry = self.geometry
        if not isinstance(function, tuple):
            function = (function,)
        if append is None:
            append = ""
        for i, f in enumerate(function):
            geometry.functions[surf_index+i] = f
            if i > 0:
                geometry.function_names[surf_index+i] = name + str(-append)
            else:
                geometry.function_names[surf_index+i] = name + str(append)
        return surf_index + len(function)