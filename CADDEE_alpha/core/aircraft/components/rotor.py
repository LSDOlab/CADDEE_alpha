import lsdo_function_spaces as lfs
from lsdo_function_spaces import FunctionSet
from CADDEE_alpha.core.component import Component
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import numpy as np
from dataclasses import dataclass
import csdl_alpha as csdl
from typing import Union


@dataclass
class RotorParameters(csdl.VariableGroup):
    radius: Union[float, int, csdl.Variable]
    hub_radius: Union[float, int, csdl.Variable] = 0.2


class Rotor(Component):
    def __init__(self, 
                 radius: Union[int, float, csdl.Variable],
                 geometry: Union[FunctionSet, None] = None,
                 **kwargs) -> None:
        csdl.check_parameter(radius, "radius", types=(float, int, csdl.Variable))
        super().__init__(geometry, **kwargs)

        self._skip_ffd = False

        self._name = f"rotor_{self._instance_count}"
        self.parameters : RotorParameters = RotorParameters(
            radius=radius,
        )

        # Make FFd block if geometry is not None
        if self.geometry is not None:
            if not isinstance(self.geometry, (FunctionSet)):
                raise TypeError(f"wing geometry must be of type {FunctionSet}, received {type(self.geometry)}")

            else:
                # Do projections for corner points
                # Find principal axis/dim
                # u-direction
                u_dim = self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])) - \
                        self._ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5]))

                # v-direction
                v_dim = self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.5])) - \
                        self._ffd_block.evaluate(parametric_coordinates=np.array([0., 1., 0.5]))
                
                # w-direction
                w_dim = self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.])) - \
                        self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.]))
                
                # Get the dimension (i.e., size) of the FFD-block
                u_norm = csdl.norm(u_dim).value
                v_norm = csdl.norm(v_dim).value
                w_norm = csdl.norm(w_dim).value
                block_dim = np.array([u_norm, v_norm, w_norm]).flatten()

                # Get the principal direction
                # self._pr_dim = np.where(block_dim == np.max(block_dim))[0][0]

                # Get smallest dimenstion
                smllst_dim = np.where(block_dim == np.min(block_dim))[0][0]
                self._pr_dim = smllst_dim
                
                if smllst_dim == 0:
                    self._corner_point_1 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.])))
                    self._corner_point_2 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.])))
                    self._corner_point_3 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0., 0.5])))
                    self._corner_point_4 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.5])))

                elif smllst_dim == 1:
                    self._corner_point_1 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])))
                    self._corner_point_2 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])))
                    self._corner_point_3 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.])))
                    self._corner_point_4 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 1.])))

                elif smllst_dim == 2:
                    self._corner_point_1 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0., 0.5, 0.5])))
                    self._corner_point_2 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([1., 0.5, 0.5])))
                    self._corner_point_3 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0., 0.5])))
                    self._corner_point_4 = geometry.project(self._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.5])))

                else:
                    raise Exception(f"Invalid smallest dimension {smllst_dim}. Needs to be 0, 1, 2. This is unlikely to be a user error")
                
    def actuate(
            self, 
            x_tilt_angle: Union[float, int, csdl.Variable, None]=None, 
            y_tilt_angle: Union[float, int, csdl.Variable, None]=None
        ):
        rotor_geometry = self.geometry
        if rotor_geometry is None:
            raise ValueError("rotor component cannot be actuated since it does not have a geometry (i.e., geometry=None)")

        if x_tilt_angle is None and y_tilt_angle is None:
            raise ValueError("Must either specify 'x_tilt_angle' or 'y_tilt_angle'")

        if x_tilt_angle is not None:    
            axis_origin_x = rotor_geometry.evaluate(self._corner_point_1)
            axis_vector_x = axis_origin_x - rotor_geometry.evaluate(self._corner_point_2)

            rotor_geometry.rotate(axis_origin_x, axis_vector_x / csdl.norm(axis_vector_x), angles=x_tilt_angle)

        if y_tilt_angle is not None:
            axis_origin_y = rotor_geometry.evaluate(self._corner_point_4)
            axis_vector_y = axis_origin_y - rotor_geometry.evaluate(self._corner_point_3)
            
            rotor_geometry.rotate(axis_origin_y, axis_vector_y / csdl.norm(axis_vector_y), angles=y_tilt_angle)

        # Re-evaluate all the meshes associated with the wing
        for mesh_name, mesh in self._discretizations.items():
            try:
                mesh = mesh._update()
                self._discretizations[mesh_name] = mesh
                print("Update rotor mesh")
            except AttributeError:
                raise Exception(f"Mesh {mesh_name} does not have an '_update' method, which is neded to" + \
                                " re-evaluate the geometry/meshes after the geometry coefficients have been changed")


    def _setup_ffd_block(self, ffd_block, parameterization_solver, plot: bool=False):
        """Set up the rotor ffd_block"""
        # Instantiate a volume sectional parameterization object
        ffd_block_sectional_parameterization = VolumeSectionalParameterization(
            name=f'{self._name}_sectional_parameterization',
            parameterized_points=ffd_block.coefficients,
            principal_parametric_dimension=self._pr_dim
        )
        if plot:
            ffd_block.plot()
            ffd_block_sectional_parameterization.plot()
    
        # Make B-spline functions for changing geometric quantities
        
        principal_b_spline = lfs.Function(
            space=self._linear_b_spline_2_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(2, ),
                value=np.array([0., 0.]),
            ),
            name=f"{self._name}_principal_b_sp_coeffs",
        )

        non_principal_1_b_spline = lfs.Function(
            space=self._constant_b_spline_1_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(1, ),
                value=0.,
            ),
            name=f"{self._name}_non_principal_1_b_sp_coeffs",
        )

        non_principal_2_b_spline = lfs.Function(
            space=self._constant_b_spline_1_dof_space,
            coefficients=csdl.ImplicitVariable(
                shape=(1, ),
                value=0.,
            ),
            name=f"{self._name}_non_principal_2_b_sp_coeffs",
        )

        # evaluate b-splines 
        num_ffd_sections = ffd_block_sectional_parameterization.num_sections
        parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))

        principal_sectional_parameters = principal_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        non_principal_1_sectional_parameters = non_principal_1_b_spline.evaluate(
            parametric_b_spline_inputs
        )
        non_principal_2_sectional_parameters = non_principal_2_b_spline.evaluate(
            parametric_b_spline_inputs
        )

        sectional_paramaters = VolumeSectionalParameterizationInputs()
        if self._pr_dim == 0:
            sectional_paramaters.add_sectional_stretch(axis=1, stretch=non_principal_1_sectional_parameters)
            sectional_paramaters.add_sectional_stretch(axis=2, stretch=non_principal_2_sectional_parameters)

        
        elif self._pr_dim == 1:
            sectional_paramaters.add_sectional_stretch(axis=0, stretch=non_principal_1_sectional_parameters)
            sectional_paramaters.add_sectional_stretch(axis=2, stretch=non_principal_2_sectional_parameters)
        
        else:
            sectional_paramaters.add_sectional_stretch(axis=0, stretch=non_principal_1_sectional_parameters)
            sectional_paramaters.add_sectional_stretch(axis=1, stretch=non_principal_2_sectional_parameters)

        ffd_coefficients = ffd_block_sectional_parameterization.evaluate(sectional_paramaters, plot=plot)
        

        # set the coefficient in the geometry
        geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
        self.geometry.set_coefficients(geometry_coefficients)

        # Add rigid body translation (without FFD)
        rigid_body_translation = csdl.ImplicitVariable(shape=(3, ), value=0.)
        for function in self.geometry.functions.values():
            shape = function.coefficients.shape
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, shape, action='j->ij')


        # Add (B-spline) coefficients to parameterization solver
        parameterization_solver.add_parameter(non_principal_1_b_spline.coefficients)
        parameterization_solver.add_parameter(non_principal_2_b_spline.coefficients)
        parameterization_solver.add_parameter(rigid_body_translation)

        return
    
    def _extract_geometric_quantities_from_ffd_block(self):
        """Extract radius."""
        # Get corner points
        p1 = self.geometry.evaluate(self._corner_point_1, plot=False)
        p2 = self.geometry.evaluate(self._corner_point_2, plot=False)

        p3 = self.geometry.evaluate(self._corner_point_3, plot=False)
        p4 = self.geometry.evaluate(self._corner_point_4, plot=False)

        radius_1 = csdl.norm(p1-p2) / 2 
        radius_2 = csdl.norm(p3-p4) / 2


        return radius_1, radius_2

    def _setup_ffd_parameterization(self, radius_1, radius_2, ffd_geometric_variables):
        """Set up the rotor parameterization."""
        radius_input = self.parameters.radius
        ffd_geometric_variables.add_variable(radius_1, radius_input)
        ffd_geometric_variables.add_variable(radius_2, radius_input)

        return
    
    def _setup_geometry(self, parameterization_solver, ffd_geometric_variables, plot=False):
        """Set up the fuselage geometry (mainly for FFD)"""

        # Get ffd block
        fuselage_ffd_block = self._ffd_block

        # Set up the ffd block
        self._setup_ffd_block(fuselage_ffd_block, parameterization_solver)

        # Get fuselage geometric quantities
        r1, r2 = self._extract_geometric_quantities_from_ffd_block()

        # Define geometric constraints
        self._setup_ffd_parameterization(r1, r2, ffd_geometric_variables)
        
        return

# option 1
        # all components are part of one FFD block 
        # pass in multiple geometries 

# option 2
        # separate components for blades, disk and hub 


# option 3
        # define a new CompositeComponent class

# option 4
        # pass in separate geometries for different sub-components 

