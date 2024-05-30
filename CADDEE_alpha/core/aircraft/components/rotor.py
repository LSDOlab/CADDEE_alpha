from CADDEE_alpha.core.component import Component
import numpy as np
from dataclasses import dataclass
import csdl_alpha as csdl
from typing import Union
from lsdo_function_spaces import FunctionSet


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

        self._name = f"rotor_{self._instance_count}"
        self.parameters : RotorParameters = RotorParameters(
            radius=radius
        )

        # Make FFd block if geometry is not None
        if self.geometry is not None:
            self._ffd_block = self._make_ffd_block(self.geometry)

            # Do projections for corner points 
            self._corner_point_1 = geometry.project(self._ffd_block.evaluate(np.array([0.5, 0.5, 0.])))
            self._corner_point_2 = geometry.project( self._ffd_block.evaluate(np.array([0.5, 0.5, 1.])))
            self._corner_point_3 = geometry.project(self._ffd_block.evaluate(np.array([0.5, 0., 0.5])))
            self._corner_point_4 = geometry.project(self._ffd_block.evaluate(np.array([0.5, 1., 0.5])))

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
                                " re-evaluate the geometry/meshes after the geometry coefficients have been changes")


# option 1
        # all components are part of one FFD block 
        # pass in multiple geometries 

# option 2
        # separate components for blades, disk and hub 


# option 3
        # define a new CompositeComponent class

# option 4
        # pass in separate geometries for different sub-components 

