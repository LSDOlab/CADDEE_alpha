from __future__ import annotations
import csdl_alpha as csdl
from CADDEE_alpha.core.mesh.mesh import Discretization, SolverMesh, DiscretizationsDict
import numpy as np
from lsdo_function_spaces import FunctionSet
from typing import Union
from dataclasses import dataclass
from CADDEE_alpha.utils.caddee_dict import CADDEEDict
from CADDEE_alpha.utils.mesh_utils import import_mesh
import lsdo_function_spaces as fs
import lsdo_geo as lg
from scipy.interpolate import interp1d
from lsdo_function_spaces import FunctionSet, Function
import warnings
from CADDEE_alpha.utils.mesh_utils import trace_intersection


def cosine_spacing(num_pts, spanwise_points=None, chord_surface=None, num_chordwise=None, flip=False):
    if spanwise_points is not None:
        i_vec = np.arange(0, num_pts)
        half_cos = 1 - np.cos(i_vec * np.pi / (2 * (num_pts - 1)))

        half_cos_array = np.ones((num_pts, 3))
        half_cos_array[:, 0] = half_cos
        half_cos_array[:, 1] = half_cos
        half_cos_array[:, 2] = half_cos

        if flip is False:
            points = spanwise_points[-1, :] - (spanwise_points[-1, :] - spanwise_points[0, :]) * half_cos_array

        else:
            points = np.flip(spanwise_points[-1, :] - (spanwise_points[-1, :] - spanwise_points[0, :]) * half_cos_array, axis=0)

        return points

    elif chord_surface is not None:
        chord_surface = chord_surface.value.reshape((num_chordwise + 1, num_pts + 1, 3))
        i_vec = np.arange(0, num_chordwise + 1)
        half_cos = 1 - np.cos(i_vec * np.pi / (2 * (num_chordwise)))

        new_chord_surface = np.zeros((num_chordwise + 1, num_pts+ 1, 3))

        new_chord_surface[:, :, 0] = (chord_surface[0, :, 0].reshape((num_pts + 1, 1)) - (chord_surface[0, :, 0] - chord_surface[-1, :, 0]).reshape((num_pts + 1, 1)) * half_cos.reshape((1, num_chordwise+1))).T
        new_chord_surface[:, :, 1] = (chord_surface[0, :, 1].reshape((num_pts + 1, 1)) - (chord_surface[0, :, 1] - chord_surface[-1, :, 1]).reshape((num_pts + 1, 1)) * half_cos.reshape((1, num_chordwise+1))).T
        new_chord_surface[:, :, 2] = (chord_surface[0, :, 2].reshape((num_pts + 1, 1)) - (chord_surface[0, :, 2] - chord_surface[-1, :, 2]).reshape((num_pts + 1, 1)) * half_cos.reshape((1, num_chordwise+1))).T

        return new_chord_surface

def make_mesh_symmetric(quantity, num_spanwise, spanwise_index=0):
    num_spanwise_half = int(num_spanwise/2 + 1)

    symmetric_quantity = csdl.Variable(shape=quantity.shape, value=0.)
    
    for i in csdl.frange(num_spanwise_half):
        # starting from wing center moving outward
        index = int(num_spanwise/2) + i
        symmetric_index = int(num_spanwise/2) - i
        
        # take the span-wise mean for x, z 
        if spanwise_index is None:
            spanwise_mean = (quantity[index] + quantity[symmetric_index]) / 2
        
        elif spanwise_index == 0:
            spanwise_mean_x = (quantity[index, 0] + quantity[symmetric_index, 0]) / 2
            spanwise_mean_z = (quantity[index, 2] + quantity[symmetric_index, 2]) / 2
            
            # in the y-direction, take mean of the absolute values
            spanwise_mean_y = ((quantity[index, 1]**2)**0.5 + (quantity[symmetric_index, 1]**2)**0.5) / 2
        
        elif spanwise_index == 1:
            spanwise_mean_x = (quantity[:, index, 0] + quantity[:, symmetric_index, 0]) / 2
            spanwise_mean_z = (quantity[:, index, 2] + quantity[:, symmetric_index, 2]) / 2
            
            # in the y-direction, take mean of the absolute values
            spanwise_mean_y = (((quantity[:, index, 1]+1e-5)**2)**0.5 + ((quantity[:, symmetric_index, 1]+1e-5)**2)**0.5) / 2
            # spanwise_mean_y = (csdl.absolute(quantity[:, index, 1]) + csdl.absolute(quantity[:, symmetric_index, 1])) / 2

        # Populate the csdl variable with the symmetric entries
        if index == symmetric_index:
            if spanwise_index is None:
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index],
                    value=spanwise_mean,
                )
            
            elif spanwise_index == 0:
                # x
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 0],
                    value=spanwise_mean_x,
                )
                # y
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 1],
                    value=spanwise_mean_y,
                )
                # z
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 2],
                    value=spanwise_mean_z,
                )

            elif spanwise_index == 1:
                # x
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 0],
                    value=spanwise_mean_x,
                )
                # y
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 1],
                    value=spanwise_mean_y,
                )
                # z
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 2],
                    value=spanwise_mean_z,
                )

        else:
            if spanwise_index is None:
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index],
                    value=spanwise_mean,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[symmetric_index],
                    value=spanwise_mean,
                )

            elif spanwise_index == 0:
                # x
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 0],
                    value=spanwise_mean_x,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[symmetric_index, 0],
                    value=spanwise_mean_x,
                )

                # y
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 1],
                    value=spanwise_mean_y,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[symmetric_index, 1],
                    value=-spanwise_mean_y,
                )

                # z
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[index, 2],
                    value=spanwise_mean_z,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[symmetric_index, 2],
                    value=spanwise_mean_z,
                )

            if spanwise_index == 1:
                # x
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 0],
                    value=spanwise_mean_x,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, symmetric_index, 0],
                    value=spanwise_mean_x,
                )

                # y
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 1],
                    value=spanwise_mean_y,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, symmetric_index, 1],
                    value=-spanwise_mean_y,
                )

                # z
                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, index, 2],
                    value=spanwise_mean_z,
                )

                symmetric_quantity = symmetric_quantity.set(
                    slices=csdl.slice[:, symmetric_index, 2],
                    value=spanwise_mean_z,
                )
    return symmetric_quantity


@dataclass
class CamberSurface(Discretization):
    _upper_wireframe_para = None
    _lower_wireframe_para = None
    _airfoil_upper_para = None
    _airfoil_lower_para = None
    _geom = None
    _num_chord_wise = None
    _num_spanwise = None
    _LE_points_para = None
    _TE_points_para = None
    _chordwise_spacing = None

    embedded_airfoil_model_Cl = None
    embedded_airfoil_model_Cd = None
    embedded_airfoil_model_Cp = None
    embedded_airfoil_model_alpha_stall = None
    reynolds_number = None
    alpha_ML_mid_panel = None
    mid_panel_chord_length = None

    def copy(self) -> CamberSurface:
        discretization = CamberSurface(
            nodal_coordinates=self.nodal_coordinates,
        )
        discretization.nodal_velocities = self.nodal_velocities
        discretization.mesh_quality = self.mesh_quality
        discretization._has_been_expanded = self._has_been_expanded

        discretization._upper_wireframe_para = self._upper_wireframe_para
        discretization._lower_wireframe_para = self._lower_wireframe_para
        discretization._geom = self._geom
        discretization._num_chord_wise = self._num_chord_wise
        discretization._num_spanwise = self._num_spanwise
        discretization._LE_points_para = self._LE_points_para
        discretization._TE_points_para = self._TE_points_para
        discretization._chordwise_spacing = self._chordwise_spacing

        discretization.embedded_airfoil_model_Cl = self.embedded_airfoil_model_Cl
        discretization.embedded_airfoil_model_Cd = self.embedded_airfoil_model_Cd
        discretization.embedded_airfoil_model_Cp = self.embedded_airfoil_model_Cp
        discretization.embedded_airfoil_model_alpha_stall = self.embedded_airfoil_model_alpha_stall
        discretization.reynolds_number = self.reynolds_number
        discretization.alpha_ML_mid_panel = self.alpha_ML_mid_panel
        discretization.mid_panel_chord_length = self.mid_panel_chord_length

        discretization._update = self._update

        return discretization

    def project_airfoil_points(self, num_points: int=120, spacing: str="sin", 
                          grid_search_density: int = 10, plot: bool=False, oml_geometry=None):
        if self._airfoil_lower_para is not None and self._airfoil_upper_para is not None:
            return self._airfoil_lower_para, self._airfoil_upper_para
        
        if self._TE_points_para is None:
            raise ValueError("VLM surface not made yet")
        
        if self._geom is None:
            raise ValueError("VLM surface not made yet")
        
        if spacing != "sin":
            raise NotImplementedError
        
        norm_chord_wise_coordinates = 0.5 + 0.5*np.sin(np.pi*(np.linspace(0., 1., num_points)-0.5))
        
        if oml_geometry is not None:
            wing_geometry = oml_geometry
        else:
            wing_geometry = self._geom
        num_spanwise = self._num_spanwise
        
        LE_points_csdl = wing_geometry.evaluate(self._LE_points_para)
        TE_points_csdl = wing_geometry.evaluate(self._TE_points_para)
        
        y_mean_spanwise = (LE_points_csdl[:, 1] + TE_points_csdl[:, 1])/ 2 
        LE_points_csdl = LE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)
        TE_points_csdl = TE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)

        LE_points_csdl_mid_panel = (LE_points_csdl[0:-1, :] + LE_points_csdl[1:, :]) / 2
        TE_points_csdl_mid_panel = (TE_points_csdl[0:-1, :] + TE_points_csdl[1:, :]) / 2

        LE_points_csdl_mid_panel = LE_points_csdl_mid_panel.set(csdl.slice[:, 0], LE_points_csdl_mid_panel[:, 0] + 0.1)
        TE_points_csdl_mid_panel = TE_points_csdl_mid_panel.set(csdl.slice[:, 0], TE_points_csdl_mid_panel[:, 0] - 0.1)

        LE_points_re_projected = wing_geometry.evaluate(wing_geometry.project(LE_points_csdl_mid_panel, grid_search_density_parameter=grid_search_density, plot=plot))
        TE_points_re_projected = wing_geometry.evaluate(wing_geometry.project(TE_points_csdl_mid_panel, grid_search_density_parameter=grid_search_density, plot=plot))
        
        num_chordwise = len(norm_chord_wise_coordinates)
        chord_surface = csdl.linear_combination(LE_points_re_projected, TE_points_re_projected, num_chordwise)

        x_interp = csdl.Variable(shape=(num_chordwise, ), value=norm_chord_wise_coordinates)
        x_interp_exp = csdl.expand(x_interp, (num_chordwise, num_spanwise), action="i->ij")

        LE_exp_x = csdl.expand(LE_points_re_projected[:, 0], (num_chordwise, num_spanwise), action="j->ij")
        TE_exp_x = csdl.expand(TE_points_re_projected[:, 0], (num_chordwise, num_spanwise), action="j->ij")

        skewed_chord_surface_x = LE_exp_x + (TE_exp_x - LE_exp_x) * x_interp_exp


        chord_surface = chord_surface.set(
            csdl.slice[:, :, 0],
            skewed_chord_surface_x,
        )

        vertical_offset_1 = csdl.expand(
            csdl.Variable(shape=(3, ), value=np.array([0., 0., 0.25])),
            chord_surface.shape, action='k->ijk'
        )

        self._airfoil_upper_para = wing_geometry.project(
            chord_surface - vertical_offset_1, 
            direction=np.array([0., 0., 1.]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density
        )

        self._airfoil_lower_para = wing_geometry.project(
            chord_surface + vertical_offset_1, 
            direction=np.array([0., 0., -1]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density,
        )

        return self._airfoil_lower_para, self._airfoil_upper_para
        

    def _update(self):
        if self._upper_wireframe_para is not None and self._lower_wireframe_para is not None:
            # Re-evaluate the geometry after coefficients have changed
            upper_surace_wireframe = self._geom.evaluate(self._upper_wireframe_para).reshape((self._num_chord_wise + 1, self._num_spanwise + 1, 3))
            lower_surace_wireframe = self._geom.evaluate(self._lower_wireframe_para).reshape((self._num_chord_wise + 1, self._num_spanwise + 1, 3))

            # compute the camber surface as the mean of the upper and lower wireframe
            camber_surface_raw = (upper_surace_wireframe + lower_surace_wireframe) / 2

            # Ensure that the mesh is symmetric across the xz-plane
            camber_surface = make_mesh_symmetric(camber_surface_raw, self._num_spanwise, spanwise_index=1)

            self.nodal_coordinates = camber_surface

            return self
        
        else:
            LE_points_csdl = self._geom.evaluate(self._LE_points_para)
            TE_points_csdl = self._geom.evaluate(self._TE_points_para)
            
            y_mean_spanwise = (LE_points_csdl[:, 1] + TE_points_csdl[:, 1])/ 2 
            LE_points_csdl = LE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)
            TE_points_csdl = TE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)
            
            if self._chordwise_spacing == "linear":
                chord_surface = csdl.linear_combination(LE_points_csdl, TE_points_csdl, self._num_chord_wise+1).reshape((self._num_chord_wise+1, self._num_spanwise+1, 3))
            
            elif self._chordwise_spacing == "cosine":
                chord_surface = cosine_spacing(
                    self._num_spanwise, 
                    None,
                    csdl.linear_combination(LE_points_csdl, TE_points_csdl, self._num_chord_wise+1),
                    self._num_chord_wise
                )

            chord_surface = chord_surface.reshape((self._num_chord_wise+1, self._num_spanwise+1, 3))
            chord_surface_sym = make_mesh_symmetric(chord_surface, self._num_spanwise, spanwise_index=1)

            self.nodal_coordinates = chord_surface_sym


            return self

class CamberSurfaceDict(DiscretizationsDict):
    def __getitem__(self, key) -> CamberSurface:
        return super().__getitem__(key)
    
    def __setitem__(self, key, value : CamberSurface):
        if not isinstance(value, CamberSurface):
            raise ValueError(f"Can only add {CamberSurface}, received {type(value)}")
        return super().__setitem__(key, value)

class VLMMesh(SolverMesh):
    discretizations : dict[CamberSurface] = CamberSurfaceDict()

def make_vlm_surface(
    wing_comp,
    num_spanwise: int,
    num_chordwise: int, 
    spacing_spanwise: str = 'linear',
    spacing_chordwise: str = 'linear',
    chord_wise_points_for_airfoil = None,
    ignore_camber: bool = False, 
    plot: bool = False,
    grid_search_density: int = 10,
    LE_interp : Union[str, None] = None,
    TE_interp : Union[str, None] = None,
) -> CamberSurface:
    """Make a VLM camber surface mesh for wing-like components. This method is NOT 
    intended for vertically oriented lifting surfaces like a vertical tail.

    Parameters
    ----------
    wing_comp : Wing
        instance of a 'Wing' component
    
    num_spanwise : int
        number of span-wise panels (note that if odd, 
        central panel will be larger)
    
    num_chordwise : int
        number of chord-wise panels
    
    spacing_spanwise : str, optional
        spacing of the span-wise panels (linear or cosine 
        currently supported), by default 'linear'
    
    spacing_chordwise : str, optional
        spacing of the chord-wise panels (linear or cosine 
        currently supported), by default 'linear'
    
    plot : bool, optional
        plot the projections, by default False
    
    grid_search_density : int, optional
        parameter to refine the quality of projections (note that the higher this parameter 
        the longer the projections will take; for finer meshes, especially with cosine 
        spacing, a value of 40-50 is recommended), by default 10

    Returns
    -------
    VLMMesh: csdl.VariableGroup
        data class storing the mesh coordinates and mesh velocities (latter will be set later)

    """
    from CADDEE_alpha.core.aircraft.components.wing import Wing
    csdl.check_parameter(wing_comp, "wing_comp", types=Wing)
    csdl.check_parameter(num_spanwise, "num_spanwise", types=int)
    csdl.check_parameter(num_chordwise, "num_chordwise", types=int)
    csdl.check_parameter(spacing_spanwise, "spacing_spanwise", values=("linear", "cosine"))
    csdl.check_parameter(spacing_chordwise, "spacing_chordwise", values=("linear", "cosine"))
    csdl.check_parameter(plot, "plot", types=bool)
    csdl.check_parameter(grid_search_density, "grid_search_density", types=int)
    csdl.check_parameter(ignore_camber, "ignore_camber", types=bool)
    csdl.check_parameter(LE_interp, "LE_interp", values=("ellipse", None))
    csdl.check_parameter(TE_interp, "TE_interp", values=("ellipse", None))

    if wing_comp.geometry is None:
        raise Exception("Cannot generate mesh for component with geoemetry=None")

    if num_spanwise % 2 != 0:
        raise Exception("Number of spanwise panels must be even.")

    wing_geometry: FunctionSet = wing_comp.geometry

    LE_left_point = wing_geometry.evaluate(wing_comp._LE_left_point).value
    LE_mid_point = wing_geometry.evaluate(wing_comp._LE_mid_point).value
    LE_right_point = wing_geometry.evaluate(wing_comp._LE_right_point).value

    TE_left_point = wing_geometry.evaluate(wing_comp._TE_left_point).value
    TE_mid_point = wing_geometry.evaluate(wing_comp._TE_mid_point).value
    TE_right_point = wing_geometry.evaluate(wing_comp._TE_right_point).value

    if spacing_spanwise == "linear":
        num_spanwise_half = int(num_spanwise / 2)
        LE_points_1 = np.linspace(LE_left_point, LE_mid_point, num_spanwise_half + 1)
        LE_points_2 = np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1)
        
        # Check whether spanwise points should projected based on ellipse
        if LE_interp is not None:
            array_to_project = np.zeros((num_spanwise + 1, 3))
            y = np.array([LE_left_point[1], LE_mid_point[1], LE_right_point[1]])
            z = np.array([LE_left_point[2], LE_mid_point[2], LE_right_point[2]])
            fz = interp1d(y, z, kind="linear")

            # Set up equation for an ellipse
            h = LE_left_point[0]
            b = 2 * (h - LE_mid_point[0]) # Semi-minor axis
            a = LE_right_point[1] # semi-major axis
            
            interp_y_1 = np.linspace(y[0], y[1], num_spanwise_half+1)
            interp_y_2 = np.linspace(y[1], y[2], num_spanwise_half+1)

            array_to_project[0:num_spanwise_half+1, 0] = (b**2 * (1 - interp_y_1**2/a**2))**0.5 + h
            array_to_project[0:num_spanwise_half+1, 1] = interp_y_1
            array_to_project[0:num_spanwise_half+1, 2] = fz(interp_y_1)

            array_to_project[num_spanwise_half:, 0] = (b**2 * (1 - interp_y_2**2/a**2))**0.5 + h
            array_to_project[num_spanwise_half:, 1] = interp_y_2
            array_to_project[num_spanwise_half:, 2] = fz(interp_y_2)

            LE_points = array_to_project
        else:
            LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

        TE_points_1 = np.linspace(TE_left_point, TE_mid_point, num_spanwise_half + 1)
        TE_points_2 = np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1)
        # Check whether spanwise points should projected based on ellipse
        if TE_interp is not None:
            array_to_project = np.zeros((num_spanwise + 1, 3))
            y = np.array([TE_left_point[1], TE_mid_point[1], TE_right_point[1]])
            z = np.array([TE_left_point[2], TE_mid_point[2], TE_right_point[2]])
            fz = interp1d(y, z, kind="linear")

            # Set up equation for an ellipse
            h = TE_left_point[0]
            b = 2 * (h - TE_mid_point[0]) # Semi-minor axis
            a = TE_right_point[1] # semi-major axis

            interp_y_1 = np.linspace(y[0], y[1], num_spanwise_half+1)
            interp_y_2 = np.linspace(y[1], y[2], num_spanwise_half+1)

            array_to_project[0:num_spanwise_half+1, 0] = -(b**2 * (1 - interp_y_1**2/a**2))**0.5 + h
            array_to_project[0:num_spanwise_half+1, 1] = interp_y_1
            array_to_project[0:num_spanwise_half+1, 2] = fz(interp_y_1)

            array_to_project[num_spanwise_half:, 0] = -(b**2 * (1 - interp_y_2**2/a**2))**0.5 + h
            array_to_project[num_spanwise_half:, 1] = interp_y_2
            array_to_project[num_spanwise_half:, 2] = fz(interp_y_2)
            
            TE_points = array_to_project
        else:
            TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
    
    elif spacing_spanwise == "cosine":
        num_spanwise_half = int(num_spanwise / 2)
        if LE_interp is not None:
            array_to_project = np.zeros((num_spanwise + 1, 3))
            y = np.array([LE_left_point[1], LE_mid_point[1], LE_right_point[1]])
            z = np.array([LE_left_point[2], LE_mid_point[2], LE_right_point[2]])
            fz = interp1d(y, z, kind="linear")

            # Set up equation for an ellipse
            h = LE_left_point[0]
            b = 2 * (h - LE_mid_point[0]) # Semi-minor axis
            a = LE_right_point[1] # semi-major axis
            
            i_vec = np.arange(0, num_spanwise_half+1)
            half_cos = np.linspace(0, 1, len(i_vec))**2
            # half_cos = 1 - np.cos(i_vec * np.pi / (2 * num_spanwise_half))

            interp_y_1 = y[0] - (y[0] - y[1]) * half_cos
            interp_y_2 = np.flip(y[2] - (y[2] - y[1]) * half_cos)

            array_to_project[0:num_spanwise_half+1, 0] = (b**2 * (1 - interp_y_1**2/a**2))**0.5 + h
            array_to_project[0:num_spanwise_half+1, 1] = interp_y_1
            array_to_project[0:num_spanwise_half+1, 2] = fz(interp_y_1)

            array_to_project[num_spanwise_half:, 0] = (b**2 * (1 - interp_y_2**2/a**2))**0.5 + h
            array_to_project[num_spanwise_half:, 1] = interp_y_2
            array_to_project[num_spanwise_half:, 2] = fz(interp_y_2)

            LE_points = array_to_project
        else:
            LE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_left_point, num_spanwise_half + 1))
            LE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1), flip=True)
            LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

        if TE_interp is not None:
            array_to_project = np.zeros((num_spanwise + 1, 3))
            y = np.array([TE_left_point[1], TE_mid_point[1], TE_right_point[1]])
            z = np.array([TE_left_point[2], TE_mid_point[2], TE_right_point[2]])
            fz = interp1d(y, z, kind="linear")

            # Set up equation for an ellipse
            h = TE_left_point[0]
            b = 2 * (h - TE_mid_point[0]) # Semi-minor axis
            a = TE_right_point[1] # semi-major axis
            
            i_vec = np.arange(0, num_spanwise_half+1)
            half_cos = np.linspace(0, 1, len(i_vec))**2
            # half_cos = 1 - np.cos(i_vec * np.pi / (2 * num_spanwise_half))

            interp_y_1 = y[0] - (y[0] - y[1]) * half_cos
            interp_y_2 = np.flip(y[2] - (y[2] - y[1]) * half_cos)

            array_to_project[0:num_spanwise_half+1, 0] = -(b**2 * (1 - interp_y_1**2/a**2))**0.5 + h
            array_to_project[0:num_spanwise_half+1, 1] = interp_y_1
            array_to_project[0:num_spanwise_half+1, 2] = fz(interp_y_1)

            array_to_project[num_spanwise_half:, 0] = -(b**2 * (1 - interp_y_2**2/a**2))**0.5 + h
            array_to_project[num_spanwise_half:, 1] = interp_y_2
            array_to_project[num_spanwise_half:, 2] = fz(interp_y_2)

            TE_points = array_to_project

        else:
            TE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_left_point, num_spanwise_half + 1))
            TE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1), flip=True)
            TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
    
    else:
        raise NotImplementedError

    LE_points_para = wing_geometry.project(LE_points, plot=plot)
    TE_points_para = wing_geometry.project(TE_points, plot=plot)

    LE_points_csdl = wing_geometry.evaluate(LE_points_para)
    TE_points_csdl = wing_geometry.evaluate(TE_points_para)
    
    y_mean_spanwise = (LE_points_csdl[:, 1] + TE_points_csdl[:, 1])/ 2 
    LE_points_csdl = LE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)
    TE_points_csdl = TE_points_csdl.set(csdl.slice[:, 1], y_mean_spanwise)

    

    if spacing_chordwise == "linear":
        chord_surface = csdl.linear_combination(LE_points_csdl, TE_points_csdl, num_chordwise+1).reshape((num_chordwise+1, num_spanwise+1, 3))
    
    elif spacing_chordwise == "cosine":
        # chord_surface = csdl.linear_combination(TE_points_csdl, LE_points_csdl, num_chordwise+1).reshape((-1, 3))
        chord_surface = cosine_spacing(
            num_spanwise, 
            None,
            csdl.linear_combination(LE_points_csdl, TE_points_csdl, num_chordwise+1),
            num_chordwise
        )

    chord_surface = chord_surface.reshape((num_chordwise+1, num_spanwise+1, 3))
    if ignore_camber:
        chord_surface_sym = make_mesh_symmetric(chord_surface, num_spanwise, spanwise_index=1)
        vlm_mesh = CamberSurface(
            nodal_coordinates=chord_surface_sym,
        )
        vlm_mesh._geom = wing_geometry
        vlm_mesh._num_chord_wise = num_chordwise
        vlm_mesh._num_spanwise = num_spanwise
        vlm_mesh._num_chord_wise = num_chordwise
        vlm_mesh._chordwise_spacing = spacing_chordwise
        vlm_mesh._LE_points_para = LE_points_para
        vlm_mesh._TE_points_para = TE_points_para

        wing_comp._discretizations[f"{wing_comp._name}_vlm_camber_mesh"] = vlm_mesh

    else:
        vertical_offset_1 = csdl.expand(
            csdl.Variable(shape=(3, ), value=np.array([0., 0., 0.25])),
            chord_surface.shape, action='k->ijk'
        )

        upper_surace_wireframe_para = wing_geometry.project(
            chord_surface - vertical_offset_1, 
            direction=np.array([0., 0., 1.]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density
        )

        lower_surace_wireframe_para = wing_geometry.project(
            chord_surface + vertical_offset_1, 
            direction=np.array([0., 0., -1]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density,
        )

        upper_surace_wireframe = wing_geometry.evaluate(upper_surace_wireframe_para).reshape((num_chordwise + 1, num_spanwise + 1, 3))
        lower_surace_wireframe = wing_geometry.evaluate(lower_surace_wireframe_para).reshape((num_chordwise + 1, num_spanwise + 1, 3))

        # compute the camber surface as the mean of the upper and lower wireframe
        camber_surface_raw = (upper_surace_wireframe + lower_surace_wireframe) / 2

        # Ensure that the mesh is symmetric across the xz-plane
        camber_surface = make_mesh_symmetric(camber_surface_raw, num_spanwise, spanwise_index=1)
    
        vlm_mesh = CamberSurface(
            nodal_coordinates=camber_surface,
        )
        vlm_mesh._geom = wing_geometry
        vlm_mesh._lower_wireframe_para = lower_surace_wireframe_para
        vlm_mesh._upper_wireframe_para = upper_surace_wireframe_para
        vlm_mesh._num_chord_wise = num_chordwise
        vlm_mesh._num_spanwise = num_spanwise

    if chord_wise_points_for_airfoil is not None:
        LE_points_csdl_mid_panel = (LE_points_csdl[0:-1, :] + LE_points_csdl[1:, :]) / 2
        TE_points_csdl_mid_panel = (TE_points_csdl[0:-1, :] + TE_points_csdl[1:, :]) / 2

        LE_points_csdl_mid_panel = LE_points_csdl_mid_panel.set(csdl.slice[:, 0], LE_points_csdl_mid_panel[:, 0] + 0.1)
        TE_points_csdl_mid_panel = TE_points_csdl_mid_panel.set(csdl.slice[:, 0], TE_points_csdl_mid_panel[:, 0] - 0.1)

        LE_points_re_projected = wing_geometry.evaluate(wing_geometry.project(LE_points_csdl_mid_panel, grid_search_density_parameter=grid_search_density, plot=plot))
        TE_points_re_projected = wing_geometry.evaluate(wing_geometry.project(TE_points_csdl_mid_panel, grid_search_density_parameter=grid_search_density, plot=plot))
        
        num_chordwise = len(chord_wise_points_for_airfoil)
        # chord_surface = csdl.linear_combination(LE_points_csdl_mid_panel, TE_points_csdl_mid_panel, num_chordwise)
        chord_surface = csdl.linear_combination(LE_points_re_projected, TE_points_re_projected, num_chordwise)

        x_interp = csdl.Variable(shape=(num_chordwise, ), value=chord_wise_points_for_airfoil)
        x_interp_exp = csdl.expand(x_interp, (num_chordwise, num_spanwise), action="i->ij")
        # LE_exp_x = csdl.expand(LE_points_csdl_mid_panel[:, 0], (num_chordwise, num_spanwise), action="j->ij")
        # TE_exp_x = csdl.expand(TE_points_csdl_mid_panel[:, 0], (num_chordwise, num_spanwise), action="j->ij")

        LE_exp_x = csdl.expand(LE_points_re_projected[:, 0], (num_chordwise, num_spanwise), action="j->ij")
        TE_exp_x = csdl.expand(TE_points_re_projected[:, 0], (num_chordwise, num_spanwise), action="j->ij")

        skewed_chord_surface_x = LE_exp_x + (TE_exp_x - LE_exp_x) * x_interp_exp


        chord_surface = chord_surface.set(
            csdl.slice[:, :, 0],
            skewed_chord_surface_x,
        )

        vertical_offset_1 = csdl.expand(
            csdl.Variable(shape=(3, ), value=np.array([0., 0., 0.25])),
            chord_surface.shape, action='k->ijk'
        )

        airfoil_upper_para = wing_geometry.project(
            chord_surface - vertical_offset_1, 
            direction=np.array([0., 0., 1.]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density
        )

        airfoil_lower_para = wing_geometry.project(
            chord_surface + vertical_offset_1, 
            direction=np.array([0., 0., -1]), 
            plot=plot, 
            grid_search_density_parameter=grid_search_density,
        )

        vlm_mesh._airfoil_upper_para = airfoil_upper_para
        vlm_mesh._airfoil_lower_para = airfoil_lower_para

        vlm_mesh.airfoil_nodes_upper = wing_geometry.evaluate(airfoil_upper_para)
        vlm_mesh.airfoil_nodes_lower = wing_geometry.evaluate(airfoil_lower_para)





    wing_comp._discretizations[f"{wing_comp._name}_vlm_camber_mesh"] = vlm_mesh

    return vlm_mesh


@dataclass
class OneDBoxBeam(Discretization):
    beam_height: csdl.Variable = None
    beam_width: csdl.Variable = None
    num_beam_nodes: int = None
    shear_web_thickness: csdl.Variable = None
    top_skin_thickness: csdl.Variable = None
    bottom_skin_thickness: csdl.Variable = None

    _geom:lg.Geometry = None
    _spar_geom = None
    _front_spar_geom = None
    _rear_spar_geom = None

    _material_properties = None

    _LE_points_parametric = None
    _TE_points_parametric = None
    _norm_node_center = None
    _norm_beam_width = None
    _node_top_parametric = None
    _node_bottom_parametric = None
    _fore_points_parametric = None
    _aft_points_parametric = None
    _top_grid_parametric = None
    _bottom_grid_parametric = None
    _front_grid_parametric = None
    _rear_grid_parametric = None
    _half_wing = False


    def _update(self):
        LE_points_csdl = self._geom.evaluate(self._LE_points_parametric).reshape((self.num_beam_nodes, 3))
        TE_points_csdl = self._geom.evaluate(self._TE_points_parametric).reshape((self.num_beam_nodes, 3))

        if self._spar_geom is not None:
            beam_width_nodal = (self._geom.evaluate(self._fore_points_parametric)[:,0] 
                                - self._geom.evaluate(self._aft_points_parametric)[:,0])
        else:
            beam_width_raw = csdl.norm((LE_points_csdl - TE_points_csdl) * self._norm_beam_width, axes=(1, ))
            if self._half_wing:
                beam_width_nodal = beam_width_raw
            else:
                beam_width_nodal = make_mesh_symmetric(beam_width_raw, self.num_beam_nodes, spanwise_index=None)
        
        
        self.beam_width = (beam_width_nodal[0:-1] + beam_width_nodal[1:]) / 2

        node_top = self._geom.evaluate(self._node_top_parametric).reshape((self.num_beam_nodes, 3))
        node_bottom = self._geom.evaluate(self._node_bottom_parametric).reshape((self.num_beam_nodes, 3))

        if self._half_wing:
            self.nodal_coordinates = (node_top + node_bottom) / 2
        else:
            self.nodal_coordinates = make_mesh_symmetric((node_top + node_bottom) / 2, self.num_beam_nodes)

        beam_height_raw = csdl.norm((node_top - node_bottom), axes=(1, ))
        if self._half_wing:
            beam_height_nodal = beam_height_raw
        else:
            beam_height_nodal = make_mesh_symmetric(beam_height_raw, self.num_beam_nodes, spanwise_index=None)
        
        self.beam_height = (beam_height_nodal[0:-1] + beam_height_nodal[1:]) / 2

        top_thickness_grid = self._material_properties.evaluate_thickness(self._top_grid_parametric)
        bottom_thickness_grid = self._material_properties.evaluate_thickness(self._bottom_grid_parametric)
        self.top_skin_thickness = csdl.average(top_thickness_grid.reshape((self.num_beam_nodes-1, -1)), axes=(1,))
        self.bottom_skin_thickness = csdl.average(bottom_thickness_grid.reshape((self.num_beam_nodes-1, -1)), axes=(1,))

        front_thickness_grid = self._material_properties.evaluate_thickness(self._front_grid_parametric)
        rear_thickness_grid = self._material_properties.evaluate_thickness(self._rear_grid_parametric)
        front_thickness = csdl.average(front_thickness_grid.reshape((self.num_beam_nodes-1, -1)), axes=(1,))
        rear_thickness = csdl.average(rear_thickness_grid.reshape((self.num_beam_nodes-1, -1)), axes=(1,))
        self.shear_web_thickness = (front_thickness + rear_thickness) / 2

class OneDBoxBeamDict(DiscretizationsDict):
    def __getitem__(self, key) -> OneDBoxBeam:
        return super().__getitem__(key)

class BeamMesh(SolverMesh):
    discretizations : dict[OneDBoxBeam] = OneDBoxBeamDict()


def make_1d_box_beam(
    wing_comp,
    num_beam_nodes: int,
    norm_node_center: float,
    norm_beam_width: float = 0.5,
    spacing: str = "linear",
    plot: bool = False,
    grid_search_density: int = 10,
    project_spars: bool = False,
    make_half_beam: bool = False,
    LE_TE_interp: str = None,
    one_side_geometry: FunctionSet=None,
    spar_search_names = ['0', '1']
) -> OneDBoxBeam:
    """Create a 1-D box beam mesh for a wing-like component. It is NOT intended
    to work for creating beam-mesh of fuselage-like or vertical tail like components.

    Parameters
    ----------
    wing_comp : Wing
        instance of a 'Wing' component
    
    num_beam_nodes : int
        must be an odd number (ensures beam node at center)
    
    norm_node_center : float
        normalized node center w.r.t. the leading edge (0 at the leading edge); 
        Note that the beam height is determined based on the node center
    
    norm_beam_width : float, optional
        normalized beam width (i.e., distance between the front and rear spar), 
        cannot be greater than 1 and should usually not be greater than 0.5, by default 0.5
    
    spacing : str, optional
        set spacing for the beam nodes - "linear" and "cosine" are allowed
        where "cosine" skews the nodes toward the wing tip, by default "linear"
    
    plot : bool, optional
        plot the projections, by default False

    grid_search_density : int, optional
        parameter to refine the quality of projections (note that the higher this parameter 
        the longer the projections will take; 

    project_spars : bool, optional
        project onto the spars to get the beam width and shear web thickness, by default False

    Returns
    -------
    OneDBoxBeam
        collection of variables including beam node locations, thickness, and width

    Raises
    ------
    ValueError
        if number of nodes is even
    """
    from CADDEE_alpha.core.aircraft.components.wing import Wing

    # Error checking 
    csdl.check_parameter(wing_comp, "wing_comp", types=Wing)
    csdl.check_parameter(num_beam_nodes, "num_beam_nodes", types=int)
    csdl.check_parameter(spacing, "spacing", values=("linear", "cosine"))
    csdl.check_parameter(plot, "plot", types=bool)
    csdl.check_parameter(grid_search_density, "grid_search_density", types=int)

    wing_geometry = wing_comp.geometry

    if wing_geometry is None:
        raise Exception("Cannot generate mesh for component with geoemetry=None")

    LE_left_point = wing_geometry.evaluate(wing_comp._LE_left_point).value
    LE_mid_point = wing_geometry.evaluate(wing_comp._LE_mid_point).value
    LE_right_point = wing_geometry.evaluate(wing_comp._LE_right_point).value

    TE_left_point = wing_geometry.evaluate(wing_comp._TE_left_point).value
    TE_mid_point = wing_geometry.evaluate(wing_comp._TE_mid_point).value
    TE_right_point = wing_geometry.evaluate(wing_comp._TE_right_point).value

    if num_beam_nodes % 2 == 0:
        raise ValueError("'num_beam_nodes' must be odd such that one beam node is always at the wing center")
        
    else:
        if spacing == "linear":
            num_spanwise_half = int(num_beam_nodes / 2)
            LE_points_1 = np.linspace(LE_left_point, LE_mid_point, num_spanwise_half + 1)
            LE_points_2 = np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1)
            
            TE_points_1 = np.linspace(TE_left_point, TE_mid_point, num_spanwise_half + 1)
            TE_points_2 = np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1)

            if LE_TE_interp is not None:
                array_to_project_LE = np.zeros((num_beam_nodes, 3))
                y_LE = np.array([LE_left_point[1], LE_mid_point[1], LE_right_point[1]])
                z_LE = np.array([LE_left_point[2], LE_mid_point[2], LE_right_point[2]])
                fz_LE = interp1d(y_LE, z_LE, kind="linear")

                # Set up equation for an ellipse
                h_LE = LE_left_point[0]
                b_LE = 2 * (h_LE - LE_mid_point[0]) # Semi-minor axis
                a_LE = LE_right_point[1] # semi-major axis
                
                interp_y_1_LE = np.linspace(y_LE[0], y_LE[1], num_spanwise_half+1)
                interp_y_2_LE = np.linspace(y_LE[1], y_LE[2], num_spanwise_half+1)

                array_to_project_LE[0:num_spanwise_half+1, 0] = (b_LE**2 * (1 - interp_y_1_LE**2/a_LE**2))**0.5 + h_LE
                array_to_project_LE[0:num_spanwise_half+1, 1] = interp_y_1_LE
                array_to_project_LE[0:num_spanwise_half+1, 2] = fz_LE(interp_y_1_LE)

                array_to_project_LE[num_spanwise_half:, 0] = (b_LE**2 * (1 - interp_y_2_LE**2/a_LE**2))**0.5 + h_LE
                array_to_project_LE[num_spanwise_half:, 1] = interp_y_2_LE
                array_to_project_LE[num_spanwise_half:, 2] = fz_LE(interp_y_2_LE)

                LE_points = array_to_project_LE

                array_to_project_TE = np.zeros((num_beam_nodes, 3))
                y_TE = np.array([TE_left_point[1], TE_mid_point[1], TE_right_point[1]])
                z_TE = np.array([TE_left_point[2], TE_mid_point[2], TE_right_point[2]])
                fz_TE = interp1d(y_TE, z_TE, kind="linear")

                # Set up equation for an ellipse
                h_TE = TE_left_point[0]
                b_TE = 2 * (h_TE - TE_mid_point[0]) # Semi-minor axis
                a_TE = TE_right_point[1] # semi-major axis
                
                interp_y_1_TE = np.linspace(y_TE[0], y_TE[1], num_spanwise_half+1)
                interp_y_2_TE = np.linspace(y_TE[1], y_TE[2], num_spanwise_half+1)

                array_to_project_TE[0:num_spanwise_half+1, 0] = -(b_TE**2 * (1 - interp_y_1_TE**2/a_TE**2))**0.5 + h_TE
                array_to_project_TE[0:num_spanwise_half+1, 1] = interp_y_1_TE
                array_to_project_TE[0:num_spanwise_half+1, 2] = fz_TE(interp_y_1_TE)

                array_to_project_TE[num_spanwise_half:, 0] = -(b_TE**2 * (1 - interp_y_2_TE**2/a_TE**2))**0.5 + h_TE
                array_to_project_TE[num_spanwise_half:, 1] = interp_y_2_TE
                array_to_project_TE[num_spanwise_half:, 2] = fz_TE(interp_y_2_TE)

                TE_points = array_to_project_TE

                if make_half_beam:
                    LE_points = LE_points[num_spanwise_half:, :]
                    TE_points = TE_points[num_spanwise_half:, :]
            
            else:
                LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))
                TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))

                if make_half_beam:
                    LE_points = LE_points_2
                    TE_points = TE_points_2
                else:
                    LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))
                    TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
        
        elif spacing == "cosine":
            num_spanwise_half = int(num_beam_nodes / 2)
            LE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_left_point, num_spanwise_half + 1))
            LE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1), flip=True)
            LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

            TE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_left_point, num_spanwise_half + 1))
            TE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1), flip=True)
            TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))

            if make_half_beam:
                LE_points = LE_points_2
                TE_points = TE_points_2
            else:
                LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))
                TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
        
        else:
            raise NotImplementedError
    

    if make_half_beam:
        num_beam_nodes = num_beam_nodes // 2 + 1

    if one_side_geometry is not None:
        wing_geometry = one_side_geometry

    LE_points_parametric = wing_geometry.project(LE_points, plot=plot, grid_search_density_parameter=grid_search_density)
    TE_points_parametric = wing_geometry.project(TE_points, plot=plot, grid_search_density_parameter=grid_search_density)

    LE_points_csdl = wing_geometry.evaluate(LE_points_parametric).reshape((num_beam_nodes, 3))
    TE_points_csdl = wing_geometry.evaluate(TE_points_parametric).reshape((num_beam_nodes, 3))

    beam_nodes_raw = csdl.linear_combination(
        LE_points_csdl, 
        TE_points_csdl,
        1, 
        np.ones((num_beam_nodes, )) * (1 - norm_node_center),
        np.ones((num_beam_nodes, )) * norm_node_center,
    ).reshape((num_beam_nodes, 3))

    if make_half_beam:
        beam_width_raw = csdl.norm((LE_points_csdl - TE_points_csdl) * norm_beam_width, axes=(1, ))
        beam_width_nodal = beam_width_raw
        beam_width = beam_width_raw
    else:
        beam_width_raw = csdl.norm((LE_points_csdl - TE_points_csdl) * norm_beam_width, axes=(1, ))
        beam_width_nodal = make_mesh_symmetric(beam_width_raw, num_beam_nodes, spanwise_index=None)
        beam_width = (beam_width_nodal[0:-1] + beam_width_nodal[1:]) / 2

    

    if project_spars:
        # use the spars to get the beam width
        fore_projection_points = beam_nodes_raw.value.copy()
        fore_projection_points[:, 0] = fore_projection_points[:, 0] + beam_width_nodal.value/2
        aft_projection_points = beam_nodes_raw.value.copy()
        aft_projection_points[:, 0] = aft_projection_points[:, 0] - beam_width_nodal.value/2

        direction = np.array([1., 0., 0.])
        spar_geometery = wing_geometry.declare_component(function_search_names=["spar"])
        fore_points_parametric = spar_geometery.project(fore_projection_points, plot=plot, direction=direction, grid_search_density_parameter=grid_search_density)
        aft_points_parametric = spar_geometery.project(aft_projection_points, plot=plot, direction=-direction, grid_search_density_parameter=grid_search_density)
        beam_width_nodal = wing_geometry.evaluate(fore_points_parametric)[:,0] - wing_geometry.evaluate(aft_points_parametric)[:,0]
        beam_width = (beam_width_nodal[0:-1] + beam_width_nodal[1:]) / 2

    offset = np.array([0., 0., 1])
    node_top_parametric = wing_geometry.project(beam_nodes_raw.value + offset, direction=np.array([0., 0., -1]),  plot=plot)
    node_bottom_parametric = wing_geometry.project(beam_nodes_raw.value - offset, direction=np.array([0., 0., 1]), plot=plot)

    node_top = wing_geometry.evaluate(node_top_parametric).reshape((num_beam_nodes, 3))
    node_bottom = wing_geometry.evaluate(node_bottom_parametric).reshape((num_beam_nodes, 3))

    # Beam nodes will be at the center vertically (i.e., at the mid-point between upper and lower wing surface)
    if make_half_beam:
        beam_nodes = (node_top + node_bottom) / 2
    else:
        beam_nodes = make_mesh_symmetric((node_top + node_bottom) / 2, num_beam_nodes)

    beam_height_raw = csdl.norm((node_top - node_bottom), axes=(1, ))
    if make_half_beam:
        beam_height = (beam_height_raw[0:-1] + beam_height_raw[1:]) / 2
    else:
        beam_height_nodal = make_mesh_symmetric(beam_height_raw, num_beam_nodes, spanwise_index=None)
        beam_height = (beam_height_nodal[0:-1] + beam_height_nodal[1:]) / 2

    # Compute top and bottom thicknesses of the beam
    material_properties = wing_comp.quantities.material_properties
    num_chordwise = 10
    num_spanwise = 10
    beam_width_offset = np.zeros((num_beam_nodes-1, 3))
    beam_width_offset[:, 0] = beam_width.value
    node_grid = np.zeros((num_beam_nodes-1, num_chordwise*num_spanwise, 3))
    for i in range(num_beam_nodes-1):
        spanwise = np.linspace(beam_nodes.value[i,:], beam_nodes.value[i+1,:], num_spanwise)
        grid = np.linspace(spanwise+beam_width_offset[i]/2, spanwise-beam_width_offset[i]/2, num_chordwise).reshape(-1,3)
        node_grid[i,:,:] = grid
    node_grid = node_grid.reshape(-1,3)
    top_grid = wing_geometry.project(node_grid + offset, direction=np.array([0., 0., -1]), plot=plot)
    bottom_grid = wing_geometry.project(node_grid - offset, direction=np.array([0., 0., 1]), plot=plot)
    top_thickness_grid = material_properties.evaluate_thickness(top_grid).reshape((num_beam_nodes-1, -1))
    bottom_thickness_grid = material_properties.evaluate_thickness(bottom_grid).reshape((num_beam_nodes-1, -1))

    # NOTE: not 100% sure this is right so if there's an issue with the beam thickness, this is the first place to check
    top_thickness = csdl.average(top_thickness_grid, axes=(1,))
    bottom_thickness = csdl.average(bottom_thickness_grid, axes=(1,))

    # Compute the spar thickness if the spars are projected
    if project_spars:
        offset = np.array([offset[2], 0., 0.])
        num_spanwise = 10
        num_heightwise = 5
        beam_thickness_offset = np.zeros((num_beam_nodes-1, 3))
        beam_thickness_offset[:, 2] = beam_height.value
        node_grid = np.zeros((num_beam_nodes-1, num_heightwise*num_spanwise, 3))
        for i in range(num_beam_nodes-1):
            spanwise = np.linspace(beam_nodes.value[i,:], beam_nodes.value[i+1,:], num_spanwise)
            grid = np.linspace(spanwise+beam_thickness_offset[i]/2, spanwise-beam_thickness_offset[i]/2, num_heightwise).reshape(-1,3)
            node_grid[i,:,:] = grid
        node_grid = node_grid.reshape(-1,3)
        # NOTE: only really works well for 2 spars, but then so does the rest of the code
        f_spar_geometry = spar_geometery.declare_component(function_search_names=spar_search_names[0])
        r_spar_geometry = spar_geometery.declare_component(function_search_names=spar_search_names[1])
        front_grid = f_spar_geometry.project(node_grid + offset, direction=np.array([-1., 0., 0.]), plot=plot)
        rear_grid = r_spar_geometry.project(node_grid - offset, direction=np.array([1., 0., 0.]), plot=plot)
        front_thickness_grid = material_properties.evaluate_thickness(front_grid).reshape((num_beam_nodes-1, -1))
        rear_thickness_grid = material_properties.evaluate_thickness(rear_grid).reshape((num_beam_nodes-1, -1))
        front_thickness = csdl.average(front_thickness_grid, axes=(1,))
        rear_thickness = csdl.average(rear_thickness_grid, axes=(1,))
        shear_web_thickness = (front_thickness + rear_thickness) / 2

    beam_mesh = OneDBoxBeam(
        nodal_coordinates=beam_nodes,
        beam_height=beam_height,
        beam_width=beam_width,
        num_beam_nodes=num_beam_nodes,
        top_skin_thickness=top_thickness,
        bottom_skin_thickness=bottom_thickness,
        shear_web_thickness=shear_web_thickness if project_spars else None,
    )

    if make_half_beam:
        beam_mesh._half_wing = True

    wing_comp._discretizations[f"{wing_comp._name}_1d_beam_mesh"] = beam_mesh

    beam_mesh._geom = wing_geometry
    beam_mesh._material_properties = material_properties
    beam_mesh._LE_points_parametric = LE_points_parametric
    beam_mesh._TE_points_parametric = TE_points_parametric
    beam_mesh._norm_beam_width = norm_beam_width
    beam_mesh._norm_node_center = norm_node_center
    beam_mesh._node_top_parametric = node_top_parametric
    beam_mesh._node_bottom_parametric = node_bottom_parametric
    beam_mesh._top_grid_parametric = top_grid
    beam_mesh._bottom_grid_parametric = bottom_grid
    if project_spars:
        beam_mesh._spar_geom = spar_geometery
        beam_mesh._front_spar_geom = f_spar_geometry
        beam_mesh._rear_spar_geom = r_spar_geometry
        beam_mesh._fore_points_parametric = fore_points_parametric
        beam_mesh._aft_points_parametric = aft_points_parametric
        beam_mesh._front_grid_parametric = front_grid
        beam_mesh._rear_grid_parametric = rear_grid

    return beam_mesh


@dataclass
class RotorDiscretization(Discretization):
    thrust_vector : Union[np.ndarray, csdl.Variable, None] = None
    thrust_origin : Union[np.ndarray, csdl.Variable, None] = None
    chord_profile : Union[np.ndarray, csdl.Variable, None] = None
    twist_profile : Union[np.ndarray, csdl.Variable, None] = None
    radius : Union[float, int, csdl.Variable, None] = None
    num_radial: Union[int, None] = None
    num_azimuthal: Union[int, None] = None
    num_blades: Union[int, None] = None
    disk_mesh: Union[csdl.Variable, np.ndarray, None] = None
    norm_hub_radius: float = 0.2

    _disk_parametric = None
    _geom = None
    _p1 = None
    _p2 = None
    _p3 = None
    _p4 = None

    def _update(self):
        if self._disk_parametric is not None:
            shape = (self.num_radial, self.num_azimuthal, 3)
            self.disk_mesh = self._geom.evaluate(self._disk_parametric).reshape(shape)

        p1 = self._geom.evaluate(self._p1)
        p2 = self._geom.evaluate(self._p2)
        p3 = self._geom.evaluate(self._p3)
        p4 = self._geom.evaluate(self._p4)
        
        # Compute thrust origin as the mean of two corner points
        self.thrust_origin = (p1 + p2) / 2 
        
        # Compute in-plane vectors from the corner points
        in_plane_x = p1 - p2
        radius = csdl.norm(in_plane_x) / 2
        in_plane_ex = in_plane_x / (2 * radius)

        in_plane_y = p3 - p4
        in_plane_ey = in_plane_y / csdl.norm(in_plane_y)
        
        # compute thrust vector
        self.thrust_vector = csdl.cross(in_plane_ey, in_plane_ex)

        return self

class RotorDiscretizationDict(DiscretizationsDict):
    def __getitem__(self, key) -> RotorDiscretization:
        return super().__getitem__(key)

class RotorMeshes(SolverMesh):
    discretizations : dict[RotorDiscretization] = RotorDiscretizationDict()

def make_rotor_mesh(
    rotor_comp,
    # blade_comp, # TODO 
    num_radial: int, 
    num_azimuthal: int = 1,
    num_blades: int = 2,
    blade_comps=None,
    plot: bool = False,
    do_disk_projections:bool = False,
) -> RotorDiscretization: 
    
    from CADDEE_alpha.core.aircraft.components.rotor import Rotor
    # Do type checking
    csdl.check_parameter(rotor_comp, "rotor_comp", types=Rotor)
    csdl.check_parameter(num_radial, "num_radial", types=int)
    csdl.check_parameter(num_azimuthal, "num_azimuthal", types=int)
    csdl.check_parameter(plot, "plot", types=bool)

    # access rotor geometry 
    rotor_geometry = rotor_comp.geometry

    if rotor_geometry is None:
        raise ValueError("Cannot compute rotor mesh parameters since the geometry is None")
    
    # Get the "corner" points from the ffd block
    p1 = rotor_geometry.evaluate(rotor_comp._corner_point_1)
    p2 = rotor_geometry.evaluate(rotor_comp._corner_point_2)
    p3 = rotor_geometry.evaluate(rotor_comp._corner_point_3)
    p4 = rotor_geometry.evaluate(rotor_comp._corner_point_4)
    
    # Compute thrust origin as the mean of two corner points
    thrust_origin = (p1 + p2) / 2 
    
    # Compute in-plane vectors from the corner points
    in_plane_x = p1 - p2
    radius = csdl.norm(in_plane_x) / 2
    in_plane_ex = in_plane_x / (2 * radius)

    in_plane_y = p3 - p4
    in_plane_ey = in_plane_y / csdl.norm(in_plane_y)
    
    # compute thrust vector
    thrust_vector = csdl.cross(in_plane_ey, in_plane_ex)

    rotor_mesh_parameters = RotorDiscretization(
        nodal_coordinates=thrust_origin,
        thrust_origin=thrust_origin,
        thrust_vector=thrust_vector,
        radius=radius,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
    )

    # Assign corner points and geometry to mesh instance
    rotor_mesh_parameters._p1 = rotor_comp._corner_point_1
    rotor_mesh_parameters._p2 = rotor_comp._corner_point_2
    rotor_mesh_parameters._p3 = rotor_comp._corner_point_3
    rotor_mesh_parameters._p4 = rotor_comp._corner_point_4
    rotor_mesh_parameters._geom = rotor_geometry
    
    # make disk mesh
    if num_azimuthal > 1 and do_disk_projections:
        p = thrust_origin
        v1 = in_plane_ex
        v2 = in_plane_ey

        norm_radius_linspace = 1.0 / num_radial / 2.0 + np.linspace(
            0.0, 1.0 - 1.0 / num_radial, num_radial
        )
        
        hub_radius = radius * rotor_comp.parameters.hub_radius
        
        radius_vec = hub_radius + (radius - hub_radius) * norm_radius_linspace

        thetha_vec = np.linspace(
            0., 2 * np.pi - 2 * np.pi / num_azimuthal, num_azimuthal
        )

        cartesian_coordinates = np.zeros((num_radial, num_azimuthal, 3))
        for i in range(num_radial):
            for j in range(num_azimuthal):
                cartesian_coordinates[i, j, :] = p.value + radius_vec.value[i] * (np.cos(thetha_vec[j]) * v1.value \
                                                    + np.sin(thetha_vec[j]) * v2.value)
        
        disk_mesh_parametric = rotor_geometry.project(cartesian_coordinates, plot=plot)
        disk_mesh = rotor_geometry.evaluate(disk_mesh_parametric).reshape((num_radial, num_azimuthal, 3))

        rotor_mesh_parameters.disk_mesh = disk_mesh
        rotor_mesh_parameters._disk_parametric = disk_mesh_parametric

    rotor_comp._discretizations[f"{rotor_comp._name}_rotor_mesh_parameters"] = rotor_mesh_parameters

    return rotor_mesh_parameters

@dataclass
class ShellDiscretization(Discretization):
    geometry:csdl.Variable=None
    connectivity:csdl.Variable=None
    nodes_parametric:csdl.Variable=None

    def _update(self):
        self.nodes = self.geometry.evaluate(self.nodes_parametric)
        return self
        
def import_shell_mesh(file_name:str, 
                      geometry,
                      plot=False,
                      rescale=[1,1,1],
                      grid_search_n = 1,
                      priority_inds=None,
                      priority_eps=1e-4,
                      force_reprojection=False,
                      dupe_tol=1e-5):
    """
    Create a shell mesh for a component using a mesh file
    """
    import CADDEE_alpha as cd
    if isinstance(geometry, cd.Component):
        geometry = geometry.geometry
    nodes, nodes_parametric, connectivity = import_mesh(file_name, 
                                                        geometry, 
                                                        rescale=rescale, 
                                                        plot=plot,
                                                        grid_search_n=grid_search_n,
                                                        priority_inds=priority_inds,
                                                        priority_eps=priority_eps,
                                                        force_reprojection=force_reprojection,
                                                        dupe_tol=dupe_tol)
    shell_mesh = ShellDiscretization(nodal_coordinates=nodes, 
                                     connectivity=connectivity,
                                     nodes_parametric=nodes_parametric,
                                     geometry=geometry)
    return shell_mesh

class ShellMesh(SolverMesh):
    def __init__(self):
        self.discretizations = DiscretizationsDict()
        
        

# def compute_component_intersection(comp_1, comp_2):
#     u_v_array = (np.array([0.5, 0.5]))

#     geom_1 : FunctionSet = comp_1.geometry
#     geom_2 : FunctionSet = comp_2.geometry

#     for fun_1 in geom_1.functions.values():
#         for fun_2 in geom_2.functions.values():
#             cps_1_max = np.max(fun_1.coefficients.value, axis=(0, 1))
#             cps_1_min = np.min(fun_1.coefficients.value, axis=(0, 1))

#             x_max_1 = cps_1_max[0]
#             y_max_1 = cps_1_max[1]
#             z_max_1 = cps_1_max[2]
#             x_min_1 = cps_1_min[0]
#             y_min_1 = cps_1_min[1]
#             z_min_1 = cps_1_min[2]

#             cps_2_max = np.max(fun_2.coefficients.value, axis=(0, 1))
#             cps_2_min = np.min(fun_2.coefficients.value, axis=(0, 1))

#             x_max_2 = cps_2_max[0]
#             y_max_2 = cps_2_max[1]
#             z_max_2 = cps_2_max[2]
#             x_min_2 = cps_2_min[0]
#             y_min_2 = cps_2_min[1]
#             z_min_2 = cps_2_min[2]

#             overlap_x = False
#             if compare_floats(x_max_1, x_min_2) and compare_floats(x_max_2, x_min_1):
#             # if x_max_1 > x_min_2 and x_min_1 < x_max_2:
#                 overlap_x = True

#             overlap_y = False
#             if compare_floats(y_max_1, y_min_2) and compare_floats(y_max_2, y_min_1):
#             # if y_max_1 > y_min_2 and y_min_1 < y_max_2:
#                 overlap_y = True

#             overlap_z = False
#             if compare_floats(z_max_1, z_min_2) and compare_floats(z_max_2, z_min_1):
#             # if z_max_1 > z_min_2 and z_min_1 < z_max_2:
#                 overlap_z = True

#             if overlap_z and overlap_y and overlap_x:
#                 additional_plotting_elements = fun_1.plot(show=False)
#                 fun_2.plot(additional_plotting_elements=additional_plotting_elements)
#                 fun_1_eval_pts = []
#                 fun_2_eval_pts = []

#                 uv_1, uv_2, uv_1_updated_minus, uv_2_updated_minus, uv_1_updated_plus, uv_2_updated_plus = find_intersection(fun_1=fun_1, fun_2=fun_2)
#                 fun_1_eval_pts.append(uv_1)
#                 fun_2_eval_pts.append(uv_2)
#                 new_starting_point_minus = np.vstack((uv_1_updated_minus, uv_2_updated_minus)).flatten()
#                 new_starting_point_plus = np.vstack((uv_1_updated_plus, uv_2_updated_plus)).flatten()

#                 print("uv_1", uv_1)
#                 print("uv_2", uv_2)
#                 for i in range(100):
#                     uv_1, uv_2, uv_1_updated_minus, uv_2_updated_minus, _, _ = find_intersection(fun_1=fun_1, fun_2=fun_2, grid_search=False, starting_point=new_starting_point_minus)
                    
#                     print("uv_1", uv_1)
#                     print("uv_2", uv_2)
#                     fun_1_eval_pts.append(uv_1)
#                     fun_2_eval_pts.append(uv_2)
#                     new_starting_point_minus = np.vstack((uv_1_updated_minus, uv_2_updated_minus)).flatten()

#                 for i in range(100):
#                     uv_1, uv_2,_, _, uv_1_updated_plus, uv_2_updated_plus = find_intersection(fun_1=fun_1, fun_2=fun_2, grid_search=False, starting_point=new_starting_point_plus)
                    
#                     print("uv_1", uv_1)
#                     print("uv_2", uv_2)
#                     fun_1_eval_pts.append(uv_1)
#                     fun_2_eval_pts.append(uv_2)
#                     new_starting_point_plus = np.vstack((uv_1_updated_plus, uv_2_updated_plus)).flatten()

#                 # print("uv_1, uv_2", uv_1, uv_2)


#                 # new_starting_point = np.vstack((uv_1_updated, uv_2_updated)).flatten()
#                 # uv_1, uv_2, uv_1_updated, uv_2_updated = find_intersection(fun_1=fun_1, fun_2=fun_2, grid_search=False, starting_point=new_starting_point)
#                 # fun_1_eval_pts.append(uv_1)
#                 # fun_2_eval_pts.append(uv_2)

#                 # print("uv_1, uv_2", uv_1, uv_2)

            
#                 fun_1.evaluate(parametric_coordinates=np.vstack(fun_1_eval_pts), plot=True)
#                 fun_2.evaluate(parametric_coordinates=np.vstack(fun_2_eval_pts), plot=True)

#                 # exit()

#                 # proj_1_list = []
#                 # proj_2_list = []
#                 # for i in range(5):
#                 #     print("i", i)
#                 #     proj_1, proj_2, new_starting_point = fixed_point_projection(fun_1=fun_1, fun_2=fun_2, starting_point=starting_point)#, starting_point=para_array)
#                 #     print("new_starting_point", new_starting_point)
#                 #     proj_1_list.append(proj_1)
#                 #     proj_2_list.append(proj_2)
#                 #     # if np.any((proj_1 == 0) | (proj_1 == 1)) or np.any((proj_2 == 0) | (proj_2 == 1)):
#                 #     if np.any((new_starting_point < 0) | (new_starting_point > 1)):
#                 #         break
#                 #     else:
#                 #         print(starting_point)
#                 #         starting_point = new_starting_point

#                 #     # exit()
#                 #     # print(eval_pt_1, eval_pt_2, new_starting_point)
                

#                 # fun_2.evaluate(parametric_coordinates=np.vstack(proj_1_list), plot=True)
#                 # fun_1.evaluate(parametric_coordinates=np.vstack(proj_2_list), plot=True)
#                 # print(proj_1_list)
#                 # print(proj_2_list)

#                 # exit()                        


                
#             # print(cps_1_max)
#             # print(cps_1_min)
#             # print(fun_2.coefficients.value)
#         print("\n")
#             # point_1 = fun_1.evaluate(parametric_coordinates=u_v_array, plot=True)
#             # proj_1 = geom_2.project(point_1, plot=True)
#             # point_2 = geom_2.evaluate(proj_1, plot=True)
#             # proj_2 = geom_1.project(point_2, plot=True)

#             # exit()


# def find_intersection(fun_1 : Function, fun_2 : Function, tol=1e-10, grid_search=True, starting_point=None):
#     if grid_search:
#         # perform grid search
#         num_grid_points = 20
#         u, v = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points), indexing="ij") 
#         para_array = np.vstack((u.flatten(), v.flatten())).T
#         para_tensor = para_array.reshape((num_grid_points, num_grid_points, 2))

#         fun_1_eval = fun_1.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))
#         fun_2_eval = fun_2.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))

#         min_dist = np.inf
#         candidates_list = []
#         for ind_1 in range(num_grid_points):
#             for ind_2 in range(num_grid_points):
#                 for ind_3 in range(num_grid_points):
#                     for ind_4 in range(num_grid_points):
#                         distance = np.linalg.norm(fun_1_eval[ind_1, ind_2, :] -  fun_2_eval[ind_3, ind_4, :])
#                         if distance < min_dist:
#                             min_dist = distance
#                             # candidates_list.append(np.array([
#                             #     para_tensor[ind_1, ind_2, 0],
#                             #     para_tensor[ind_1, ind_2, 1],
#                             #     para_tensor[ind_3, ind_4, 0],
#                             #     para_tensor[ind_3, ind_4, 1],
#                             # ])
#                             # )
#                             starting_point = np.array([
#                                 para_tensor[ind_1, ind_2, 0],
#                                 para_tensor[ind_1, ind_2, 1],
#                                 para_tensor[ind_3, ind_4, 0],
#                                 para_tensor[ind_3, ind_4, 1],
#                             ])

#     # print(candidates_list)
#     # print(len(candidates_list))
#     # exit()
#     uv_1_list = []
#     uv_2_list = []
#     # for starting_point in candidates_list:
#     # start newton optimization iteration
#     print("starting_point", starting_point)
#     for iteration in range(200):
#         uv_1 = starting_point[0:2]
#         uv_2 = starting_point[2:]
        
#         # Compute residual B1(u1, v1)P1 - B2(u2, v2)P2
#         residual = fun_1.evaluate(parametric_coordinates=uv_1).value - \
#             fun_2.evaluate(parametric_coordinates=uv_2).value
        
#         # Compute the square of the euclidean distance (OBJECTIVE)
#         distance_squared = residual.T @ residual

#         # Compute gradient/hessian of squared distance w.r.t. u1, v1, u2, v2 --> (4, ) & (4, 4)
        
#         # 1) gradient of distance w.r.t. entries of residual --> (1, 3)
#         grad_d_x = 2 * np.array([[residual[0], residual[1], residual[2]]])
        
#         # 2) gradient of cartesian coords. w.r.t. parametric coords. (Jacobian) --> (3, 4)
#         grad_x_u = np.zeros((3, 4))
#         hessian_x_u = np.zeros((3, 4, 4))
#         for i in range(2):
#             parametric_der_order = np.zeros((2, ), dtype=int)
#             parametric_der_order[i] = 1

#             grad_x_u[:, i] = fun_1.space.compute_basis_matrix(
#                 uv_1, parametric_derivative_orders=parametric_der_order
#             ).dot(fun_1.coefficients.value.reshape((-1, 3)))

#             grad_x_u[:, i+2] = -fun_2.space.compute_basis_matrix(
#                 uv_2, parametric_derivative_orders=parametric_der_order
#             ).dot(fun_2.coefficients.value.reshape((-1, 3)))
            
#             # Extra dimension for Hessian (tensor)
#             for j in range(2):
#                 parametric_der_order_hessian = np.zeros((2, ), dtype=int)
#                 if i==j:
#                     parametric_der_order_hessian[i] = 2
#                 else:
#                     parametric_der_order_hessian[i] = 1
#                     parametric_der_order_hessian[j] = 1

#                 # Note that fun_1 does not depend on u2, v2 (parameters of fun_2)
#                 hessian_x_u[:, j, i] = fun_1.space.compute_basis_matrix(
#                     uv_1, parametric_derivative_orders=parametric_der_order_hessian
#                 ).dot(fun_1.coefficients.value.reshape((-1, 3)))

#                 # Note that fun_2 does not depend on u1, v1 (parameters of fun_1)
#                 hessian_x_u[:, j+2, i+2] = -fun_2.space.compute_basis_matrix(
#                     uv_2, parametric_derivative_orders=parametric_der_order_hessian
#                 ).dot(fun_2.coefficients.value.reshape((-1, 3)))
                


#         # 3) apply chain (and product) rule to compute grad_d_u and hess_d_u
#         grad_d_u = grad_d_x @ grad_x_u
#         hessian_d_u = 2 * (grad_x_u.T @ grad_x_u) + np.einsum('i,ijk->jk', grad_d_x.flatten(), hessian_x_u)

#         # 4) Identify the active set
#         # NOTE: this is a very ineqality constraint (design variable bounds)
#         # instead of adding Lagrange multipliers we remove the design variable 
#         # (i.e., parametric coordinate) when it's zero or one and the gradient 
#         # tries to "push" it outside of the bounds
#         remove_dvs_lower_bound = np.logical_and(starting_point == 0, grad_d_u > 0)
#         remove_dvs_upper_bound = np.logical_and(starting_point == 1, grad_d_u < 0)
#         remove_dvs = np.logical_or(remove_dvs_lower_bound, remove_dvs_upper_bound)

#         # NOTE: remove zero columns in the Hessian in edge cases
#         # (e.g., if there are infinitely many optimal solution)
#         # NOTE: This could potentially mask bugs
#         remove_cols_hessian = np.where(~hessian_d_u.any(axis=1))[0]
#         remove_dvs[remove_cols_hessian] = True
#         keep_dvs = np.logical_not(remove_dvs).flatten()

#         reduced_gradient = grad_d_u.flatten()[keep_dvs]
#         reduced_hessian = hessian_d_u[keep_dvs][:, keep_dvs]

#         if np.linalg.norm(reduced_gradient) < 1e-7:
#             break
        
#         # compute step direction
#         step = np.linalg.solve(reduced_hessian, -reduced_gradient.flatten())

#         # Update starting point with step
#         starting_point[keep_dvs] = starting_point[keep_dvs] + step.flatten()
        
#         # clamp parameters between 0, 1
#         starting_point = np.clip(starting_point, 0, 1)

#         # uv_1_list.append(uv_1)
#         # uv_2_list.append(uv_2)
#         # print(np.linalg.norm(grad_d_u))
#         # print("\n")

#     # uv_1_array = np.unique(np.round(np.array(uv_1_list), decimals=6), axis=0)
#     # uv_2_array = np.unique(np.round(np.array(uv_2_list), decimals=6), axis=0)
    

#     # Start tracing intersection
#     # 1) Finding tangent and curvature vectors at u, v coordinates of both patches
#     tangent_vectors_fun_1 = np.zeros((2, 3))
#     tangent_vectors_fun_2 = np.zeros((2, 3))

#     curvature_hessian_fun_1 = np.zeros((3, 2, 2))
#     curvature_hessian_fun_2 = np.zeros((3, 2, 2))
#     for p in range(2):
#         parametric_derivative_orders = np.zeros((fun_1.space.num_parametric_dimensions, ), dtype=int)
#         parametric_derivative_orders_curvature = np.zeros((fun_1.space.num_parametric_dimensions, ), dtype=int)
#         parametric_derivative_orders[p] = 1
#         parametric_derivative_orders_curvature[p] = 2

#         tangent_vectors_fun_1[p, :] = fun_1.space.compute_basis_matrix(
#             uv_1, parametric_derivative_orders=parametric_derivative_orders
#         ).dot(fun_1.coefficients.value.reshape((-1, 3)))

#         tangent_vectors_fun_2[p, :] = fun_2.space.compute_basis_matrix(
#             uv_2, parametric_derivative_orders=parametric_derivative_orders
#         ).dot(fun_2.coefficients.value.reshape((-1, 3)))

#         for q in range(2):
#             parametric_der_order_hessian = np.zeros((2, ), dtype=int)
#             if p==q:
#                 parametric_der_order_hessian[p] = 2
#             else:
#                 parametric_der_order_hessian[p] = 1
#                 parametric_der_order_hessian[q] = 1

#             curvature_hessian_fun_1[:, p, q] = fun_1.space.compute_basis_matrix(
#                 uv_1, parametric_derivative_orders=parametric_der_order_hessian,
#             ).dot(fun_1.coefficients.value.reshape((-1, 3)))

#             curvature_hessian_fun_2[:, p, q] = fun_2.space.compute_basis_matrix(
#                 uv_2, parametric_derivative_orders=parametric_der_order_hessian,
#             ).dot(fun_2.coefficients.value.reshape((-1, 3)))

#     # 2) Compute surface normal for each patch
#     N1 = np.cross(tangent_vectors_fun_1[0, :], tangent_vectors_fun_1[1, :])
#     N2 = np.cross(tangent_vectors_fun_2[0, :], tangent_vectors_fun_2[1, :])

#     # 3) Compute direction of intersection (normal of the two normals)
#     T = np.cross(N1, N2)
    
#     # First order Taylor series approximation (use as starting point for Newton iteration)
#     delta_para_1 = np.linalg.solve(tangent_vectors_fun_1 @ tangent_vectors_fun_1.T, tangent_vectors_fun_1 @ T).reshape((-1, 2))
#     delta_para_2 = np.linalg.solve(tangent_vectors_fun_2 @ tangent_vectors_fun_2.T, tangent_vectors_fun_2 @ T).reshape((-1, 2))

#     delta_uv = np.zeros((4, ))
#     delta_uv[0:2] = delta_para_1.flatten()
#     delta_uv[2:] = delta_para_2.flatten()

#     converged = False
#     for i in range(50):
#         first_order_term_1 = np.einsum('i,ij->j', delta_uv[0:2], tangent_vectors_fun_1 )
#         second_order_term_intermediate_1 = np.einsum('j,ijk->ik', delta_uv[0:2], curvature_hessian_fun_1)
#         second_order_term_1 = np.einsum('ik,k->i', second_order_term_intermediate_1, delta_uv[0:2])

#         first_order_term_2 = np.einsum('i,ij->j', delta_uv[2:], tangent_vectors_fun_2 )
#         second_order_term_intermediate_2 = np.einsum('j,ijk->ik', delta_uv[2:], curvature_hessian_fun_2)
#         second_order_term_2 = np.einsum('ik,k->i', second_order_term_intermediate_2, delta_uv[2:])

#         residual_1 = first_order_term_1 + 0.5 * second_order_term_1 - T
#         residual_2 = first_order_term_2 + 0.5 * second_order_term_2 - T

#         residual = np.vstack((residual_1, residual_2)).flatten()

#         jacobian = np.zeros((6, 4)) 
#         jacobian[0:3, 0:2] = tangent_vectors_fun_1.T + second_order_term_intermediate_1
#         jacobian[3:, 2:] = tangent_vectors_fun_2.T + second_order_term_intermediate_2

#         delta_uv_updated = np.linalg.solve(jacobian.T @ jacobian, - jacobian.T @ residual)
#         if np.linalg.norm(delta_uv_updated) < 1e-10:
#             converged = True
#             break

#         delta_uv += delta_uv_updated

#     if converged:
#         delta_uv_norm = delta_uv / np.linalg.norm(delta_uv)
#     else:
#         print("NOT CONVERGED")
#         delta_uv = np.zeros((4, ))
#         delta_uv[0:2] = delta_para_1.flatten()
#         delta_uv[2:] = delta_para_2.flatten()
#         delta_uv_norm = delta_uv / np.linalg.norm(delta_uv)

#     step = 0.1/np.linalg.norm(T)
    
#     uv_1_updated_minus = uv_1 - 0.01 * delta_uv_norm[0:2] #, delta_uv[0:2] #  0.01 * delta_uv_norm[0:2] # 0.01 * delta_para_1_norm
#     uv_2_updated_minus = uv_2 - 0.01 * delta_uv_norm[2:] #delta_uv[2:] # 0.01 * delta_uv_norm[2:] # 0.01 * delta_para_2_norm

#     uv_1_updated_plus = uv_1 + 0.01 * delta_uv_norm[0:2] # delta_uv[0:2] #0.01 * delta_uv_norm[0:2] # 0.01 * delta_para_1_norm
#     uv_2_updated_plus = uv_2 + 0.01 * delta_uv_norm[2:] #delta_uv[2:] # 0.01 * delta_uv_norm[2:] # 0.01 * delta_para_2_norm

#     return uv_1, uv_2, np.clip(uv_1_updated_minus, 0, 1), np.clip(uv_2_updated_minus, 0, 1), np.clip(uv_1_updated_plus, 0, 1), np.clip(uv_2_updated_plus, 0, 1)


def compute_component_intersection(comp_1, comp_2):
    geom_1 : FunctionSet = comp_1.geometry
    geom_2 : FunctionSet = comp_2.geometry

    intersection_parametric_1 = []
    intersection_parametric_2 = []

    for fun_ind_1, fun_1 in geom_1.functions.items():
        for fun_ind_2, fun_2 in geom_2.functions.items():
            cps_1_max = np.max(fun_1.coefficients.value, axis=(0, 1))
            cps_1_min = np.min(fun_1.coefficients.value, axis=(0, 1))

            x_max_1 = cps_1_max[0]
            y_max_1 = cps_1_max[1]
            z_max_1 = cps_1_max[2]
            x_min_1 = cps_1_min[0]
            y_min_1 = cps_1_min[1]
            z_min_1 = cps_1_min[2]

            cps_2_max = np.max(fun_2.coefficients.value, axis=(0, 1))
            cps_2_min = np.min(fun_2.coefficients.value, axis=(0, 1))

            x_max_2 = cps_2_max[0]
            y_max_2 = cps_2_max[1]
            z_max_2 = cps_2_max[2]
            x_min_2 = cps_2_min[0]
            y_min_2 = cps_2_min[1]
            z_min_2 = cps_2_min[2]

            overlap_x = False
            if compare_floats(x_max_1, x_min_2) and compare_floats(x_max_2, x_min_1):
            # if x_max_1 > x_min_2 and x_min_1 < x_max_2:
                overlap_x = True

            overlap_y = False
            if compare_floats(y_max_1, y_min_2) and compare_floats(y_max_2, y_min_1):
            # if y_max_1 > y_min_2 and y_min_1 < y_max_2:
                overlap_y = True

            overlap_z = False
            if compare_floats(z_max_1, z_min_2) and compare_floats(z_max_2, z_min_1):
            # if z_max_1 > z_min_2 and z_min_1 < z_max_2:
                overlap_z = True

            if overlap_z and overlap_y and overlap_x:
                # new_space = fs.BSplineSpace(num_parametric_dimensions=2, degree=3, coefficients_shape=(10, 10))
                # fun_1 = fun_1.refit(new_function_space=new_space)
                # geom_1.functions[fun_ind_1] = fun_1

                # new_space = fs.BSplineSpace(num_parametric_dimensions=2, degree=3, coefficients_shape=(10, 10))
                # fun_2 = fun_2.refit(new_function_space=new_space)
                # geom_2.functions[fun_ind_2] = fun_2

                additional_plotting_elements = fun_1.plot(show=False)
                fun_2.plot(additional_plotting_elements=additional_plotting_elements)
                
                components_intersect, uv_1_coords, uv_2_coords = trace_intersection(surface_fun_1=fun_1, surface_fun_2=fun_2)

                if components_intersect:
                    intersection_parametric_1.append((fun_ind_1, uv_1_coords))
                    intersection_parametric_2.append((fun_ind_2, uv_2_coords))
                else:
                    pass
                
                
                # exit()
                # fun_1_eval_pts = []
                # fun_2_eval_pts = []

                # uv_1, uv_2, uv_1_updated_minus, uv_2_updated_minus, uv_1_updated_plus, uv_2_updated_plus, _, _ = find_intersection(fun_1=fun_1, fun_2=fun_2)
                # if uv_1 is None:
                #     pass 
                # else:
                #     point_on_intersection = fun_1.evaluate(parametric_coordinates=uv_1).value
                #     fun_1_eval_pts.append(uv_1)
                #     fun_2_eval_pts.append(uv_2)
                #     new_starting_point_minus = np.vstack((uv_1_updated_minus, uv_2_updated_minus)).flatten()
                #     new_starting_point_plus = np.vstack((uv_1_updated_plus, uv_2_updated_plus)).flatten()

                #     step_multiplier = 1

                #     for i in range(8000):
                #         # print("fun_1_eval_pts", fun_1_eval_pts)
                #         # print("fun_2_eval_pts", fun_2_eval_pts)
                #         # print("\n")
                        
                #         uv_1, uv_2, uv_1_updated_minus, uv_2_updated_minus, _, _, old_starting_point, newton_converged = find_intersection(fun_1=fun_1, fun_2=fun_2, grid_search=False, starting_point=new_starting_point_minus, step_multiplier=step_multiplier, point_on_intersection=point_on_intersection)
                #         if newton_converged:
                #             point_on_intersection = fun_1.evaluate(parametric_coordinates=uv_1).value
                #             fun_1_eval_pts.append(uv_1)
                #             fun_2_eval_pts.append(uv_2)
                            
                #             new_starting_point_minus = np.vstack((uv_1_updated_minus, uv_2_updated_minus)).flatten()
                            
                #             if np.linalg.norm(new_starting_point_minus - old_starting_point)==0:
                #                 print("increase_step_multipler")
                #                 step_multiplier *= 1.1
                #             else:
                #                 step_multiplier = 1

                #             if (uv_1 == 1).any() or (uv_1 == 0).any(): 
                #                 print("uv_1", uv_1)
                #                 break
                #             if (uv_2 == 1).any() or (uv_2 == 0).any(): 
                #                 print("uv_2", uv_2)
                #                 break
                #         #     print(fun_1_eval_pts)
                #         #     print(fun_2_eval_pts)
                #         #     eval_points_fun_1_numpy = np.vstack(fun_1_eval_pts)
                #         #     fun_1.evaluate(parametric_coordinates=eval_points_fun_1_numpy, plot=True)
                #         #     point_on_intersection = None
                #         #     new_starting_point_minus = old_starting_point
                #         else:
                #             break


                #     eval_points_fun_1_numpy = np.vstack(fun_1_eval_pts)
                #     eval_points_fun_2_numpy = np.vstack(fun_2_eval_pts)
                #     eval_points_fun_2_numpy_sorted = eval_points_fun_2_numpy[eval_points_fun_2_numpy[:, 1].argsort()]
                #     eval_points_fun_1_numpy_sorted = eval_points_fun_1_numpy[eval_points_fun_1_numpy[:, 0].argsort()]

                    
                #     cartesian_coords_2 = fun_2.evaluate(parametric_coordinates=eval_points_fun_2_numpy_sorted, plot=True).value
                #     ordered_cartesian_2 = order_points_by_proximity(cartesian_coords_2)
                #     resampled_points_2 = resample_curve(points=ordered_cartesian_2, num_points=100)
                #     fun_2_reprojected_points = fun_2.project(points=resampled_points_2, plot=True, grid_search_density_parameter=50)
            
                #     cartesian_coords_1 = fun_1.evaluate(parametric_coordinates=eval_points_fun_1_numpy_sorted, plot=True).value
                #     ordered_cartesian_1 = order_points_by_proximity(cartesian_coords_1)
                #     resampled_points_1 = resample_curve(points=ordered_cartesian_1, num_points=100)
                #     fun_1_reprojected_points = fun_1.project(points=resampled_points_1, plot=True, grid_search_density_parameter=50)
                    
                #     intersection_parametric_1.append((fun_ind_1, fun_1_reprojected_points))
                #     intersection_parametric_2.append((fun_ind_2, fun_2_reprojected_points))
                    
    para_dict_patches_1 = {}
    for counter, para in enumerate(intersection_parametric_1):
        index = para[0]
        coords = para[1]
        if index in para_dict_patches_1:
            existing_coords = para_dict_patches_1[index][1]
            para_dict_patches_1[index] = (index, np.vstack((existing_coords, coords)))
        else:
            para_dict_patches_1[index] = para
    
    intersection_list_1 = list(para_dict_patches_1.values())

    para_dict_patches_2 = {}
    for counter, para in enumerate(intersection_parametric_2):
        index = para[0]
        coords = para[1]
        if index in para_dict_patches_2:
            existing_coords = para_dict_patches_2[index][1]
            para_dict_patches_2[index] = (index, np.vstack((existing_coords, coords)))
        else:
            para_dict_patches_2[index] = para
    
    intersection_list_2 = list(para_dict_patches_2.values())


    for counter, entry in enumerate(intersection_list_1):
        index = entry[0]
        coords = entry[1]
        fun = geom_1.functions[index]

        fun.evaluate(parametric_coordinates=coords, plot=True)

    for counter, entry in enumerate(intersection_list_2):
        index = entry[0]
        coords = entry[1]
        fun = geom_2.functions[index]

        fun.evaluate(parametric_coordinates=coords, plot=True)


    return intersection_parametric_1, intersection_parametric_2, intersection_list_1, intersection_list_2

def find_intersection(fun_1 : Function, fun_2 : Function, tol=1e-10, grid_search=True, starting_point=None, step_multiplier=1, point_on_intersection=None, distance_between_points=0.05):
    if grid_search:
        # # perform grid search
        # num_grid_points = 100
        # u, v = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points), indexing="ij") 
        # para_array = np.vstack((u.flatten(), v.flatten())).T
        # para_tensor = para_array.reshape((num_grid_points, num_grid_points, 2))

        # fun_1_eval = fun_1.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))
        # fun_2_eval = fun_2.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))

        # # Calculate pairwise distances using broadcasting
        # distances = np.linalg.norm(fun_1_eval[:, :, None, None, :] - fun_2_eval[None, None, :, :, :], axis=-1)

        # # Find minimum distance and corresponding indices
        # min_dist = np.min(distances)
        # if min_dist > 0.1:
        #     return None, None, None, None, None, None, None, None

        # ind_1, ind_2, ind_3, ind_4 = np.unravel_index(np.argmin(distances), distances.shape)

        # # Retrieve starting points using indices
        # starting_point = np.array([
        #     para_tensor[ind_1, ind_2, 0],
        #     para_tensor[ind_1, ind_2, 1],
        #     para_tensor[ind_3, ind_4, 0],
        #     para_tensor[ind_3, ind_4, 1],
        # ])
       
        num_grid_points = 100
        u, v = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points), indexing="ij")
        para_tensor = np.stack((u, v), axis=-1)
        para_array = para_tensor.reshape(-1, 2)

        # Identify boundary points for both surfaces
        boundary_mask = (u == 0) | (u == 1) | (v == 0) | (v == 1)
        boundary_points = para_tensor[boundary_mask]  # Boundary points (parametric space)

        # Evaluate both functions
        fun_1_eval = fun_1.evaluate(parametric_coordinates=para_array).value  # Shape: (num_grid_points^2, 3)
        fun_2_eval = fun_2.evaluate(parametric_coordinates=para_array).value  # Shape: (num_grid_points^2, 3)

        # Evaluate boundary points of fun_1 with all points of fun_2
        fun_1_boundary_eval = fun_1.evaluate(parametric_coordinates=boundary_points).value
        distances_1 = np.linalg.norm(fun_1_boundary_eval[:, None, :] - fun_2_eval[None, :, :], axis=-1)

        # Evaluate boundary points of fun_2 with all points of fun_1
        fun_2_boundary_eval = fun_2.evaluate(parametric_coordinates=boundary_points).value
        distances_2 = np.linalg.norm(fun_1_eval[:, None, :] - fun_2_boundary_eval[None, :, :], axis=-1)

        # Combine the distances
        combined_distances = np.vstack((distances_1.reshape(-1), distances_2.T.reshape(-1)))

        # Find the minimum distance and corresponding indices
        min_index = np.argmin(combined_distances)
        min_dist = np.min(combined_distances)

        print("min_dist", min_dist)
        exit()

        if min_dist > 0.1:
            return None, None, None, None, None, None, None, None
        else:
            if min_index < distances_1.size:
                ind_1, ind_2 = np.unravel_index(min_index, distances_1.shape)
                starting_point = np.array([
                    boundary_points[ind_1, 0],
                    boundary_points[ind_1, 1],
                    para_array[ind_2, 0],
                    para_array[ind_2, 1],
                ])
            else:
                ind_1, ind_2 = np.unravel_index(min_index - distances_1.size, distances_2.shape)
                starting_point = np.array([
                    para_array[ind_1, 0],
                    para_array[ind_1, 1],
                    boundary_points[ind_2, 0],
                    boundary_points[ind_2, 1],
                ])
        # print("min_dist", min_dist)
        # print("starting_point", starting_point)
        # exit()

    uv_1_list = []
    uv_2_list = []


    # start newton optimization iteration
    starting_point_copy = starting_point.copy()
    newton_opt_converged = False
    constraint = None
    if point_on_intersection is not None:
        lambda_0 = 0.001
    for iteration in range(200):
        uv_1 = starting_point[0:2]
        uv_2 = starting_point[2:]
        
        # Compute residual B1(u1, v1)P1 - B2(u2, v2)P2
        residual = fun_1.evaluate(parametric_coordinates=uv_1).value - \
            fun_2.evaluate(parametric_coordinates=uv_2).value
        
        if point_on_intersection is not None:
            constraint = fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection
            constraint_distance = np.linalg.norm(fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection)**2 - distance_between_points**2
        
        # Compute the square of the euclidean distance (OBJECTIVE)
        distance_squared = residual.T @ residual

        # Compute gradient/hessian of squared distance w.r.t. u1, v1, u2, v2 --> (4, ) & (4, 4)
        
        # 1) gradient of distance w.r.t. entries of residual --> (1, 3)
        grad_d_x = 2 * np.array([[residual[0], residual[1], residual[2]]])
        if point_on_intersection is not None:
            grad_c_x = 2 * np.array([[constraint[0], constraint[1], constraint[2]]])

        # 2) gradient of cartesian coords. w.r.t. parametric coords. (Jacobian) --> (3, 4)
        grad_x_u = np.zeros((3, 4))
        hessian_x_u = np.zeros((3, 4, 4))
        for i in range(2):
            parametric_der_order = np.zeros((2, ), dtype=int)
            parametric_der_order[i] = 1

            grad_x_u[:, i] = fun_1.space.compute_basis_matrix(
                uv_1, parametric_derivative_orders=parametric_der_order
            ).dot(fun_1.coefficients.value.reshape((-1, 3)))

            grad_x_u[:, i+2] = -fun_2.space.compute_basis_matrix(
                uv_2, parametric_derivative_orders=parametric_der_order
            ).dot(fun_2.coefficients.value.reshape((-1, 3)))
            
            # Extra dimension for Hessian (tensor)
            for j in range(2):
                parametric_der_order_hessian = np.zeros((2, ), dtype=int)
                if i==j:
                    parametric_der_order_hessian[i] = 2
                else:
                    parametric_der_order_hessian[i] = 1
                    parametric_der_order_hessian[j] = 1

                # Note that fun_1 does not depend on u2, v2 (parameters of fun_2)
                hessian_x_u[:, j, i] = fun_1.space.compute_basis_matrix(
                    uv_1, parametric_derivative_orders=parametric_der_order_hessian
                ).dot(fun_1.coefficients.value.reshape((-1, 3)))

                # Note that fun_2 does not depend on u1, v1 (parameters of fun_1)
                hessian_x_u[:, j+2, i+2] = -fun_2.space.compute_basis_matrix(
                    uv_2, parametric_derivative_orders=parametric_der_order_hessian
                ).dot(fun_2.coefficients.value.reshape((-1, 3)))
                


        # 3) apply chain (and product) rule to compute grad_d_u and hess_d_u
        grad_d_u = grad_d_x @ grad_x_u
        hessian_d_u = 2 * (grad_x_u.T @ grad_x_u) + np.einsum('i,ijk->jk', grad_d_x.flatten(), hessian_x_u)
        if point_on_intersection is not None:
            grad_c_u = np.zeros((1, 4))
            grad_c_u[0, 0:2] = grad_c_x @ grad_x_u[:, 0:2]

            hessian_x_u_for_c = np.zeros((3, 4, 4))
            hessian_x_u_for_c[:, 0:2, 0:2] = hessian_x_u[:, 0:2, 0:2]
            grad_x_u_for_c = np.zeros((3, 4))
            grad_x_u_for_c[:, 0:2] = grad_x_u[:, 0:2]

            hessian_c_u = 2 * (grad_x_u_for_c.T @ grad_x_u_for_c) + np.einsum('i,ijk->jk', grad_c_x.flatten(), hessian_x_u_for_c)

            grad_L_u = grad_d_u + lambda_0 * grad_c_u
            hessian_L_u = hessian_d_u + lambda_0 * hessian_c_u

            gradient = grad_L_u
            hessian = hessian_L_u

        else:
            gradient = grad_d_u
            hessian = hessian_d_u

        # 4) Identify the active set
        # NOTE: this is a very ineqality constraint (design variable bounds)
        # instead of adding Lagrange multipliers we remove the design variable 
        # (i.e., parametric coordinate) when it's zero or one and the gradient 
        # tries to "push" it outside of the bounds
        remove_dvs_lower_bound = np.logical_and(starting_point == 0, gradient > 0)
        remove_dvs_upper_bound = np.logical_and(starting_point == 1, gradient < 0)
        remove_dvs = np.logical_or(remove_dvs_lower_bound, remove_dvs_upper_bound)

        # # NOTE: remove zero columns in the Hessian in edge cases
        # # (e.g., if there are infinitely many optimal solution)
        # # NOTE: This could potentially mask bugs
        remove_cols_hessian = np.where(~hessian.any(axis=1))[0]
        try:
            remove_dvs[remove_cols_hessian] = True
        except:
            break

        keep_dvs = np.logical_not(remove_dvs).flatten()

        reduced_gradient = gradient.flatten()[keep_dvs]
        reduced_hessian = hessian[keep_dvs][:, keep_dvs]

        if point_on_intersection is not None:
            grad_c_u = grad_c_u.flatten()[keep_dvs]

        if np.linalg.norm(residual) < 1e-12:
            newton_opt_converged = True
            break
        
        # compute step direction

        if point_on_intersection is not None:
            dim = len(reduced_hessian)
            newton_system_lhs = np.zeros((dim+1, dim+1))
            newton_system_lhs[0:dim, 0:dim] = reduced_hessian
            newton_system_lhs[dim, 0:dim] = grad_c_u
            newton_system_lhs[0:dim, dim] = grad_c_u

            newton_system_rhs = np.zeros((dim+1, ))
            newton_system_rhs[0:dim] = reduced_gradient
            newton_system_rhs[dim] = constraint_distance
        else:
            newton_system_lhs = reduced_hessian
            newton_system_rhs = reduced_gradient

        # step = np.linalg.solve(reduced_hessian, -reduced_gradient.flatten())
        step = np.linalg.solve(newton_system_lhs, -newton_system_rhs.flatten())

        if point_on_intersection is not None:
            new_uv_coords = step[0:dim]
            lambda_0 = step[dim]
        else:
            new_uv_coords = step

        # Update starting point with step
        starting_point[keep_dvs] = starting_point[keep_dvs] + new_uv_coords.flatten()
        # starting_point = starting_point + new_uv_coords.flatten()
        
        # clamp parameters between 0, 1
        starting_point = np.clip(starting_point, 0, 1)

    if newton_opt_converged is False:
        print("residual", np.linalg.norm(residual))
        print("starting_point", starting_point)
        print("------------------NEWTON OPT DID NOT CONVERGE------------------")
        return  None, None, None, None, None, None, starting_point, newton_opt_converged
        starting_point = starting_point_copy

    uv_1 = starting_point[0:2]
    uv_2 = starting_point[2:]

    stalled_trace = False
    if np.linalg.norm(starting_point - starting_point_copy) == 0:
        stalled_trace = True
    else:
        pass
        # print(starting_point, starting_point_copy)

    # Start tracing intersection
    # 1) Finding tangent and curvature vectors at u, v coordinates of both patches
    tangent_vectors_fun_1 = np.zeros((2, 3))
    tangent_vectors_fun_2 = np.zeros((2, 3))

    curvature_hessian_fun_1 = np.zeros((3, 2, 2))
    curvature_hessian_fun_2 = np.zeros((3, 2, 2))
    for p in range(2):
        parametric_derivative_orders = np.zeros((fun_1.space.num_parametric_dimensions, ), dtype=int)
        parametric_derivative_orders_curvature = np.zeros((fun_1.space.num_parametric_dimensions, ), dtype=int)
        parametric_derivative_orders[p] = 1
        parametric_derivative_orders_curvature[p] = 2

        tangent_vectors_fun_1[p, :] = fun_1.space.compute_basis_matrix(
            uv_1, parametric_derivative_orders=parametric_derivative_orders
        ).dot(fun_1.coefficients.value.reshape((-1, 3)))

        tangent_vectors_fun_2[p, :] = fun_2.space.compute_basis_matrix(
            uv_2, parametric_derivative_orders=parametric_derivative_orders
        ).dot(fun_2.coefficients.value.reshape((-1, 3)))

        for q in range(2):
            parametric_der_order_hessian = np.zeros((2, ), dtype=int)
            if p==q:
                parametric_der_order_hessian[p] = 2
            else:
                parametric_der_order_hessian[p] = 1
                parametric_der_order_hessian[q] = 1

            curvature_hessian_fun_1[:, p, q] = fun_1.space.compute_basis_matrix(
                uv_1, parametric_derivative_orders=parametric_der_order_hessian,
            ).dot(fun_1.coefficients.value.reshape((-1, 3)))

            curvature_hessian_fun_2[:, p, q] = fun_2.space.compute_basis_matrix(
                uv_2, parametric_derivative_orders=parametric_der_order_hessian,
            ).dot(fun_2.coefficients.value.reshape((-1, 3)))

    # 2) Compute surface normal for each patch
    N1 = np.cross(tangent_vectors_fun_1[0, :], tangent_vectors_fun_1[1, :])
    N2 = np.cross(tangent_vectors_fun_2[0, :], tangent_vectors_fun_2[1, :])

    # 3) Compute direction of intersection (normal of the two normals)
    T = np.cross(N1, N2)
    
    # First order Taylor series approximation (use as starting point for Newton iteration)
    reg_param = 1e-5
    delta_para_1 = np.linalg.solve(tangent_vectors_fun_1 @ tangent_vectors_fun_1.T + reg_param * np.eye(2), tangent_vectors_fun_1 @ T).reshape((-1, 2))
    delta_para_2 = np.linalg.solve(tangent_vectors_fun_2 @ tangent_vectors_fun_2.T + reg_param * np.eye(2), tangent_vectors_fun_2 @ T).reshape((-1, 2))

    # delta_uv = np.zeros((4, ))
    # delta_uv[0:2] = delta_para_1.flatten()
    # delta_uv[2:] = delta_para_2.flatten()

    # def compute_residual(x):
    #     first_order_term_1 = np.einsum('i,ij->j', x[0:2], tangent_vectors_fun_1)
    #     second_order_term_intermediate_1 = np.einsum('j,ijk->ik', x[0:2], curvature_hessian_fun_1)
    #     second_order_term_1 = np.einsum('ik,k->i', second_order_term_intermediate_1, x[0:2])

    #     first_order_term_2 = np.einsum('i,ij->j', x[2:], tangent_vectors_fun_2)
    #     second_order_term_intermediate_2 = np.einsum('j,ijk->ik', x[2:], curvature_hessian_fun_2)
    #     second_order_term_2 = np.einsum('ik,k->i', second_order_term_intermediate_2, x[2:])

    #     residual_1 = first_order_term_1 + 0.5 * second_order_term_1 - T
    #     residual_2 = first_order_term_2 + 0.5 * second_order_term_2 - T

    #     residual = np.vstack((residual_1, residual_2)).flatten()

    #     return residual, second_order_term_intermediate_1, second_order_term_intermediate_2

    # def compute_jacobian(second_order_term_intermediate_1, second_order_term_intermediate_2):
    #     jacobian = np.zeros((6, 4)) 
    #     jacobian[0:3, 0:2] = tangent_vectors_fun_1.T + second_order_term_intermediate_1
    #     jacobian[3:, 2:] = tangent_vectors_fun_2.T + second_order_term_intermediate_2

    #     return jacobian

    delta_uv = np.zeros((4, ))
    delta_uv[0:2] = delta_para_1.flatten()
    delta_uv[2:] = delta_para_2.flatten()
 
    converged = False
    for i in range(20):
        first_order_term_1 = np.einsum('i,ij->j', delta_uv[0:2], tangent_vectors_fun_1 )
        second_order_term_intermediate_1 = np.einsum('j,ijk->ik', delta_uv[0:2], curvature_hessian_fun_1)
        second_order_term_1 = np.einsum('ik,k->i', second_order_term_intermediate_1, delta_uv[0:2])
 
        first_order_term_2 = np.einsum('i,ij->j', delta_uv[2:], tangent_vectors_fun_2 )
        second_order_term_intermediate_2 = np.einsum('j,ijk->ik', delta_uv[2:], curvature_hessian_fun_2)
        second_order_term_2 = np.einsum('ik,k->i', second_order_term_intermediate_2, delta_uv[2:])
 
        residual_1 = first_order_term_1 + 0.5 * second_order_term_1 - T
        residual_2 = first_order_term_2 + 0.5 * second_order_term_2 - T
 
        residual = np.vstack((residual_1, residual_2)).flatten()
 
        jacobian = np.zeros((6, 4))
        jacobian[0:3, 0:2] = tangent_vectors_fun_1.T + second_order_term_intermediate_1
        jacobian[3:, 2:] = tangent_vectors_fun_2.T + second_order_term_intermediate_2
 
        reg_param = 1e-8
        delta_uv_updated = np.linalg.solve(jacobian.T @ jacobian + reg_param * np.eye(4), - jacobian.T @ residual)
        if np.linalg.norm(delta_uv_updated) < 1e-10:
            converged = True
            break
 
        delta_uv += delta_uv_updated


    # converged = False
    # x = delta_uv
    # lambd = 0 #1e-3
    # for i in range(500):
    #     residual, second_order_term_intermediate_1, second_order_term_intermediate_2 = compute_residual(x)
    #     jacobian = compute_jacobian(second_order_term_intermediate_1, second_order_term_intermediate_2)
        
    #     residual_norm = np.linalg.norm(residual)
    #     delta_uv_updated = np.linalg.solve(jacobian.T @ jacobian + lambd * np.eye(4), - jacobian.T @ residual)


    #     if np.linalg.norm(delta_uv_updated) < 1e-10:
    #         converged = True
    #         break

    #     delta_uv += delta_uv_updated

        # new_residual, _, _  = compute_residual(delta_uv)

        # if np.linalg.norm(new_residual) < residual_norm:
        #     # Update successful, accept the step and reduce lambda
        #     x = delta_uv
        #     lambd *= 0.8
        # else:
        #     # Update unsuccessful, reject the step and increase lambda
        #     lambd *= 2

    if converged:
        delta_uv_norm = delta_uv / np.linalg.norm(delta_uv + 1e-6)
    else:
        delta_uv = np.zeros((4, ))
        delta_uv[0:2] = delta_para_1.flatten()
        delta_uv[2:] = delta_para_2.flatten()
        delta_uv_norm = delta_uv / np.linalg.norm(delta_uv + 1e-8)


    # Computing step length
    if stalled_trace:
        print("stalled trace")
        # step = 1.01 * step_multiplier * 2 * 0.0005 # /np.linalg.norm(T)
        step = 1.0 * step_multiplier * 1 * 0.02 # /np.linalg.norm(T)
    else:
        step = step_multiplier *  1 *  0.02 # / np.linalg.norm(T)
    
    
    uv_1_updated_minus = uv_1 + step * delta_uv_norm[0:2] #, delta_uv[0:2] #  0.01 * delta_uv_norm[0:2] # 0.01 * delta_para_1_norm
    uv_2_updated_minus = uv_2 + step * delta_uv_norm[2:] #delta_uv[2:] # 0.01 * delta_uv_norm[2:] # 0.01 * delta_para_2_norm

    uv_1_updated_plus = uv_1 - step * delta_uv_norm[0:2] # delta_uv[0:2] #0.01 * delta_uv_norm[0:2] # 0.01 * delta_para_1_norm
    uv_2_updated_plus = uv_2 - step * delta_uv_norm[2:] #delta_uv[2:] # 0.01 * delta_uv_norm[2:] # 0.01 * delta_para_2_norm


    return uv_1, uv_2, np.clip(uv_1_updated_minus, 0, 1), np.clip(uv_2_updated_minus, 0, 1), np.clip(uv_1_updated_plus, 0, 1), np.clip(uv_2_updated_plus, 0, 1), starting_point, newton_opt_converged


def fixed_point_projection(fun_1 : Function, fun_2 : Function, starting_point=np.array([0.5, 0.5]), tol=1e-6):
    n = 0
    residual = fun_1.evaluate(parametric_coordinates=starting_point) - \
        fun_2.evaluate(parametric_coordinates=starting_point)
    
    
    while True:
        eval_pt_1 = fun_1.evaluate(parametric_coordinates=starting_point).value
        proj_1 = fun_2.project(eval_pt_1, plot=False, grid_search_density_parameter=1)
        eval_pt_2 = fun_2.evaluate(parametric_coordinates=proj_1).value
        proj_2 = fun_1.project(eval_pt_2, plot=False, grid_search_density_parameter=1)
        starting_point = proj_2
        n += 1
        eval_pt_norm = np.linalg.norm(eval_pt_1-eval_pt_2)
        # print(eval_pt_norm)
        if eval_pt_norm < tol:
            break

        if n > 100:
            warnings.warn(f"fixed_point_projection iteration did not converge to {tol}. Norm is {eval_pt_norm}")
            break


    d_fun_1_d_para = np.zeros((2, 3))
    d_fun_2_d_para = np.zeros((2, 3))
    for k in range(fun_1.space.num_parametric_dimensions):
        parametric_derivative_orders = np.zeros((fun_1.space.num_parametric_dimensions, ), dtype=int)
        parametric_derivative_orders[k] = 1
        d_fun_1_d_para[k, :] = fun_1.space.compute_basis_matrix(
            proj_2, parametric_derivative_orders=parametric_derivative_orders
        ).dot(fun_1.coefficients.value.reshape((-1, 3)))

        d_fun_2_d_para[k, :] = fun_2.space.compute_basis_matrix(
            proj_1, parametric_derivative_orders=parametric_derivative_orders
        ).dot(fun_2.coefficients.value.reshape((-1, 3)))
    
    N1 = np.cross(d_fun_1_d_para[0, :], d_fun_1_d_para[1, :])
    N2 = np.cross(d_fun_2_d_para[0, :], d_fun_2_d_para[1, :])

    T = np.cross(N1, N2)

    delta_para_1 = np.linalg.solve(d_fun_1_d_para @ d_fun_1_d_para.T, d_fun_1_d_para @ T).reshape((-1, 2))
    delta_para_2 = np.linalg.solve(d_fun_2_d_para @ d_fun_2_d_para.T, d_fun_2_d_para @ T).reshape((-1, 2))

    
    delta_para_1_norm = delta_para_1 / np.linalg.norm(delta_para_1)
    delta_para_2_norm = delta_para_2 / np.linalg.norm(delta_para_2)

    print("delta_para_1_norm", delta_para_1_norm)
    print("proj_1", proj_1)
    print("proj_2", proj_2)
    # exit()

    # print(delta_para_1_norm)
    # print(delta_para_2_norm)
    step = 1e-2
    # print(proj_1, proj_1 + step * delta_para_1_norm)
    # print(proj_2, proj_2 + step * delta_para_2_norm)
    # print("\n")
    # exit()

    return proj_1, proj_2, proj_1 + step * delta_para_1_norm


def compare_floats(x, y, decimals=5):
    # Round both numbers to the specified number of decimal places
    x_rounded = round(x, decimals)
    y_rounded = round(y, decimals)
    
    # Perform the comparison
    return x_rounded > y_rounded