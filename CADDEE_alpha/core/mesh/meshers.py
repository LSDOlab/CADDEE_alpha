import csdl_alpha as csdl
from CADDEE_alpha.core.mesh.mesh import Discretization, SolverMesh, DiscretizationsDict
import numpy as np
from typing import Union
from dataclasses import dataclass
from CADDEE_alpha.utils.caddee_dict import CADDEEDict
import lsdo_function_spaces as fs


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
            spanwise_mean_y = ((quantity[:, index, 1]**2)**0.5 + (quantity[:, symmetric_index, 1]**2)**0.5) / 2

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
    _geom = None
    _num_chord_wise = None
    _num_spanwise = None
    _LE_points_para = None
    _TE_points_para = None
    _chordwise_spacing = None

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
            if self._chordwise_spacing == "linear":
                chord_surface = csdl.linear_combination(TE_points_csdl, LE_points_csdl, self._num_chord_wise+1).reshape((self._num_chord_wise+1, self._num_spanwise+1, 3))
            
            elif self._chordwise_spacing == "cosine":
                chord_surface = cosine_spacing(
                    self._num_spanwise, 
                    None,
                    csdl.linear_combination(TE_points_csdl, LE_points_csdl, self._num_chord_wise+1),
                    self._num_chord_wise
                )

            chord_surface = chord_surface.reshape((self._num_chord_wise+1, self._num_spanwise+1, 3))
            chord_surface_sym = make_mesh_symmetric(chord_surface, self._num_spanwise, spanwise_index=1)

            self.nodal_coordinates = chord_surface_sym

            return self

class CamberSurfaceDict(DiscretizationsDict):
    def __getitem__(self, key) -> CamberSurface:
        return super().__getitem__(key)

class VLMMesh(SolverMesh):
    discretizations : dict[CamberSurface] = CamberSurfaceDict()

def make_vlm_surface(
    wing_comp,
    num_spanwise: int,
    num_chordwise: int, 
    spacing_spanwise: str = 'linear',
    spacing_chordwise: str = 'linear',
    ignore_camber: bool = False, 
    plot: bool = False,
    grid_search_density: int = 10,
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

    if wing_comp.geometry is None:
        raise Exception("Cannot generate mesh for component with geoemetry=None")

    if num_spanwise % 2 != 0:
        raise Exception("Number of spanwise panels must be even.")

    wing_geometry = wing_comp.geometry

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
        LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

        TE_points_1 = np.linspace(TE_left_point, TE_mid_point, num_spanwise_half + 1)
        TE_points_2 = np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1)
        TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
    
    elif spacing_spanwise == "cosine":
        num_spanwise_half = int(num_spanwise / 2)
        LE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_left_point, num_spanwise_half + 1))
        LE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1), flip=True)
        LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

        TE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_left_point, num_spanwise_half + 1))
        TE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1), flip=True)
        TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
    else:
        raise NotImplementedError

    LE_points_para = wing_geometry.project(LE_points, plot=plot)
    TE_points_para = wing_geometry.project(TE_points, plot=plot)

    LE_points_csdl = wing_geometry.evaluate(LE_points_para)
    TE_points_csdl = wing_geometry.evaluate(TE_points_para)

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
        vlm_mesh = CamberSurface(nodal_coordinates=chord_surface_sym)
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
            csdl.Variable(shape=(3, ), value=np.array([0., 0., 0.5])),
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
    
        vlm_mesh = CamberSurface(nodal_coordinates=camber_surface)
        vlm_mesh._geom = wing_geometry
        vlm_mesh._lower_wireframe_para = lower_surace_wireframe_para
        vlm_mesh._upper_wireframe_para = upper_surace_wireframe_para
        vlm_mesh._num_chord_wise = num_chordwise
        vlm_mesh._num_spanwise = num_spanwise

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

    _geom = None
    _LE_points_parametric = None
    _TE_points_parametric = None
    _norm_node_center = None
    _norm_beam_width = None
    _node_top_parametric = None
    _node_bottom_parametric = None


    def _update(self):
        LE_points_csdl = self._geom.evaluate(self._LE_points_parametric).reshape((self.num_beam_nodes, 3))
        TE_points_csdl = self._geom.evaluate(self._TE_points_parametric).reshape((self.num_beam_nodes, 3))

        beam_width_raw = csdl.norm((LE_points_csdl - TE_points_csdl) * self._norm_beam_width, axes=(1, ))
        beam_width_nodal = make_mesh_symmetric(beam_width_raw, self.num_beam_nodes, spanwise_index=None)
        self.beam_width = (beam_width_nodal[0:-1] + beam_width_nodal[1:]) / 2

        node_top = self._geom.evaluate(self._node_top_parametric).reshape((self.num_beam_nodes, 3))
        node_bottom = self._geom.evaluate(self._node_bottom_parametric).reshape((self.num_beam_nodes, 3))

        self.nodal_coordinates = make_mesh_symmetric((node_top + node_bottom) / 2, self.num_beam_nodes)

        beam_height_raw = csdl.norm((node_top - node_bottom), axes=(1, ))
        beam_height_nodal = make_mesh_symmetric(beam_height_raw, self.num_beam_nodes, spanwise_index=None)
        self.beam_height = (beam_height_nodal[0:-1] + beam_height_nodal[1:]) / 2


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
        set spacing for the beam nodes, by default "linear"
    
    plot : bool, optional
        plot the projections, by default False

    grid_search_density : int, optional
        parameter to refine the quality of projections (note that the higher this parameter 
        the longer the projections will take; 

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
            LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

            TE_points_1 = np.linspace(TE_left_point, TE_mid_point, num_spanwise_half + 1)
            TE_points_2 = np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1)
            TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
        
        elif spacing == "cosine":
            num_spanwise_half = int(num_beam_nodes / 2)
            LE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_left_point, num_spanwise_half + 1))
            LE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(LE_mid_point, LE_right_point, num_spanwise_half + 1), flip=True)
            LE_points = np.vstack((LE_points_1, LE_points_2[1:, :]))

            TE_points_1 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_left_point, num_spanwise_half + 1))
            TE_points_2 = cosine_spacing(num_spanwise_half + 1, np.linspace(TE_mid_point, TE_right_point, num_spanwise_half + 1), flip=True)
            TE_points = np.vstack((TE_points_1, TE_points_2[1:, :]))
        else:
            raise NotImplementedError
    
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

    beam_width_raw = csdl.norm((LE_points_csdl - TE_points_csdl) * norm_beam_width, axes=(1, ))
    beam_width_nodal = make_mesh_symmetric(beam_width_raw, num_beam_nodes, spanwise_index=None)
    beam_width = (beam_width_nodal[0:-1] + beam_width_nodal[1:]) / 2

    offset = np.array([0., 0., 2])
    node_top_parametric = wing_geometry.project(beam_nodes_raw.value + offset, direction=np.array([0., 0., -1]),  plot=plot)
    node_bottom_parametric = wing_geometry.project(beam_nodes_raw.value - offset, direction=np.array([0., 0., 1]), plot=plot)

    node_top = wing_geometry.evaluate(node_top_parametric).reshape((num_beam_nodes, 3))
    node_bottom = wing_geometry.evaluate(node_bottom_parametric).reshape((num_beam_nodes, 3))

    # Beam nodes will be at the center vertically (i.e., at the mid-point between upper and lower wing surface)
    beam_nodes = make_mesh_symmetric((node_top + node_bottom) / 2, num_beam_nodes)

    beam_height_raw = csdl.norm((node_top - node_bottom), axes=(1, ))
    beam_height_nodal = make_mesh_symmetric(beam_height_raw, num_beam_nodes, spanwise_index=None)
    beam_height = (beam_height_nodal[0:-1] + beam_height_nodal[1:]) / 2

    # Compute top and bottom thicknesses of the beam
    # TODO: handle more than 1 material? maybe not needed
    if wing_comp.quantities.material_properties.material is not None:
        thickness = wing_comp.quantities.material_properties.thickness
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
        top_thickness_grid = thickness.evaluate(top_grid).reshape((num_beam_nodes-1, num_chordwise*num_spanwise, 1))
        bottom_thickness_grid = thickness.evaluate(bottom_grid).reshape((num_beam_nodes-1, num_chordwise*num_spanwise, 1))

        top_thickness = csdl.Variable(shape=(num_beam_nodes-1,), value=0.)
        bottom_thickness = csdl.Variable(shape=(num_beam_nodes-1,), value=0.)
        for i in csdl.frange(num_beam_nodes-1):
            top_thickness = top_thickness.set(csdl.slice[i], csdl.average(top_thickness_grid[i,:,:]))
            bottom_thickness = bottom_thickness.set(csdl.slice[i], csdl.average(bottom_thickness_grid[i,:,:]))



    beam_mesh = OneDBoxBeam(
        nodal_coordinates=beam_nodes,
        beam_height=beam_height,
        beam_width=beam_width,
        num_beam_nodes=num_beam_nodes,
        top_skin_thickness=top_thickness,
        bottom_skin_thickness=bottom_thickness,
    )

    wing_comp._discretizations[f"{wing_comp._name}_1d_beam_mesh"] = beam_mesh

    beam_mesh._geom = wing_geometry
    beam_mesh._LE_points_parametric = LE_points_parametric
    beam_mesh._TE_points_parametric = TE_points_parametric
    beam_mesh._norm_beam_width = norm_beam_width
    beam_mesh._norm_node_center = norm_node_center
    beam_mesh._node_top_parametric = node_top_parametric
    beam_mesh._node_bottom_parametric = node_bottom_parametric

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
    plot: bool = False
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
    if num_azimuthal > 1:
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
                cartesian_coordinates[i, j, :] = p.value + radius_vec[i].value * (np.cos(thetha_vec[j]) * v1.value \
                                                    + np.sin(thetha_vec[j]) * v2.value)
        
        disk_mesh_parametric = rotor_geometry.project(cartesian_coordinates, plot=plot)
        disk_mesh = rotor_geometry.evaluate(disk_mesh_parametric).reshape((num_radial, num_azimuthal, 3))

        rotor_mesh_parameters.disk_mesh = disk_mesh
        rotor_mesh_parameters._disk_parametric = disk_mesh_parametric

    rotor_comp._discretizations[f"{rotor_comp._name}_rotor_mesh_parameters"] = rotor_mesh_parameters

    return rotor_mesh_parameters
        
