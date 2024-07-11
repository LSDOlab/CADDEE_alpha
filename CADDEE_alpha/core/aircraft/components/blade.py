import lsdo_function_spaces as lfs
from lsdo_function_spaces import FunctionSet
from CADDEE_alpha.core.component import Component
from CADDEE_alpha.core.aircraft.components.wing import Wing, WingParameters
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import time
import numpy as np
from dataclasses import dataclass
import csdl_alpha as csdl
from typing import Union
import lsdo_geo as lg
import gmsh




class Blade(Wing):
    """The Blade component class.
    
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
        geometry : Union[lfs.FunctionSet, None]=None,
        tight_fit_ffd: bool = False,
        orientation: str = "horizontal",
        **kwargs
    ) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super(Wing, self).__init__(geometry=geometry, **kwargs)
        
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

        # Check if wing is over-parameterized
        if all(arg is not None for arg in [AR, S_ref, span]):
            raise Exception("Wing comp over-parameterized: Cannot specifiy AR, S_ref, and span at the same time.")
        # Check if wing is under-parameterized
        if sum(1 for arg in [AR, S_ref, span] if arg is None) >= 2:
            raise Exception("Wing comp under-parameterized: Must specify two out of three: AR, S_ref, and span.")
        
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
                t1 = time.time()
                # Set the wetted area
                self.parameters.S_wet = self.quantities.surface_area
                t2 = time.time()
                # print("time for copmuting wetted area", t2-t1)

                t3 = time.time()
                # Make the FFD block upon instantiation
                ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd, degree=(1, 2, 1), num_coefficients=(2, 2, 2))
                # ffd_block.plot()
                t4 = time.time()
                print("time for making ffd_block", t4-t3)
                # ffd_block.plot()

                t5 = time.time()
                # Compute the corner points of the wing 
                self._LE_base_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0., 1.])), plot=False, extrema=True)
                self._LE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 1.])), plot=False, extrema=True)

                self._TE_base_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.])), plot=False, extrema=True)
                self._TE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.])), plot=False, extrema=True)
                t6 = time.time()

                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=False)

                # print("time for computing corner points", t6-t5)
            # internal geometry projection info
            self._dependent_geometry_points = [] # {'parametric_points', 'function_space', 'fitting_coords', 'mirror'}
            self._base_geometry = self.geometry.copy()
    
    def construct_cross_section(
            self,
            top_geometry:lg.Geometry,
            bottom_geometry:lg.Geometry,
            spar_locations:np.ndarray=None,
            spar_function_space:lfs.FunctionSpace=None,
            surf_index:int=1000,
            offset:np.ndarray=np.array([0., 0., .23]), 

        ):
        '''
        Construct the internal geometry of the rotor blade based on input 
        parameters and the upper and lower surfaces of the blade 
        '''
        csdl.check_parameter(spar_locations, "spar_locations", types=np.ndarray, allow_none=True)
        csdl.check_parameter(surf_index, "surf_index", types=int)

        # Check if spar and rib locations are between 0 and 1
        if spar_locations is not None:
            if not np.all((spar_locations > 0) & (spar_locations < 1)):
                raise ValueError("all spar locations must be between 0 and 1 (excluding the endpoints)")
        if spar_locations is None:
            spar_locations = np.array([0.25, 0.75])
        
        blade = self

        #get surface indices
        top_index =list(top_geometry.function_names.keys())[0]
        bot_index =list(bottom_geometry.function_names.keys())[0]
        front_spar_index = surf_index
        rear_spar_index = surf_index+1
        
        num_spars = spar_locations.shape[0]
        if num_spars == 1:
            raise Exception("Cannot have single spar. Provide at least two normalized spar locations.")
        
        num_spanwise = 10
        if spar_function_space is None:
            spar_function_space = lfs.BSplineSpace(2, (1, 1), (num_spanwise, 2))

        #TODO: need to be able to determine where the root and where the tip is...
        # gather important points 
        root_te = blade.geometry.evaluate(blade._TE_tip_point, non_csdl=True)
        root_le = blade.geometry.evaluate(blade._LE_tip_point, non_csdl=True)
        r_tip_te = blade.geometry.evaluate(blade._TE_base_point, non_csdl=True)
        r_tip_le = blade.geometry.evaluate(blade._LE_base_point, non_csdl=True)
        
        # get spar start/end points (root and tip)
        root_tip_pts = np.zeros((num_spars, 2, 3))
        for i in range(num_spars):
            root_tip_pts[i,0] = (1-spar_locations[i]) * root_le + spar_locations[i] * root_te
            root_tip_pts[i,1] = (1-spar_locations[i]) * r_tip_le + spar_locations[i] * r_tip_te

        # #compute projection direction (perpendicular to the chordline)
        # chord_direction_root = root_le-root_te
        # chord_direction_tip = r_tip_le-r_tip_te
        spanwise_direction = np.array([0,1,0])
        # proj_direction_root = np.cross(chord_direction_root,spanwise_direction)
        # proj_direction_tip = np.cross(chord_direction_tip,spanwise_direction)
        
        # ribs_top_root = top_geometry.project(root_tip_pts[:,0], direction=proj_direction_root, grid_search_density_parameter=10,plot=False)
        # ribs_bottom_root = bottom_geometry.project(root_tip_pts[:,0], direction=-proj_direction_root, grid_search_density_parameter=10,plot=False)
        # ribs_top_tip = top_geometry.project(root_tip_pts[:,1], direction=proj_direction_tip, grid_search_density_parameter=10,plot=False)
        # ribs_bottom_tip = bottom_geometry.project(root_tip_pts[:,1], direction=-proj_direction_tip, grid_search_density_parameter=10,plot=False)

        u_coords=np.linspace(0,1,num_spanwise)
        # v_coords = 

        parametric_LE = np.vstack([u_coords,np.ones((num_spanwise))]).T
        parametric_TE = np.vstack([u_coords,np.zeros((num_spanwise))]).T

        physical_LE = blade.geometry.evaluate([(top_index,parametric_LE)],non_csdl=True)
        physical_TE = blade.geometry.evaluate([(top_index,parametric_TE)],non_csdl=True)
        
        #determine chord direction for each section
        chord_direction = physical_LE-physical_TE
        #determine direction perpendicular to chord direction in section plane
        proj_direction = np.cross(chord_direction,np.tile(spanwise_direction,(num_spanwise,1)))


        fwd_spar_chord_loc = (1-spar_locations[0]) * physical_LE + spar_locations[0] * physical_TE
        rear_spar_chord_loc = (1-spar_locations[1]) * physical_LE + spar_locations[1] * physical_TE
        # print("Fwd spar locations")
        # print(fwd_spar_chord_loc)
        # print("Rear spar locations")
        # print(rear_spar_chord_loc)

        # get parametric coordinates for spars
        # spar_top_front = np.zeros((num_spanwise,2))
        # spar_bot_front = np.zeros((num_spanwise,2))
        # spar_top_rear = np.zeros((num_spanwise,2))
        # spar_bot_rear = np.zeros((num_spanwise,2))
        
        spar_top_front = np.empty((num_spanwise),dtype='O,O')
        spar_bot_front = np.empty((num_spanwise),dtype='O,O')
        spar_top_rear = np.empty((num_spanwise),dtype='O,O')
        spar_bot_rear = np.empty((num_spanwise),dtype='O,O')
        for i in range(num_spanwise):
            # [(_,spar_top_front[i,:])] = top_geometry.project(fwd_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)
            # [(_,spar_top_rear[i,:])] = top_geometry.project(rear_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)
            # [(_,spar_bot_rear[i,:])] = bottom_geometry.project(rear_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)
            # [(_,spar_bot_front[i,:])] = bottom_geometry.project(fwd_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)
            
            spar_top_front[i] = top_geometry.project(fwd_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)[0]
            # print('spar_top_front')
            # print(spar_top_front[i])
            spar_top_rear[i] = top_geometry.project(rear_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)[0]
            spar_bot_rear [i]= bottom_geometry.project(rear_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)[0]
            spar_bot_front[i] = bottom_geometry.project(fwd_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)[0]
        
        # spars_physical=geometry.evaluate([(174,spar_top_front),
        #                                     (174,spar_bot_front),
        #                                     (175,spar_top_rear),
        #                                     (175,spar_bot_rear)],plot=True)

        # spar_top_front_physical=geometry.evaluate(spar_top_front,plot=True)
        # spar_top_rear_physical=geometry.evaluate(spar_top_rear,plot=True)
        # spar_bot_front_physical=geometry.evaluate(spar_bot_front,plot=True)
        # spar_bot_rear_physical=geometry.evaluate(spar_bot_rear,plot=True)
        
        # ribs_top_front_physical=geometry.evaluate([(174,ribs_bot_front)],plot=True)
        # ribs_top_front_physical=geometry.evaluate([(174,ribs_bot_front)],plot=True)
        
        # print(spar_top_front)
        # print(type(spar_top_front))
        # print(spar_top_rear)

        #create spar surfaces

        #front spar
        parametric_points_front = spar_top_front.tolist()+spar_bot_front.tolist()
        fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
        fitting_values = self.geometry.evaluate(parametric_points_front)
        coefficients = spar_function_space.fit(fitting_values, fitting_coords)
        front_spar = lfs.Function(spar_function_space, coefficients)
        front_spar_name="Blade_spar_front"
        self._add_geometry(front_spar_index, front_spar, front_spar_name)

        #rear spar
        parametric_points_rear = spar_top_rear.tolist()+spar_bot_rear.tolist()
        fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
        fitting_values = self.geometry.evaluate(parametric_points_rear)
        coefficients = spar_function_space.fit(fitting_values, fitting_coords)
        rear_spar = lfs.Function(spar_function_space, coefficients)
        rear_spar_name = "Blade_spar_rear"
        self._add_geometry(rear_spar_index, rear_spar, rear_spar_name)

        front_spar_geometry = self.create_subgeometry(search_names=front_spar_name)
        rear_spar_geometry = self.create_subgeometry(search_names=rear_spar_name)

        return front_spar_geometry,rear_spar_geometry
    
    def create_beam_xs_meshes(
            self,
            top_geometry:lg.Geometry,
            bottom_geometry:lg.Geometry,
            front_spar_geometry:lg.Geometry=None,
            rear_spar_geometry:lg.Geometry=None,
            num_spanwise=10
        ):

        #get surface indices
        top_index =list(top_geometry.function_names.keys())[0]
        bottom_index =list(bottom_geometry.function_names.keys())[0]
        if front_spar_geometry is not None:
            front_spar_index =list(front_spar_geometry.function_names.keys())[0]
        if rear_spar_geometry is not None:
            rear_spar_index =list(rear_spar_geometry.function_names.keys())[0]

        u_coords = np.linspace(0,1,num_spanwise)
        v_coords = np.linspace(0,1,10)
        xs = []
        for u_coord in u_coords:
            parametric_top = [(top_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            parametric_bot = [(bottom_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            top_pts = self.geometry.evaluate(parametric_top, plot=False)
            bot_pts = self.geometry.evaluate(parametric_bot, plot=False)

            pts=[top_pts,bot_pts]
            if front_spar_geometry is not None:
                insertion_pts = []
                parametric_front_spar = [(front_spar_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
                front_spar_pts = self.geometry.evaluate(parametric_front_spar, plot=False)
                pts.append(front_spar_pts)
                #add the start and end points to the top and bottom surface to enforce intersection
                
                #front spar point added to top curve
                sorted_indices=np.argsort(np.linalg.norm(top_pts.value - front_spar_pts.value[0,:],axis=1))
                insert_idx,insert_idx_end = np.sort(sorted_indices[0:2])
                # if (insert_idx_end-insert_idx) != 1:
                #     raise Exception('Invalid front spar and upper surface intersection')
                
                insertion_pts.append(insert_idx)

                # top_pts = csdl.vstack([top_pts[:insert_start,:],
                #                        front_spar_pts[0:1,:],
                #                        top_pts[insert_start:,:]])
                # # print("After insertion:")
                # # print(top_pts.value)
                # pts[0]=top_pts
                # print('top pts new len')
                # print(top_pts.shape)

                #front spar point added to bottom curve
                insert_idx=np.argmin(np.linalg.norm(bot_pts.value - front_spar_pts.value[-1,:],axis=1))
                # insert_start,insert_end = np.sort(sorted_indices[0:2])
                #check if the insert idx is at the start or the end of the list
                if insert_idx == 0:
                    insert_start = 0
                    insert_end = 1
                if insert_idx == bot_pts.shape[0]-1:
                    insert_start = insert_idx-1
                    insert_end = insert_idx
                #check surrounding points to determine which side of closest point to insert spar point
                surrounding_indices = [insert_idx-1,insert_idx+1]
                #determine which of the surrounding indices 
                surrounding_to_insert = np.linalg.norm(bot_pts.value[surrounding_indices,:] - bot_pts.value[insert_idx,:],axis=1)
                surrounding_to_spar = np.linalg.norm(bot_pts.value[surrounding_indices,:] - front_spar_pts.value[-1,:],axis=1)
                print('indices:')
                print(surrounding_indices)
                print("s to i")
                print(surrounding_to_insert)
                print("s to s")
                print(surrounding_to_spar)
                print("distances")
                print(surrounding_to_insert-surrounding_to_spar)
                print(np.argmin(surrounding_to_insert-surrounding_to_spar))

                #TODO: NEED TO FIX THIS logic to make sure we are inserting the b-spline pt at the right spot
                # surrounding_idx = surrounding_indices[np.argmin(surrounding_to_insert-surrounding_to_spar)]
                if np.argmin(surrounding_to_insert-surrounding_to_spar) == 0:
                    insert_idx = insert_idx+1
                elif np.argmin(surrounding_to_insert-surrounding_to_spar) == 1:
                    insert_idx=insert_idx

                # print(surrounding_idx)

                # sorted_indices=np.sort([insert_idx,surrounding_idx])
                # insert_start = sorted_indices[0]
                # insert_end = sorted_indices[1]

                # if (insert_end-insert_start) != 1:
                #     raise Exception('Invalid front spar and lower surface intersection')

                insertion_pts.append(insert_idx)

                # print('Before insertion:')
                # print(top_pts.value)
                # bot_pts = csdl.vstack([bot_pts[:insert_start,:],front_spar_pts[-1:,:],bot_pts[insert_end:,:]])

                # pts[1]=bot_pts

            if rear_spar_geometry is not None:
                parametric_rear_spar = [(rear_spar_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
                rear_spar_pts = self.geometry.evaluate(parametric_rear_spar, plot=False)
                pts.append(rear_spar_pts)
            
            xs.append(pts)
            print(insertion_pts)

        gmsh.initialize()
        for pts in xs:
            top_pts = pts[0]
            bot_pts = pts[1]
            lc =0.001
            top_pts_list=[]
            bot_pts_list=[]
            for top_pt in top_pts.value:
                top_pts_list.append(gmsh.model.occ.add_point(top_pt[0],top_pt[1],top_pt[2]))
            for bot_pt in bot_pts.value:
                bot_pts_list.append(gmsh.model.occ.add_point(bot_pt[0],bot_pt[1],bot_pt[2]))
            
            #check that xc_pts1 and 2 have the same start and end points
            # if, not add it to one of the lower surface
            if not np.equal(top_pts[0,:].value,bot_pts[-1,:].value).all():
                pt = top_pts[0,:].value
                bot_pts_list.append(gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))
            if not np.equal(top_pts[-1,:].value,bot_pts[0,:].value).all():
                pt = top_pts[-1,:].value
                bot_pts_list.insert(0,gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))
                insertion_pts[1]+=1
                       
            # add curve loop
            # CL1 = gmsh.model.occ.add_curve_loop([spline1,spline2])

            # try to offset, but if offset returns nothing, make a solid section
            # CL2_tags_raw = gmsh.model.occ.offset_curve(CL1,.005)

            # gmsh.model.geo.add_plane_surface([spline])
            # gmsh.model.occ.add_plane_surface([CL1])
            # gmsh.model.occ.add_plane_surface([CL2])

            # #if the offset curve process fails, create a solid section
            # try:
            #     #need to get proper list of curves by removing dim part of each tuple
            #     CL2_tags = [curve[1] for curve in CL2_tags_raw ]
            #     CL2 = gmsh.model.occ.add_curve_loop(CL2_tags)
            #     gmsh.model.occ.add_plane_surface([CL1,CL2])
            # except:
            #     gmsh.model.occ.add_plane_surface([CL1])
            
            #if there are spars, add their geometry to the occ represenation
            if rear_spar_geometry and front_spar_geometry is not None:
                front_spar_pts = pts[2]
                front_spar_pts_list=[]
                for pt3 in front_spar_pts.value:
                    front_spar_pts_list.append(gmsh.model.occ.add_point(pt3[0],pt3[1],pt3[2]))
                spline3 = gmsh.model.occ.add_spline(front_spar_pts_list)

                top_pts_list.insert(insertion_pts[0],front_spar_pts_list[0])
                bot_pts_list.insert(insertion_pts[1],front_spar_pts_list[-1])

                # gmsh.model.occ.add_plane_surface([CL1])

                # print(CL1)
                # print(CL2_tags_raw)
                # output = gmsh.model.occ.intersect(CL2_tags_raw,[(1,spline3)],False,False)
                # output = gmsh.model.occ.fragment([(1,spline1),(1,spline2)],[(1,spline3)])
                # output = gmsh.model.occ.cut([(1,spline1)],[(1,spline3)],removeObject=False,removeTool=False)
                # output2 = gmsh.model.occ.cut([(1,spline2)],[(1,spline3)],removeObject=False,removeTool=False)
                # print(output)
                # print(output2)
                # curve_tags = [curve[1] for curve in output[0]]
                # curve_tags.append(spline3)
                # FS_CL1 = gmsh.model.occ.add_curve_loop(curve_tags)
                # CL2_tags_raw = gmsh.model.occ.offset_curve(FS_CL1,-.005)
                
                rear_spar_pts = pts[3]
                pts4=[]
                for pt4 in rear_spar_pts.value:
                    pts4.append(gmsh.model.occ.add_point(pt4[0],pt4[1],pt4[2]))
                spline4 = gmsh.model.occ.add_spline(pts4)

            # bspline = gmsh.model.geo.add_compound_bspline([spline])
            #add the upper and lower skins splines
            print
            # spline1 = gmsh.model.occ.add_spline(top_pts_list)
            spline2 = gmsh.model.occ.add_spline(bot_pts_list)

            gmsh.model.occ.synchronize()

            gmsh.fltk.run()


        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005)

        # gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        gmsh.fltk.run()

        gmsh.finalize()


        return
    
# class CrossSection(Component):
#     def __init__(self, **kwargs):