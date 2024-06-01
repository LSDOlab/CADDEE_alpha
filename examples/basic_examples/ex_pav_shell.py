import CADDEE_alpha as cd
import csdl_alpha as csdl
from VortexAD.core.vlm.vlm_solver import vlm_solver
import numpy as np
import lsdo_function_spaces as fs
import lsdo_geo as lg

recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

# Import and plot the geometry
pav_geometry = cd.import_geometry('pav.stp')


def define_base_config(caddee: cd.CADDEE):
    aircraft = cd.aircraft.components.Aircraft(geometry=pav_geometry)

    # aircraft.quantities.mass_properties.mass = 815
    # aircraft.quantities.mass_properties.cg_vector = np.array([-1, 1., -1])

    # wing
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    

    wing = cd.aircraft.components.Wing(
        AR=7, S_ref=15, geometry=wing_geometry, tight_fit_ffd=False,
    )
    # wing.plot()


    aircraft.comps["wing"] = wing

    wing.quantities.mass_properties.mass = 250
    wing.quantities.mass_properties.cg_vector = np.array([-3, 0., -1])

    pav_vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_surface(
        wing, 30, 5, grid_search_density=20, ignore_camber=False,
    )
    # pav_geometry.plot_meshes(wing_camber_surface.nodal_coordinates.value)


    # tail
    tail_geometry = aircraft.create_subgeometry(search_names=["Stabilizer"])
    tail = cd.aircraft.components.Wing(
        AR=5, S_ref=2, geometry=tail_geometry, tight_fit_ffd=False,
    )
    aircraft.comps["tail"] = tail

    tail_camber_surface = cd.mesh.make_vlm_surface(
        tail, 30, 6, grid_search_density=10, ignore_camber=True,
    )
    # pav_geometry.plot_meshes(tail_camber_surface.nodal_coordinates.value)

    tail.quantities.mass_properties.mass = 250
    tail.quantities.mass_properties.cg_vector = np.array([-3, 0., -0])

    pav_vlm_mesh.discretizations["wing_camber_surface"] = wing_camber_surface
    pav_vlm_mesh.discretizations["tail_camber_surface"] = tail_camber_surface

    

    base_config = cd.Configuration(aircraft)
    mesh_container = base_config.mesh_container
    mesh_container["vlm_mesh"] = pav_vlm_mesh
    caddee.base_configuration = base_config


def define_conditions(caddee: cd.CADDEE):
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    pitch_angle = csdl.ImplicitVariable(shape=(1, ), value=np.deg2rad(2.))
    cruise = cd.aircraft.conditions.CruiseCondition(
        altitude=1e3,
        range=60e3,
        speed=50.,
        pitch_angle=pitch_angle,
    )
    cruise.configuration = base_config
    conditions["cruise"] = cruise
    return pitch_angle


def define_analysis(caddee: cd.CADDEE, pitch_angle=None):
    cruise = caddee.conditions["cruise"]
    cruise_config = cruise.configuration
    mesh_container = cruise_config.mesh_container
    
    aircraft = cruise_config.system
    tail = aircraft.comps["tail"]
    elevator = csdl.ImplicitVariable(shape=(1, ), value=0.)
    tail.actuate(elevator)


    cruise.finalize_meshes()

    # VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]

    wing_camber_surface = vlm_mesh.discretizations["wing_camber_surface"]
    tail_camber_surface = vlm_mesh.discretizations["tail_camber_surface"]

    camber_surface_coordinates = [wing_camber_surface.nodal_coordinates, tail_camber_surface.nodal_coordinates]
    camber_surface_nodal_velocities = [wing_camber_surface.nodal_velocities, tail_camber_surface.nodal_velocities]

    vlm_outputs = vlm_solver(camber_surface_coordinates, camber_surface_nodal_velocities, alpha_ML=None)



def construct_internal_geometry():
    aircraft = cd.aircraft.components.Aircraft(geometry=pav_geometry)
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    wing = cd.aircraft.components.Wing(
        AR=7, S_ref=15, geometry=wing_geometry, tight_fit_ffd=False,
    )

    # TODO: consult Andrew regarding whether the function coefficients should be csdl variables or not
    
    # gather important points (right only)
    root_te = wing_geometry.evaluate(wing._TE_mid_point).value
    root_le = wing_geometry.evaluate(wing._LE_mid_point).value
    r_tip_te = wing_geometry.evaluate(wing._TE_right_point).value
    r_tip_le = wing_geometry.evaluate(wing._LE_right_point).value

    root_25 = (3 * root_le + root_te) / 4
    root_75 = (root_le + 3 * root_te) / 4
    tip_25 = (3 * r_tip_le + r_tip_te) / 4
    tip_75 = (r_tip_le + 3 * r_tip_te) / 4

    # project (right) rib top/bottom points
    num_ribs = 6
    num_rib_pts = 20
    f_spar_projection_points = np.array([root_25,root_25+0.1143*(tip_25-root_25),root_25+0.2115*(tip_25-root_25),root_25+0.4944*(tip_25-root_25),root_25+0.7772*(tip_25-root_25),tip_25])
    r_spar_projection_points = np.array([root_75,root_75+0.1143*(tip_75-root_75),root_75+0.2115*(tip_75-root_75),root_75+0.4944*(tip_75-root_75),root_75+0.7772*(tip_75-root_75),tip_75])

    rib_projection_points = np.linspace(f_spar_projection_points, r_spar_projection_points, num_rib_pts)

    offset = np.array([0., 0., .23])
    direction = np.array([0., 0., 1.])

    ribs_top = wing_geometry.project(rib_projection_points+offset, direction=-direction, grid_search_density_parameter=10)
    ribs_bottom = wing_geometry.project(rib_projection_points-offset, direction=direction, grid_search_density_parameter=10)
    # order of ribs is across then back


    # make right spars from front/back of rib points
    spar_fitting_points_parametric = np.zeros((num_ribs*2, 2))
    f_spar_points_parametric = []
    r_spar_points_parametric = []
    j = int(len(ribs_top)-num_ribs)
    for i in range(num_ribs):
        f_spar_points_parametric.append(ribs_top[i])
        f_spar_points_parametric.append(ribs_bottom[i])
        spar_fitting_points_parametric[2*i] = [i/(num_ribs-1),0]
        spar_fitting_points_parametric[2*i+1] = [i/(num_ribs-1),1]
        r_spar_points_parametric.append(ribs_top[j+i])
        r_spar_points_parametric.append(ribs_bottom[j+i])
    f_spar_points = wing_geometry.evaluate(f_spar_points_parametric)
    r_spar_points = wing_geometry.evaluate(r_spar_points_parametric)

    spar_function_space = fs.BSplineSpace(2, 1, (num_ribs, 2))
    f_spar = spar_function_space.fit_function(f_spar_points, spar_fitting_points_parametric)
    r_spar = spar_function_space.fit_function(r_spar_points, spar_fitting_points_parametric)
    
    # make left spars from right spars
    f_spar_left_coeffs = csdl.blockmat([f_spar.coefficients[:,0].reshape((num_ribs*2,1)),     
                                        -f_spar.coefficients[:,1].reshape((num_ribs*2,1)), 
                                        f_spar.coefficients[:,2].reshape((num_ribs*2,1))])
    r_spar_left_coeffs = csdl.blockmat([r_spar.coefficients[:,0].reshape((num_ribs*2,1)),     
                                        -r_spar.coefficients[:,1].reshape((num_ribs*2,1)), 
                                        r_spar.coefficients[:,2].reshape((num_ribs*2,1))])
    f_spar_left = fs.Function(spar_function_space, f_spar_left_coeffs)
    r_spar_right = fs.Function(spar_function_space, r_spar_left_coeffs)

    # add spars to wing geometry
    surf_index = 1000
    pav_geometry.functions[surf_index] = f_spar
    pav_geometry.function_names[surf_index] = "Wing_f_l_spar"
    surf_index += 1
    pav_geometry.functions[surf_index] = r_spar
    pav_geometry.function_names[surf_index] = "Wing_r_l_spar"
    surf_index += 1
    pav_geometry.functions[surf_index] = f_spar_left
    pav_geometry.function_names[surf_index] = "Wing_f_r_spar"
    surf_index += 1
    pav_geometry.functions[surf_index] = r_spar_right
    pav_geometry.function_names[surf_index] = "Wing_r_r_spar"
    surf_index += 1
    # wing_geometry.plot(opacity=0.3)

    # make ribs from rib points
    rib_function_space = fs.BSplineSpace(2, (5,1), (num_rib_pts, 2))
    for i in range(num_ribs):
        rib_points_parametric = []
        rib_parametric_fitting_points = np.zeros((num_rib_pts*2, 2))
        k = 0
        for j in range(0, len(ribs_top), num_ribs):
            rib_points_parametric.append(ribs_top[j+i])
            rib_points_parametric.append(ribs_bottom[j+i])
            rib_parametric_fitting_points[2*k] = [k/(num_rib_pts-1),0]
            rib_parametric_fitting_points[2*k+1] = [k/(num_rib_pts-1),1]
            k += 1
        rib_fitting_points = wing_geometry.evaluate(rib_points_parametric)
        rib = rib_function_space.fit_function(rib_fitting_points, rib_parametric_fitting_points)
        pav_geometry.functions[surf_index] = rib
        pav_geometry.function_names[surf_index] = "Wing_rib_"+str(i)
        surf_index += 1
        # mirror for right rib (skip center)
        if i > 0:        
            r_rib_coeffs = csdl.blockmat([rib.coefficients[:,0].reshape((num_rib_pts*2,1)),
                                        -rib.coefficients[:,1].reshape((num_rib_pts*2,1)),
                                        rib.coefficients[:,2].reshape((num_rib_pts*2,1))])
            r_rib = fs.Function(rib_function_space, r_rib_coeffs)
            pav_geometry.functions[surf_index] = r_rib
            pav_geometry.function_names[surf_index] = "Wing_rib_"+str(-i)
            surf_index += 1
        
    # pav_geometry.plot(opacity=0.3)


    
construct_internal_geometry()


define_base_config(caddee)
cd.mesh_utils.import_mesh('pav_wing_6rib_caddee_mesh_2374_quad.xdmf', 
                          caddee.base_configuration.system.comps['wing'].geometry,
                          plot=True,
                          rescale=[-1,1,-1])


pitch_angle = define_conditions(caddee)


define_analysis(caddee, pitch_angle)
