'''Example pav with shell'''
import CADDEE_alpha as cd
import csdl_alpha as csdl
from VortexAD.core.vlm.vlm_solver import vlm_solver
from femo.rm_shell.rm_shell_model import RMShellModel
from femo.fea.utils_dolfinx import readMesh
import numpy as np
import lsdo_function_spaces as fs
import lsdo_geo as lg

# fs.num_workers = 1     # uncommont this if projections break
plot=False

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
    # TODO: get rid of non-rib rib stuff in the original geometry maybe
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing"])
    
    wing = cd.aircraft.components.Wing(
        AR=7, S_ref=15, geometry=wing_geometry, tight_fit_ffd=False,
    )
    rib_locations = np.array([0, 0.1143, 0.2115, 0.4944, 0.7772, 1])
    wing.construct_ribs_and_spars(aircraft.geometry, rib_locations=rib_locations)

    # wing.plot()

    # wing material
    in2m = 0.0254
    aluminum = cd.materials.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, 
                                              density=2700, nu=0.33)
    thickness_space = wing_geometry.create_parallel_space(fs.ConstantSpace(2))
    thickness_var, thickness_function = thickness_space.initialize_function(
                                                1, value=0.05*in2m)
    wing.quantities.material_properties.set_material(aluminum, 
                                                    thickness_function)

    # wing state spaces
    force_space = fs.IDWFunctionSpace(num_parametric_dimensions=2, 
                                        grid_size=4, order=2, conserve=True)
    wing.quantities.force_space = wing_geometry.create_parallel_space(force_space)

    pressure_space = fs.BSplineSpace(2, (5,5), (7,7))
    wing.quantities.pressure_space = wing_geometry.create_parallel_space(
                                                    pressure_space)

    displacement_space = fs.BSplineSpace(2, (1,1), (3,3))
    wing.quantities.displacement_space = wing_geometry.create_parallel_space(
                                                    displacement_space)


    aircraft.comps["wing"] = wing

    wing.quantities.mass_properties.mass = 250
    wing.quantities.mass_properties.cg_vector = np.array([-3, 0., -1])

    pav_vlm_mesh = cd.mesh.VLMMesh()
    wing_camber_surface = cd.mesh.make_vlm_surface(
        wing, 30, 5, grid_search_density=20, ignore_camber=False,
    )
    # pav_geometry.plot_meshes(wing_camber_surface.nodal_coordinates.value)

    filename = './pav_wing_6rib_caddee_mesh_2374_quad.xdmf'
    wing_shell_discritization = cd.mesh.import_shell_mesh(
        filename, 
        wing,
        plot=plot,
        rescale=[-1,1,-1]
    )

    # store the xdmf mesh object for shell analysis
    wing_shell_mesh_dolfinx = readMesh(filename)
    wing_shell_discritization.fea_mesh = wing_shell_mesh_dolfinx

    pav_shell_mesh = cd.mesh.ShellMesh()
    pav_shell_mesh.discretizations['wing'] = wing_shell_discritization

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
    mesh_container["shell_mesh"] = pav_shell_mesh
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
    wing = aircraft.comps["wing"]
    elevator = csdl.ImplicitVariable(shape=(1, ), value=0.)
    tail.actuate(elevator)

    cruise.finalize_meshes()

    # prep transfer mesh for wing (SIFR)
    transfer_mesh_para = wing.geometry.generate_parametric_grid((20, 20))
    transfer_mesh_phys = wing.geometry.evaluate(transfer_mesh_para)

    # VLM analysis
    vlm_mesh = mesh_container["vlm_mesh"]

    wing_camber_surface = vlm_mesh.discretizations["wing_camber_surface"]
    tail_camber_surface = vlm_mesh.discretizations["tail_camber_surface"]

    camber_surface_coordinates = [wing_camber_surface.nodal_coordinates, 
                                    tail_camber_surface.nodal_coordinates]
    camber_surface_nodal_velocities = [wing_camber_surface.nodal_velocities, 
                                    tail_camber_surface.nodal_velocities]

    vlm_outputs = vlm_solver(camber_surface_coordinates, 
                             camber_surface_nodal_velocities, alpha_ML=None)

    # VLM to framework (SIFR)
    areas = vlm_outputs.surface_panel_areas[0]
    forces = vlm_outputs.surface_panel_forces[0]
    pressures = csdl.reshape(csdl.norm(forces, axes=(3,)) / areas, 
                             (np.prod(forces.shape[0:-1]), 1))
    forces = csdl.reshape(forces, (np.prod(forces.shape[0:-1]), 3))
    camber_force_points = vlm_outputs.surface_panel_force_points[0].value.reshape(-1,3)
    split_fp = np.vstack((camber_force_points + np.array([0, 0, -1]), 
                            camber_force_points + np.array([0, 0, 1])))

    oml_fp_para = wing.geometry.project(split_fp, plot=plot)
    force_function = wing.quantities.force_space.fit_function_set(
                            csdl.blockmat([[forces/2], [forces/2]]), oml_fp_para)
    wing.quantities.force_function = force_function
    pressure_function = wing.quantities.pressure_space.fit_function_set(
                            csdl.blockmat([[pressures/2], [pressures/2]]), 
                            oml_fp_para, regularization_parameter=1)


    # framework to shell (SIFR)
    pav_shell_mesh = mesh_container["shell_mesh"]
    wing_shell_mesh = pav_shell_mesh.discretizations['wing']
    nodes = wing_shell_mesh.nodal_coordinates
    nodes_parametric = wing_shell_mesh.nodes_parametric
    connectivity = wing_shell_mesh.connectivity
    shell_forces = force_function.evaluate(nodes_parametric)

    wing_shell_mesh_dolfinx = wing_shell_mesh.fea_mesh
    # gather material info
    # TODO: make an evaluate that spits out a list of material and a variable 
    #       for thickness (for varying mat props)
    #       This works fine for a single material
    material = wing.quantities.material_properties.material
    thickness = wing.quantities.material_properties.evaluate_thickness(
                                                        nodes_parametric)

    # RM shell analysis
    # g = 9.81 # unit: m/s^2
    E, nu, G = material.from_compliance()
    density = material.density
    shell_locs = {
            'y_root': -1E-6, # location of the clamped Dirichlet BC
            'y_tip': -4.2672,} # location of the wing tip 


    shell_model = RMShellModel(wing_shell_mesh_dolfinx, 
                               shell_locs,  # set up bc locations
                               record = False) # record flag for saving the structure outputs as # xdmf files
    shell_outputs = shell_model.evaluate(shell_forces, # force_vector
                            thickness, E, nu, density, # material properties
                            debug_mode=False)          # debug mode flag

    # Demostrate all the shell outputs even if they might not be used
    disp_solid = shell_outputs.disp_solid # displacement on the shell mesh
    compliance = shell_outputs.compliance # compliance of the structure
    mass = shell_outputs.mass             # total mass of the structure
    elastic_energy = shell_outputs.elastic_energy  # total mass of the structure
    wing_von_Mises_stress = shell_outputs.stress # von Mises stress on the shell
    wing_aggregated_stress = shell_outputs.aggregated_stress # aggregated stress
    disp_extracted = shell_outputs.disp_extracted # extracted displacement 
                                            # for deformation of the OML mesh

    print("Wing tip deflection (m):",max(abs(disp_solid.value)))
    print("Extracted wing tip deflection (m):",max(abs(disp_extracted.value[:,2])))
    print("Wing total mass (kg):", mass.value)
    print("Wing Compliance (N*m):", compliance.value)
    print("Wing elastic energy (J):", elastic_energy.value)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)
    print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress.value))

define_base_config(caddee)

pitch_angle = define_conditions(caddee)

define_analysis(caddee, pitch_angle)
