"""Generate internal geometry"""
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = False

# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def define_base_config(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # Fuselage
    fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselage"])
    fuselage = cd.aircraft.components.Fuselage(length=10., geometry=fuselage_geometry)
    airframe.comps["fuselage"] = fuselage

    # Main wing
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"])
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2, sweep=np.deg2rad(-20),
                                       geometry=wing_geometry, tight_fit_ffd=True)
    # wing.geometry.plot(opacity=0.3)
    # exit()
    wing.construct_ribs_and_spars(
        wing_geometry, 
        num_ribs=9,
        LE_TE_interpolation="ellipse",
        plot_projections=False, 
        export_wing_box=True
    )
    wing.geometry.plot(opacity=0.3)
    wing.geometry.export_iges("lpc_wing_geometry.igs")
    
    airframe.comps["wing"] = wing

    base_config = cd.Configuration(system=aircraft)
    base_config.setup_geometry()
    caddee.base_configuration = base_config
    wing.geometry.plot(opacity=0.3)

    exit()

    # Horizontal tail
    h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_1"])
    h_tail = cd.aircraft.components.Wing(AR=4.3, S_ref=3.7, 
                                         taper_ratio=0.6, geometry=h_tail_geometry)
    airframe.comps["h_tail"] = h_tail

    # Pusher prop
    pusher_prop_geometry = aircraft.create_subgeometry(search_names=[
        "Rotor-9-disk",
        "Rotor_9_blades",
        "Rotor_9_Hub",
    ])
    pusher_prop = cd.aircraft.components.Rotor(radius=2.74/2.5, geometry=pusher_prop_geometry)
    airframe.comps["pusher_prop"] = pusher_prop

    # Lift rotors
    lift_rotors = []
    for i in range(8):
        rotor_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_disk",
            f"Rotor_{i+1}_Hub",
            f"Rotor_{i+1}_blades",]
        )
        rotor = cd.aircraft.components.Rotor(radius=3.048/2.5, geometry=rotor_geometry)
        lift_rotors.append(rotor)
        airframe.comps[f"rotor_{i+1}"] = rotor

    # Booms
    for i in range(8):
        boom_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_Support",
        ])
        boom = cd.Component(geometry=boom_geometry)
        airframe.comps[f"boom_{i+1}"] = boom

    # ::::::::::::::::::::::::::: Make meshes :::::::::::::::::::::::::::
    if make_meshes:
    # wing + tail
        vlm_mesh = cd.mesh.VLMMesh()
        wing_chord_surface = cd.mesh.make_vlm_surface(
            wing, 32, 15, LE_interp="ellipse", TE_interp="ellipse", ignore_camber=False,
        )
        vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
        
        tail_surface = cd.mesh.make_vlm_surface(
            h_tail, 14, 6, ignore_camber=True
        )
        vlm_mesh.discretizations["tail_surface"] = tail_surface

        lpc_geom.plot_meshes([wing_chord_surface.nodal_coordinates, tail_surface.nodal_coordinates])

        # rotors
        rotor_meshes = cd.mesh.RotorMeshes()
        # pusher prop
        pusher_prop_mesh = cd.mesh.make_rotor_mesh(
            pusher_prop, num_radial=30, num_azimuthal=1, num_blades=4, plot=True
        )
        rotor_meshes.discretizations["pusher_prop_mesh"] = pusher_prop_mesh

        # lift rotors
        for i in range(8):
            rotor_mesh = cd.mesh.make_rotor_mesh(
                lift_rotors[i], num_radial=30, num_blades=2,
            )
            rotor_meshes.discretizations[f"rotor_{i+1}_mesh"] = rotor_mesh

    # aircraft.geometry.plot()

    # Make base configuration    
    base_config = cd.Configuration(system=aircraft)
    exit()
    base_config.setup_geometry()
    caddee.base_configuration = base_config

    # Store meshes
    if make_meshes:
        mesh_container = base_config.mesh_container
        mesh_container["vlm_mesh"] = vlm_mesh
        mesh_container["rotor_meshes"] = rotor_meshes


define_base_config(caddee)

