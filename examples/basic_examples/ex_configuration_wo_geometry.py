'''Example config w/o geometry'''
import CADDEE_alpha as cd
from CADDEE_alpha.core.configuration import Configuration
from CADDEE_alpha.core.component import Component
from CADDEE_alpha.core.aircraft.components.aircraft import Aircraft
from CADDEE_alpha.core.aircraft.components.wing import Wing
from CADDEE_alpha.core.aircraft.components.fuselage import Fuselage
from CADDEE_alpha.core.aircraft.conditions.aircraft_condition import CruiseCondition, ClimbCondition
from CADDEE_alpha.core.aircraft.models.weights.weights_solver import WeightsSolverModel
from CADDEE_alpha.core.aircraft.models.weights.general_aviation.general_aviation_weights import *
from CADDEE_alpha.utils.units import Units
import numpy as np
import csdl_alpha as csdl


recorder = csdl.Recorder(inline=True)
recorder.start()


units = Units()

def define_base_config(caddee : cd.CADDEE):
    """ Build the base configuration."""

    # Make aircraft component and assign airframe
    c172 = Aircraft()
    c172.comps["airframe"] = airframe = Component()

    # Geometric parameters
    # Wing
    S_ref_wing = csdl.Variable(shape=(1,), value=174)  # sq ft
    S_wet_wing = 2.05 * S_ref_wing # sq ft
    wing_AR = csdl.Variable(shape=(1,), value=7.34)
    wing_taper_ratio = csdl.Variable(shape=(1,), value=0.75)
    wing_thickness_to_chord = csdl.Variable(shape=(1,), value=0.12)

    main_wing = Wing(
        AR=wing_AR,
        S_ref=S_ref_wing,
        S_wet=S_wet_wing,
        sweep=0.,
        taper_ratio=wing_taper_ratio,
        thickness_to_chord_ratio=wing_thickness_to_chord,
    )
    airframe.comps["wing"] = main_wing

    # horizontal tail
    S_ref_h_tail = csdl.Variable(shape=(1,), value=21.56) # sq ft
    span_h_tail = csdl.Variable(shape=(1,), value=11.12) # ft
    tail_thickness_to_chord = csdl.Variable(shape=(1,), value=0.12)

    h_tail = Wing(
        AR=None, 
        S_ref=S_ref_h_tail,
        span=span_h_tail,
        taper_ratio=0.75,
        sweep=0.,
        thickness_to_chord_ratio=tail_thickness_to_chord,
    )
    airframe.comps["h_tail"] = h_tail


    # vertical tail
    S_ref_v_tail = csdl.Variable(shape=(1,), value=11.2) # sq ft
    v_tail_AR = csdl.Variable(shape=(1,), value=2.)
    v_tail_sweep = csdl.Variable(shape=(1,), value=np.deg2rad(15))

    v_tail = Wing(
        AR=v_tail_AR, 
        S_ref=S_ref_v_tail,
        sweep=v_tail_sweep,
    )
    airframe.comps["v_tail"] = v_tail

    # fuselage
    fuselage_length = csdl.Variable(shape=(1,), value=28) # ft
    cabin_depth = csdl.Variable(shape=(1,), value=10) # ft
    max_width = csdl.Variable(shape=(1,), value=3.2) # ft
    S_wet_fuselage = csdl.Variable(shape=(1,), value=310) # sq ft

    fuselage = Fuselage(
        length=fuselage_length,
        max_width=max_width,
        max_height=4.5,
        cabin_depth=10.,
        S_wet=S_wet_fuselage,
    )
    airframe.comps["fuselage"] = fuselage

    # Components without pre-specified parameters
    # Main landing gear
    main_landing_gear = Component()
    airframe.comps["main_landing_gear"] = main_landing_gear

    # Avionics
    avionics = Component()
    fuselage.comps["avionics"] = avionics

    # Instruments
    instruments = Component()
    fuselage.comps["instruments"] = instruments

    # Engine
    engine = Component()
    c172.comps["engine"] = engine

    # Fuel
    fuel = Component()
    main_wing.comps["fuel"] = fuel

    payload = Component()
    fuselage.comps["payload"] = payload

    # set the base configuration
    c172_base_config = Configuration(c172)
    caddee.base_configuration = c172_base_config


def define_conditions(caddee : cd.CADDEE):
    """Define the design conditions."""
    conditions = caddee.conditions
    base_config = caddee.base_configuration

    # Climb 
    climb = ClimbCondition(
        initial_altitude=500, 
        final_altitude=3000,
        pitch_angle=np.deg2rad(10),
        fligth_path_angle=np.deg2rad(5),
        mach_number=0.18,
    )
    climb.configuration = base_config.copy()
    conditions["climb"] = climb

    # Cruise 1 (normal speed)
    cruise_1 = CruiseCondition(
        altitude=3000.,
        mach_number=0.19,
        speed=None,
        range=600 * units.length.nautical_mile_to_m,
        pitch_angle=0.,
    )
    cruise_1.configuration = base_config.copy()
    conditions["cruise_1"] = cruise_1

    # Cruise 2 (high speed)
    cruise_2 = CruiseCondition(
        altitude=3000.,
        mach_number=0.22,
        speed=None,
        range=100 * units.length.kilometer_to_m,
        pitch_angle=0.,
    )
    cruise_2.configuration = base_config.copy()
    conditions["cruise_2"] = cruise_2

    # Descent
    descent = ClimbCondition(
        initial_altitude=3000, 
        final_altitude=200,
        pitch_angle=np.deg2rad(-3),
        fligth_path_angle=np.deg2rad(-5),
        mach_number=0.17,
    )
    descent.configuration = base_config.copy()
    conditions["descent"] = descent

def define_mass_properties(caddee : cd.CADDEE):
    """Define vehicle-level mass properties of the base configuration."""
    base_config = caddee.base_configuration
    conditions = caddee.conditions

    # get some operational variables from the cruise condition
    cruise = conditions["cruise_1"]
    fast_cruise = conditions["cruise_2"]
    rho_imperial = cruise.quantities.atmos_states.density * (1 / units.mass.slug_to_kg) / (1 / units.length.foot_to_m)**3
    speed_imperial = cruise.parameters.speed * (1 / units.speed.ftps_to_mps)
    q_cruise = 0.5 * rho_imperial * speed_imperial**2
    range_imperial = cruise.parameters.range * (1/ units.length.nautical_mile_to_m)
    max_mach = fast_cruise.parameters.mach_number

    #Access the base config and the its components
    aircraft = base_config.system
    airframe = aircraft.comps["airframe"]

    wing = airframe.comps["wing"]
    fuselage = airframe.comps["fuselage"]
    h_tail = airframe.comps["h_tail"]
    v_tail = airframe.comps["v_tail"]
    main_landing_gear = airframe.comps["main_landing_gear"]
    avionics = fuselage.comps["avionics"]
    instruments = fuselage.comps["instruments"]
    engine = aircraft.comps["engine"]
    fuel = wing.comps["fuel"]
    payload = fuselage.comps["payload"]

    # design gross weight estimate
    dg_est = csdl.ImplicitVariable(shape=(1, ), value=2200)

    fuel_weight = csdl.Variable(shape=(1,), value=500)
    engine_weight = csdl.Variable(shape=(1,), value=250)
    payload_weight =  csdl.Variable(shape=(1,), value=500)

    # wing mass
    wing_weight_inputs = GAWingWeightInputs(
        S_ref=wing.parameters.S_ref,
        W_fuel=fuel_weight,
        AR=wing.parameters.AR,
        sweep_c4=0,
        taper_ratio=wing.parameters.taper_ratio,
        thickness_to_chord=wing.parameters.thickness_to_chord,
        dynamic_pressure=q_cruise,
        W_gross_design=dg_est,
        correction_factor=0.8,
    )
    wing_weight_model = GAWingWeightModel()
    wing_weight = wing_weight_model.evaluate(wing_weight_inputs)

    # fuselage mass
    fuselage_weight_inputs = GAFuselageWeightInputs(
        S_wet=fuselage.parameters.S_wet,
        q_cruise=q_cruise,
        W_gross_design=dg_est,
        xl=fuselage.parameters.length, 
        d_av=fuselage.parameters.max_width,
        correction_factor=2.2,
    )
    fuselage_weight_model = GAFuselageWeigthModel()
    fuselage_weight = fuselage_weight_model.evaluate(fuselage_weight_inputs)

    # h tail mass
    h_tail_weight_inputs = GAHorizontalTailInputs(
        S_ref=h_tail.parameters.S_ref,
        W_gross_design=dg_est,
        q_cruise=q_cruise,
    )
    h_tail_weight_model = GAHorizontalTailWeigthModel()
    h_tail_weight = h_tail_weight_model.evaluate(h_tail_weight_inputs)

    # v tail mass
    v_tail_weight_inputs = GAVerticalTailInputs(
        S_ref=v_tail.parameters.S_ref,
        AR=v_tail.parameters.AR,
        W_gross_design=dg_est,
        q_cruise=q_cruise,
        t_o_c=0.12,
        sweep_c4=np.deg2rad(30),
    )
    v_tail_weight_model = GAVerticalTailWeigthModel()
    v_tail_weight = v_tail_weight_model.evaluate(v_tail_weight_inputs)

    # avionics mass
    avionics_weight_inputs = GAAvionicsWeightInputs(
        design_range=range_imperial,
        num_flight_crew=1,
        S_fuse_planform=3.2 * 7.,
        correction_factor=0.8,
    )
    avionics_weight_model =  GAAvionicsWeightModel()
    avionics_weight = avionics_weight_model.evaluate(avionics_weight_inputs)

    # instruments mass
    instruments_weight_inputs = GAInstrumentsWeightInputs(
        mach_max=max_mach,
        num_flight_crew=1., 
        num_wing_mounted_engines=0,
        num_fuselage_mounted_engines=1,
        S_fuse_planform=3.2 * 7.,
    )
    instruments_weight_model = GAInstrumentsWeightModel()
    instruments_weight = instruments_weight_model.evaluate(instruments_weight_inputs)

    # Landing gear mass
    landing_gear_weights_inputs = GAMainLandingGearWeightInputs(
        fuselage_length=fuselage.parameters.length,
        design_range=range_imperial,
        W_ramp=dg_est * 1.05,
        correction_factor=0.15,
    )
    landing_gear_weight_model = GAMainLandingGearWeightModel()
    landing_gear_weight = landing_gear_weight_model.evaluate(landing_gear_weights_inputs)


    weights_solver = WeightsSolverModel()
    weights_solver.evaluate(
        dg_est, wing_weight, fuselage_weight, h_tail_weight, v_tail_weight, avionics_weight, instruments_weight, engine_weight, fuel_weight, payload_weight, landing_gear_weight
    )


    wing.quantities.mass_properties.mass = wing_weight * units.mass.pound_to_kg
    wing.quantities.mass_properties.cg_vector = np.array([9.649 * units.length.foot_to_m, 0. , 2. * units.length.foot_to_m])

    fuselage.quantities.mass_properties.mass = fuselage_weight * units.mass.pound_to_kg
    fuselage.quantities.mass_properties.cg_vector = np.array([9.649 * units.length.foot_to_m, 0. , 0.])

    h_tail.quantities.mass_properties.mass = h_tail_weight * units.mass.pound_to_kg
    h_tail.quantities.mass_properties.cg_vector = np.array([25.137 * units.length.foot_to_m, 0., 0.051 * units.length.foot_to_m])

    v_tail.quantities.mass_properties.mass = h_tail_weight * units.mass.pound_to_kg
    v_tail.quantities.mass_properties.cg_vector = np.array([25.137 * units.length.foot_to_m, 0., 1.51 * units.length.foot_to_m])

    instruments.quantities.mass_properties.mass = instruments_weight * units.mass.pound_to_kg
    instruments.quantities.mass_properties.cg_vector = np.array([1.5 * units.length.foot_to_m, 0., 0])

    avionics.quantities.mass_properties.mass = avionics_weight * units.mass.pound_to_kg
    avionics.quantities.mass_properties.cg_vector = np.array([1.2 * units.length.foot_to_m, 0., 0])

    engine.quantities.mass_properties.mass = engine_weight * units.mass.pound_to_kg
    engine.quantities.mass_properties.cg_vector = np.array([0.8 * units.length.foot_to_m, 0., 0])

    fuel.quantities.mass_properties.mass = fuel_weight * units.mass.pound_to_kg
    fuel.quantities.mass_properties.cg_vector = np.array([9.649 * units.length.foot_to_m, 0. , 2. * units.length.foot_to_m])

    payload.quantities.mass_properties.mass = payload_weight * units.mass.pound_to_kg
    payload.quantities.mass_properties.cg_vector = np.array([9.649 * units.length.foot_to_m, 0. , 0.75 * units.length.foot_to_m])

    main_landing_gear.quantities.mass_properties.mass = landing_gear_weight * units.mass.pound_to_kg
    main_landing_gear.quantities.mass_properties.cg_vector = np.array([6.649 * units.length.foot_to_m, 0. , -0.75 * units.length.foot_to_m])

    print(aircraft.quantities.mass_properties.mass)

    print(id(payload.quantities.mass_properties))
    print(id(aircraft.quantities.mass_properties))

    # aircraft.quantities.mass_properties.mass = 1100
    # aircraft.quantities.mass_properties.cg_vector = np.array([2.2513916, 0., 0.216399])
    # aircraft.quantities.mass_properties.inertia_tensor = np.zeros((3, 3))


    base_config.assemble_system_mass_properties()

    print(aircraft.quantities.mass_properties.mass.value)
    print(base_config.system.quantities.mass_properties)

def define_sub_configurations(caddee : cd.CADDEE):
    base_config = caddee.base_configuration
    conditions = caddee.conditions
    cruise_configuration = conditions["cruise_1"].configuration

    aircraft_in_cruise = cruise_configuration.system
    payload_in_cruise = aircraft_in_cruise.comps["airframe"].comps["fuselage"].comps["payload"]

    print("\n")
    print(base_config.system.quantities.mass_properties)
    print(aircraft_in_cruise.comps["airframe"].comps["fuselage"].comps["payload"].quantities.mass_properties.mass.value)
    
    cruise_configuration.remove_component(payload_in_cruise)
    # print(aircraft_in_cruise.comps["airframe"].comps["fuselage"].comps["payload"])
    cruise_configuration.assemble_system_mass_properties()
    print(cruise_configuration.system.quantities.mass_properties.mass.value)
    
    
if __name__ == "__main__":
    caddee = cd.CADDEE()

    define_base_config(caddee)

    define_conditions(caddee)

    define_mass_properties(caddee)

    define_sub_configurations(caddee)


    

