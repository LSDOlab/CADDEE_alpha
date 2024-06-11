'''Building a component hierarchy'''
# Imports
import CADDEE_alpha as cd
import csdl_alpha as csdl


# Instantiate and start csdl Recorder with inline set to True
recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

# system component & airframe
aircraft = cd.aircraft.components.Aircraft()
airframe = aircraft.comps["airframe"] = cd.Component()
powertrain = aircraft.comps["powertrain"] = cd.aircraft.components.Powertrain()

# Fuselage
fuselage = cd.aircraft.components.Fuselage(length=9.144)
airframe.comps["fuselage"] = fuselage

# Main wing
wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2)
airframe.comps["wing"] = wing

# Empennage
empennage = cd.Component()
airframe.comps["empennage"] = empennage

# Horizontal tail
h_tail = cd.aircraft.components.Wing(AR=4.3, S_ref=3.7, 
                                        taper_ratio=0.6)
empennage.comps["h_tail"] = h_tail

# Vertical tail
v_tail = cd.aircraft.components.Wing(AR=1.17, S_ref=2.54)
empennage.comps["v_tail"] = v_tail

# motor group
power_density = 5000
motors = cd.Component() # Create a parent component for all the motors
powertrain.comps["motors"] = motors
pusher_motor = cd.Component(power_density=power_density)
motors.comps["pusher_motor"] = pusher_motor
for i in range(8):
    motor = cd.Component(power_density=power_density)
    motors.comps[f"motor_{i+1}"] = motor # add motors to parent

# Pusher prop 
rotors = cd.Component() # Create a parent component for all the rotors
pusher_prop = cd.aircraft.components.Rotor(radius=2.74/2)
rotors.comps["pusher_prop"] = pusher_prop

# Lift rotors / motors
lift_rotors = []
airframe.comps["rotors"] = rotors
for i in range(8):
    rotor = cd.aircraft.components.Rotor(radius=3.048/2)
    lift_rotors.append(rotor)
    rotors.comps[f"rotor_{i+1}"] = rotor

# Booms
booms = cd.Component() # Create a parent component for all the booms
airframe.comps["booms"] = booms
for i in range(8):
    boom = cd.Component()
    booms.comps[f"boom_{i+1}"] = boom

# battery
battery = cd.Component(energy_density=380)
powertrain.comps["battery"] = battery

# payload
payload = cd.Component()
aircraft.comps["payload"] = payload

passenger_1 = cd.Component()
payload.comps["passenger_1"] = passenger_1

passenger_2 = cd.Component()
payload.comps["passenger_2"] = passenger_2

baggage = cd.Component()
payload.comps["baggage"] = baggage

# systems
systems = cd.Component()
airframe.comps["systems"] = systems

avionics = cd.Component()
systems.comps["avionics"] = avionics

hydraulics = cd.Component()
systems.comps["hydraulics"] = hydraulics

# create Configuration instance and assign it to CADDEE instance
base_config = cd.Configuration(system=aircraft)
caddee.base_configuration = base_config

# visualization: This makes a .png file of the component hierarchy
base_config.visualize_component_hierarchy(show=True)