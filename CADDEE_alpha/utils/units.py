from dataclasses import dataclass

@dataclass
class Length:
    foot_to_m = 0.3048
    mile_to_m = 1609.34
    nautical_mile_to_m = 1852
    yard_to_m = 0.9144
    inch_to_m = 0.0254
    kilometer_to_m = 1e3
    nanometer_to_m = 1e-9
    micrometer_to_m = 1e-6
    millimeter_to_m = 1e-3
    centimeter_to_m = 1e-2

@dataclass
class Area:
    sq_ft_to_sq_m = 0.092903
    sq_in_to_sq_m = 0.0006451597222199104
    sq_mile_to_sq_m = 2589986.9951907191426

@dataclass
class Time:
    hour_to_sec = 3600
    minutes_to_sec = 60
    millisec_to_sec = 1e-3

@dataclass
class Speed:
    mph_to_mps = 0.44704
    kph_to_mps = 0.277778
    ftps_to_mps = 0.3048
    knots_to_mps = 0.514444

@dataclass
class Pressure:
    lb_per_sq_in_to_Pa = 6894.76
    bar_to_Pa = 100000.0393

@dataclass
class Mass:
    pound_to_kg = 0.453592
    ounce_to_kg = 0.0283495
    slug_to_kg = 14.5939

@dataclass
class Units:
    length = Length()
    area = Area()
    time = Time()
    mass = Mass()
    pressure = Pressure()
    speed = Speed()


