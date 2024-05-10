from __future__ import annotations
from CADDEE_alpha.utils.var_groups import MassProperties
from typing import TypeVar, List, Type


T = TypeVar('T')

class CADDEEDict(dict):
    """Modified dictionary class for storing data with type checking.
    """
    def __init__(self, types: T = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(types, list):
            self.types = (types, )
        else:
            self.types = types
    
    def __setitem__(self, key, value, allow_overwrite=False):
        # check if types is empty
        if not self.types: 
            super().__setitem__(key, value)
        
        # Check type
        elif not isinstance(value, tuple(self.types)):
            raise TypeError(f"Value must be of type(s) {self.types}")
        
        # Check if key is already specified
        elif key in self:
            if allow_overwrite is False:
                raise Exception(f"The value for key {key} has already been set")
            else:
                super().__setitem__(key, value)

        # Set item otherwise
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key) -> T:
        # Check if key exists
        if key not in self:
            raise KeyError(f"The key '{key}' does not exist. Existing keys: {list(self.keys())}")
        else:
            return super().__getitem__(key)
        
class ComponentQuantities(CADDEEDict):
    mass_properties = MassProperties()
        





if __name__ == "__main__":
    import caddee as cd
    import lsdo_geo as lg

    geometry = lg.import_file()

    # ----

    aircraft = cd.Aircraft(geometry)
    base_config = cd.Configuration(aircraft)

    base_config.set_system(aircraft)


    wing_geometry = aircraft.create_subgeometry(search_name = 'wing')
    wing = cd.Wing(
        geometry = wing_geometry,
        AR=...,
        S_ref=...,
    )
    wing.plot()
    aircraft.comps['wing'] = wing

    aileron_geometry = wing.create_subgeometry(search_name='aileraon')

    base_config.run_geometry_solver()



    wing = cd.Wing(
        geo_keyword = 'wing',
        AR=...,
    )
    aircraft.comps['wing'] = wing
    wing.plot()

    wing = cd.Wing(
        AR=..., 
        S_ref=...,
    )
    wing.setup_geometry(
        geometry.b_spline_search("Wing")
    )
    wing.plot()
    aircraft.comps["wing"] = wing


    aircraft = cd.Aircraft(geometry=geometry)

    wing = aircraft.declare_component(
        "wing",
        cd.Wing(),
    )


    wing = cd.Wing()
