import copy
from CADDEE_alpha.core.component import Component, ComponentDict


def copy_comps(comp : Component, system_geometry=None):
    """Create copies of components and their attributes.
    """
    # 1) Create a shallow copy of the component itself
    comp_copy = copy.copy(comp)
    comp_copy._is_copy = True

    # 2) Create shallow copy of the comp's geometry
    if comp.geometry is not None:
        if system_geometry is not None:
            geometry_copy = comp_copy.geometry.copy() 
            comp_copy.geometry = geometry_copy
            for ind in geometry_copy.functions.keys():
                comp_copy.geometry.functions[ind] = system_geometry.functions[ind]
        else:
            geometry_copy = comp.geometry.copy()
            comp_copy.geometry = geometry_copy

    # 3) Create shallow copy of the comp's children 
    # TODO: dictionary's __getitem__ error message is not copied right now
    # Possible solution: Make new ComponentDict object and populate with the component's children
    children_comps_copy = comp.comps.copy() # copy.copy(comp.comps) #
    comp_copy.comps : ComponentDict = children_comps_copy

    # 4) Create shallow copy of the comp's quantities
    quantities_copy = copy.copy(comp.quantities)
    # 4a) Copy mass properties 
    mps_copy = copy.copy(quantities_copy.mass_properties)
    quantities_copy.mass_properties = mps_copy
    # 4b) Copy material properties
    mat_copy = copy.copy(quantities_copy.material_properties)
    quantities_copy.material_properties = mat_copy
    # 4c) copy the utils dictionary
    utils_copy = quantities_copy.utils.copy()
    quantities_copy.utils = utils_copy
    
    comp_copy.quantities = quantities_copy

    # 5) Create shallow copy of the comp's parameters
    parameters_copy = copy.copy(comp.parameters)
    comp_copy.parameters = parameters_copy

    # # 6) discretizations
    # discretizations_copy = comp_copy._discretizations.copy()
    # for discr_name, discr in discretizations_copy.items():
    #     discretizations_copy[discr_name] = copy.copy(discr)
    # comp_copy._discretizations = discretizations_copy

    # 7) Recursively copy children of comp
    if system_geometry is None:
        system_geometry_input = comp_copy.geometry
    else:
        system_geometry_input = system_geometry
        
    for child_comp_copy_name, child_comp_copy in children_comps_copy.items():
        child_comp_copy_copy = copy_comps(child_comp_copy, system_geometry_input)
        
        # Set the value of the comps dictionary
        children_comps_copy[child_comp_copy_name] = child_comp_copy_copy
        
    return comp_copy


def convert_vect_mp_to_arrays():
    pass