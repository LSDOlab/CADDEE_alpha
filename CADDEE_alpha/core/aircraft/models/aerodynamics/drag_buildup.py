import csdl_alpha as csdl
from CADDEE_alpha.core.component import Component
from CADDEE_alpha.utils.var_groups import AircaftStates, AtmosphericStates
from typing import List, Union
from CADDEE_alpha.utils.coordinate_transformations import perform_local_to_body_transformation


def compute_drag_build_up(
    ac_states: AircaftStates, 
    atmos_states: AtmosphericStates, 
    S_ref: Union[csdl.Variable, int, float],
    components: List[Component]
):
    csdl.check_parameter(components, "components", types=list)

    # Extract relevant aircraft states
    u = ac_states.u
    v = ac_states.v
    w = ac_states.w

    num_nodes = u.shape[0]

    # Extract atmospheric parameters
    rho = atmos_states.density
    mu = atmos_states.dynamic_viscosity
    speed_of_sound = atmos_states.speed_of_sound

    # Compute freestream velocity and Mach
    V_inf = (u**2 +  v**2 + w**2)**0.5
    Mach = V_inf / speed_of_sound

    # Loop over all components and compute drag area
    drag_area = 0

    for comp in components:
        if not isinstance(comp, Component):
            raise TypeError(f"At least one invalid component: elements of 'components' argument must be of type 'Component'; received {type(comp)}")
        
        # Extract drag are
        comp_drag_area = comp.quantities.drag_parameters.drag_area
        
        # If drag area is provided, add it directly        
        if comp_drag_area is not None:
            drag_area = drag_area + comp_drag_area
        
        else:
            # Extract key quantities from component
            S_wet = comp.quantities.surface_area
            ff = comp.quantities.drag_parameters.form_factor
            interference_factor = comp.quantities.drag_parameters.interference_factor
            length = comp.quantities.drag_parameters.characteristic_length

            if any([S_wet, ff, length]) is None:
                raise TypeError(f"At least one component quantitiy ('surface_area', 'form_factor', 'characteristic_length') of component {comp} is None.")
        
            # Compute Re
            Re = rho * V_inf * length / mu

            # Compute Cf
            per_lam = comp.quantities.drag_parameters.percent_laminar
            per_turb = comp.quantities.drag_parameters.percent_turbulent

            if per_lam + per_turb != 100:
                raise ValueError("'percent_laminar' and 'percent_turbulent' must add to 100 (%)")

            per_lam *= 1e-2
            per_turb *= 1e-2

            Cf_fun_lam = comp.quantities.drag_parameters.cf_laminar_fun
            Cf_fun_turb = comp.quantities.drag_parameters.cf_turbulent_fun

            Cf = per_lam * Cf_fun_lam(Re) + per_turb * Cf_fun_turb(Re, Mach)

            # Rayer drag build up: Cd_0 = Cf * FF * Q * S_wet / S_ref
            drag_area = drag_area + Cf * ff * interference_factor * S_wet # Non-dimensionalize later

    
    Cd_0 = drag_area / S_ref

    drag = - Cd_0 * 0.5 * rho * V_inf**2 * S_ref

    forces = csdl.Variable(shape=(num_nodes, 3), value=0.)
    forces = forces.set(csdl.slice[:, 0], drag)
    forces = perform_local_to_body_transformation(
        phi=ac_states.phi,
        theta=ac_states.theta,
        psi=ac_states.psi,
        vectors=forces,
    )

    return forces


