import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs
import CADDEE_alpha as cd


def load_thickness_vars(fname, group):
    inputs = csdl.inline_import(fname, group)
    t_vars = {key:val for key, val in inputs.items() if 'thickness' in key}
    return t_vars

def construct_bay_condition(upper, lower):
    if upper == 1:
        geq = True
    else:
        geq = False
    def condition(parametric_coordinates:np.ndarray):
        if geq:
            out = np.logical_and(np.greater_equal(parametric_coordinates[:,0], lower), np.less_equal(parametric_coordinates[:,0], upper))
        else:
            out = np.logical_and(np.greater(parametric_coordinates[:,0], lower), np.less_equal(parametric_coordinates[:,0], upper))
        if len(out.shape) == 1:
            out.reshape(-1, 1)
        # if len(out.shape) == 1:
        #     out.reshape(1, 1)
        return out.T
    return condition

def construct_thickness_function(wing, num_ribs, top_array, bottom_array, material, 
                                 t_vars=None, skin_t=0.01, spar_t=0.01, rib_t=0.01, 
                                 minimum_thickness=0.0003, add_dvs=True):
    bay_eps = 1e-2
    t_out = {}
    for i in range(num_ribs-1):
        # upper wing bays
        lower = top_array[:, i]
        upper = top_array[:, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in upper])) - bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps

        if l_surf_ind == u_surf_ind:
            condition = construct_bay_condition(u_coord_u, l_coord_u)
            if t_vars:
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                thickness.value = t_vars['upper_wing_thickness_'+str(i)]
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            if t_vars:
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                thickness.value = t_vars['upper_wing_thickness_'+str(i)]
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=skin_t, name='upper_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function1 = lfs.Function(lfs.ConditionalSpace(2, condition1), thickness)
            function2 = lfs.Function(lfs.ConditionalSpace(2, condition2), thickness)
            functions = {l_surf_ind: function1, u_surf_ind: function2}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)

        # lower wing bays
        lower = bottom_array[:, i]
        upper = bottom_array[:, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in upper])) - bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0, 0]
        # u_coord_u = upper[0][1][0, 0]
        if l_surf_ind == u_surf_ind:
            condition = construct_bay_condition(u_coord_u, l_coord_u)
            if t_vars:
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                thickness.value = t_vars['lower_wing_thickness_'+str(i)]
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            if t_vars:
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                thickness.value = t_vars['lower_wing_thickness_'+str(i)]
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=skin_t, name='lower_wing_thickness_'+str(i))
                t_out[thickness.name] = thickness
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function1 = lfs.Function(lfs.ConditionalSpace(2, condition1), thickness)
            function2 = lfs.Function(lfs.ConditionalSpace(2, condition2), thickness)
            functions = {l_surf_ind: function1, u_surf_ind: function2}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)


    # ribs
    rib_geometry = wing.create_subgeometry(search_names=["rib"])
    rib_fsp = lfs.ConstantSpace(2)
    for ind in rib_geometry.functions:
        name = rib_geometry.function_names[ind]
        if "-" in name:
            pass
        else:
            if t_vars:
                thickness = csdl.Variable(value=rib_t, name=name+'_thickness')
                thickness.value = t_vars[name+'_thickness']
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=rib_t, name=name+'_thickness')
                t_out[thickness.name] = thickness
            if add_dvs:
                pass
                # thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(rib_fsp, thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)

    # spars
    u_cords = np.linspace(0, 1, num_ribs)
    spar_geometry = wing.create_subgeometry(search_names=["spar"], ignore_names=['_r_'])
    spar_inds = list(spar_geometry.functions)
    for i in range(num_ribs-1):
        lower = u_cords[i] - bay_eps
        upper = u_cords[i+1] - bay_eps
        if i == num_ribs-2:
            upper = 1 + 2*bay_eps
        condition = construct_bay_condition(upper, lower)
        spar_num = 0
        for ind in spar_inds:
            if t_vars:
                thickness = csdl.Variable(value=spar_t, name=f'spar_{spar_num}_thickness_{i}')
                thickness.value = t_vars[f'spar_{spar_num}_thickness_{i}']
                t_out[thickness.name] = thickness
            else:
                thickness = csdl.Variable(value=spar_t, name=f'spar_{spar_num}_thickness_{i}')
                t_out[thickness.name] = thickness
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
            spar_num += 1

    return t_out
    

def construct_plate_condition(upper, lower, forward, backward, ind):
    if upper == 1:
        geq = True
    else:
        geq = False
    def condition(parametric_coordinate:tuple[int, np.ndarray]):
        index = parametric_coordinate[0]
        if index != ind:
            return False
        coord = parametric_coordinate[1]
        if geq:
            out = np.logical_and(np.greater_equal(coord[:,0], lower), np.less_equal(coord[:,0], upper))
            out = np.logical_and(out, np.logical_and(np.greater_equal(coord[:,1], backward), np.less_equal(coord[:,1], forward)))
        else:
            out = np.logical_and(np.greater(coord[:,0], lower), np.less_equal(coord[:,0], upper))
            out = np.logical_and(out, np.logical_and(np.greater(coord[:,1], backward), np.less_equal(coord[:,1], forward)))
        return out[0]
    return condition

def construct_plates(num_ribs, top_array, bottom_array):
    bays = {}
    bay_eps = 1e-2
    for i in range(num_ribs-2):
        # upper wing bays
        lower = top_array[1:3, i]
        upper = top_array[1:3, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0,0] for (ind, coord) in upper])) + bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0,0]
        # u_coord_u = upper[0][1][0,0]
        if l_surf_ind == u_surf_ind:
            f_coord_v = (lower[0][1][0,1] + upper[0][1][0,1])/2 + bay_eps
            b_coord_v = (lower[1][1][0,1] + upper[1][1][0,1])/2 - bay_eps
            condition = construct_plate_condition(u_coord_u, l_coord_u, f_coord_v, b_coord_v, l_surf_ind)
            bays['upper_wing_bay_'+str(i)] = [condition]
        else:
            condition1 = construct_plate_condition(1, l_coord_u, lower[0][1][0,1]+bay_eps, lower[1][1][0,1]-bay_eps, l_surf_ind)
            condition2 = construct_plate_condition(u_coord_u, 0, upper[0][1][0,1]+bay_eps, upper[1][1][0,1]-bay_eps, u_surf_ind)
            bays['upper_wing_bay_'+str(i)] = [condition1, condition2]

        # lower wing bays
        lower = bottom_array[1:3, i]
        upper = bottom_array[1:3, i+1]
        l_surf_ind = lower[0][0]
        u_surf_ind = upper[0][0]
        l_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in lower])) - bay_eps
        u_coord_u = np.mean(np.array([coord[0, 0] for (ind, coord) in upper])) + bay_eps
        if i == num_ribs-2:
            u_coord_u = 1 + bay_eps
        # l_coord_u = lower[0][1][0, 0]
        # u_coord_u = upper[0][1][0, 0]
        if l_surf_ind == u_surf_ind:
            f_coord_v = (lower[0][1][0,1] + upper[0][1][0,1])/2 - bay_eps
            b_coord_v = (lower[1][1][0,1] + upper[1][1][0,1])/2 + bay_eps
            condition = construct_plate_condition(u_coord_u, l_coord_u, b_coord_v, f_coord_v, l_surf_ind)
            bays['lower_wing_bay_'+str(i)] = [condition]
        else:
            condition1 = construct_plate_condition(1, l_coord_u, lower[1][1][0,1]+bay_eps, lower[0][1][0,1]-bay_eps, l_surf_ind)
            condition2 = construct_plate_condition(u_coord_u, 0, upper[1][1][0,1]+bay_eps, upper[0][1][0,1]-bay_eps, u_surf_ind)
            bays['lower_wing_bay_'+str(i)] = [condition1, condition2]
    return bays

def compute_buckling_loads(wing, material, point_array, t_vars):
    compression_k_lookup = {0.2:22.2, 0.3:10.9, 0.4:6.92, 0.6:4.23, 0.8:3.45, 
                        1.0:3.29, 1.2:3.40, 1.4:3.68, 1.6:3.45, 1.8:3.32, 
                        2.0:3.29, 2.2:3.32, 2.4:3.40, 2.7:3.32, 3.0:3.29}
    shear_k_lookup = {1.0:7.75, 1.2:6.58, 1.4:6.00, 1.5:5.84, 1.6:5.76, 
                    1.8:5.59, 2.0:5.43, 2.5:5.18, 3.0:5.02}
    E, nu, G = material.get_constants()
    sigma_cr = []
    tau_cr = []
    for i in range(point_array.shape[1]-1):
        # get relevant parametric points
        lower = point_array[:, i].tolist()   # [s1, s2]
        upper = point_array[:, i+1].tolist() # [s1, s2]

        # get thickness
        t = t_vars['upper_wing_thickness_'+str(i)]

        # approximate the bay as a rectangle between ribs (lower and upper) and the spars (s1 and s2)
        # compute the side lengths of the rectangle
        corner_points_parametric = lower + upper
        corner_points = wing.geometry.evaluate(corner_points_parametric, non_csdl=True)
        b1 = np.linalg.norm(corner_points[0] - corner_points[1])
        b2 = np.linalg.norm(corner_points[2] - corner_points[3])
        a1 = np.linalg.norm(corner_points[0] - corner_points[2])
        a2 = np.linalg.norm(corner_points[1] - corner_points[3])
        a = (a1 + a2)/2
        b = (b1 + b2)/2
        aspect_ratio = a/b
        compression_k = compression_k_lookup[min(compression_k_lookup, key=lambda x:abs(x-aspect_ratio))]
        if aspect_ratio < 1:
            aspect_ratio = 1/aspect_ratio
        shear_k = shear_k_lookup[min(shear_k_lookup, key=lambda x:abs(x-aspect_ratio))]

        sigma_cr.append(compression_k*E/(1-nu**2)*(t/b)**2)
        tau_cr.append(shear_k*E/(1-nu**2)*(t/b)**2)
    return sigma_cr, tau_cr

def compute_curved_buckling_loads(wing, material, point_array, t_vars, surface="upper"):
    if surface not in ["upper", "lower"]:
        raise ValueError("'surface' must either be 'upper' or 'lower'")
    
    E, nu, G = material.get_constants()
    if isinstance(E, csdl.Variable):
        E = E.value
        nu = nu.value
        G = G.value
    sigma_cr = csdl.Variable(shape=(point_array.shape[1]-1, ), value=0)
    for i in range(point_array.shape[1]-1):
        # get relevant parametric points
        lower = point_array[:, i].tolist()
        upper = point_array[:, i+1].tolist()
        lower_middle = (lower[0][0], (lower[0][1]+lower[1][1])/2)
        upper_middle = (upper[0][0], (upper[0][1]+upper[1][1])/2)

        # get thickness
        t = t_vars[f'{surface}_wing_thickness_'+str(i)]

        # approximate the bay as a rectangle between ribs (lower and upper) and the spars (s1 and s2)
        # compute the side lengths of the rectangle
        corner_points_parametric = lower + upper
        corner_points = wing.geometry.evaluate(corner_points_parametric, non_csdl=True)
        b1 = np.linalg.norm(corner_points[0] - corner_points[1])
        b2 = np.linalg.norm(corner_points[2] - corner_points[3])
        a1 = np.linalg.norm(corner_points[0] - corner_points[2])
        a2 = np.linalg.norm(corner_points[1] - corner_points[3])
        a = (a1 + a2)/2
        b = (b1 + b2)/2
        
        # compute average radius of curvature
        r = np.sum(roc(wing, lower+upper+[lower_middle, upper_middle]))/6

        # sigma_cr.append(1/6*E/(1-nu**2)*((12*(1-nu**2)*(t/r)**2+(np.pi*t/b)**4)**(1/2)+(np.pi*t/b)**2))
        sigma_cr = sigma_cr.set(
            csdl.slice[i], 
            1/6*E/(1-nu**2)*((12*(1-nu**2)*(t/r)**2+(np.pi*t/b)**4)**(1/2)+(np.pi*t/b)**2)
        )
    return sigma_cr

def roc(wing:cd.Component, point):
    u_prime = wing.geometry.evaluate(point, parametric_derivative_orders=(0,1), non_csdl=True)[:,0]
    u_double_prime = wing.geometry.evaluate(point, parametric_derivative_orders=(0,2), non_csdl=True)[:,0]
    return np.abs(1+np.abs(u_prime)**(1/2)/u_double_prime)


def load_dv_values(fname, group):
    inputs = csdl.inline_import(fname, group)
    recorder = csdl.get_current_recorder()
    dvs = recorder.design_variables
    for var in dvs:
        var.set_value(inputs[var.name].value)
        scale = 1/np.linalg.norm(inputs[var.name].value)
        dvs[var] = (scale, dvs[var][1], dvs[var][2])