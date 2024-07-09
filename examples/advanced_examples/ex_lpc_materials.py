import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs


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

def construct_thickness_function(wing, num_ribs, top_array, bottom_array, material, initial_thickness, minimum_thickness, dv_dict=None, add_dvs=True):
    bay_eps = 1e-2
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
        # l_coord_u = lower[0][1][0,0]
        # u_coord_u = upper[0][1][0,0]
        if l_surf_ind == u_surf_ind:
            condition = construct_bay_condition(u_coord_u, l_coord_u)
            thickness = csdl.Variable(value=initial_thickness, name='upper_wing_thickness_'+str(i))
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            thickness = csdl.Variable(value=initial_thickness, name='upper_wing_thickness_'+str(i))
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
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
            thickness = csdl.Variable(value=initial_thickness, name='lower_wing_thickness_'+str(i))
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {l_surf_ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
        else:
            condition1 = construct_bay_condition(1, l_coord_u)
            condition2 = construct_bay_condition(u_coord_u, 0)
            thickness = csdl.Variable(value=initial_thickness, name='lower_wing_thickness_'+str(i))
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
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
            thickness = csdl.Variable(value=initial_thickness, name=name+'_thickness')
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
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
            thickness = csdl.Variable(value=initial_thickness, name=f'spar_{spar_num}_thickness_{i}')
            if dv_dict is not None:
                thickness.value = dv_dict[thickness.name]
            if add_dvs:
                thickness.set_as_design_variable(upper=0.05, lower=minimum_thickness, scaler=1e3)
            function = lfs.Function(lfs.ConditionalSpace(2, condition), thickness)
            functions = {ind: function}
            thickness_fs = lfs.FunctionSet(functions)
            wing.quantities.material_properties.add_material(material, thickness_fs)
            spar_num += 1