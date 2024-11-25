import numpy as np
import meshio
import time
import lsdo_geo as lg
from lsdo_function_spaces import Function
import copy


def import_mesh(file, component:lg.Geometry, rescale:list=[1,1,1], remove_dupes=True, plot=False, grid_search_n:int=5, force_reprojection=False,
                priority_inds=None, priority_eps=1e-4, dupe_tol=1e-5):
    '''
    Read mesh file (from any format meshio supports) and convert into mapped array + connectivity
    ------------
    Parameters:
        file: str, name of mesh file
        ms: mechanical structure object 
        ...
    
    '''
    optimize_projection = False
    ms = None
    am = None
    tol = None
    mesh = meshio.read(file)
    nodes = np.array([mesh.points[:,0]*rescale[0], mesh.points[:,1]*rescale[1], mesh.points[:,2]*rescale[2]]).T

    if remove_dupes:
        nnodes = nodes.shape[0]
        # remove duplicate nodes
        tol_decimals = int(-np.log10(dupe_tol))
        rounded_nodes = np.round(nodes,decimals=tol_decimals)
        _, o_index, index = np.unique(rounded_nodes, return_index=True, return_inverse=True, axis=0)
        nodes = nodes[o_index]
        #print('dupes removed in ' + str(end-start) + ' seconds')
        # map indices in cells to new indices
        cells = []
        connectivity = None
        nquads = 0
        for cell in mesh.cells:
            if cell.type == 'vertex':
                continue
            elif cell.type == 'line':
                continue
            elif cell.type == 'quad':     #TODO: add aditional 2D element types
                for n in range(cell.data.shape[0]):
                    row = cell.data[n,:]
                    # row = np.array([cind2[i] for i in row])
                    row = np.array([index[i] for i in row])
                    if connectivity is None:
                        connectivity = row.reshape((1,4))
                    else:
                        connectivity = np.vstack((connectivity,row.reshape((1,4))))
                    cell.data[n,:] = row
                cells.append(cell)
                nquads += cell.data.shape[0]
            elif cell.type == 'triangle':
                for n in range(cell.data.shape[0]):
                    row = cell.data[n,:]
                    # row = np.array([cind2[i] for i in row])
                    row = np.array([index[i] for i in row])
                    if connectivity is None:
                        connectivity = row.reshape((1,3))
                    else:
                        connectivity = np.vstack((connectivity,row.reshape((1,3))))
                    cell.data[n,:] = row
                cells.append(cell)
                nquads += cell.data.shape[0]
            else:
                print('cell type ' + cell.type + ' not supported')
        print('number of duplicates removed is ' + str(nnodes-nodes.shape[0]))
        nnodes = nodes.shape[0]
    else:
        connectivity = np.ndarray((0,4))
        nquads = 0
        for cell in mesh.cells:
            if cell.type == 'quad':     #TODO: add aditional 2D element types
                for row in cell.data:
                    connectivity = np.vstack((connectivity,row.reshape((1,4))))
                nquads += cell.data.shape[0]

    if optimize_projection:
        # assign nodes to their cells, cell to a surface in ms
        # TODO: look into vectorization for this
        if targets is None:
            if component is None:
                targets = list(ms.primitives.values())
            else:
                targets = list(component.get_primitives())
        targets = np.array(targets)
        cells = np.array(cells)
        cell_targets = np.full(targets.shape[0], None)
        node_cells = np.full(nodes.shape[0], None)
        cell_num = 0
        for cell in cells:
            included_nodes = []
            for element in cell.data:
                for node in element:
                    if node_cells[node] is None:
                        node_cells[node] = cell_num
                        included_nodes.append(node)
            included_nodes = np.array(included_nodes)
            valid_targets = np.full(targets.shape[0],True,dtype=bool)
            i = 0
            while np.count_nonzero(valid_targets) > 1 and i < included_nodes.shape[0]:
                n = 0
                for target in targets:
                    if valid_targets[n]:
                        proj_point = target.project(nodes[included_nodes[i]])
                        if np.linalg.norm(proj_point.value - nodes[included_nodes[i]]) > tol:
                            valid_targets[n] = False
                    n += 1
                i += 1
            cell_targets[cell_num] = np.nonzero(valid_targets)[0][0]
            cell_num += 1
        # project nodes onto assigned targets
        ma_nodes = ms.project(nodes[0], targets = [targets[cell_targets[node_cells[0]]]])
        for i in range(1,nnodes):
            ma_new = ms.project(nodes[i], targets = [targets[cell_targets[node_cells[i]]]])
            ma_nodes = am.vstack((ma_nodes,ma_new))
        if plot:
            ms.plot_meshes(ma_nodes, mesh_plot_types=['point_cloud'])
    else:
        ma_nodes_parametric = component.project(nodes, grid_search_density_parameter=grid_search_n, plot=plot, force_reprojection=force_reprojection,
                                                priority_inds=priority_inds, priority_eps=priority_eps)
        ma_nodes = component.evaluate(ma_nodes_parametric)

    return ma_nodes, ma_nodes_parametric, connectivity


def compute_objective(
        surface_fun_1 : Function, 
        surface_fun_2 : Function, 
        uv_1,
        uv_2,
    ):
        residual = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - \
            surface_fun_2.evaluate(parametric_coordinates=uv_2).value
        
        return residual.T @ residual

def compute_constraint(
    surface_fun_1, 
    uv_1, 
    point_on_intersection,
    distance_between_intersection_points,
):
    constraint = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection
    constraint_distance = np.linalg.norm(surface_fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection)**2 - distance_between_intersection_points**2

    return constraint_distance

def compute_objective_gradient_and_hessian(
    surface_fun_1 : Function, 
    surface_fun_2 : Function, 
    uv_1,
    uv_2,
):
    # Compute residual B1(u1, v1)P1 - B2(u2, v2)P2
    residual = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - \
        surface_fun_2.evaluate(parametric_coordinates=uv_2).value
    
    # 1) gradient of distance w.r.t. entries of residual --> (1, 3)
    grad_d_x = 2 * np.array([[residual[0], residual[1], residual[2]]])

    # 2) gradient of cartesian coords. w.r.t. parametric coords. (Jacobian) --> (3, 4)
    grad_x_u = np.zeros((3, 4))
    hessian_x_u = np.zeros((3, 4, 4))
    for i in range(2):
        parametric_der_order = np.zeros((2, ), dtype=int)
        parametric_der_order[i] = 1

        grad_x_u[:, i] = surface_fun_1.space.compute_basis_matrix(
            uv_1, parametric_derivative_orders=parametric_der_order
        ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

        grad_x_u[:, i+2] = -surface_fun_2.space.compute_basis_matrix(
            uv_2, parametric_derivative_orders=parametric_der_order
        ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))
        
        # Extra dimension for Hessian (tensor)
        for j in range(2):
            parametric_der_order_hessian = np.zeros((2, ), dtype=int)
            if i==j:
                parametric_der_order_hessian[i] = 2
            else:
                parametric_der_order_hessian[i] = 1
                parametric_der_order_hessian[j] = 1

            # Note that surface_fun_1 does not depend on u2, v2 (parameters of surface_fun_2)
            hessian_x_u[:, j, i] = surface_fun_1.space.compute_basis_matrix(
                uv_1, parametric_derivative_orders=parametric_der_order_hessian
            ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

            # Note that surface_fun_2 does not depend on u1, v1 (parameters of fun_1)
            hessian_x_u[:, j+2, i+2] = -surface_fun_2.space.compute_basis_matrix(
                uv_2, parametric_derivative_orders=parametric_der_order_hessian
            ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))

    # 3) apply chain (and product) rule to compute grad_d_u and hess_d_u
    grad_d_u = grad_d_x @ grad_x_u
    hessian_d_u = 2 * (grad_x_u.T @ grad_x_u) + np.einsum('i,ijk->jk', grad_d_x.flatten(), hessian_x_u)

    return grad_d_u, hessian_d_u, grad_x_u, hessian_x_u

def compute_constraint_gradient_and_hessian(
    surface_fun_1 : Function, 
    uv_1,
    grad_x_u,
    hessian_x_u,
    point_on_intersection, 
):
    constraint = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection
    grad_c_x = 2 * np.array([[constraint[0], constraint[1], constraint[2]]])

    grad_c_u = np.zeros((1, 4))
    grad_c_u[0, 0:2] = grad_c_x @ grad_x_u[:, 0:2]

    hessian_x_u_for_c = np.zeros((3, 4, 4))
    hessian_x_u_for_c[:, 0:2, 0:2] = hessian_x_u[:, 0:2, 0:2]
    grad_x_u_for_c = np.zeros((3, 4))
    grad_x_u_for_c[:, 0:2] = grad_x_u[:, 0:2]

    hessian_c_u = 2 * (grad_x_u_for_c.T @ grad_x_u_for_c) + np.einsum('i,ijk->jk', grad_c_x.flatten(), hessian_x_u_for_c)

    return grad_c_u, hessian_c_u


def find_point_on_intersection(
        surface_fun_1 : Function, 
        surface_fun_2 : Function, 
        starting_point=None, 
        point_on_intersection=None, 
        distance_between_intersection_points=0.05, 
        num_iteration=100
    ):
    uv_1 = starting_point[0:2]
    uv_2 = starting_point[2:]
    
    # Set converged flag to false
    newton_opt_converged = False

    # Initial lagrange multiplier if point on intersection is given

    for iteration in range(num_iteration):
        uv_1 = starting_point[0:2]
        uv_2 = starting_point[2:]

        
        f = compute_objective(surface_fun_1, surface_fun_2, uv_1, uv_2)
        df, d2f, grad_x_u, hessian_x_u = compute_objective_gradient_and_hessian(surface_fun_1, surface_fun_2, uv_1, uv_2)
        
        if point_on_intersection is not None:
            lambda_0 = 0.
            c = compute_constraint(surface_fun_1, uv_1, point_on_intersection, distance_between_intersection_points)
            dc, d2c = compute_constraint_gradient_and_hessian(surface_fun_1, uv_1, grad_x_u, hessian_x_u, point_on_intersection)

            gradient = df + lambda_0 * dc
            hessian = d2f + lambda_0 * d2c

        else:
            gradient = df
            hessian = d2f
       

        # 4) Identify the active set
        # NOTE: this is a very ineqality constraint (design variable bounds)
        # instead of adding Lagrange multipliers we remove the design variable 
        # (i.e., parametric coordinate) when it's zero or one and the gradient 
        # tries to "push" it outside of the bounds
        remove_dvs_lower_bound = np.logical_and(starting_point == 0, gradient > 0)
        remove_dvs_upper_bound = np.logical_and(starting_point == 1, gradient < 0)
        remove_dvs = np.logical_or(remove_dvs_lower_bound, remove_dvs_upper_bound)

        # # NOTE: remove zero columns in the Hessian in edge cases
        # # (e.g., if there are infinitely many optimal solution)
        # # NOTE: This could potentially mask bugs
        remove_cols_hessian = np.where(~hessian.any(axis=1))[0]
        try:
            remove_dvs[remove_cols_hessian] = True
        except:
            break

        keep_dvs = np.logical_not(remove_dvs).flatten()

        reduced_gradient = gradient.flatten()[keep_dvs]
        reduced_hessian = hessian[keep_dvs][:, keep_dvs]

        if point_on_intersection is not None:
            dc = dc.flatten()[keep_dvs]

        if np.linalg.norm(f) < 1e-12:
            newton_opt_converged = True
            break
        
        # compute step direction

        if point_on_intersection is not None:
            dim = len(reduced_hessian)
            newton_system_lhs = np.zeros((dim+1, dim+1))
            newton_system_lhs[0:dim, 0:dim] = reduced_hessian
            newton_system_lhs[dim, 0:dim] = dc
            newton_system_lhs[0:dim, dim] = dc

            newton_system_rhs = np.zeros((dim+1, ))
            newton_system_rhs[0:dim] = reduced_gradient
            newton_system_rhs[dim] = c
        else:
            newton_system_lhs = reduced_hessian
            newton_system_rhs = reduced_gradient

        # step = np.linalg.solve(reduced_hessian, -reduced_gradient.flatten())
        try:
            step = np.linalg.solve(newton_system_lhs, -newton_system_rhs.flatten())
        except:
            print("starting_point", starting_point)


        step_length = 1

        if point_on_intersection is not None:
            new_uv_coords = step[0:dim]
            lambda_0 = step[dim]
        else:
            new_uv_coords = step

        # Update starting point with step
        starting_point[keep_dvs] = starting_point[keep_dvs] + step_length * new_uv_coords.flatten()
        # starting_point = starting_point + new_uv_coords.flatten()
        
        # clamp parameters between 0, 1
        starting_point = np.clip(starting_point, 0, 1)

    if point_on_intersection is not None:
        return starting_point, newton_opt_converged, f, c**0.5
    else:
        return starting_point, newton_opt_converged, f, None

def trace_intersection(surface_fun_1 : Function, surface_fun_2 : Function):
    components_intersect = False
    
    # Perform grid search with one edge of the intersecting components fixed
    num_grid_points = 50
    u, v = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points), indexing="ij") 
    para_array = np.vstack((u.flatten(), v.flatten())).T
    para_tensor = para_array.reshape((num_grid_points, num_grid_points, 2))

    fun_1_eval = surface_fun_1.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))
    fun_2_eval = surface_fun_2.evaluate(parametric_coordinates=para_array).value.reshape((num_grid_points, num_grid_points, 3))

    # Calculate pairwise distances using broadcasting
    distances = np.linalg.norm(fun_1_eval[:, :, None, None, :] - fun_2_eval[None, None, :, :, :], axis=-1)

    # Find minimum distance and corresponding indices
    min_dist = np.min(distances)
    if min_dist > 0.1:
        return None, None, None, None, None, None, None, None

    ind_1, ind_2, ind_3, ind_4 = np.unravel_index(np.argmin(distances), distances.shape)

    # Retrieve starting points using indices
    starting_point = np.array([
        para_tensor[ind_1, ind_2, 0],
        para_tensor[ind_1, ind_2, 1],
        para_tensor[ind_3, ind_4, 0],
        para_tensor[ind_3, ind_4, 1],
    ])

    # num_grid_points = 200
    # u, v = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points), indexing="ij")
    # para_tensor = np.stack((u, v), axis=-1)
    # para_array = para_tensor.reshape(-1, 2)

    # # Identify boundary points for both surfaces
    # boundary_mask = (u == 0) | (u == 1) | (v == 0) | (v == 1)
    # boundary_points = para_tensor[boundary_mask]  # Boundary points (parametric space)

    # # Evaluate both functions
    # surface_fun_1_eval = surface_fun_1.evaluate(parametric_coordinates=para_array).value  # Shape: (num_grid_points^2, 3)
    # surface_fun_2_eval = surface_fun_2.evaluate(parametric_coordinates=para_array).value  # Shape: (num_grid_points^2, 3)

    # # Evaluate boundary points of surface_fun_1 with all points of surface_fun_2
    # surface_fun_1_boundary_eval = surface_fun_1.evaluate(parametric_coordinates=boundary_points).value
    # distances_1 = np.linalg.norm(surface_fun_1_boundary_eval[:, None, :] - surface_fun_2_eval[None, :, :], axis=-1)

    # # Evaluate boundary points of surface_fun_2 with all points of surface_fun_1
    # surface_fun_2_boundary_eval = surface_fun_2.evaluate(parametric_coordinates=boundary_points).value
    # distances_2 = np.linalg.norm(surface_fun_1_eval[:, None, :] - surface_fun_2_boundary_eval[None, :, :], axis=-1)

    # # Combine the distances
    # combined_distances = np.vstack((distances_1.reshape(-1), distances_2.T.reshape(-1)))

    # # Find the minimum distance and corresponding indices
    # min_index = np.argmin(combined_distances)
    # min_dist = np.min(combined_distances)

    # if min_dist > 0.05:
    #     return components_intersect, _, _

    # else:
    #     if min_index < distances_1.size:
    #         ind_1, ind_2 = np.unravel_index(min_index, distances_1.shape)
    #         starting_point = np.array([
    #             boundary_points[ind_1, 0],
    #             boundary_points[ind_1, 1],
    #             para_array[ind_2, 0],
    #             para_array[ind_2, 1],
    #         ])
    #     else:
    #         ind_1, ind_2 = np.unravel_index(min_index - distances_1.size, distances_2.shape)
    #         starting_point = np.array([
    #             para_array[ind_1, 0],
    #             para_array[ind_1, 1],
    #             boundary_points[ind_2, 0],
    #             boundary_points[ind_2, 1],
    #         ])

    uv_intersection_points_1 = []
    uv_intersection_points_2 = []

    initial_point_on_intersection, newton_opt_converged, _, _ = find_point_on_intersection(
        surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, starting_point=starting_point,
    )


    point_on_intersection_cartesian = surface_fun_1.evaluate(initial_point_on_intersection[0:2]).value
    point_on_intersection_cartesian_copy = copy.copy(point_on_intersection_cartesian)

    if newton_opt_converged:
        uv_intersection_points_1.append(initial_point_on_intersection[0:2])
        uv_intersection_points_2.append(initial_point_on_intersection[2:])
    else:
        raise Exception(f"Cannot find an intersection point that lies on any of the edges of the intersecting B-splines. Point found is {starting_point}.")

    initial_direction = None
    # next_parametric_coords, new_direction = compute_step_in_direction_of_intersection(surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, direction=initial_direction, point_on_intersection=point_on_intersection)
    next_parametric_coords_plus, next_parametric_coords_minus = compute_step_in_direction_of_intersection(surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, direction=initial_direction, point_on_intersection=initial_point_on_intersection)

    step_multiplier = 1
    for i in range(400):
        # plus
        point_on_intersection_plus, newton_opt_converged_plus, res, const = find_point_on_intersection(
            surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, starting_point=next_parametric_coords_plus, point_on_intersection=point_on_intersection_cartesian
        )
        point_on_intersection_cartesian = surface_fun_1.evaluate(point_on_intersection_plus[0:2]).value

        next_parametric_coords_plus, _ = compute_step_in_direction_of_intersection(surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, point_on_intersection=point_on_intersection_plus, step_multiplier=step_multiplier)

        if newton_opt_converged_plus:
            uv_1_inter = point_on_intersection_plus[0:2]
            cartesian_inter = surface_fun_1.evaluate(parametric_coordinates=uv_1_inter).value
            uv_2_inter = point_on_intersection_plus[2:]
            uv_intersection_points_1.append(uv_1_inter)
            uv_intersection_points_1_ordered = order_points_by_proximity(surface_fun_1.evaluate(parametric_coordinates=np.vstack(uv_intersection_points_1)).value)
            if np.array_equal(cartesian_inter, uv_intersection_points_1_ordered[0]):
                uv_intersection_points_2.append(uv_2_inter)
                step_multiplier = 1
            elif np.array_equal(cartesian_inter,  uv_intersection_points_1_ordered[-1]):
                uv_intersection_points_2.append(uv_2_inter)
                step_multiplier = 1
            else:
                uv_intersection_points_1.pop()
                step_multiplier *= 2
                print("increase step multiplier")
            # uv_intersection_points_2.append(point_on_intersection_plus[2:])
        else:
            point_on_intersection_plus, newton_opt_converged_plus, res, _ = find_point_on_intersection(
                surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, starting_point=next_parametric_coords_plus, point_on_intersection=None
            )
            if newton_opt_converged_plus:
                uv_1_inter = point_on_intersection_plus[0:2]
                uv_2_inter = point_on_intersection_plus[2:]
                if any(np.array_equal(uv_1_inter, arr) for arr in uv_intersection_points_1):
                    step_multiplier *= 2
                else:
                    step_multiplier = 1
                    uv_intersection_points_1.append(uv_1_inter)
                    uv_intersection_points_2.append(uv_2_inter)
                    break

            else:
                raise Exception(f"Cannot find last intersection point. Point found is {point_on_intersection_plus}.")

    step_multiplier = 1
    for i in range(400):
        # minus
        point_on_intersection_minus, newton_opt_converged_minus, res, const = find_point_on_intersection(
            surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, starting_point=next_parametric_coords_minus, point_on_intersection=point_on_intersection_cartesian_copy
        )
        point_on_intersection_cartesian_copy = surface_fun_1.evaluate(point_on_intersection_minus[0:2], plot=False).value

        _, next_parametric_coords_minus = compute_step_in_direction_of_intersection(surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, point_on_intersection=point_on_intersection_minus, step_multiplier=step_multiplier)


        if newton_opt_converged_minus:
            uv_1_inter = point_on_intersection_minus[0:2]
            cartesian_inter = surface_fun_1.evaluate(parametric_coordinates=uv_1_inter).value
            uv_2_inter = point_on_intersection_minus[2:]
            uv_intersection_points_1.append(uv_1_inter)
            uv_intersection_points_1_ordered = order_points_by_proximity(surface_fun_1.evaluate(parametric_coordinates=np.vstack(uv_intersection_points_1)).value)
            if np.array_equal(cartesian_inter, uv_intersection_points_1_ordered[0]):
                uv_intersection_points_2.append(uv_2_inter)
                step_multiplier = 1
            elif np.array_equal(cartesian_inter, uv_intersection_points_1_ordered[-1]):
                uv_intersection_points_2.append(uv_2_inter)
                step_multiplier = 1
            else:
                uv_intersection_points_1.pop()
                print("increase step multiplier")
                step_multiplier *= 2
        else:
            point_on_intersection_minus, newton_opt_converged_minus, res, _ = find_point_on_intersection(
                surface_fun_1=surface_fun_1, surface_fun_2=surface_fun_2, starting_point=next_parametric_coords_minus, point_on_intersection=None
            )
            if newton_opt_converged_minus:
                uv_intersection_points_1.append(point_on_intersection_minus[0:2])
                uv_intersection_points_2.append(point_on_intersection_minus[2:])

                uv_1_coords = np.vstack(uv_intersection_points_1)
                # surface_fun_1.evaluate(parametric_coordinates=uv_1_coords, plot=True)

                uv_2_coords = np.vstack(uv_intersection_points_2)
                # surface_fun_2.evaluate(parametric_coordinates=uv_2_coords, plot=True)

                components_intersect = True
                return components_intersect, uv_1_coords, uv_2_coords
            
            else:
                raise Exception(f"Cannot find last intersection point. Point found is {point_on_intersection_minus}.")


    return components_intersect, None, None

def compute_step_in_direction_of_intersection(surface_fun_1 : Function, surface_fun_2: Function, point_on_intersection, step_length=0.02, direction=None, step_multiplier=1):
    uv_1 = point_on_intersection[0:2]
    uv_2 = point_on_intersection[2:]
    
    tangent_vectors_fun_1 = np.zeros((2, 3))
    tangent_vectors_fun_2 = np.zeros((2, 3))

    curvature_hessian_fun_1 = np.zeros((3, 2, 2))
    curvature_hessian_fun_2 = np.zeros((3, 2, 2))
    for p in range(2):
        parametric_derivative_orders = np.zeros((surface_fun_1.space.num_parametric_dimensions, ), dtype=int)
        parametric_derivative_orders_curvature = np.zeros((surface_fun_1.space.num_parametric_dimensions, ), dtype=int)
        parametric_derivative_orders[p] = 1
        parametric_derivative_orders_curvature[p] = 2

        tangent_vectors_fun_1[p, :] = surface_fun_1.space.compute_basis_matrix(
            uv_1, parametric_derivative_orders=parametric_derivative_orders
        ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

        tangent_vectors_fun_2[p, :] = surface_fun_2.space.compute_basis_matrix(
            uv_2, parametric_derivative_orders=parametric_derivative_orders
        ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))

        for q in range(2):
            parametric_der_order_hessian = np.zeros((2, ), dtype=int)
            if p==q:
                parametric_der_order_hessian[p] = 2
            else:
                parametric_der_order_hessian[p] = 1
                parametric_der_order_hessian[q] = 1

            curvature_hessian_fun_1[:, p, q] = surface_fun_1.space.compute_basis_matrix(
                uv_1, parametric_derivative_orders=parametric_der_order_hessian,
            ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

            curvature_hessian_fun_2[:, p, q] = surface_fun_2.space.compute_basis_matrix(
                uv_2, parametric_derivative_orders=parametric_der_order_hessian,
            ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))

    # 2) Compute surface normal for each patch
    N1 = np.cross(tangent_vectors_fun_1[0, :], tangent_vectors_fun_1[1, :])
    N2 = np.cross(tangent_vectors_fun_2[0, :], tangent_vectors_fun_2[1, :])

    # 3) Compute direction of intersection (normal of the two normals)
    T = np.cross(N1, N2)
    T_norm = T / np.linalg.norm(T)

    # First order Taylor series approximation (use as starting point for Newton iteration)
    reg_param = 1e-5
    delta_para_1 = np.linalg.solve(tangent_vectors_fun_1 @ tangent_vectors_fun_1.T + reg_param * np.eye(2), tangent_vectors_fun_1 @ T).reshape((-1, 2))
    delta_para_2 = np.linalg.solve(tangent_vectors_fun_2 @ tangent_vectors_fun_2.T + reg_param * np.eye(2), tangent_vectors_fun_2 @ T).reshape((-1, 2))

    delta_uv = np.zeros((4, ))
    delta_uv[0:2] = delta_para_1.flatten()
    delta_uv[2:] = delta_para_2.flatten()
    delta_uv_norm = delta_uv / np.linalg.norm(delta_uv)

    step = 0.01 * step_multiplier

    
    uv_1_updated_minus = uv_1 - step * delta_uv_norm[0:2]
    uv_2_updated_minus = uv_2 - step * delta_uv_norm[2:] 

    uv_1_updated_plus = uv_1 + step * delta_uv_norm[0:2]
    uv_2_updated_plus = uv_2 + step * delta_uv_norm[2:] 

    next_parametric_coords_plus = np.clip(np.array([uv_1_updated_plus, uv_2_updated_plus]).flatten(), 0, 1)
    next_parametric_coords_minus = np.clip(np.array([uv_1_updated_minus, uv_2_updated_minus]).flatten(), 0, 1)

    return next_parametric_coords_plus, next_parametric_coords_minus #next_parametric_coords, direction

def update_step(uv_1, uv_2, para_step, step_direction, function_1: Function, function_2: Function, cartesian_step_length):
    cartesian_point = function_1.evaluate(parametric_coordinates=uv_1).value
    para_step_length = 0.0001

    if step_direction is not None:
        if step_direction == "plus":
            while True:
                new_para_point_1 = uv_1 + para_step_length * para_step[0:2]
                new_cartesian_point = function_1.evaluate(parametric_coordinates=np.clip(new_para_point_1, 0, 1)).value
                if (np.linalg.norm(new_cartesian_point-cartesian_point) - cartesian_step_length) >= 0.:
                    new_para_point_2 = uv_2 + para_step_length * para_step[2:]
                    break
                else:
                    para_step_length *= 2
        else:
            while True:
                new_para_point_1 = uv_1 - para_step_length * para_step[0:2]
                new_cartesian_point = function_1.evaluate(parametric_coordinates=np.clip(new_para_point_1, 0, 1)).value
                if (np.linalg.norm(new_cartesian_point-cartesian_point) - cartesian_step_length) >= 0.:
                    new_para_point_2 = uv_2 - para_step_length * para_step[2:]
                    break
                else:
                    para_step_length *= 2

    else: 
        new_para_point_1 = uv_1 + para_step_length * para_step[0:2]
        new_para_point_2 = uv_2 + para_step_length * para_step[2:]

        if (new_para_point_1 < 0).any() or (new_para_point_1 > 1).any():
            step_direction = "minus"
        elif (new_para_point_2 < 0).any() or (new_para_point_2 > 1).any():
            step_direction = "minus"
        else:
            step_direction = "plus"

        update_step(uv_1=uv_1, uv_2=uv_2, para_step=para_step, step_direction=step_direction, function_1=function_1, function_2=function_2, cartesian_step_length=cartesian_step_length)



    return np.vstack((new_para_point_1, new_para_point_2)).flatten(), step_direction
    

def resample_curve(points, num_points):
    # Calculate cumulative arc length
    cumulative_lengths = compute_cumulative_lengths(points)
    total_length = cumulative_lengths[-1]
    
    # Generate target arc lengths
    target_lengths = np.linspace(0, total_length, num_points)
    
    # Interpolate to find new points at target lengths
    resampled_points = []
    for s_k in target_lengths:
        # Find where target length falls between cumulative lengths
        i = np.searchsorted(cumulative_lengths, s_k) - 1
        i = min(max(i, 0), len(points) - 2)  # Clamp index to valid range
        
        # Linear interpolation
        t = (s_k - cumulative_lengths[i]) / (cumulative_lengths[i+1] - cumulative_lengths[i])
        new_point = points[i] + t * (points[i+1] - points[i])
        resampled_points.append(new_point)
    
    return np.array(resampled_points)

def compute_cumulative_lengths(points):
    # Calculate cumulative arc length for each point
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = np.hstack(([0], np.cumsum(distances)))
    return cumulative_lengths

def order_points_by_proximity(points):
    # Copy the original points to avoid modifying the input
    points = np.array(points)
    
    starting_point = points[np.where(points[:, 0] == np.max(points[:, 0]))]
    
    # Initialize ordered points with the first point and mark it as visited
    ordered_points = [starting_point.flatten()]
    remaining_points = set(range(1, len(points)))  # Track indices of unvisited points
    
    # Continue ordering until all points are added
    while remaining_points:
        last_point = ordered_points[-1]
        
        # Find the closest unvisited point
        closest_index = min(remaining_points, key=lambda i: np.linalg.norm(points[i] - last_point))
        ordered_points.append(points[closest_index])
        remaining_points.remove(closest_index)
    
    return np.array(ordered_points)[1:, :]



        # # Compute residual B1(u1, v1)P1 - B2(u2, v2)P2
        # residual = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - \
        #     surface_fun_2.evaluate(parametric_coordinates=uv_2).value
        
        # if point_on_intersection is not None:
        #     constraint = surface_fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection
        #     constraint_distance = np.linalg.norm(surface_fun_1.evaluate(parametric_coordinates=uv_1).value - point_on_intersection)**2 - distance_between_intersection_points**2
        
        # # Compute the square of the euclidean distance (OBJECTIVE)
        # distance_squared = residual.T @ residual

        # # Compute gradient/hessian of squared distance w.r.t. u1, v1, u2, v2 --> (4, ) & (4, 4)
        
        # # 1) gradient of distance w.r.t. entries of residual --> (1, 3)
        # grad_d_x = 2 * np.array([[residual[0], residual[1], residual[2]]])
        # if point_on_intersection is not None:
        #     grad_c_x = 2 * np.array([[constraint[0], constraint[1], constraint[2]]])

        # # 2) gradient of cartesian coords. w.r.t. parametric coords. (Jacobian) --> (3, 4)
        # grad_x_u = np.zeros((3, 4))
        # hessian_x_u = np.zeros((3, 4, 4))
        # for i in range(2):
        #     parametric_der_order = np.zeros((2, ), dtype=int)
        #     parametric_der_order[i] = 1

        #     grad_x_u[:, i] = surface_fun_1.space.compute_basis_matrix(
        #         uv_1, parametric_derivative_orders=parametric_der_order
        #     ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

        #     grad_x_u[:, i+2] = -surface_fun_2.space.compute_basis_matrix(
        #         uv_2, parametric_derivative_orders=parametric_der_order
        #     ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))
            
        #     # Extra dimension for Hessian (tensor)
        #     for j in range(2):
        #         parametric_der_order_hessian = np.zeros((2, ), dtype=int)
        #         if i==j:
        #             parametric_der_order_hessian[i] = 2
        #         else:
        #             parametric_der_order_hessian[i] = 1
        #             parametric_der_order_hessian[j] = 1

        #         # Note that surface_fun_1 does not depend on u2, v2 (parameters of surface_fun_2)
        #         hessian_x_u[:, j, i] = surface_fun_1.space.compute_basis_matrix(
        #             uv_1, parametric_derivative_orders=parametric_der_order_hessian
        #         ).dot(surface_fun_1.coefficients.value.reshape((-1, 3)))

        #         # Note that surface_fun_2 does not depend on u1, v1 (parameters of fun_1)
        #         hessian_x_u[:, j+2, i+2] = -surface_fun_2.space.compute_basis_matrix(
        #             uv_2, parametric_derivative_orders=parametric_der_order_hessian
        #         ).dot(surface_fun_2.coefficients.value.reshape((-1, 3)))
                


        # # 3) apply chain (and product) rule to compute grad_d_u and hess_d_u
        # grad_d_u = grad_d_x @ grad_x_u
        # hessian_d_u = 2 * (grad_x_u.T @ grad_x_u) + np.einsum('i,ijk->jk', grad_d_x.flatten(), hessian_x_u)
        # if point_on_intersection is not None:
        #     grad_c_u = np.zeros((1, 4))
        #     grad_c_u[0, 0:2] = grad_c_x @ grad_x_u[:, 0:2]

        #     hessian_x_u_for_c = np.zeros((3, 4, 4))
        #     hessian_x_u_for_c[:, 0:2, 0:2] = hessian_x_u[:, 0:2, 0:2]
        #     grad_x_u_for_c = np.zeros((3, 4))
        #     grad_x_u_for_c[:, 0:2] = grad_x_u[:, 0:2]

        #     hessian_c_u = 2 * (grad_x_u_for_c.T @ grad_x_u_for_c) + np.einsum('i,ijk->jk', grad_c_x.flatten(), hessian_x_u_for_c)

        #     grad_L_u = grad_d_u + lambda_0 * grad_c_u
        #     hessian_L_u = hessian_d_u + lambda_0 * hessian_c_u

        #     gradient = grad_L_u
        #     hessian = hessian_L_u

        # else:
        #     gradient = grad_d_u
        #     hessian = hessian_d_u


