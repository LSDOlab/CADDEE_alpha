import numpy as np
import meshio
import time
import lsdo_geo as lg


def import_mesh(file, component:lg.Geometry, rescale:list=[1,1,1], remove_dupes=True, plot=False, grid_search_n:int=5, force_reprojection=False):
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
        nodes, index = np.unique(nodes, return_inverse=True, axis=0)
        #print('dupes removed in ' + str(end-start) + ' seconds')
        # map indices in cells to new indices
        cells = []
        connectivity = np.ndarray((0,4))
        nquads = 0
        for cell in mesh.cells:
            if cell.type == 'quad':     #TODO: add aditional 2D element types
                for n in range(cell.data.shape[0]):
                    row = cell.data[n,:]
                    # row = np.array([cind2[i] for i in row])
                    row = np.array([index[i] for i in row])
                    connectivity = np.vstack((connectivity,row.reshape((1,4))))
                    cell.data[n,:] = row
                cells.append(cell)
                nquads += cell.data.shape[0]
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
        ma_nodes_parametric = component.project(nodes, grid_search_density_parameter=grid_search_n, plot=plot, force_reprojection=force_reprojection)
        ma_nodes = component.evaluate(ma_nodes_parametric)

    return ma_nodes, ma_nodes_parametric, connectivity