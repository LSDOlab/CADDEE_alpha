import csdl_alpha as csdl
from typing import Union
import numpy as np


def perform_local_to_body_transformation(
    phi: Union[csdl.Variable, float, int],
    theta: Union[csdl.Variable, float, int],
    psi: Union[csdl.Variable, float, int],
    vectors: Union[np.ndarray, csdl.Variable],

) -> csdl.Variable: 
    """Perform a roll -> pitch -> yaw coordinate transformation from the local
    to body-fixed frame based on Euler angles. 

    Parameters
    ----------
    phi : Union[csdl.Variable, float, int]
        roll angle (rad)
    
    theta : Union[csdl.Variable, float, int]
        pitch angle (rad)
    
    psi : Union[csdl.Variable, float, int]
        yaw angle (rad)
    
    vectors : csdl.Variable
        The vectors to be rotated; stored in an 
        array of size (num_nodes, 3)

    Returns
    -------
    csdl.Variable
        Rotated vectors; stored in array of size 
        (num_nodes, 3)
    """

    csdl.check_parameter(phi, "phi", types=(csdl.Variable, float, int, np.ndarray))
    csdl.check_parameter(theta, "theta", types=(csdl.Variable, float, int, np.ndarray))
    csdl.check_parameter(psi, "psi", types=(csdl.Variable, float, int, np.ndarray))
    csdl.check_parameter(vectors, "vectors", types=(csdl.Variable, np.ndarray), allow_none=True)

    # check the euler angles all have the same shape
    if isinstance(phi, (int, float)):
        phi_shape = (1, )
    else:
        phi_shape = phi.shape

    if isinstance(theta, (int, float)):
        theta_shape = (1, )
    else:
        theta_shape = theta.shape
    
    if isinstance(psi, (int, float)):
        psi_shape = (1, )
    else:
        psi_shape = psi.shape

    if not phi_shape == theta_shape == psi_shape:
        raise Exception("Euler angles of different shapes. They should be scalars or vectors of size (num_nodes, )")

    angles_shape = phi_shape
    if len(angles_shape) > 2:
        raise Exception(f"Euler angles must be of shape (num_nodes, ), (num_nodes, 1) or (1, num_nodes). Received shape {angles_shape}")
    
    if len(angles_shape) == 2:
        if angles_shape[0] != 1 or angles_shape[1] != 1:
            raise Exception(f"Euler angles must be of shape (num_nodes, ), (num_nodes, 1) or (1, num_nodes). Received shape {angles_shape}")

    # get the number of nodes
    num_nodes = max(angles_shape)

    phi = phi.reshape((num_nodes, ))
    theta = theta.reshape((num_nodes, ))
    psi = psi.reshape((num_nodes, ))

    if vectors is not None:
        # check if the vector shape is compatible
        vector_shape = vectors.shape
        if vector_shape[-1] != 3 or len(vector_shape) > 2:
            raise Exception(f"'vectors' must be a vector of size (3, ) or a 2d array of shape {(num_nodes, 3)}. Received shape {vector_shape}")

        # Reshape or expand vectors to (num_nodes, 3) if possible
        try:
            vectors = vectors.reshape((num_nodes, 3))
        except:
            try:
                vectors = csdl.expand(vectors, (num_nodes, 3), action='j->ij')
            except:
                raise Exception(f"'vectors' must be a vector of size (3, ) or a 2d array of shape {(num_nodes, 3)}. Received shape {vector_shape}")


        transformed_vec = csdl.Variable(shape=(num_nodes, 3), value=0.)

    L2B_tensor = csdl.Variable(shape=(num_nodes, 3, 3), value=0.)

    for i in range(num_nodes):
        L2B_mat = csdl.Variable(shape=(3, 3), value=0.)
        L2B_mat = L2B_mat.set(
            slices=csdl.slice[0, 0],
            value=csdl.cos(theta[i]) * csdl.cos(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[0, 1],
            value=csdl.cos(theta[i]) * csdl.sin(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[0, 2],
            value= -csdl.sin(theta[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[1, 0],
            value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) - csdl.cos(phi[i]) * csdl.sin(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[1, 1],
            value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) + csdl.cos(phi[i]) * csdl.cos(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[1, 2],
            value=csdl.sin(phi[i]) * csdl.cos(theta[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[2, 0],
            value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) + csdl.sin(phi[i]) * csdl.sin(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[2, 1],
            value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) - csdl.sin(phi[i]) * csdl.cos(psi[i]),
        )

        L2B_mat = L2B_mat.set(
            slices=csdl.slice[2, 2],
            value=csdl.cos(phi[i]) * csdl.cos(theta[i]),
        )
        
        L2B_tensor = L2B_tensor.set(
            slices=csdl.slice[i, :, :],
            value=L2B_mat
        )

        if vectors is not None:
            transformed_vec = transformed_vec.set(
                csdl.slice[i, :], csdl.matvec(L2B_mat, vectors[i, :])
            )
        
        
    if vectors is not None:
        return transformed_vec
    
    else:
        return L2B_tensor


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    phi = csdl.Variable(shape=(2, ), value=np.deg2rad(10))
    theta = np.deg2rad(np.array([4, 7]))
    psi = np.deg2rad(np.array([2, 6]))

    vector = np.array([10, 5, 0])


    rotated_vector = perform_local_to_body_transformation(
        phi, theta, psi, vector
    )

    print(rotated_vector.value)