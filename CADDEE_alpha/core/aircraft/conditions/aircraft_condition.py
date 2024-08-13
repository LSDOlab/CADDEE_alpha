from CADDEE_alpha.core.condition import Condition
from CADDEE_alpha.utils.var_groups import AircaftStates, LongitudinalStabilityMetrics
from CADDEE_alpha.utils.coordinate_transformations import perform_local_to_body_transformation
from typing import Union, Tuple
from CADDEE_alpha.core.aircraft.models.atmosphere.simple_atmosphere_model import AtmosphericStates, SimpleAtmosphereModel
import csdl_alpha as csdl
from dataclasses import dataclass, asdict, fields
import numpy as np
import warnings
from CADDEE_alpha.core.mesh.meshers import CamberSurface, RotorDiscretization


@dataclass
class HoverParameters(csdl.VariableGroup):
    altitude : Union[float, int, csdl.Variable, np.ndarray]
    time : Union[float, int, csdl.Variable, np.ndarray]


@dataclass
class ClimbParameters:
    initial_altitude : Union[float, int, csdl.Variable, np.ndarray] 
    final_altitude : Union[float, int, csdl.Variable, np.ndarray] 
    pitch_angle : Union[float, int, csdl.Variable, np.ndarray]
    climb_gradient : Union[float, int, csdl.Variable, None, np.ndarray] 
    rate_of_climb : Union[float, int, csdl.Variable, None, np.ndarray]
    speed : Union[float, int, csdl.Variable, None, np.ndarray]
    mach_number : Union[float, int, csdl.Variable, None, np.ndarray]
    flight_path_angle : Union[float, int, csdl.Variable, None, np.ndarray] 
    time : Union[float, int, csdl.Variable, None, np.ndarray]

    # @property
    # def initial_altitude(self):
    #     return self._initial_altitude
    
    # @initial_altitude.setter
    # def initial_altitude(self, new_value):
    #     self._initial_altitude = new_value

    # @property
    # def final_altitude(self):
    #     return self._final_altitude
    
    # @final_altitude.setter
    # def final_altitude(self, new_value):
    #     self._final_altitude = new_value

    # @property
    # def pitch_angle(self):
    #     return self._pitch_angle
    
    # @pitch_angle.setter
    # def pitch_angle(self, new_value):
    #     self._pitch_angle = new_value

    # @property
    # def climb_gradient(self):
    #     return self._climb_gradient
    
    # @climb_gradient.setter
    # def climb_gradient(self, new_value):
    #     self._climb_gradient = new_value

    # @property
    # def climb_gradient(self):
    #     return self._climb_gradient
    
    # @climb_gradient.setter
    # def climb_gradient(self, new_value):
    #     self._climb_gradient = new_value

    # @property
    # def climb_gradient(self):
    #     return self._climb_gradient
    
    # @climb_gradient.setter
    # def climb_gradient(self, new_value):
    #     self._climb_gradient = new_value

    # @property
    # def climb_gradient(self):
    #     return self._climb_gradient
    
    # @climb_gradient.setter
    # def climb_gradient(self, new_value):
    #     self._climb_gradient = new_value

    # @property
    # def climb_gradient(self):
    #     return self._climb_gradient
    
    # @climb_gradient.setter
    # def climb_gradient(self, new_value):
    #     self._climb_gradient = new_value
    

@dataclass
class ACQuantities(csdl.VariableGroup):
    ac_states: AircaftStates = None
    atmos_states: AtmosphericStates = None
    inertial_forces = None
    inertial_moments = None

@dataclass
class CruiseParameters:
    altitude : Union[float, int, csdl.Variable, np.ndarray]
    speed : Union[float, int, csdl.Variable, np.ndarray]
    mach_number : Union[float, int, csdl.Variable, np.ndarray]
    pitch_angle : Union[float, int, csdl.Variable, csdl.ImplicitVariable, np.ndarray]
    range : Union[float, int, csdl.Variable, np.ndarray]
    time : Union[float, int, csdl.Variable, np.ndarray]


class AircraftCondition(Condition):
    """General aircraft condition."""
    def __init__(self, u, v, w, p, q, r, phi, theta, psi, x, y, z, atmos_model=SimpleAtmosphereModel()) -> None:
        self.quantities : ACQuantities = ACQuantities()
        self.quantities.ac_states = AircaftStates(
            u=u, v=v, w=w, p=p, q=q, r=r,
            phi=phi, theta=theta, psi=psi, 
            x=x, y=y, z=z,
        )
        # Check shape consistency
        self._check_shape_consistency(self.quantities.ac_states)

        # Expand states
        self._expand_variables("ac_states")

        # Compute atmospheric states (with expanded variables)
        self.quantities.atmos_states = atmos_model.evaluate(self.quantities.ac_states.z)


    def _expand_variables(self, qty):
        if qty == "ac_states":
            variables = self.quantities.ac_states
        elif qty == "parameters":
            variables = self.parameters
        else:
            raise NotImplementedError
        
        # Loop over parameters or ac states
        for field in fields(variables):
            var_name = field.name
            var = getattr(variables, var_name)
            # check the type of the variable
            skip_flag = False
            if var is None:
                skip_flag = True
            elif not isinstance(var, (csdl.Variable, csdl.ImplicitVariable)):
                if isinstance(var, (int, float)):
                    csdl_var = csdl.Variable(shape=(1, ), value=var)
                else:
                    csdl_var = csdl.Variable(shape=var.shape, value=var)
            else:
                csdl_var = var

            if not skip_flag:
                if csdl_var.shape == (1, ):
                    csdl_var_exp = csdl.expand(csdl_var, (self._num_nodes, ))
                    setattr(variables, var_name, csdl_var_exp)
                elif csdl_var.shape == (1, 1):
                    csdl_var_exp = csdl.expand(csdl_var.reshape((1, )), (self._num_nodes, ))
                    setattr(variables, var_name, csdl_var_exp)
                else:
                    setattr(variables, var_name, csdl_var.reshape((self._num_nodes, )))
    
    def _check_shape_consistency(self, variables):
        # Get shapes of all aircraft states
        variable_shapes = []
        for var_name in variables.__annotations__.keys():
            var = getattr(variables, var_name)
            if isinstance(var, (int, float)):
                variable_shapes.append((1, ))
            elif isinstance(var, np.ndarray):
                variable_shapes.append(var.shape)
            elif isinstance(var, csdl.Variable):
                variable_shapes.append(var.shape)
            else:
                variable_shapes.append(())
        # variable_shapes = [np.asanyarray(getattr(variables, variable)).shape for variable in variables.__annotations__.keys()]
        # variable_shapes = [np.asarray(getattr(variables, variable)).shape for variable in variables.__annotations__.keys()]
        variables = [variable for variable in variables.__annotations__.keys()]

        # Check if all shapes are the same or broadcastable
        for i, shape in enumerate(variable_shapes):
            if not self._is_vector(shape):
                raise ValueError(f"Received invalid shape for variable {variables[i]} {shape}. Can only accept vectors.")
            for j, other_shape in enumerate(variable_shapes):
                if i != j:
                    if shape != other_shape and not self._is_broadcastable(shape, other_shape):
                        raise ValueError(f"Inconsistent shapes for variables: {variables[i]} {shape} and {variables[j]} {other_shape}")
    
        num_nodes = max([max(var_shape) for var_shape in variable_shapes if var_shape])
        self._num_nodes = num_nodes

    def _is_vector(self, shape):
        if len(shape) > 2:
            return False
        else:
            if len(shape) == 2:
                if shape[0] == 1 or shape[1] == 1:
                    return True
                else:
                    return False
            else:
                return True
    
    def _is_broadcastable(self, shape1, shape2):
        """Check if two shapes are broadcastable (only for vectors)."""
        for s1, s2 in zip(reversed(shape1), reversed(shape2)):
            if s1 != 1 and s2 != 1 and s1 != s2:
                return False
        return True

    def update(self, state : str, value : Union[float, int, np.ndarray, csdl.Variable]):
        ac_states = self.ac_states.__annotations__.keys()
        if state not in ac_states:
            raise KeyError(f"Unknown state {state}. Acceptable states are {ac_states}")
        # TODO: warning when updating csdl variables that coudl be design variables
        else:
            raise NotImplementedError
    
    def finalize_meshes(self, ignore_body_rotations=False) -> None:
        """Expand meshes along the "num_nodes" axis, compute mesh velcoties and assemble
        meshes into isntance of MeshContainer.
        
        Solvers may be vectorized or their inputs discretized in time ("num_nodes"). As such,
        meshes may need to be expanded along the "num_nodes" axis. In addition, aerodynamic 
        solvers require nodal mesh velocities, which are in terms of the linear and angular 
        velocities (due to vehicle rotations p, q, r) in the body-fixed frame. This function 
        computes and sets the "nodal_velocities" attribute of the mesh class. 
        """
        from CADDEE_alpha.core.configuration import VectorizedConfig

        normal_config = self.configuration
        vectorized_config = self.vectorized_configuration
        
        if normal_config is None and vectorized_config is None:
            raise Exception("Cannot finalize meshes since its configuration or vectorized_configuration has not been set.")

        if normal_config is not None:
            config = normal_config
        else:
            config = vectorized_config

        if isinstance(config, VectorizedConfig):
            cg_vec = None # TODO: MPS for Vectroized config

        else:
            ac_mps = config.system.quantities.mass_properties
            cg_vec = ac_mps.cg_vector

            if isinstance(cg_vec, list):
                if not [cg for cg in cg_vec if cg is not None]:
                    cg_vec = None
                else:
                    raise NotImplementedError()

            if cg_vec is None:
                if ignore_body_rotations:
                    print("No mass properties defined; ignore any body rotations in mesh velocities")
                else:
                    config.assemble_system_mass_properties()
                    if cg_vec is None:
                        warnings.warn("No mass properties defined; ignore any body rotations in mesh velocities")

        mesh_container = config.mesh_container
        
        # If no meshes are in the container, raise exception
        if not mesh_container:
            raise Exception("No meshes associated with configuration.")

        # Assemble the linear and angular velocities
        u = self.quantities.ac_states.u
        v = self.quantities.ac_states.v
        w = self.quantities.ac_states.w
        p = self.quantities.ac_states.p
        q = self.quantities.ac_states.q
        r = self.quantities.ac_states.r

        V_vec = csdl.Variable(shape=(self._num_nodes, 3), value=0)
        V_vec = V_vec.set(csdl.slice[:, 0], u)
        V_vec = V_vec.set(csdl.slice[:, 1], v)
        V_vec = V_vec.set(csdl.slice[:, 2], w)
        omega_vec = csdl.Variable(shape=(self._num_nodes, 3), value=0)
        omega_vec = omega_vec.set(csdl.slice[:, 0], p)
        omega_vec = omega_vec.set(csdl.slice[:, 1], q)
        omega_vec = omega_vec.set(csdl.slice[:, 2], r)

        # Loop over meshes in mesh_container
        for mesh_name, mesh in mesh_container.items():
            for discretization_name, discretization in mesh.discretizations.items():
                # update mesh
                discretization._update()

                # get mesh coordinates
                initial_nodal_coordinates = discretization.nodal_coordinates


                if isinstance(initial_nodal_coordinates, list): # This will be the case for vect. configs.
                    num_nodes_config = len(initial_nodal_coordinates)
                    if self._num_nodes != num_nodes_config:
                        raise Exception(f"'num_nodes' of the aircraft condition ({self._num_nodes}) and vectorized configuration ({num_nodes_config}) must be equal")

                    coordinates_shape = initial_nodal_coordinates[0].shape
                    stacked_nodal_coordiantes = csdl.Variable(shape=(num_nodes_config, ) + coordinates_shape, value=0.)
                    
                    # Can't use frange since 'initial_nodal_coordinates' is a list for vect. configs.
                    for i in range(num_nodes_config):
                        stacked_nodal_coordiantes = stacked_nodal_coordiantes.set(
                            slices=csdl.slice[i, :], 
                            value=initial_nodal_coordinates[i],
                        )

                    initial_nodal_coordinates = stacked_nodal_coordiantes
                    discretization.nodal_coordinates = initial_nodal_coordinates
                    discretization._has_been_expanded = True

                if discretization._has_been_expanded:
                    shape_exp = discretization.nodal_coordinates.shape

                else:
                    mesh_shape = discretization.nodal_coordinates.shape
                    shape_exp = (self._num_nodes, ) + mesh_shape
                    mesh_action_string = convert_shape_to_action_string(mesh_shape, None, "mesh")
                    nodal_coordinates_exp = csdl.expand(initial_nodal_coordinates, shape_exp, mesh_action_string)
                    discretization.nodal_coordinates = nodal_coordinates_exp

                # expand linear and angular veclocities
                V_vec_action_string = convert_shape_to_action_string(None, shape_exp, "vel_vec")
                omega_vec_action_string = convert_shape_to_action_string(None, shape_exp, "vel_vec")
                V_vec_exp = csdl.expand(V_vec, shape_exp, V_vec_action_string)
                omega_vec_exp = csdl.expand(omega_vec, shape_exp, omega_vec_action_string)

                if cg_vec is not None:
                    if ignore_body_rotations:
                        nodal_velocities = V_vec_exp
                    else:
                        # expand cg_vec
                        cg_vec_action_string = convert_shape_to_action_string(None, shape_exp, "cg_vec")
                        r_vec_exp = nodal_coordinates_exp - csdl.expand(cg_vec, shape_exp, cg_vec_action_string)

                        # Compute mesh velocities
                        nodal_velocities = V_vec_exp + csdl.cross(omega_vec_exp, r_vec_exp, axis=len(shape_exp) -1)

                else:
                    nodal_velocities = V_vec_exp

                discretization.nodal_velocities = nodal_velocities


                # # Set the nodal_velocities in the mesh data class instance
                # # rotation_tensor = None
                # if rotation_tensor is None:
                #     discretization.nodal_velocities = nodal_velocities
                # else:
                #     discretization.nodal_velocities = csdl.einsum(rotation_tensor, nodal_velocities, action='iklm,ijkl->ijkm')


    def assemble_forces_and_moments(self, 
            aero_propulsive_forces: list[csdl.Variable],
            aero_propulsive_moments: list[csdl.Variable],
            load_factor: Union[int, float, csdl.Variable] = 1,
            ref_point: Union[csdl.Variable, np.ndarray] = np.array([0., 0., 0.]),
            ac_mps=None,
        ) -> tuple[csdl.Variable]:
        from CADDEE_alpha.core.configuration import VectorizedConfig

        """Compute the total forces and moments acting on the aircraft.
        
        This method sums the total aero-propulsive forces and inertial forces
        and rotates them into the body-fixed frame bsed on the Euler angles
        (i.e., phi, theta, psi) stored in the aircraft states.

        The inertial forces AND moments are ignored if the configuration of 
        the condition has not been set.

        The inertial moments will be ignored if the center of gravity of the aircraft 
        cannot be computed.

        Parameters
        ----------
        aero_propulsive_forces : list[csdl.Variable]
            list of aero-propulsive forces
        
        aero_propulsive_moments : list[csdl.Variable]
            list of aero-propulsive moments
        
        load_factor : Union[int, float, csdl.Variable], optional
            factor by which the gravitational constant is multiplied, by default 1
        
        ref_point : Union[csdl.Variable, np.ndarray], optional
            reference points for computing inertial moments, by default np.array([0., 0., 0.])

        Returns
        -------
        tuple[csdl.Variable]
            (total forces, total moments)

        Raises
        ------
        TypeError
            if forces or moments are not csdl Variables
        Exception
            forces and moments are not of shape (num_nodes, 3)
        """
        csdl.check_parameter(load_factor, "load_factor", types=(int, float, csdl.Variable))
        csdl.check_parameter(aero_propulsive_moments, "aero_propulsive_moments", types=list)
        csdl.check_parameter(aero_propulsive_forces, "aero_propulsive_forces", types=list)

        num_nodes = self._num_nodes
        total_forces = csdl.Variable(shape=(self._num_nodes, 3), value=0.)
        total_moments = csdl.Variable(shape=(self._num_nodes, 3), value=0.)

        # sum of the aero-propulsive forces and moments
        for i in range(len(aero_propulsive_forces)):
            force = aero_propulsive_forces[i]
            # Check that the forces are csdl variables and have the right shape
            if not isinstance(force, csdl.Variable):
                raise TypeError(f"Received invalid type {force}. Forces must be of type {csdl.Variable}")
            if not force.shape == (num_nodes, 3):
                raise Exception(f"Shape mismatch. Forces must be shape (nun_nodes, 3)={(num_nodes, 3)}. Received shape {force.shape}")
            
            total_forces =  total_forces + force

        for i in range(len(aero_propulsive_moments)):
            moment = aero_propulsive_moments[i]
            # Check that the forces are csdl variables and have the right shape
            if not isinstance(moment, csdl.Variable):
                raise TypeError(f"Received invalid type {moment}. Moments must be of type {csdl.Variable}")
            if not force.shape == (num_nodes, 3):
                raise Exception(f"Shape mismatch. Moments must be shape (nun_nodes, 3)={(num_nodes, 3)}. Received shape {moment.shape}")
            
            total_moments =  total_moments + moment
        
        # Get Euler angles from the ac states
        ac_states = self.quantities.ac_states
        phi = ac_states.phi
        theta = ac_states.theta
        psi = ac_states.psi


        # Check if the configuration has any mass properties
        ignore_inertial_loads = False
        ignore_inertial_moments = False
        norm_config = self.configuration
        vectorized_config = self.vectorized_configuration

        if norm_config is not None and vectorized_config is not None:
            raise Exception(f"Condition {self} has configuration and vectorized_configuration. Cannot have both.")
        
        if norm_config is not None:
            config = norm_config
        else:
            config = vectorized_config

        if config is None:
            warnings.warn(f"Configuration has not been set for condition {self}. Will ignore inertial loads")
            ignore_inertial_loads = True
        
        elif isinstance(config, VectorizedConfig):
            mass = ac_mps.mass
            cg = ac_mps.cg_vector            
        
        else:
            # Compute configuration mass properties
            config.assemble_system_mass_properties()
            ac_mps = config.system.quantities.mass_properties
            cg = ac_mps.cg_vector
            mass = ac_mps.mass

            if cg is None and mass is None:
                warnings.warn(f"Configuration's mass properties are {None}. Will ignore inertial loads")
                ignore_inertial_loads = True
            elif cg is None:
                warnings.warn(f"Configuration's cg is {None}. Will ignore inertial moments")
                ignore_inertial_moments = True


        g = 9.81 * load_factor # TODO: compute g as a function of altitude

        # If there are no inertial loads, total forces = aero-propulsive forces
        if ignore_inertial_loads:

            # total_forces_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_forces
            # )

            # total_moments_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_moments
            # )

            total_forces_body_fixed = total_forces
            total_moments_body_fixed = total_moments

        # If we only have mass (and no cg), only add inertial forces
        elif ignore_inertial_moments:
            inertial_forces = csdl.Variable(shape=(num_nodes, 3), value=0)
            inertial_forces = inertial_forces.set(
                csdl.slice[:, 2], mass * g
            )
            inertial_forces_body_fixed = perform_local_to_body_transformation(
                phi, theta, psi, inertial_forces
            )
            
            total_forces = total_forces + inertial_forces_body_fixed

            total_forces_body_fixed = total_forces
            total_moments_body_fixed = total_moments

            # total_forces_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_forces
            # )

            # total_moments_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_moments
            # )

        # Otherwise also compute inertial moments
        else:
            # Compute moment arm
            cg_exp = csdl.expand(cg, (num_nodes, 3), 'i->ji')
            ref_point_ex = csdl.expand(ref_point, (num_nodes, 3), 'i->ji')
            r_exp = cg_exp - ref_point_ex

            r_exp_body_fixed = perform_local_to_body_transformation(
                phi, theta, psi, r_exp
            )

            inertial_forces = csdl.Variable(shape=(num_nodes, 3), value=0)
            inertial_forces = inertial_forces.set(
                csdl.slice[:, 2], mass * g
            )

            inertial_forces_body_fixed = perform_local_to_body_transformation(
                phi, theta, psi, inertial_forces
            )

            # print(inertial_forces_body_fixed.value)
            # exit()

            inertial_moments_body_fixed = csdl.cross(r_exp_body_fixed, inertial_forces_body_fixed, axis=1)

            total_forces = total_forces + inertial_forces_body_fixed
            total_moments = total_moments + inertial_moments_body_fixed

            total_forces_body_fixed = total_forces
            total_moments_body_fixed = total_moments

            # total_forces_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_forces
            # )

            # total_moments_body_fixed = perform_local_to_body_transformation(
            #     phi, theta, psi, total_moments
            # )
        
        return total_forces_body_fixed, total_moments_body_fixed


    def perform_linear_stability_analysis(
        self,
        total_forces,
        total_moments,
        ac_states,
        mass_properties,
        g0=9.81,
    ) -> LongitudinalStabilityMetrics:
        m = mass_properties.mass
        iyy = mass_properties.inertia_tensor[1, 1]

        num_nodes = total_forces.shape[0]
        for i in range(num_nodes):
            A_mat = csdl.Variable(shape=(4, 4), value=0)

            # Aircraft states
            u = ac_states.u # [i]
            w = ac_states.w # [i]
            q = ac_states.q # [i]
            theta = ac_states.theta #[i]

            # Aerodynamic forces and moments
            X = total_forces[i, 0]
            Y = total_forces[i, 1]
            Z = total_forces[i, 2]

            L = total_moments[i, 0]
            M = total_moments[i, 1]
            N = total_moments[i, 2]

            # stability derivatives
            X_u = csdl.derivative(ofs=X, wrts=u)
            X_w = csdl.derivative(ofs=X, wrts=w)
            Z_u = csdl.derivative(ofs=Z, wrts=u)
            Z_w = csdl.derivative(ofs=Z, wrts=w)
            Z_q = csdl.derivative(ofs=Z, wrts=q)
            M_u = csdl.derivative(ofs=M, wrts=u)
            M_w = csdl.derivative(ofs=M, wrts=w)
            M_q = csdl.derivative(ofs=M, wrts=q)

            # Construct A matrix
            A_mat = A_mat.set(csdl.slice[0, 0], X_u / m)
            A_mat = A_mat.set(csdl.slice[0, 1], X_w / m)
            A_mat = A_mat.set(csdl.slice[0, 3], -g0 * csdl.cos(theta))

            A_mat = A_mat.set(csdl.slice[1, 0], Z_u / m)
            A_mat = A_mat.set(csdl.slice[1, 1], Z_w / m)
            A_mat = A_mat.set(csdl.slice[1, 2], (Z_q + m * u) / m)
            A_mat = A_mat.set(csdl.slice[1, 3], -g0 * csdl.sin(theta))

            A_mat = A_mat.set(csdl.slice[2, 0], M_u / iyy)
            A_mat = A_mat.set(csdl.slice[2, 1], M_w / iyy)
            A_mat = A_mat.set(csdl.slice[2, 2], M_q / iyy)

            A_mat = A_mat.set(csdl.slice[3, 2], 1.)

            eig_val_operation = EigenValueOperation()
            eig_real, eig_imag = eig_val_operation.evaluate(A_mat)

            print(eig_real.value)
            print(eig_imag.value)
            import matplotlib.pyplot as plt
            plt.scatter(eig_real[0:2].value, eig_imag[0:2].value, label="short-period")
            plt.scatter(eig_real[2:].value, eig_imag[2:].value, label="long-period (Phugoid)")
            plt.axhline(0, color='black',linewidth=0.5)
            plt.axvline(0, color='black',linewidth=0.5)
            plt.xlabel(r'$Re$', fontsize=14)
            plt.ylabel(r'$Im$', fontsize=14)
            plt.xlim([-3.5, 0.1])
            plt.ylim([-10, 10])
            plt.legend(loc="best",fontsize=14)
            # plt.gca().set_aspect('equal', adjustable='box')

            # Show grid
            plt.grid(True)
            plt.show()
            # exit()


            # Short period (more stable; eigenvalues come out sorted)
            lambda_sp_real = eig_real[0]
            lambda_sp_imag = eig_imag[0]
            sp_omega_n = ((lambda_sp_real**2 + lambda_sp_imag**2) + 1e-10)**0.5
            sp_damping_ratio = -lambda_sp_real / sp_omega_n
            sp_time_2_double = np.log(2) / (lambda_sp_real**2 + 1e-10)**0.5

            # Phugoid (long period) (less stable)
            lambda_ph_real = eig_real[2]
            lambda_ph_imag = eig_imag[2]
            ph_omega_n = ((lambda_ph_real**2 + lambda_ph_imag**2) + 1e-10)**0.5
            ph_damping_ratio = -lambda_ph_real / ph_omega_n
            ph_time_2_double = np.log(2) / (lambda_ph_real**2 + 1e-10)**0.5

            long_stability_results = LongitudinalStabilityMetrics(
                A_mat_longitudinal=A_mat,
                real_eig_vals_short_period=lambda_sp_real,
                imag_eig_vals_short_period=lambda_sp_imag,
                nat_freq_short_period=sp_omega_n,
                damping_ratio_short_period=sp_damping_ratio,
                time_2_double_short_period=sp_time_2_double,
                real_eig_vals_phugoid=lambda_ph_real,
                imag_eig_vals_phugoid=lambda_ph_imag,
                nat_freq_phugoid=ph_omega_n,
                damping_ratio_phugoid=ph_damping_ratio,
                time_2_double_phugoid=ph_time_2_double,
            )

        return long_stability_results

class EigenValueOperation(csdl.CustomExplicitOperation):
    def __init__(self):
        super().__init__()

    def evaluate(self, mat):
        shape = mat.shape
        size = shape[0]

        self.declare_input("mat", mat)
        eig_real = self.create_output("eig_vals_real", shape=(size, ))
        eig_imag = self.create_output("eig_vals_imag", shape=(size, ))

        self.declare_derivative_parameters("eig_vals_real", "mat")
        self.declare_derivative_parameters("eig_vals_imag", "mat")

        return eig_real, eig_imag

    def compute(self, inputs, outputs):
        mat = inputs["mat"]

        eig_vals, eig_vecs = np.linalg.eig(mat)
    
        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        outputs['eig_vals_real'] = np.real(eig_vals)
        outputs['eig_vals_imag'] = np.imag(eig_vals)

    def compute_derivatives(self, inputs, outputs, derivatives):
        mat = inputs["mat"]
        size = mat.shape[0]

        eig_vals, eig_vecs = np.linalg.eig(mat)
        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        # v inverse transpose
        v_inv_T = (np.linalg.inv(eig_vecs)).T

        # preallocate Jacobian: n outputs, n^2 inputs
        temp_r = np.zeros((size, size*size))
        temp_i = np.zeros((size, size*size))

        for j in range(len(eig_vals)):
            # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
            partial = np.outer(eig_vecs[:, j], v_inv_T[:, j]).flatten(order='F')
            # Note that the order of flattening matters, hence argument in flatten()

            # Set jacobian rows
            temp_r[j, :] = np.real(partial)
            temp_i[j, :] = np.imag(partial)

        # Set Jacobian
        derivatives['eig_vals_real', 'mat'] = temp_r
        derivatives['eig_vals_imag', 'mat'] = temp_i


def convert_shape_to_action_string(old_shape, new_shape, type_: str):
    csdl.check_parameter(old_shape, "old_shape", types=tuple, allow_none=True)
    csdl.check_parameter(type_, "type_", values=["mesh", "cg_vec", "vel_vec"])
    
    # mesh expansion from mesh shape -> (num_nodes, ) + mesh_shape
    if type_ == "mesh":
        base_shape_str = ''.join(chr(97 + num ) for num in range(len(old_shape)))
        exp_shape_str = chr(97 + len(old_shape)) + base_shape_str
    
        action_str = base_shape_str + f'->{exp_shape_str}'
    
    # cg_vec expansion from (3, ) -> (num_nodes, ) + mesh_shape
    elif type_ == "cg_vec":
        base_shape_str = "a"
        exp_shape_str = "".join(chr(98 + num) for num in range(len(new_shape)-1)) + base_shape_str 

        action_str = base_shape_str + f'->{exp_shape_str}'

    # vel_vec expansion from (num_nodes, 3) -> (num_nodes, ) + mesh_shape
    else: 
        base_shape_str = "ab"
        exp_shape_str = "".join(chr(99 + num ) for num in range(len(new_shape)-2)) 

        action_str = base_shape_str + f'->{base_shape_str[0]}{exp_shape_str}{base_shape_str[1]}'


    return action_str

class CruiseCondition(AircraftCondition):
    """Cruise condition: intended for steady analyses.
    
    Parameters
    ----------
    - speed : int | float | np.ndarray | csdl.Variable

    - mach_number : Union[float, int, csdl.Variable]

    - time : int | float | np.ndarray | csdl.Variable

    - pitch_angle : int | float | np.ndarray | csdl.Variable

    - range : int | float | np.ndarray | csdl.Variable
    
    - altitude : int | float | np.ndarray | csdl.Variable

    Note: cannot specify all parameters at once 
    (e.g., cannot specify speed and mach_number at the same time)
    """

    def __init__(self, 
                 altitude : Union[float, int, csdl.Variable, np.ndarray],
                 range : Union[float, int, csdl.Variable, np.ndarray, None],
                 pitch_angle : Union[float, int, csdl.Variable, np.ndarray]=0.,
                 speed : Union[float, int, csdl.Variable, np.ndarray, None]=None,
                 mach_number : Union[float, int, csdl.Variable, None, np.ndarray]=None,
                 time : Union[float, int, csdl.Variable, None, np.ndarray]=None,
                 atmos_model = SimpleAtmosphereModel()
                 ):
        csdl.check_parameter(altitude, "altitude", types=(float, int, csdl.Variable, np.ndarray))
        csdl.check_parameter(range, "range", types=(float, int, csdl.Variable, np.ndarray), allow_none=True)
        csdl.check_parameter(pitch_angle, "pitch_angle", types=(float, int, csdl.Variable, csdl.ImplicitVariable, np.ndarray), allow_none=True)
        csdl.check_parameter(speed, "speed", types=(float, int, csdl.Variable, np.ndarray), allow_none=True)
        csdl.check_parameter(mach_number, "mach_number", types=(float, int, csdl.Variable, np.ndarray), allow_none=True)
        csdl.check_parameter(time, "time", types=(float, int, csdl.Variable, np.ndarray), allow_none=True)
        
        self.parameters : CruiseParameters = CruiseParameters(
            altitude=altitude,
            speed=speed,
            range=range,
            pitch_angle=pitch_angle,
            mach_number=mach_number,
            time=time,
        )
        self.quantities = ACQuantities()
        self._atmos_model = atmos_model

        # Check shapes
        self._check_shape_consistency(self.parameters)

        # Expand parameters to (num_nodes, )
        self._expand_variables("parameters")

        # Compute ac states
        self._setup_condition()

        

    def _setup_condition(self):
        # Different combinations of conflicting attributes
        conflicting_attributes_1 = ["speed", "mach_number"]
        conflicting_attributes_2 = ["speed", "time", "range"]
        conflicting_attributes_3 = ["mach_number", "time", "range"]

        # Check for conflicting attributes:
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_1]):
            raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_2]):
            raise Exception("Cannot specify 'speed', 'time', and 'range' at the same time")
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_3]):
            raise Exception("Cannot specify 'mach_number', 'time', and 'range' at the same time")

        # zero aircraft states in climb
        x = y = v = phi = psi = p = q = r = csdl.Variable(shape=(self._num_nodes, ), value=0.)

        # set z to altitude and evaluate atmosphere model
        z = self.parameters.altitude
        atmos_states = self._atmos_model.evaluate(z)

        # set theta to pitch_angle
        theta = self.parameters.pitch_angle
        

        # Compute or set speed, mach, range, and time
        mach_number = self.parameters.mach_number
        speed = self.parameters.speed
        time = self.parameters.time
        range = self.parameters.range

        if mach_number is not None and range is not None:
            V = atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            time = range / V
            self.parameters.time = time
        elif mach_number is not None and time is not None:
            V = atmos_states.speed_of_sound * mach_number
            self.parameters.speed = V
            range = V * time
            self.parameters.range = range
        elif speed is not None and range is not None:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            time = range / V
            self.parameters.time = time
        elif speed is not None and time is not None:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            range = V * time 
            self.parameters.range = range
        else:
            raise NotImplementedError

        # Compute u and w from V and theta
        u = V * csdl.cos(theta)
        w = V * csdl.sin(theta)

        # Set aircraft and atmospheric states
        self.quantities.ac_states = AircaftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, 
            phi=phi, theta=theta, psi=psi,
            x=x, y=y, z=z,
        )
        self.quantities.atmos_states = atmos_states
    
    def update(self, parameter : str, value : Union[float, int, np.ndarray]):
        """Update a parameter for the cruise condition"""
        parameters = self.parameters.__annotations__.keys()
        if parameter not in parameters:
            raise KeyError(f"Unkown parameter {parameter}. Acceptable cruise parameters are {parameters}")
        
        if isinstance(value, csdl.Variable):
            warnings.warn(f"Update parameter {parameter} with a csdl Variable; If this is a design variable, this might cause issues during optimization")

        if parameter == "mach_number":
            self.parameters.speed = None
            self.parameters.time = None
        elif parameter == "speed":
            self.parameters.mach_number = None
            self.parameters.time = None
        elif parameter == "time":
            self.parameters.range = None
            self.parameters.speed = None
        else:
            self.parameters.mach_number = None
            self.parameters.time = None

        setattr(self.parameters, parameter, value)
        self._setup_condition()


class ClimbCondition(AircraftCondition):
    """Climb condition (intended for steady analyses)
    
    Parameters
    ----------
    TODO
    """
    def __init__(self, 
                 initial_altitude : Union[float, int, csdl.Variable], 
                 final_altitude : Union[float, int, csdl.Variable],
                 pitch_angle : Union[float, int, csdl.Variable],
                 fligth_path_angle : Union[float, int, csdl.Variable, None],
                 speed : Union[float, int, csdl.Variable, None]=None,
                 mach_number : Union[float, int, csdl.Variable, None]=None,
                 time : Union[float, int, csdl.Variable, None]=None,
                 climb_gradient : Union[float, int, csdl.Variable, None]=None,
                 rate_of_climb : Union[float, int, csdl.Variable, None]=None,
                 atmos_model = SimpleAtmosphereModel()
                 ) -> None:
        if rate_of_climb is not None or climb_gradient is not None:
            raise NotImplementedError("Climb parameterization in terms of climb gradient or rate of climb not yet implemented.")

        self.parameters : ClimbParameters = ClimbParameters(
            initial_altitude=initial_altitude,
            final_altitude=final_altitude,
            pitch_angle=pitch_angle,
            speed=speed,
            mach_number=mach_number,
            flight_path_angle=fligth_path_angle,
            time=time,
            rate_of_climb=rate_of_climb,
            climb_gradient=climb_gradient,
        )
        self.quantities = ACQuantities()

        self._atmos_model = atmos_model

        # Check shapes
        self._check_shape_consistency(self.parameters)

        # Expand parameters to (num_nodes, )
        self._expand_variables("parameters")

        # Compute ac states
        self._setup_condition()

    def _setup_condition(self):
        # Different combinations of conflicting attributes
        conflicting_attributes_1 = ["speed", "mach_number"]
        conflicting_attributes_2 = ["speed", "time"]
        conflicting_attributes_3 = ["mach_number", "time"]

        # Check for conflicting attributes:
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_1]):
            raise Exception("Cannot specify 'mach_number' and 'speed' at the same time")
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_2]):
            raise Exception("Cannot specify 'speed' and 'time' at the same time")
        if all([attr if getattr(self.parameters, attr) is not None else False for attr in conflicting_attributes_3]):
            raise Exception("Cannot specify 'mach_number', 'time' at the same time")

        # zero aircraft states in climb
        x = y = v = phi = psi = p = q = r = csdl.Variable(shape=(self._num_nodes, ), value=0.)

        # Non-zero aircraft states in climb
        # Compute mean altitude
        hi = self.parameters.initial_altitude
        hf = self.parameters.final_altitude
        h_mean = z = 0.5 * (hi + hf)

        # Get pitch agle and flight path angle 
        theta = self.parameters.pitch_angle
        gamma = self.parameters.flight_path_angle

        # Compute atmospheric states for mean altitude
        atmos_states = self._atmos_model.evaluate(altitude=h_mean)

        # Compute speed from mach number or time or set V = speed
        mach_number = self.parameters.mach_number
        speed = self.parameters.speed
        time = self.parameters.time
        if mach_number is not None:
            V = mach_number * atmos_states.speed_of_sound
            self.parameters.speed = V
        elif speed is not None:
            V = speed
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
        else:
            w = (hf - hi) / time
            u = w / csdl.tan(gamma)
            V = (u**2 + w**2)**0.5
            mach_number = V / atmos_states.speed_of_sound
            self.parameters.mach_number = mach_number
            self.parameters.speed = V

        # Compute time spent in climb if time is None
        if time is None:
            h = ((hf - hi)**2)**0.5 # avoid getting a negative time 
            d = h / csdl.tan(gamma)
            time = (d**2 + h**2)**0.5 / V
            self.parameters.time = time

        # Compute a.o.a and compute u, w knowing the speed
        alfa = theta - gamma
        u = V * csdl.cos(alfa)
        w = V * csdl.sin(alfa)

        # Set aircraft and atmospheric states
        self.quantities.ac_states = AircaftStates(
            u=u, v=v, w=w, p=p, q=q, r=r, 
            phi=phi, theta=theta, psi=psi,
            x=x, y=y, z=z,
        )
        self.quantities.atmos_states = atmos_states

    def update(self, parameter : str, value : Union[float, int, np.ndarray]):
        """Update a parameter for the climb condition"""
        parameters = self.parameters.__annotations__.keys()
        if parameter not in parameters:
            raise KeyError(f"Unkown parameter {parameter}. Acceptable climb parameters are {parameters}")
        
        if isinstance(value, csdl.Variable):
            warnings.warn(f"Update parameter {parameter} with a csdl Variable; If this is a design variable, this might cause issues during optimization")

        if parameter == "mach_number":
            self.parameters.speed = None
            self.parameters.time = None
        elif parameter == "speed":
            self.parameters.mach_number = None
            self.parameters.time = None
        elif parameter == "time":
            self.parameters.mach_number = None
            self.parameters.speed = None
        else:
            self.parameters.mach_number = None
            self.parameters.time = None

        setattr(self.parameters, parameter, value)
        self._setup_condition()


class HoverCondition(AircraftCondition):
    """Hover condition (intended for steady analyses)
    
    Parameters
    ----------
    - altitude : Union[float, int, csdl.Variable]

    - time : Union[float, int, csdl.Variable]
    """
    
    def __init__(self, 
                 altitude : Union[float, int, csdl.Variable], 
                 time : Union[float, int, csdl.Variable],
                 atmos_model=SimpleAtmosphereModel()
                 ) -> None:

        self.parameters : HoverParameters = HoverParameters(
            altitude=altitude,
            time=time,
        )
        self.quantities = ACQuantities()
        self._atmos_model = atmos_model

        # Check shapes
        self._check_shape_consistency(self.parameters)

        # Expand parameters to (num_nodes, )
        self._expand_variables("parameters")

        # Compute ac states
        self._setup_condition()

    def _setup_condition(self):
        hover_parameters = self.parameters
        
        # All aircraft states except z will be zero
        u = v = w = p = q = r = phi = theta = psi = x = y = csdl.Variable(shape=(self._num_nodes, ), value=0.)
        z = hover_parameters.altitude

        # Evaluate atmospheric states
        atmos_states = self._atmos_model.evaluate(z)

        # Set aircraft and atmospheric states
        self.quantities.ac_states = AircaftStates(
            u=u, v=v, w=w, p=p, q=q, r=r,
            phi=phi, theta=theta, psi=psi,
            x=x, y=y, z=z
        )

        self.quantities.atmos_states = atmos_states

    def update(self, parameter: str, value: Union[float , int , np.ndarray , csdl.Variable]):
        """Update a parameter for the hover condition"""
        parameters = self.parameters.__annotations__.keys()
        if parameter not in parameters:
            raise KeyError(f"Unkown parameter {parameter}. Acceptable hover parameters are {parameters}")
        
        setattr(self.parameters, parameter, value)
        self._setup_condition()


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    # old_shape = (5, 3)
    # new_shape = (5, 10, 20, 3)
    # print(convert_shape_to_action_string(old_shape, new_shape, type_="vel_vec"))
    # exit()

    cruise = CruiseCondition(
        mach_number=0.2,
        altitude=3e3,
        pitch_angle=np.deg2rad(5),
        range=5e5,
    )


    print("Cruise altitude: ", cruise.parameters.altitude.value)
    print("Cruise time:     ", cruise.parameters.time.value)
    print("Cruise u:        ", cruise.quantities.ac_states.u.value)
    print("Cruise w:        ", cruise.quantities.ac_states.w.value)
    print("Cruise rho:      ", cruise.quantities.atmos_states.density.value)
    print("Cruise mu:       ", cruise.quantities.atmos_states.dynamic_viscosity.value)
    print("Cruise pressure: ", cruise.quantities.atmos_states.pressure.value)
    print("Cruise a:        ", cruise.quantities.atmos_states.speed_of_sound.value)
    print("Cruise temp:     ", cruise.quantities.atmos_states.temperature.value)
    # print(cruise.ac_states)

    # test_tensor = csdl.Variable(shape=(2, 4, 5, 3), value=np.random.randn(2, 4, 5, 3))

    # print(test_tensor[-1, 0].shape)

    # print(convert_shape_to_action_string(test_tensor.shape))
    # exit()
    
    print("\n")
    print("\n")
    
    
    print('Test aircraft condition')
    aircraft_condition = AircraftCondition(
        u=np.array([0., 10., 20., 30.]), v=0., w=5.,
        p=0., q=0., r=0., phi=0., theta=0., psi=0., x=0., y=0., 
        z= np.array([100, 200, 300, 400]))
    
    print(aircraft_condition.quantities.ac_states)
    print(aircraft_condition.quantities.atmos_states)
    print("\n")
    print("\n")
    
    print('Test climb condition')
    climb_condition = ClimbCondition(
        initial_altitude=1000, 
        final_altitude=2000,
        mach_number=0.2,
        pitch_angle=np.deg2rad(10),
        fligth_path_angle=np.deg2rad(5),
    )
    print(climb_condition.parameters)
    print(climb_condition.quantities.ac_states)
    print(climb_condition.quantities.atmos_states)

    climb_condition.update(parameter='final_altitude', value=500)

    print("\n")
    print("\n")
    print(climb_condition.parameters)
    print(climb_condition.quantities.ac_states)
    print(climb_condition.quantities.atmos_states)
    print("\n")
    print("\n")

    print('Test cruise condition')
    climb_condition = CruiseCondition(
        altitude=1000, 
        mach_number=0.2,
        speed=None,
        range=50000,
        pitch_angle=np.deg2rad(10),
    )

    print(climb_condition.parameters)
    print(climb_condition.quantities.ac_states)
    print(climb_condition.quantities.atmos_states)

    climb_condition.update(parameter='altitude', value=500)

    print("\n")
    print("\n")
    print(climb_condition.parameters)
    print(climb_condition.quantities.ac_states)
    print(climb_condition.quantities.atmos_states)

    climb_condition.update(parameter='mach_number', value=0.2)

    print("\n")
    print("\n")
    print(climb_condition.parameters)
    print(climb_condition.quantities.ac_states)
    print(climb_condition.quantities.atmos_states)

    print("\n")
    print("\n")
    print("test hover condition")
    hover_condition = HoverCondition(
        altitude=1000,
        time=100
    )

    print(hover_condition.parameters)
    print(hover_condition.quantities.ac_states)
    print(hover_condition.quantities.atmos_states)

    hover_condition.update(parameter='altitude', value=500)

    print("\n")
    print("\n")
    print(hover_condition.parameters)
    print(hover_condition.quantities.ac_states)
    print(hover_condition.quantities.atmos_states)