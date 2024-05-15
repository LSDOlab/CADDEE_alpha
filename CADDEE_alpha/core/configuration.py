from __future__ import annotations
from CADDEE_alpha.core.component import Component
from CADDEE_alpha.core.mesh.mesh import MeshContainer
from lsdo_geo import BSplineSet
import numpy as np 
import csdl_alpha as csdl
import warnings
import copy 


class Configuration:
    """The configurations class"""
    def __init__(self, system : Component) -> None:
        # Check whether system if a component
        if not isinstance(system, Component):
            raise TypeError(f"system must by of type {Component}")
        # Check that if system geometry is not None, it is of the correct type
        if not isinstance(system.geometry, BSplineSet ) and system.geometry is not None:
            raise TypeError(f"If system geometry is not None, it must be of type '{BSplineSet}', received object of type '{type(system.geometry)}'")
        self.system = system
        self.mesh_container : MeshContainer = MeshContainer()

        self._is_copy : bool = False

    def assemble_meshes(self):
        """Assemble all component meshes into the mesh container."""
        def add_mesh_to_container(comp: Component):
            for mesh_name, mesh in comp._discretizations.items():
                self.mesh_container[mesh_name] = mesh
            if comp.comps:
                for sub_comp_name, sub_comp in comp.comps.items():
                    add_mesh_to_container(sub_comp)
        
        for comp_name, comp in self.system.comps.items():
            add_mesh_to_container(comp)
            

    def copy(self) -> Configuration:
        """Copy a configuration."""
        def copy_comps(comp : Component):
            """Create copies of components and their attributes.
            """
            # 1) Create a shallow copy of the component itself
            comp_copy = copy.copy(comp)
            comp_copy._is_copy = True

            # 2) Create shallow copy of the comp's geometry
            geometry_copy = copy.copy(comp.geometry)
            comp_copy.geometry = geometry_copy

            # 3) Create shallow copy of the comp's children
            children_comps_copy = comp.comps.copy() # copy.copy(comp.comps) #
            comp_copy.comps = children_comps_copy

            # 4) Create shallow copy of the comp's quantities
            quantities_copy = copy.copy(comp.quantities)
            comp_copy.quantities = quantities_copy

            # 5) Create shallow copy of the comp's parameters
            parameters_copy = copy.copy(comp.parameters)
            comp_copy.parameters = parameters_copy

            # 6) TODO: meshes

            # 7) Recursively copy children of comp
            for child_comp_copy_name, child_comp_copy in children_comps_copy.items():
                child_comp_copy_copy = copy_comps(child_comp_copy)
                
                # Set the value of the comps dictionary
                children_comps_copy[child_comp_copy_name] = child_comp_copy_copy
                
            return comp_copy

        system_copy = copy_comps(self.system)
        copied_config = Configuration(system_copy)
        
        return copied_config
    
    def remove_component(self, comp : Component):
        """Remove a component from a configuration."""
        
        # Check that comp is the right type
        if not isinstance(comp, Component):
            raise TypeError(f"Can only remove components. Receieved type {type(comp)}")
        
        # Check if comp is the system itself
        if comp == self.system:
            raise Exception("Cannot remove system component.")
        
        def remove_comp_from_dictionary(comp, comp_dict)-> bool:
            """
            Remove a component from a dictionary.

            Arguments
            ---------
            - comp : original component to be removed

            - comp_dict : component dictionary (will be changed if comp not in comp_dict)

            """
            for sub_comp_name, sub_comp in comp_dict.items():
                if comp == sub_comp:
                    # del comp_dict[sub_comp_name]
                    comp_dict.pop(sub_comp_name)
                    return True
                else:
                    if remove_comp_from_dictionary(comp, sub_comp.comps):
                        return True
            return False
        
        comp_exists = remove_comp_from_dictionary(comp, self.system.comps)

        if comp_exists:
            return
        else:
            raise Exception(f"Cannot remove component {comp._name} of type {type(comp)} from the configuration since it does not exists.")

    def assemble_system_mass_properties(self, point : np.ndarray=np.array([0., 0., 0.])):
        """Compute the mass properties of the configuration.
        """
        # TODO: how to handle case if children components and parent components both have mass properties:
        #   Ex: if user assigns airframe mass and then also masses for airframe components
        system = self.system
        system_comps = system.comps

         # TODO: allow for parallel axis theorem
        if not np.array_equal(point, np.array([0. ,0., 0.])):
            raise NotImplementedError

        def _add_component_mps_to_system_mps(
                comps, 
                system_mass=0., 
                system_cg=np.zeros((3, )), 
                system_inertia_tensor=np.zeros((3, 3))
            ):
            """Add component-level mass properties to the system-level mass properties. 
            Only called internally by 'assemble_system_mass_properties'"""
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")

                    system_mass = system_mass * 1
                    system_cg = system_cg * 1
                    system_inertia_tensor = system_inertia_tensor * 1

                # otherwise add the mass properties
                else:
                    m = mass_props.mass
                    cg = mass_props.cg_vector
                    it = mass_props.inertia_tensor

                    if m is not None:
                        system_mass = system_mass + m
                    if cg is not None:
                        system_cg = (system_cg * system_mass + m * cg) / (system_mass +  m)
                    if it is not None:
                        system_inertia_tensor = system_inertia_tensor + it
                    elif m is not None and cg is not None:
                        x = cg[0]
                        y = cg[1]
                        z = cg[2]
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        it = np.array([
                            [ixx, ixy, ixz],
                            [iyx, iyy, iyz],
                            [izx, izy, izz],
                        ])
                        system_inertia_tensor = system_inertia_tensor + it
                
                # Check if the component has children
                if not comp.comps:
                    pass

                # If their children, add their mass properties via a private method
                else:
                    system_mass, system_cg, system_inertia_tensor = \
                        _add_component_mps_to_system_mps(comp.comps, system_mass, system_cg, system_inertia_tensor)

            return system_mass, system_cg, system_inertia_tensor

        # Check if mass properties of system has already been set/computed
        # 1) masss, cg, and inertia tensor have all been defined
        system_mps = system.quantities.mass_properties
        if all(getattr(system_mps, mp) is not None for mp in system_mps.__annotations__):
            # Check if the system is a copy
            if system._is_copy:
                system_mass, system_cg, system_inertia_tensor = \
                _add_component_mps_to_system_mps(system_comps)

                system.quantities.mass_properties.mass = system_mass
                system.quantities.mass_properties.cg_vector = system_cg
                system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

            else:
                warnings.warn(f"System already has define mass properties: {system_mps}")
                return

        # 2) mass and cg have been defined and inertia tensor is None
        elif system_mps.mass is not None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            system_inertia_tensor = np.zeros((3, 3))
            warnings.warn(f"System already has defined mass and cg vectotr; will compute inertia tensor based on point mass assumption")
            x = system_mps.cg_vector[0]
            y = system_mps.cg_vector[1]
            z = system_mps.cg_vector[2]
            m = system_mps.mass
            ixx = m * (y**2 + z**2)
            ixy = -m * (x * y)
            ixz = -m * (x * z)
            iyx = ixy 
            iyy = m * (x**2 + z**2)
            iyz = -m * (y * z)
            izx = ixz 
            izy = iyz 
            izz = m * (x**2  + y**2)
            system_mps.inertia_tensor = np.array([
                [ixx, ixy, ixz],
                [iyx, iyy, iyz],
                [izx, izy, izz],
            ])
            return 
        
        # 3) only mass has been defined
        elif system_mps.mass is not None and system_mps.cg_vector is None and system_mps.inertia_tensor is None:
            raise Exception("Partially defined system mass properties; only system mass has been set. Need at least mass and the cg vector.")
        
        # 4) only cg vector has been defined
        elif system_mps.mass is None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            raise Exception("Partially defined system mass properties; only system cg_vector has been set. Need at least mass and the cg vector.")

        # 5) only inertia tensor vector has been defined
        elif system_mps.mass is None and system_mps.cg_vector is None and system_mps.inertia_tensor is not None:
            raise Exception("Partially defined system mass properties; only system inertia_tensor has been set. Need to also specify mass and cg vector.")

        # 6) Inertia tensor is not None and cg vector is not None
        elif system_mps.mass is None and system_mps.cg_vector is not None and system_mps.inertia_tensor is not None:
            raise Exception("Partially defined system mass properties; only system cg_vector and inertia_tensor has been set. Mass has not been set.")

        else:
            # check if system has any components
            if not system_comps:
                raise Exception("System does not have any subcomponents and does not have any mass properties. Cannot assemble mass properties.")
            
            # loop over all components and sum the mass properties
            system_mass = 0
            system_cg = np.array([0., 0., 0.])
            system_inertia_tensor = np.zeros((3, 3))
    
            system_mass, system_cg, system_inertia_tensor = \
                _add_component_mps_to_system_mps(system_comps, system_mass, system_cg, system_inertia_tensor)

            system.quantities.mass_properties.mass = system_mass
            system.quantities.mass_properties.cg_vector = system_cg
            system.quantities.mass_properties.inertia_tensor = system_inertia_tensor


    def setup_geometry(self, run_ffd : bool=True, plot : bool=False):
        """Run the geometry parameterization solver. 
        
        Note: This is only allowed on the based configuration.
        """
        if self._is_copy and run_ffd:
            raise Exception("Cannot call setup_geometry with run_ffd=True on a copy of a configuration")
        
        from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
        parameterization_solver = ParameterizationSolver()
        system_geometry = self.system.geometry

        if system_geometry is None:
            raise TypeError("'setup_geometry' cannot be called because the geometry asssociated with the system component is None")

        if not isinstance(system_geometry, BSplineSet):
            raise TypeError(f"The geometry of the system must be of type {BSplineSet}")

        parameterization_inputs = {}
        def setup_geometries(component : Component):
            if component.geometry is not None:
                try: # NOTE: this might be potentially dangerous because it can obscure errors in lower-level implementations
                    parameterization_inputs.update(
                        component._setup_geometry(parameterization_solver, system_geometry, plot=plot)
                    )
                except:
                    warnings.warn(f"'_setup_geometry' has not been implemented for component {type(component)}")

            if component.comps is not None:
                for comp_name, comp in component.comps.items():
                    setup_geometries(comp)
            return parameterization_inputs

        parameterization_inputs = setup_geometries(self.system)
        
        if run_ffd:
            if not parameterization_inputs:
                raise Exception(f"Cannot run FFD since there are no parameterization inputs. Make sure that your system component knows about all its children component" + \
                                " by calling system.comps['child_comp'] = child_comp. Your system component is {self.system} and has children components {self.system.comps}.")

            parameterization_solver_states = parameterization_solver.evaluate(parameterization_inputs)

            def re_assign_coefficients(component : Component):
                if component.geometry is not None:
                    if hasattr(component, "_update_dict"):
                        component._update_coefficients_after_inner_optimization(
                            parameterization_solver_states, system_geometry, component._update_dict
                        )

                if component.comps:
                    for comp_name, comp in component.comps.items():
                        re_assign_coefficients(comp)

            re_assign_coefficients(self.system)

            system_geometry.plot()

        else:
            system_geometry.plot()
