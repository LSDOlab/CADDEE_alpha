from __future__ import annotations
from CADDEE_alpha.core.component import Component, VectorizedComponent
from CADDEE_alpha.core.mesh.mesh import MeshContainer, SolverMesh
from CADDEE_alpha.utils.copy_comps import copy_comps
from lsdo_function_spaces import FunctionSet
import numpy as np 
import csdl_alpha as csdl
import warnings
import copy
from typing import Union, List
import time


class VectorizedConfig:
    def __init__(self, config : Configuration, num_nodes : int) -> None:
        from CADDEE_alpha.core.mesh.mesh import VectorizedDiscretization

        self.system : VectorizedComponent = VectorizedComponent(config.system, num_nodes=num_nodes)
        new_mesh_container = config.mesh_container.copy()

        for mesh_name, mesh in new_mesh_container.items():
            mesh_copy = copy.copy(mesh)

            discretizations_copy = mesh_copy.discretizations.copy()

            for discr_name, discr in discretizations_copy.items():
                geom_list =  [comp.geometry for comp in self.system.comp_list]
                discr_copy = VectorizedDiscretization(discr, geom_list, num_nodes)
                discretizations_copy[discr_name] = discr_copy

            mesh_copy.discretizations = discretizations_copy

            new_mesh_container[mesh_name] = mesh_copy

        #     new_mesh_container[mesh_name] = copy.copy(mesh)
        #     for discretization_name, discretization in mesh.discretizations.items():
        #         discretization_copy = discretization.copy()
        #         new_mesh_container[mesh_name].discretizations[discretization_name] = discretization_copy
        #         new_discretization = new_mesh_container[mesh_name].discretizations[discretization_name]

        #         new_discretization.nodal_coordinates = [discretization_copy.nodal_coordinates for i in range(num_nodes)]
        #         new_discretization.nodal_velocities = [discretization_copy.nodal_velocities for i in range(num_nodes)]
        #         new_discretization.mesh_quality = [discretization_copy.mesh_quality for i in range(num_nodes)]
        #         new_discretization._has_been_expanded = [discretization_copy._has_been_expanded for i in range(num_nodes)]
        
        self.mesh_container = new_mesh_container


    def assemble_system_mass_properties(
            self, 
            point : np.ndarray = np.array([0., 0., 0.]),
            update_copies: bool = False
        ):
        """Compute the mass properties of the configuration.
        """
        csdl.check_parameter(update_copies, "update_copies", types=bool)

        # TODO: how to handle case if children components and parent components both have mass properties:
        #   Ex: if user assigns airframe mass and then also masses for airframe components
        system = self.system
        system_comps = system.comps

        # TODO: allow for parallel axis theorem
        if not np.array_equal(point, np.array([0. ,0., 0.])):
            raise NotImplementedError("Mass properties taken w.r.t. a specific point not yet implemented.")

        def _add_component_mps_to_system_mps(
                comps, 
                system_mass=0., 
                system_cg=np.zeros((3, )), 
                system_inertia_tensor=np.zeros((3, 3))
            ):
            """Add component-level mass properties to the system-level mass properties. 
            Only called internally by 'assemble_system_mass_properties'"""
            # first, compute total mass
            for comp_name, comp in comps.items():
                print("mass", comp.quantities.mass_properties.mass)
                print("cg_vector", comp.quantities.mass_properties.cg_vector)
                print("\n")
                mass_props = comp.quantities.mass_properties
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")
                    system_mass = system_mass * 1

                # otherwise add the mass properties
                else:
                    m = mass_props.mass
                    if isinstance(m, list):
                        raise NotImplementedError("vectorized mps not implemented yet")
                        if all(mass is None for mass in m):
                            m = None
                        else:
                            exit()
                    
                    if m is not None:
                        system_mass = system_mass + m
                    
            # second, compute total cg
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_cg = system_cg * 1

                # Otherwise add component cg contribution
                else:
                    cg = mass_props.cg_vector
                    m = mass_props.mass
                    if isinstance(cg, list):
                        raise NotImplementedError("vectorized mps not implemented yet")
                        if all(cg_vec is None for cg_vec in cg):
                            cg = None
                        else:
                            exit()
                    if cg is not None:
                        system_cg = system_cg + m * cg / system_mass

            # Third, compute total cg and inertia tensor 
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_inertia_tensor = system_inertia_tensor * 1
                
                else:
                    it = mass_props.inertia_tensor
                    if isinstance(it, list):
                        raise NotImplementedError("vectorized mps not implemented yet")
                        if all(it_tens is None for it_tens in it):
                            it = None
                        else:
                            exit()
                    
                    m = mass_props.mass
                    x = system_cg[0]
                    y = system_cg[1]
                    z = system_cg[2]

                    if it is not None:
                        if m is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no mass. Cannot apply parallel axis theorem.")
                        
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = it[0, 0] + m * (y**2 + z**2)
                        ixy = it[0, 1] - m * (x * y)
                        ixz = it[0, 2] - m * (x * z)
                        iyx = ixy 
                        iyy = it[1, 1] + m * (x**2 + z**2)
                        iyz = it[1, 2] - m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = it[2, 2] + m * (x**2  + y**2)

                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
                        
                        system_inertia_tensor = system_inertia_tensor + it

                    elif m is not None:
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        
                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
                        system_inertia_tensor = system_inertia_tensor + it
                    
                    else:
                        pass


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
        if all(getattr(system_mps, mp) is not None for mp in system_mps.__dict__):
            # Check if the system is a copy
            if system._is_copy:
                system_mass, system_cg, system_inertia_tensor = \
                _add_component_mps_to_system_mps(system_comps)

                system.quantities.mass_properties.mass = system_mass
                system.quantities.mass_properties.cg_vector = system_cg
                system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

            else:
                warnings.warn(f"System already has defined mass properties: {system_mps}")
                return

        # 2) mass and cg have been defined and inertia tensor is None
        elif system_mps.mass is not None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            system_inertia_tensor = np.zeros((3, 3))
            warnings.warn(f"System already has defined mass and cg vector; will compute inertia tensor based on point mass assumption")
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


class Configuration:
    """The configurations class"""
    def __init__(self, system : Component) -> None:
        # Check whether system if a component
        if not isinstance(system, Component):
            raise TypeError(f"system must by of type {Component}")
        # Check that if system geometry is not None, it is of the correct type
        if system.geometry is not None:
            if not isinstance(system.geometry, FunctionSet ):
                raise TypeError(f"If system geometry is not None, it must be of type '{FunctionSet}', received object of type '{type(system.geometry)}'")
        self.system = system
        self.mesh_container : MeshContainer = MeshContainer()

        self._is_copy : bool = False
        self._geometry_setup_has_been_called = False
        self._geometric_connections = []

        self._config_copies: List[self] = []

    def visualize_component_hierarchy(
            self, 
            file_name: str="component_hierarchy", 
            file_format: str="png",
            show: bool=False,
        ):
        csdl.check_parameter(file_name, "file_name", types=str)
        csdl.check_parameter(file_format, "file_format", values=("png", "pdf"))
        csdl.check_parameter(show, "show", types=bool)
        try:
            from graphviz import Graph
        except ImportError:
            raise ImportError("Must install graphviz. Can do 'pip install graphviz'")
        
        # make graph object
        graph = Graph(comment="Compopnent Hierarchy")

        # Go through component hierarchy and components to nodes
        def add_component_to_graph(comp: Component, comp_name: str, parent_name=None):
            graph.node(comp_name, comp_name)
            if parent_name is not None:
                graph.edge(parent_name, comp_name)
            for child_name, child in comp.comps.items():
                add_component_to_graph(child, child_name, comp_name)

        add_component_to_graph(self.system, "system")
        graph.render("component_hierarchy", format=file_format, view=show)

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

    def vectorized_copy(self, num_nodes : int) -> VectorizedConfig:
        csdl.check_parameter(num_nodes, "num_nodes", types=int)

        if num_nodes <= 1:
            raise ValueError("'num_nodes' must be an integer greater than 1")

        vectorized_config = VectorizedConfig(config=self, num_nodes=num_nodes)

        self._config_copies.append(vectorized_config)
        
        return vectorized_config

    def copy(self) -> Configuration:
        """Copy a configuration."""

        system_copy = copy_comps(self.system)

        # Copy the mesh containers
        # 1) copy the dictionary of meshes
        mesh_container_copy = self.mesh_container.copy()

        # 2) copy the meshes themselves
        for mesh_name, mesh in mesh_container_copy.items():
            mesh_copy = copy.copy(mesh)
            
            # 2a) copy the discretizations dict
            discretizations_copy = mesh_copy.discretizations.copy()
            
            # 2b) loop over and copy the individual discretizations
            for discr_name, discr in discretizations_copy.items():
                discr_copy = copy.copy(discr)
                discr_copy._geom = system_copy.geometry
                discretizations_copy[discr_name] = discr_copy

            # 2c) assign the copied discretizations to the mesh copy
            mesh_copy.discretizations = discretizations_copy

            # 2d) updated the copied mesh
            mesh_container_copy[mesh_name] = mesh_copy

        # Make a new instance of a Configuration
        copied_config = Configuration(system_copy)
        copied_config.mesh_container = mesh_container_copy

        self._config_copies.append(copied_config)

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

    def assemble_system_mass_properties(
            self, 
            point : np.ndarray = np.array([0., 0., 0.]),
            update_copies: bool = False
        ):
        """Compute the mass properties of the configuration.
        """
        csdl.check_parameter(update_copies, "update_copies", types=bool)

        # TODO: how to handle case if children components and parent components both have mass properties:
        #   Ex: if user assigns airframe mass and then also masses for airframe components
        system = self.system
        system_comps = system.comps

        # TODO: allow for parallel axis theorem
        if not np.array_equal(point, np.array([0. ,0., 0.])):
            raise NotImplementedError("Mass properties taken w.r.t. a specific point not yet implemented.")

        def _sum_component_masses(
            comps,
            system_mass=0.,
        ):
            """Sum all component mass to compute system mass.

            Parameters
            ----------
            comps : _type_
                dictionary of children components
            system_mass : _type_, optional
                Initial system mass, by default 0.
            """
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")
                    system_mass = system_mass * 1

                # Add component mass to system mass
                else:
                    m = mass_props.mass
                    if m is not None:
                        system_mass = system_mass + m

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_mass = \
                        _sum_component_masses(comp.comps, system_mass)

            return system_mass
        
        def _sum_component_cgs(
            comps,
            system_mass,
            system_cg=np.zeros((3, ))
        ):
            # second, compute total cg 
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_cg = system_cg * 1

                # otherwise add component cg contribution
                else:
                    cg = mass_props.cg_vector
                    m = mass_props.mass
                    if cg is not None:
                        system_cg = system_cg + m * cg / system_mass

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_cg = \
                        _sum_component_cgs(comp.comps, system_mass, system_cg)

            return system_cg
        
        def _sum_component_inertias(
            comps,
            system_cg,
            system_inertia_tensor=np.zeros((3, 3)),
        ):
            x = system_cg[0]
            y = system_cg[1]
            z = system_cg[2]

            # Third, compute total cg and inertia tensor
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_inertia_tensor = system_inertia_tensor * 1

                else:
                    it = mass_props.inertia_tensor
                    m = mass_props.mass

                    # use given inertia if provided
                    if it is not None:
                        if m is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no mass. Cannot apply parallel axis theorem.")
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = it[0, 0] + m * (y**2 + z**2)
                        ixy = it[0, 1] - m * (x * y)
                        ixz = it[0, 2] - m * (x * z)
                        iyx = ixy 
                        iyy = it[1, 1] + m * (x**2 + z**2)
                        iyz = it[1, 2] - m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = it[2, 2] + m * (x**2  + y**2)

                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)

                        system_inertia_tensor = system_inertia_tensor + it
                    
                    # point mass assumption
                    elif m is not None: 
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        
                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
                        system_inertia_tensor = system_inertia_tensor + it

                # Check if the component has children
                if not comp.comps:
                    pass

                # If comp has children, add their mass recursively 
                else:
                    system_inertia_tensor = \
                        _sum_component_inertias(comp.comps, system_cg, system_inertia_tensor)

            return system_inertia_tensor

        def _add_component_mps_to_system_mps(
                comps, 
                system_mass=0., 
                system_cg=np.zeros((3, )), 
                system_inertia_tensor=np.zeros((3, 3))
            ):
            """Add component-level mass properties to the system-level mass properties. 
            Only called internally by 'assemble_system_mass_properties'"""
            # first, compute total mass
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                # Check if mass_properties have been set/computed
                if mass_props is None:
                    warnings.warn(f"Component {comp} has no mass properties")

                    system_mass = system_mass * 1

                else:
                    m = mass_props.mass
                    
                    if m is not None:
                        system_mass = system_mass + m
        
            # second, compute total cg 
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_cg = system_cg * 1
                    system_inertia_tensor = system_inertia_tensor * 1

                # otherwise add component cg contribution
                else:
                    cg = mass_props.cg_vector
                    m = mass_props.mass
                    if cg is not None:
                        system_cg = system_cg + m * cg / system_mass
           
            # Third, compute total cg and inertia tensor
            for comp_name, comp in comps.items():
                mass_props = comp.quantities.mass_properties
                if mass_props is None:
                    system_inertia_tensor = system_inertia_tensor * 1

                else:
                    it = mass_props.inertia_tensor
                    m = mass_props.mass

                    x = system_cg[0]
                    y = system_cg[1]
                    z = system_cg[2]
                    
                    # use given mps
                    if it is not None:
                        if m is None:
                            raise Exception(f"Component {comp_name}, has an inertia tensor but no mass. Cannot apply parallel axis theorem.")
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = it[0, 0] + m * (y**2 + z**2)
                        ixy = it[0, 1] - m * (x * y)
                        ixz = it[0, 2] - m * (x * z)
                        iyx = ixy 
                        iyy = it[1, 1] + m * (x**2 + z**2)
                        iyz = it[1, 2] - m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = it[2, 2] + m * (x**2  + y**2)

                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)

                        system_inertia_tensor = system_inertia_tensor + it
                    
                    # point mass assumption
                    elif m is not None: 
                        # Apply parallel axis theorem to get inertias w.r.t to global cg
                        ixx = m * (y**2 + z**2)
                        ixy = -m * (x * y)
                        ixz = -m * (x * z)
                        iyx = ixy 
                        iyy = m * (x**2 + z**2)
                        iyz = -m * (y * z)
                        izx = ixz 
                        izy = iyz 
                        izz = m * (x**2  + y**2)
                        
                        it = csdl.Variable(shape=(3, 3), value=0.)
                        it = it.set(csdl.slice[0, 0], ixx)
                        it = it.set(csdl.slice[0, 1], ixy)
                        it = it.set(csdl.slice[0, 2], ixz)
                        it = it.set(csdl.slice[1, 0], iyx)
                        it = it.set(csdl.slice[1, 1], iyy)
                        it = it.set(csdl.slice[1, 2], iyz)
                        it = it.set(csdl.slice[2, 0], izx)
                        it = it.set(csdl.slice[2, 1], izy)
                        it = it.set(csdl.slice[2, 2], izz)
                        
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
        if all(getattr(system_mps, mp) is not None for mp in system_mps.__dict__):
            # Check if the system is a copy
            if system._is_copy:
                system_mass = _sum_component_masses(system_comps)
                system_cg = _sum_component_cgs(system_comps, system_mass=system_mass)
                system_inertia_tensor = _sum_component_inertias(system_comps, system_cg=system_cg)
                
                # system_mass, system_cg, system_inertia_tensor = \
                # _add_component_mps_to_system_mps(system_comps)

                system.quantities.mass_properties.mass = system_mass
                system.quantities.mass_properties.cg_vector = system_cg
                system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

            else:
                warnings.warn(f"System already has defined mass properties: {system_mps}")
                return

        # 2) mass and cg have been defined and inertia tensor is None
        elif system_mps.mass is not None and system_mps.cg_vector is not None and system_mps.inertia_tensor is None:
            system_inertia_tensor = np.zeros((3, 3))
            warnings.warn(f"System already has defined mass and cg vector; will compute inertia tensor based on point mass assumption")
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

            system_mass = _sum_component_masses(system_comps,system_mass=system_mass)
            system_cg = _sum_component_cgs(system_comps, system_mass=system_mass, system_cg=system_cg)
            system_inertia_tensor = _sum_component_inertias(system_comps, system_cg=system_cg, system_inertia_tensor=system_inertia_tensor)
    
            # system_mass, system_cg, system_inertia_tensor = \
            #     _add_component_mps_to_system_mps(system_comps, system_mass, system_cg, system_inertia_tensor)

            system.quantities.mass_properties.mass = system_mass
            system.quantities.mass_properties.cg_vector = system_cg
            system.quantities.mass_properties.inertia_tensor = system_inertia_tensor

        from CADDEE_alpha.core.component import VectorizedAttributes, VectorizedComponent
        # Update mass properties for any copied configurations
        if update_copies:

            def _update_comp_copy_mps(component_copy: Component, original_component: Component):
                # Get original mass properties
                original_mps = original_component.quantities.mass_properties

                # Copy original mass properties and set them as new ones
                component_copy.quantities.mass_properties = copy.copy(original_mps)

                # print(isinstance(component_copy, VectorizedComponent))
                # mass = component_copy.quantities.mass_properties.mass
                # if isinstance(mass, csdl.Variable):
                #     print(component_copy._name, mass.value)
                # elif isinstance(mass, VectorizedAttributes):
                #     mass_list = mass.attribute_list
                #     print(component_copy._name, [val.value for val in mass_list])
                # else:
                #     print(component_copy._name, mass)

                # Repeat if component has children
                for child_name, child_copy in component_copy.comps.items():
                    if child_name in original_component.comps.keys():
                        original_child = original_component.comps[child_name]
                        _update_comp_copy_mps(child_copy, original_child)

            def _update_config_copy_mps(config_copy: self):
                # print("Config copy type", type(config_copy))
                config_copy_system = config_copy.system
                original_system = system
                _update_comp_copy_mps(config_copy_system, original_system)


            for config_copy in self._config_copies:
                _update_config_copy_mps(config_copy)

                # _print_existing_mps(config_copy.system)

    def connect_component_geometries(
        self,
        comp_1: Component,
        comp_2: Component,
        connection_point: Union[csdl.Variable, np.ndarray, None]=None,
    ):
        """Connect the geometries of two components.

        Function to ensure a component geometries can rigidly 
        translate if the component it is connected to moves.

        Parameters
        ----------
        comp_1 : Component
            the first component to be connected
        comp_2 : Component
            the second component to be connected
        connection_point : Union[csdl.Variable, np.ndarray, None], optional
            the point with respect to which the connection is defined. E.g., 
            if the wing and fuselage geometries are connected, this point 
            could be the quarter chord of the wing. This means that the distance
            between the point and the two component will remain constant, 
            by default None

        Raises
        ------
        Exception
            If 'comp_1' is not an instances of Compone
        Exception
            If 'comp_2' is not an instances of Compone
        Exception
            If 'connection_point' is not of shape (3, )
        """
        csdl.check_parameter(comp_1, "comp_1", types=Component)
        csdl.check_parameter(comp_2, "comp_2", types=Component)
        csdl.check_parameter(connection_point, "connection_point" ,
                             types=(csdl.Variable, np.ndarray), allow_none=True)

        # Check that comp_1 and comp_2 have geometries
        if comp_1.geometry is None:
            raise Exception(f"Comp {comp_1.name} does not have a geometry.")
        if comp_2.geometry is None:
            raise Exception(f"Comp {comp_2.name} does not have a geometry.")
        
        # If connection point provided, check that its shape is (3, )
        if connection_point is not None:
            try:
                connection_point.reshape((3, ))
            except:
                raise Exception(f"'connection_point' must be of shape (3, ) or reshapable to (3, ). Received shape {connection_point.shape}")

            projection_1 = comp_1.geometry.project(connection_point)
            projection_2 = comp_2.geometry.project(connection_point)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))
        
        # Else choose the center points of the FFD block
        else:
            point_1 = comp_1._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))
            point_2 = comp_2._ffd_block.evaluate(parametric_coordinates=np.array([0.5, 0.5, 0.5]))

            projection_1 = comp_1.geometry.project(point_1)
            projection_2 = comp_2.geometry.project(point_2)

            self._geometric_connections.append((projection_1, projection_2, comp_1, comp_2))
        
        return

    def setup_geometry(self, run_ffd : bool=True, plot : bool=False, recorder: csdl.Recorder =None):
        """Run the geometry parameterization solver. 
        
        Note: This is only allowed on the based configuration.
        """
        self._geometry_setup_has_been_called = True

        if self._is_copy and run_ffd:
            raise Exception("With curent version of CADDEE, Cannot call setup_geometry with run_ffd=True on a copy of a configuration.")
        
        from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
        parameterization_solver = ParameterizationSolver()
        ffd_geometric_variables = GeometricVariables()
        system_geometry = self.system.geometry

        if system_geometry is None:
            raise TypeError("'setup_geometry' cannot be called because the geometry asssociated with the system component is None")

        if not isinstance(system_geometry, FunctionSet):
            raise TypeError(f"The geometry of the system must be of type {FunctionSet}. Received {type(system_geometry)}")

        def setup_geometries(component: Component):
            # If component has a geometry, set up its geometry
            if component.geometry is not None:
                if component._skip_ffd is True:
                    pass
                else:
                    try: # NOTE: might cause some issues because try/except might hide some errors that shouldn't be hidden
                        component._setup_geometry(parameterization_solver, ffd_geometric_variables, plot=plot)

                    except NotImplementedError:
                        warnings.warn(f"'_setup_geometry' has not been implemented for component {component._name} of {type(component)}")
            
            # If component has children, set up their geometries
            if component.comps:
                for comp_name, comp in component.comps.items():
                    setup_geometries(comp)

            return
    
        setup_geometries(self.system)

        for connection in self._geometric_connections:
            projection_1 = connection[0]
            projection_2 = connection[1]
            comp_1 : Component = connection[2]
            comp_2 : Component = connection[3]
            if isinstance(projection_1, list):
                connection = comp_1.geometry.evaluate(parametric_coordinates=projection_1) - comp_2.geometry.evaluate(parametric_coordinates=projection_2)
            elif isinstance(projection_1, np.ndarray):
                connection = comp_1._ffd_block.evaluate(parametric_coordinates=projection_1) - comp_2._ffd_block.evaluate(parametric_coordinates=projection_2)
            else:
                print(f"wrong type {type(projection_1)} for projection")
                raise NotImplementedError
            
            ffd_geometric_variables.add_variable(connection, connection.value)

        # Evalauate the parameterization solver
        t1 = time.time()
        if recorder is not None:
            print("Setting 'inline' to false for inner optimization")
            recorder.inline = False
        parameterization_solver.evaluate(ffd_geometric_variables)
        t2 = time.time()
        print("time for inner optimization", t2-t1)

        print("update meshes after inner optimization")
        for mesh_name, mesh in self.mesh_container.items():
            for discretization_name, discretization in mesh.discretizations.items():
                discretization._update()
        
        if plot:
            system_geometry.plot(show=True)

        # TODO: re-evaluate meshes
