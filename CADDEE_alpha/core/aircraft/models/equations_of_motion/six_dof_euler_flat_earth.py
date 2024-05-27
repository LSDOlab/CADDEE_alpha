import csdl_alpha as csdl
import numpy as np
from CADDEE_alpha.core.aircraft.conditions.aircraft_condition import AircaftStates
from CADDEE_alpha.utils.var_groups import MassProperties
from typing import Union
from dataclasses import dataclass


@dataclass
class LinAngAccel(csdl.VariableGroup):
    du_dt: csdl.Variable
    dv_dt: csdl.Variable
    dw_dt: csdl.Variable
    dp_dt: csdl.Variable
    dq_dt: csdl.Variable
    dr_dt: csdl.Variable
    accel_norm: csdl.Variable


class SixDofEulerFlatEarthModel:
    def __init__(self, num_nodes: int = 1, stability_flag: bool = False):
        self.num_nodes = num_nodes
        self.stability_flag = stability_flag
        csdl.check_parameter(num_nodes, "num_nodes", types=int)
        csdl.check_parameter(stability_flag, "stability_flag", types=bool)

    def evaluate(self, 
        total_forces: csdl.Variable, 
        total_moments: csdl.Variable, 
        ac_states: AircaftStates,
        ac_mass_properties: MassProperties,
        ref_pt: Union[csdl.Variable, np.ndarray] = np.array([0., 0., 0.])
    ):
        
        # NOTE: this was rewritten based on Darshan's implementation from TC1
        # Decompose forces and moments into components
        Fx = total_forces[:, 0]
        Fy = total_forces[:, 1]
        Fz = total_forces[:, 2]
        Mx = total_moments[:, 0] 
        My = total_moments[:, 1] 
        Mz = total_moments[:, 2] 

        # Get mass, cg, I and decompose into components
        m = ac_mass_properties.mass
        cg_vector =  ac_mass_properties.cg_vector
        inertia_tensor = ac_mass_properties.inertia_tensor

        cgx = cg_vector[0]
        cgy = cg_vector[1]
        cgz = cg_vector[2]

        Ixx = inertia_tensor[0, 0]
        Iyy = inertia_tensor[1, 1]
        Izz = inertia_tensor[2, 2]
        Ixy = inertia_tensor[0, 1]
        Ixz = inertia_tensor[0, 2]
        Iyz = inertia_tensor[1, 2]

        # Get aircraft states
        u = ac_states.u
        v = ac_states.v
        w = ac_states.w
        p = ac_states.p
        q = ac_states.q
        r = ac_states.r
        phi = ac_states.phi
        theta = ac_states.theta
        psi = ac_states.psi
        x = ac_states.x
        y = ac_states.y
        z = ac_states.z

        Idot = csdl.Variable(shape=(3, 3), value=0.)

        # cg offset from reference point
        Rbcx = cgx - ref_pt[0]
        Rbcy = cgy - ref_pt[1]
        Rbcz = cgz - ref_pt[2]

        xcgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        ycgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        zcgdot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        xcgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        ycgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)
        zcgddot = csdl.Variable(shape=(self.num_nodes, ), value=0.)

        # fill in (6 x 6) mp matrix
        mp_matrix = csdl.Variable(shape=(6, 6), value=0)
        
        mp_matrix = mp_matrix.set(csdl.slice[0, 0], m)
        mp_matrix = mp_matrix.set(csdl.slice[0, 4], m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[0, 5], -m * Rbcy)

        mp_matrix = mp_matrix.set(csdl.slice[1, 1], m)
        mp_matrix = mp_matrix.set(csdl.slice[1, 3], -m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[1, 5], m * Rbcx)

        mp_matrix = mp_matrix.set(csdl.slice[2, 2], m)
        mp_matrix = mp_matrix.set(csdl.slice[2, 3], m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[2, 4], -m * Rbcx)

        mp_matrix = mp_matrix.set(csdl.slice[3, 1], -m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[3, 2], m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[3, 3], Ixx)
        mp_matrix = mp_matrix.set(csdl.slice[3, 4], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[3, 5], Ixz)

        mp_matrix = mp_matrix.set(csdl.slice[4, 0], m * Rbcz)
        mp_matrix = mp_matrix.set(csdl.slice[4, 2], -m * Rbcx)
        mp_matrix = mp_matrix.set(csdl.slice[4, 3], Ixy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 4], Iyy)
        mp_matrix = mp_matrix.set(csdl.slice[4, 5], Iyz)

        mp_matrix = mp_matrix.set(csdl.slice[5, 0], -m * Rbcy)
        mp_matrix = mp_matrix.set(csdl.slice[5, 1], m * Rbcx)
        mp_matrix = mp_matrix.set(csdl.slice[5, 3], Ixz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 4], Iyz)
        mp_matrix = mp_matrix.set(csdl.slice[5, 5], Izz)


        lambda_x = Fx + m * (r * v - q * w - xcgdot - 2 * q * zcgdot
                            + 2 * r * ycgdot + Rbcx * (q ** 2 + r ** 2)
                            - Rbcy * p * q - Rbcz * p * r)

        lambda_y = Fy + m * (p * w - r * u - ycgddot - 2 * r * xcgdot 
                            + 2 * p * zcgdot - Rbcx * p * q
                            + Rbcy * (p ** 2 + r ** 2) - Rbcz * q * r)

        lambda_z = Fz + m * (q * u - p * v - zcgddot - 2 * p * ycgdot 
                            + 2 * q * xcgdot - Rbcx * p * r 
                            - Rbcy * q * r + Rbcz * (p ** 2 + q ** 2))
        

        ang_vel_vec = csdl.Variable(shape=(self.num_nodes, 3), value=0.)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 0], p)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 1], q)
        ang_vel_vec = ang_vel_vec.set(csdl.slice[:, 2], r)


        angvel_ssym = csdl.Variable(shape=(self.num_nodes, 3, 3), value=0.)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 0, 1], -r)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 0, 2], q)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 1, 0], r)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 1, 2], -p)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 2, 0], -q)
        angvel_ssym = angvel_ssym.set(csdl.slice[:, 2, 1], p)


        Rbc_ssym = csdl.Variable(shape=(self.num_nodes, 3, 3), value=0.)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 0, 1], -Rbcz)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 0, 2], Rbcy)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 1, 0], Rbcz)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 1, 2], -Rbcx)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 2, 0], -Rbcy)
        Rbc_ssym = Rbc_ssym.set(csdl.slice[:, 2, 1], Rbcx)


        mu_vec = csdl.Variable(shape=(self.num_nodes, 3), value=0.)
        for i in csdl.frange(self.num_nodes):
            t1 = csdl.matvec(Idot, ang_vel_vec[i, :])
            
            var_1 = csdl.matmat(angvel_ssym[i, :, :], inertia_tensor)

            var_2 = csdl.matvec(var_1, ang_vel_vec[i, :])

            var_3 = csdl.matmat(angvel_ssym[i, :, :], Rbc_ssym[i, :, :])

            var_4 = csdl.matvec(var_3, ang_vel_vec[i, :])

            var_5 = m * var_4

            mu_vec = mu_vec.set(
                slices=csdl.slice[i, :],
                value=total_moments[i, :] - t1 - var_2 - var_5,
            )

        
        # Assemble the right hand side vector
        rhs = csdl.Variable(shape=(self.num_nodes, 6), value=0.)
        rhs = rhs.set(csdl.slice[:, 0], lambda_x)
        rhs = rhs.set(csdl.slice[:, 1], lambda_y)
        rhs = rhs.set(csdl.slice[:, 2], lambda_z)
        rhs = rhs.set(csdl.slice[:, 3], mu_vec[:, 0])
        rhs = rhs.set(csdl.slice[:, 4], mu_vec[:, 1])
        rhs = rhs.set(csdl.slice[:, 5], mu_vec[:, 2])

        # Initialize the state vector (acceleration) and the residual
        state = csdl.ImplicitVariable(shape=(6, self.num_nodes), value=0.)
        residual = mp_matrix @ state - rhs.T()

        solver = csdl.nonlinear_solvers.Newton(tolerance=1e-12)
        solver.add_state(state, residual)
        solver.run()

        lin_and_ang_accel = state.T()

        lin_and_ang_accel_output = LinAngAccel(
            du_dt=lin_and_ang_accel[:, 0],
            dv_dt=lin_and_ang_accel[:, 1],
            dw_dt=lin_and_ang_accel[:, 2],
            dp_dt=lin_and_ang_accel[:, 3],
            dq_dt=lin_and_ang_accel[:, 4],
            dr_dt=lin_and_ang_accel[:, 5],
            accel_norm=csdl.norm(
                lin_and_ang_accel,
                axes=(1, ),
            )
        )

        return lin_and_ang_accel_output


        

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_nodes = 5
    total_forces= csdl.Variable(shape=(num_nodes, 3), value=100.) 
    total_moments= csdl.Variable(shape=(num_nodes, 3), value=20)
    ac_states= AircaftStates(
        u=csdl.Variable(shape=(num_nodes, ), value=0),
        v=csdl.Variable(shape=(num_nodes, ), value=0),
        w=csdl.Variable(shape=(num_nodes, ), value=5),
        p=csdl.Variable(shape=(num_nodes, ), value=0),
        q=csdl.Variable(shape=(num_nodes, ), value=0),
        r=csdl.Variable(shape=(num_nodes, ), value=0),
        phi=csdl.Variable(shape=(num_nodes, ), value=np.deg2rad(5)),
        theta=csdl.Variable(shape=(num_nodes, ), value=np.deg2rad(10)),
        psi=csdl.Variable(shape=(num_nodes, ), value=0),
        x=csdl.Variable(shape=(num_nodes, ), value=0),
        y=csdl.Variable(shape=(num_nodes, ), value=0),
        z=csdl.Variable(shape=(num_nodes, ), value=0),
    )
    ac_mass_properties= MassProperties(
        mass=csdl.Variable(shape=(1, ), value=7126.1992),
        cg_vector=csdl.Variable(shape=(3, ), value=np.array([12.57675332, 0., 7.084392152])),
        inertia_tensor=csdl.Variable(shape=(3, 3), value=np.array([
            [4376.344208, 0, 213.8989507],
            [0., 2174.842852, 0],
            [213.8989507, 0, 6157.83761],
        ])
    ))

    eom_model = SixDofEulerFlatEarthModel(num_nodes=num_nodes)
    eom_model.evaluate(
        total_forces=total_forces,
        total_moments=total_moments,
        ac_states=ac_states,
        ac_mass_properties=ac_mass_properties,
    )

