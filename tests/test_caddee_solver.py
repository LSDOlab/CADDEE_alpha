import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
from CADDEE_alpha.utils.var_groups import AircaftStates, MassProperties
import pytest


@pytest.fixture(scope="class")
def setup_test_class():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Define L+C parameters
    AR = 12.5
    S_ref = 19.5
    m_bat = 800.
    V_cr = 66.
    fuselage_length = 9.5

    S_ref_h_tail = 3.7
    AR_h_tail = 4.3
    S_ref_v_tail = 2.54
    AR_v_tail = 1.17


    # Make components
    wing = cd.aircraft.components.Wing(AR=AR, S_ref=S_ref)
    h_tail = cd.aircraft.components.Wing(AR=AR_h_tail, S_ref=S_ref_h_tail)
    v_tail = cd.aircraft.components.Wing(AR=AR_v_tail, S_ref=S_ref_v_tail)
    fuselage = cd.aircraft.components.Fuselage(length=fuselage_length)
    booms = cd.Component()

    return {
        "wing" : wing,
        "h_tail": h_tail,
        "v_tail": v_tail,
        "fuselage": fuselage,
        "booms": booms,
        "m_bat": m_bat,
        "V_cr": V_cr,
    }

@pytest.mark.usefixtures("setup_test_class")
class TestCADDEESolvers:
    @pytest.fixture(autouse=True)
    def _setup(self, setup_test_class):
        self.wing = setup_test_class["wing"]
        self.h_tail = setup_test_class["h_tail"]
        self.v_tail = setup_test_class["v_tail"]
        self.fuselage = setup_test_class["fuselage"]
        self.booms = setup_test_class["booms"]
        self.m_bat = setup_test_class["m_bat"]
        self.V_cr = setup_test_class["V_cr"]

    # M4 regressions
    def define_mass_properties(self):
        nasa_lpc_weights = cd.aircraft.models.weights.nasa_lpc

        self.wing.quantities.mass_properties = nasa_lpc_weights.compute_wing_mps(
        wing_AR=self.wing.parameters.AR,
        wing_area=self.wing.parameters.S_ref,
        fuselage_length=self.fuselage.parameters.length,
        battery_mass=self.m_bat,
        cruise_speed=self.V_cr,
        )

    # EoM model
    def evaluate_euler_6_dof_eom_model(self):
        num_nodes = 1
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

        eom_model = cd.aircraft.models.eom.SixDofEulerFlatEarthModel(num_nodes=num_nodes)
        accel = eom_model.evaluate(
            total_forces=total_forces,
            total_moments=total_moments,
            ac_states=ac_states,
            ac_mass_properties=ac_mass_properties,
        )

        return accel

    def test_m4_wing_mass(self):
        """Test that wing regression works."""

        desired_wing_mass = 296.60345850400006
        desired_wing_cg = np.array([-3.64567517, 0., -2.54192697])
        desired_wing_it_00 = 3739.447260189365 
        desired_wing_it_11 = 47.97886827466999
        desired_wing_it_22 = 3711.269379490369


        self.define_mass_properties()
        # Test mass
        np.testing.assert_almost_equal(
            self.wing.quantities.mass_properties.mass.value, 
            desired_wing_mass,
            decimal=8,
        )

        # Test cg
        np.testing.assert_almost_equal(
            self.wing.quantities.mass_properties.cg_vector.value[0],
            desired_wing_cg[0], 
            decimal=8
        )

        # Test inertia
        np.testing.assert_almost_equal(
            self.wing.quantities.mass_properties.inertia_tensor.value[0, 0],
            desired_wing_it_00,
            decimal=8
        )

        np.testing.assert_almost_equal(
            self.wing.quantities.mass_properties.inertia_tensor.value[1, 1],
            desired_wing_it_11,
            decimal=8
        )

        np.testing.assert_almost_equal(
            self.wing.quantities.mass_properties.inertia_tensor.value[2, 2],
            desired_wing_it_22,
            decimal=8
        )

    def test_euler_6_dof(self):
        """Test the euler 6-dof EoM model."""

        desired_norm = 0.01986085
        desired_du_dt = 0.01675262
        desired_dv_dt = -2.34017885e-05
        desired_dw_dt = 0.00920416
        desired_dp_dt = 0.00413169
        desired_dq_dt = -0.00038393
        desired_dr_dt = 0.00344498

        accelerations = self.evaluate_euler_6_dof_eom_model()

        np.testing.assert_almost_equal(
            accelerations.accel_norm.value,
            desired_norm,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.du_dt.value,
            desired_du_dt,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.dv_dt.value,
            desired_dv_dt,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.dw_dt.value,
            desired_dw_dt,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.dp_dt.value,
            desired_dp_dt,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.dq_dt.value,
            desired_dq_dt,
            decimal=8
        )

        np.testing.assert_almost_equal(
            accelerations.dr_dt.value,
            desired_dr_dt,
            decimal=8
        )

    def test_condition_parameterization(self):
        """Test that conditions are parameterized correctly.
        
        There will be a separte test file for testing 
        the Condition class more comprhensively/
        """

        # cruise
        cruise = cd.aircraft.conditions.CruiseCondition(
            mach_number=0.2,
            altitude=3e3,
            pitch_angle=np.deg2rad(5),
            range=5e5,
        )

        np.testing.assert_almost_equal(
            cruise.parameters.time.value,
            7609.10430225,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.ac_states.u.value,
            65.46070723,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.ac_states.w.value,
            5.72706979,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.atmos_states.density.value,
            0.90909145,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.atmos_states.dynamic_viscosity.value,
            1.64224975e-05,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.atmos_states.pressure.value,
            70095.87788256,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.atmos_states.speed_of_sound.value,
            328.55378251,
        )

        np.testing.assert_almost_equal(
            cruise.quantities.atmos_states.temperature.value,
            268.66,
        )
        