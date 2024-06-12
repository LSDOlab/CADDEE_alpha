import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import pytest


@pytest.fixture(scope="class")
def setup_test_class():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    wing_geometry = cd.import_geometry("simple_wing.stp")

    wing = cd.aircraft.components.Wing(
        S_ref=90,
        AR=10.2,
        taper_ratio=0.5,
        geometry=wing_geometry,
        root_twist_delta=np.deg2rad(-10),
        tip_twist_delta=np.deg2rad(10),
        dihedral=np.deg2rad(10),
        sweep=np.deg2rad(-30),
        tight_fit_ffd=False,
    )

    return {"wing" : wing}

@pytest.mark.usefixtures("setup_test_class")
class TestFFD:
    @pytest.fixture(autouse=True)
    def _setup(self, setup_test_class):
        self.wing = setup_test_class["wing"]
        self.wing_geometry = self.wing.geometry
        

    def test_initial_geometry(self):
        """Test that initial geometry is imported correctly."""
        
        initial_coeff_norm_sum_expected = 1357.73155717
        actual_coeff_norm_sum = 0
        for fun in self.wing_geometry.functions.values():
            actual_coeff_norm_sum += csdl.norm(fun.coefficients).value

        np.testing.assert_almost_equal(
            actual_coeff_norm_sum,
            initial_coeff_norm_sum_expected,
            decimal=7,
        )

    def test_ffd_block(self):
        """Test that initial FFD block has correct coefficient norm."""
        initial_ffd_block_coeff_norm_expected = 29.65823832
        ffd_block_coeffs = self.wing._ffd_block.coefficients
        actual_ffd_block_coeff_norm = csdl.norm(ffd_block_coeffs).value

        np.testing.assert_almost_equal(
            actual_ffd_block_coeff_norm,
            initial_ffd_block_coeff_norm_expected,
            decimal=7,
        )

    def test_inner_optimization_wing(self):
        """Test that inner optimization can manipulate a wing with comprehensive parameterization."""
        desired_ffd_block_coeff_norm_after_inner_opt = 48.21388335

        config = cd.Configuration(system=self.wing)
        config.setup_geometry()
        ffd_block_coeffs = self.wing._ffd_block.coefficients
        actual_ffd_block_coeff_norm = csdl.norm(ffd_block_coeffs).value

        np.testing.assert_almost_equal(
            actual_ffd_block_coeff_norm,
            desired_ffd_block_coeff_norm_after_inner_opt,
            decimal=7,
        )

