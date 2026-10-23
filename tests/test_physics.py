"""Physics sanity checks: outputs obey the constraints expected from the underlying physics."""

import numpy as np

from snudd.geometry import SolarAngles
from snudd.models import SM
from snudd.nsi import oscillation as osc
from snudd.nsi.nsi_probabilities import (
    DensityMatrixCalculator,
    SolarDensityMatrixCalculator,
)
from snudd.targets import helm_form_factor


def _assert_density_matrix(rho):
    """Check that each rho[i] is Hermitian, unit-trace and positive semi-definite."""
    np.testing.assert_allclose(rho, rho.conj().transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(
        np.trace(rho, axis1=1, axis2=2), 1.0, atol=1e-12
    )
    assert np.all(np.linalg.eigvalsh(rho) > -1e-12)


def test_pmns_unitary():
    """The PMNS mixing matrix must be unitary to conserve probability."""
    U = osc.UPMNS(osc.osc_params_best)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(3), atol=1e-12)


def test_helm_form_factor():
    """The Helm form factor should be 1 at zero momentum transfer and fall off with increasing momentum transfer."""
    q = np.array([0.0, 0.01, 0.1, 0.5])
    F = helm_form_factor(q, 131)
    assert F[0] == 1.0
    assert np.all(np.abs(F) <= 1.0)
    assert np.all(np.diff(F) < 0)


def test_solar_density_matrix(model, e_nus):
    """The solar density matrix must be a valid density matrix: Hermitian, unit trace and positive semi-definite."""
    _assert_density_matrix(
        SolarDensityMatrixCalculator(model).density(e_nus, "8B")
    )


def test_earth_density_matrix(nsi_model, e_nus):
    """Evolution through the Earth is unitary, so the density matrix must stay valid after it."""
    _assert_density_matrix(
        DensityMatrixCalculator(nsi_model).density_earth(
            e_nus, [0.5], [1.0], nu="8B"
        )
    )


def test_solar_survival_probability_msw(e_nus):
    """P_ee should drop from the vacuum-dominated to the matter-dominated regime."""
    p_ee = (
        SolarDensityMatrixCalculator(SM()).density(e_nus, "8B")[:, 0, 0].real
    )
    assert np.all(np.diff(p_ee) < 0)
    assert 0.5 < p_ee[0] < 0.6
    assert 0.3 < p_ee[-1] < 0.35


def test_cnadir_weights_normalised():
    """The flux-weighted nadir-angle histogram should be normalised over a full year of data taking."""
    _, weights = SolarAngles(latitude=42.47, t0=0, T=365).cnadir_hist(bins=10)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-3)


def test_cevns_spectrum_kinematic_cutoff(xe_nucleus):
    """The CEvNS spectrum should be non-zero below the maximum kinematically allowed recoil energy and zero above it."""
    xe_nucleus.prepare_density()
    E_max = xe_nucleus.E_max / 1e6  # keV -> GeV
    E_Rs = np.array([0.5, 0.9, 1.1, 2.0]) * E_max
    spec = xe_nucleus.spectrum(E_Rs)
    assert np.all(spec[:2] > 0)
    np.testing.assert_array_equal(spec[2:], 0.0)
