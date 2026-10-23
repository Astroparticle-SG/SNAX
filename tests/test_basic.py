"""Basic correctness checks: modules import, and the main entry points run and return well-formed output."""
import importlib

import numpy as np
import pytest

from snudd.geometry import SolarAngles
from snudd.models import SM, GeneralNSI
from snudd.nsi.nsi_probabilities import DensityMatrixCalculator, SolarDensityMatrixCalculator


MODULES = [
    "snudd.config",
    "snudd.binding",
    "snudd.efficiencies",
    "snudd.geometry",
    "snudd.models",
    "snudd.quenching",
    "snudd.resolution",
    "snudd.rrpa",
    "snudd.spectrum",
    "snudd.targets",
    "snudd.bkg.dw_background",
    "snudd.bkg.lz_background",
    "snudd.bkg.xnt_background",
    "snudd.nsi.earth_matter",
    "snudd.nsi.flux_dists",
    "snudd.nsi.nsi_probabilities",
    "snudd.nsi.oscillation",
    "snudd.nsi.solar_profiles",
    "snudd.nsi.utils",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    """Every module should import cleanly, catching missing dependencies or broken module-level code."""
    importlib.import_module(module)


def test_solar_density_shape(model, e_nus):
    """The solar density matrix should be a finite (N, 3, 3) array, one matrix per neutrino energy."""
    rho = SolarDensityMatrixCalculator(model).density(e_nus, "8B")
    assert rho.shape == (len(e_nus), 3, 3)
    assert np.all(np.isfinite(rho))


def test_earth_density_shape(nsi_model, e_nus):
    """The Earth-matter evolved density matrix should also be a finite (N, 3, 3) array."""
    rho = DensityMatrixCalculator(nsi_model).density_earth(e_nus, [0.5], [1.0], nu="8B")
    assert rho.shape == (len(e_nus), 3, 3)
    assert np.all(np.isfinite(rho))


def test_cnadir_hist_shape():
    """The nadir-angle histogram should return one cos(nadir) and one weight per bin, with valid cosines."""
    cnadirs, weights = SolarAngles(latitude=42.47, t0=0, T=365).cnadir_hist(bins=10)
    assert cnadirs.shape == weights.shape == (10,)
    assert np.all(np.abs(cnadirs) <= 1)


@pytest.mark.parametrize("target", ["xe_nucleus", "xe_electron"])
def test_spectrum(target, request):
    """Nuclear and electron recoil spectra should be finite, non-negative and match the length of the recoil energies."""
    target = request.getfixturevalue(target)
    target.prepare_density()
    E_Rs = np.geomspace(1, 10, 4) / 1e6
    spec = target.spectrum(E_Rs)
    assert spec.shape == E_Rs.shape
    assert np.all(np.isfinite(spec))
    assert np.all(spec >= 0)


def test_sm_equals_zero_nsi(xe_nucleus):
    """The SM model and an NSI model with all couplings zero go through different code paths but should agree."""
    E_Rs = np.geomspace(1e-2, 10, 5) / 1e6
    spectra = []
    for model in (SM(), GeneralNSI(np.zeros((3, 3)), np.pi / 4, 0)):
        xe_nucleus.update_model(model)
        xe_nucleus.prepare_density()
        spectra.append(xe_nucleus.spectrum(E_Rs))
    np.testing.assert_allclose(spectra[0], spectra[1], rtol=1e-12)
