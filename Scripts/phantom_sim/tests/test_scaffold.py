import importlib
import numpy as np


def test_fibre_packer_vendored():
    mod = importlib.import_module("phantom_sim.third_party.fibre_pack.fibre_packer")
    assert hasattr(mod, "from_fvf") or hasattr(mod, "FibrePacker")


def test_synth_fixture_shapes(synth_config):
    configuration, radii = synth_config
    Z, two, N = configuration.shape
    assert two == 2
    assert radii.shape == (N,)
    assert Z >= 2
