"""Reject unverified strip-transform axis conventions, retain ordinary pdata."""
import nmrglue as ng
import numpy as np
import pytest

from spintegrate.core import load_pdata


def make_fixture(path, updates):
    path.mkdir()
    values = np.arange(32, dtype=float)
    procs = dict(_coreheader=['##TITLE= Synthetic processed spectrum'], _comments=[],
                 SI=32, XDIM=32, PPARMOD=0, BYTORDP=0, DTYPP=0, NC_proc=0,
                 SF=400., SW_p=4000., OFFSET=12., PHC0=12., PHC1=34., **updates)
    ng.bruker.write_pdata(str(path), {'procs': procs}, values,
                         scale_data=True, write_procs=True, isfloat=False)
    return values


@pytest.mark.parametrize('updates', [
    dict(STSR=4, STSI=32, FTSIZE=64),
    dict(STSR=0, STSI=32, FTSIZE=64),
    dict(STSR=0, STSI=16, FTSIZE=32),
])
def test_cropped_pdata_fails_explicitly_instead_of_guessing_axis(tmp_path, updates):
    make_fixture(tmp_path/'pdata', updates)
    with pytest.raises(ValueError, match='STSR/STSI/FTSIZE'):
        load_pdata(tmp_path/'pdata')


@pytest.mark.parametrize('updates', [dict(STSR=0, STSI=32, FTSIZE=32),
                                    dict(STSR=0, STSI=0, FTSIZE=32),
                                    dict(STSR=0, STSI=0, FTSIZE=0)])
def test_ordinary_pdata_with_explicit_strip_defaults_keeps_saved_axis(tmp_path, updates):
    values = make_fixture(tmp_path/'pdata', updates)
    spectrum = load_pdata(tmp_path/'pdata')
    np.testing.assert_array_equal(spectrum.intensity, values[::-1])
    np.testing.assert_array_equal(spectrum.ppm, (12.-np.arange(32)*4000/(400*32))[::-1])
