from pyscf.acmp.ac_interpolators import BasicACW, MOD_ISI_ACW, PURE_MOD_ISI_ACW
from pyscf import gto, scf, dft
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ac_ump2 import ACUMP2
from pyscf.mp.mp2 import MP2
from pyscf.acmp.pt2 import PT2
from pyscf.acmp.ft_mp2 import make_ftmp2
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.scf.addons import smearing
import unittest
from numpy.testing import assert_allclose


class KnownValues(unittest.TestCase):
    def test_singles(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-svp")

        df_codes = [("MGGA", funcs.mgga_epc_winfp)]
        silim = ("MGGA", funcs.mgga_epc_winf)
        acw = MOD_ISI_ACW()
        interpolator = get_interpolator(acw=acw, mode="M")

        rks = scf.RKS(mol, xc="PBE")
        rks.conv_tol = 1e-12
        rks.kernel()
        uks = scf.UKS(mol, xc="PBE")
        uks.conv_tol = 1e-12
        uks.kernel()

        for with_singles in [False, True]:
            mypt = ACMP2(rks)
            mypt.ac_interpolator = interpolator
            mypt.df_codes = df_codes
            mypt.si_limit = silim
            mypt.with_singles = with_singles
            mypt.kernel()
            et_r, ec_r = mypt.e_tot, mypt.e_corr

            mypt = ACUMP2(uks)
            mypt.ac_interpolator = interpolator
            mypt.df_codes = df_codes
            mypt.si_limit = silim
            mypt.with_singles = with_singles
            mypt.kernel()
            et_u, ec_u = mypt.e_tot, mypt.e_corr

            assert_allclose(et_r, et_u, rtol=0, atol=1e-8)
            assert_allclose(ec_r, ec_u, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
