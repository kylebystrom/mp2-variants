from pyscf.acmp.ac_interpolators import BasicACW, MOD_ISI_ACW, PURE_MOD_ISI_ACW
from pyscf import gto, scf, dft
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.mp.mp2 import MP2
from pyscf.acmp.ft_mp2 import make_ftmp2
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.scf.addons import smearing
import unittest
from numpy.testing import assert_allclose


class KnownValues(unittest.TestCase):
    def test_ft_mp2(self):
        # TODO split this into multiple tests

        # Make a simple molecule and zero-T PBE calculation
        mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-tzvp")
        mf = dft.RKS(mol, xc="PBE").newton()
        mf.kernel()
        mymp = make_ftmp2(MP2(mf), beta=10000)
        mymp.kernel()

        df_codes = [] # [("GGA", funcs.gga_pch_winfp_v2)]
        silim = ("GGA", funcs.gga_pch_winf_v2)
        acw = BasicACW()
        interpolator = get_interpolator(acw=acw, mode="M")
        mymp = ACMP2(mf)
        if False:
            mymp = mymp.set_frozen()
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel(with_t2=False)

        # Perform a high-temperature PBE calculation
        # beta = 1/T
        beta = 40.0
        ftmf = smearing(dft.RKS(mol, xc="PBE"), sigma=1.0 / beta)
        ftmf.kernel()
        print("MF ENS", ftmf.e_tot, ftmf.e_free, ftmf.e_zero)

        mymp = make_ftmp2(ACMP2(ftmf), beta=beta,
                          particle_fix=None,
                          with_singles=False,
                          ecorr_method="zeroth_order")
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel()
        assert_allclose(mymp.e_tot, -100.391027865593, atol=1e-8, rtol=0)
        assert_allclose(mymp.e_corr, -0.38009583997081, atol=1e-8, rtol=0)

        mymp = make_ftmp2(ACMP2(ftmf), beta=beta,
                  particle_fix=None,
                  with_singles=False,
                  ecorr_method="finite_difference")
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel()
        #assert_allclose(mymp.e_tot, -100.338659133308, atol=1e-8, rtol=0)
        #assert_allclose(mymp.e_corr, -0.327727107685651, atol=1e-8, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta,
                  particle_fix=None,
                  ecorr_method="analytical")
        mymp.kernel()
        ec = mymp.e_corr
        #assert_allclose(mymp.e_tot, -100.173149236711, atol=1e-8, rtol=0)
        #assert_allclose(mymp.e_corr, -0.162217211088837, atol=1e-8, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta,
                  particle_fix=None,
                  ecorr_method="finite_difference")
        mymp.kernel()
        print(mymp.e_corr, ec)
        assert_allclose(mymp.e_corr, ec, atol=1e-6, rtol=0)
        #assert_allclose(mymp.f_corr, ef, atol=1e-6, rtol=0)

        if False:
            # NOTE this fd does not work currently because
            # e_corr for ensemble=None contains terms from
            # both the internal energy and grand potential,
            # so differentiating the whole thing by finite
            # wrt beta and mu does not yield the entropy
            # and particle number terms of the internal energy.
            f0 = mymp.f_corr
            e0 = mymp.e_corr
            mu = mymp.mu_opt

            delta = 0.000001
            mymp = FTMP2(ftmf, beta=beta*(1+0.5*delta), mu0=mu)
            mymp.kernel()
            f1 = mymp.e_corr

            mymp = FTMP2(ftmf, beta=beta*(1-0.5*delta), mu0=mu)
            mymp.kernel()
            f2 = mymp.e_corr

            mymp = FTMP2(ftmf, beta=beta, mu0=mu+0.5*delta)
            mymp.kernel()
            f3 = mymp.e_corr

            mymp = FTMP2(ftmf, beta=beta, mu0=mu-0.5*delta)
            mymp.kernel()
            f4 = mymp.e_corr

            print("FINAL ENS", e0, f0 + (f1-f2)/delta + mu*(f4-f3)/delta)
            assert_allclose(e0, f0 + (f1-f2)/delta + mu*(f4-f3)/delta,
                            rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
