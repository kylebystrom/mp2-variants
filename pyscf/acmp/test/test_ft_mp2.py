from pyscf.acmp.ac_interpolators import BasicACW, MOD_ISI_ACW, PURE_MOD_ISI_ACW
from pyscf import gto, scf, dft
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.mp.mp2 import MP2
from pyscf.acmp.pt2 import PT2
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
        XC = "PBE"
        mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-tzvp")
        # mol = gto.M(atom="H 0 0 0; F 0 0 0.9168", basis="sto-3g")
        mf = dft.RKS(mol, xc=XC).newton()
        mf.kernel()
        mymp = make_ftmp2(MP2(mf), beta=10000)
        mymp.kernel()

        df_codes = []  # [("GGA", funcs.gga_pch_winfp_v2)]
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
        # beta = 40.0
        beta = 40.0
        # beta = 1 / 0.0316679
        ftmf = smearing(dft.RKS(mol, xc=XC), sigma=1.0 / beta)
        # ftmf = smearing(dft.RKS(mol, xc=XC), sigma=0.0000001)
        # ftmf = dft.RKS(mol, xc=XC)
        ftmf.kernel()
        # print("MF ENS", mf.e_tot, ftmf.e_tot, ftmf.e_free, ftmf.e_zero)

        mymp = make_ftmp2(ACMP2(ftmf), beta=beta,
                          particle_fix=None,
                          with_singles=False,
                          ecorr_method="zeroth_order")
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel()
        if XC == "PBE" and beta == 40:
            assert_allclose(mymp.e_tot, -100.391027865593, atol=1e-8, rtol=0)
            assert_allclose(mymp.e_corr, -0.38009583997081, atol=1e-8, rtol=0)

        mymp = make_ftmp2(ACMP2(ftmf), beta=beta, particle_fix=None,
                          with_singles=False, ecorr_method="finite_difference")
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel()
        assert_allclose(mymp.e_tot, -100.386074249698, atol=1e-8, rtol=0)
        assert_allclose(mymp.e_corr, -0.375142224075233, atol=1e-8, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix=None,
                          ecorr_method="analytical")
        mymp.kernel()
        ec = mymp.e_corr
        assert_allclose(mymp.e_tot, -100.173149236711, atol=1e-8, rtol=0)
        assert_allclose(mymp.e_corr, -0.162217211088837, atol=1e-8, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix=None,
                          ecorr_method="finite_difference")
        mymp.kernel()
        print(mymp.e_corr, ec)
        assert_allclose(mymp.e_corr, ec, atol=1e-6, rtol=0)
        # assert_allclose(mymp.f_corr, ef, atol=1e-6, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="iter",
                          ecorr_method="analytical")
        mymp.kernel()

        print(mymp.e_tot, mymp.e_corr)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="iter_fd",
                          ecorr_method="analytical")
        mymp.kernel()

        print(mymp.e_tot, mymp.e_corr)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="pt",
                          ecorr_method="finite_difference", occ_tol=0)
        mymp.kernel()

        print()
        res1 = [
            mymp.e_tot, mymp.e_free, mymp.e_corr,
            mymp.results["e0"], mymp.results["e1"], mymp.results["e2"],
            mymp.results["gp1"], mymp.results["gp2"],
            mymp.results["mu1"], mymp.results["mu2"],
        ]
        print("LOOK1", mymp.e_tot, mymp.e_free, mymp.e_corr,
              mymp.results["e0"], mymp.results["e1"], mymp.results["e2"],
              mymp.results["gp1"], mymp.results["gp2"],
              mymp.results["mu1"], mymp.results["mu2"])
        print()

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="pt",
                          ecorr_method="analytical", occ_tol=0)
        mymp.kernel()

        print()
        res2 = [
            mymp.e_tot, mymp.e_free, mymp.e_corr,
            mymp.results["e0"], mymp.results["e1"], mymp.results["e2"],
            mymp.results["gp1"], mymp.results["gp2"],
            mymp.results["mu1"], mymp.results["mu2"],
        ]
        print("LOOK2", mymp.e_tot, mymp.e_free, mymp.e_corr,
              mymp.results["e0"], mymp.results["e1"], mymp.results["e2"],
              mymp.results["gp1"], mymp.results["gp2"],
              mymp.results["mu1"], mymp.results["mu2"])
        print(mymp.mu_opt)
        print()
        for r1, r2 in zip(res1, res2):
            assert_allclose(r1, r2, atol=1e-7, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix=None,
                          ecorr_method="analytical")
        mymp.kernel()
        print(mymp.mu_opt)

        res1 = [mymp.e_tot, mymp.e_corr]
        print(mymp.e_tot, mymp.e_corr)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix=None,
                          ecorr_method="finite_difference")
        mymp.kernel()

        res2 = [mymp.e_tot, mymp.e_corr]
        print(mymp.e_tot, mymp.e_corr)

        for r1, r2 in zip(res1, res2):
            assert_allclose(r1, r2, atol=1e-7, rtol=0)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="pt",
                          ecorr_method="finite_difference")
        mymp.kernel()

        print(mymp.e_tot, mymp.e_corr)

        mymp = make_ftmp2(MP2(ftmf), beta=beta, particle_fix="dv",
                          ecorr_method="finite_difference")
        mymp.kernel()

        print(mymp.e_tot, mymp.e_corr)
        print("\n\n\n")

        mymp = make_ftmp2(MP2(ftmf), beta=1000, particle_fix=None,
                          ecorr_method="finite_difference")
        mymp.kernel()

        print(mymp.e_tot, mymp.e_corr)

        if False:
            # NOTE this fd does not work currently because
            # e_corr for ensemble=None contains terms from
            # both the internal energy and grand potential,
            # so differentiating the whole thing by finite difference
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

    def _check_ft_ac_pt2(self, beta=80):
        df_codes = []  # [("GGA", funcs.gga_pch_winfp_v2)]
        silim = ("GGA", funcs.gga_pch_winf_v2)
        acw = BasicACW()
        interpolator = get_interpolator(acw=acw, mode="M")

        beta = 80

        # Make a simple molecule and zero-T PBE calculation
        XC = "PBE"
        mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-svp")
        mf = dft.RKS(mol, xc=XC)
        mf.kernel()
        mymp = ACMP2(mf)
        mymp.ac_interpolator = interpolator
        mymp.df_codes = df_codes
        mymp.si_limit = silim
        mymp.kernel()
        mymp.with_singles = True
        mymp.kernel()
        et, ec = mymp.e_tot, mymp.e_corr

        ftmf = smearing(mf, sigma=1/beta)
        ftmf.kernel()
        print(mf.mo_occ)
        print(ftmf.mo_occ)

        mymp = make_ftmp2(mymp, beta=1000,
                          particle_fix=None,
                          ecorr_method="finite_difference")
        mymp.kernel()

        assert_allclose(mymp.e_tot, et, atol=1e-5, rtol=0)
        assert_allclose(mymp.e_corr, ec, atol=1e-5, rtol=0)

        mymp.with_singles = False
        mymp.kernel()

        # TODO test no singles

        mymp.particle_fix = "pt"
        mymp.with_singles = True
        mymp.kernel()

        assert_allclose(mymp.e_tot, et, atol=1e-5, rtol=0)
        assert_allclose(mymp.e_corr, ec, atol=1e-5, rtol=0)

        print("CHECK", mf.e_tot, ftmf.e_tot, ftmf.e_free, ftmf.e_zero)
        if True:
            mymp = make_ftmp2(PT2(mf), beta=10000, with_singles=True,
                              particle_fix="pt", ecorr_method="analytical")
            mymp.kernel()
            mymp = PT2(ftmf)
        else:
            mymp = ACMP2(mf)
            mymp.ac_interpolator = interpolator
            mymp.df_codes = df_codes
            mymp.si_limit = silim
            mymp.with_singles = True
            mymp.kernel()
            mymp = ACMP2(ftmf)
            mymp.ac_interpolator = interpolator
            mymp.df_codes = df_codes
            mymp.si_limit = silim
        mymp = make_ftmp2(mymp, beta=beta,
                          # particle_fix="iter_fd",
                          particle_fix="iter",
                          ecorr_method="finite_difference",
                          # ecorr_method="zeroth_order",
                          occ_tol=1e-8)
        print(mymp._init_smearing())
        mymp.with_singles = True
        mymp.kernel()
        print(mymp._init_smearing(), mymp.mu_opt, mymp.get_e_hf())
        return mymp.e_zero

    def test_ft_ac_pt2(self):
        ens = []
        betas = [60, 70, 80, 90, 100]
        for beta in betas:
            self._check_ft_ac_pt2(beta)


if __name__ == "__main__":
    unittest.main()
