from pyscf.acmp.ueg import driver, fast_driver, fast_driver_ft
from itertools import product
import numpy as np
import unittest
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import MOD_ISI_ACW, get_interpolator
from pyscf.acmp.ueg.magic_numbers import MAGIC_NUMBERS
from numpy.testing import assert_allclose


class PyscfVsFastDriver(unittest.TestCase):
    def _check_pyscf_vs_fast_driver(self, method1, method2, vcuts, beta=None):
        h0s = ["fock", "kinetic"]
        rss = [0.01, 0.25, 0.5, 1.0, 1.3, 1.8, 4.5, 100]
        nelec_indices = [1, 2]
        nbas_adds = [1, 2, 3]
        options = [h0s, rss, nelec_indices, nbas_adds, vcuts]

        settings = {}
        results = []
        for h0, rs, nelec_index, nbas_add, vcut in product(*options):
            settings["h0"] = h0
            settings["ws_radius"] = rs
            settings["nelec_index"] = nelec_index
            settings["nbas_index"] = nelec_index + nbas_add
            settings["vcut"] = vcut
            settings["mp2_init"] = method1
            k, x, c = driver.run_ueg_calc(**settings)
            settings["method"] = method2
            fres = fast_driver.run_ueg_calc(**settings)
            fc = fast_driver.post_process(method2, fres)
            fk, fx = fres[:2]
            fk = np.mean(fk)
            fx = np.mean(fx)
            if np.isnan([k, x, c, fk, fx, fc]).any():
                print("ERROR", settings, k, x, c, fk, fx, fc)
                assert False
            results.append([k, x, c, fk, fx, fc])

        results = np.array(results)
        k_err = results[:, 0] - results[:, 3]
        x_err = results[:, 1] - results[:, 4]
        c_err = results[:, 2] - results[:, 5]

        if len(vcuts) == 2:
            c_err2 = results[::2, 2] - results[1::2, 2]
            c_err3 = results[::2, 5] - results[1::2, 5]
            for err, res in zip([c_err2, c_err3], [results[:, 2], results[:, 5]]):
                print(
                    np.max(np.abs(err)),
                    np.max(np.abs(err / res[::2])),
                    np.max(np.abs(err / res[1::2]))
                )

        for res, err in zip([results[:, 0], results[:, 1], results[:, 2]],
                            [k_err, x_err, c_err]):
            max_err = np.max(np.abs(err))
            max_rel_err = np.max(np.abs(err) / np.abs(res))
            assert_allclose(res + err, res, atol=1e-8, rtol=1e-8)
            print(max_err, max_rel_err)

    def test_ueg_mf(self):
        NELEC = 54
        VCUT = True
        for h0 in ["kinetic", "fock"]:
            mf, ek, ex = driver.get_ueg_mf(
                NELEC, MAGIC_NUMBERS[6], 2.7,
                h0, VCUT
            )
            ftmf, ftek, ftex = driver.get_ueg_mf(
                NELEC, MAGIC_NUMBERS[6], 2.7,
                h0, VCUT, beta=10
            )
            print("EK", ek, ftek)
            print("EX", ex, ftex)
            ftmf, ftek1, ftex1 = driver.get_ueg_mf(
                NELEC + 3.5, MAGIC_NUMBERS[6], 2.7,
                h0, VCUT, beta=10
            )
            print("EK", ftek, ftek1)
            print("EX", ftex, ftex1)
            ftmf, ftek1, ftex1 = driver.get_ueg_mf(
                NELEC - 3.5, MAGIC_NUMBERS[6], 2.7,
                h0, VCUT, beta=10
            )
            print("EK", ftek, ftek1)
            print("EX", ftex, ftex1)
            print()

    def test_kappa_mp2(self):
        method1 = lambda mf: driver.UEGKappaMP2(mf, kappa=1.5, damping="kappa")
        method2 = {"name": "kappa", "param": 1.5}
        self._check_pyscf_vs_fast_driver(method1, method2, [False])

    def test_lambda_mp2(self):
        omega_code = ("LDA", funcs.lda_plasma_frequency)
        df_codes = []

        def method1(mf):
            mymp = driver.UEGLambdaMP2(mf, gap_model=None)
            mymp.omega_code = omega_code
            mymp.df_codes = df_codes
            return mymp

        method2 = {"name": "lambda", "gap_model": None,
                   "omega_code": omega_code,
                   "df_codes": df_codes}
        self._check_pyscf_vs_fast_driver(method1, method2, [False])

    def test_ac_mp2(self):
        df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
        silim = ("GGA", funcs.gga_pch_winf_v2)
        acw = MOD_ISI_ACW()
        aci = get_interpolator(acw=acw, mode="M")

        def method1(mf):
            mymp = driver.UEGACMP2(mf)
            mymp.ac_interpolator = aci
            mymp.si_limit = silim
            mymp.df_codes = df_codes
            return mymp

        method2 = {
            "name": "ac",
            "silim": silim,
            "df_codes": df_codes,
            "aci": aci,
        }
        self._check_pyscf_vs_fast_driver(method1, method2, [False])

    def test_ft_mp2(self, h0="fock"):
        from pyscf.acmp.ft_mp2 import make_ftmp2

        nelecs = [44, 54, 64, 74]
        beta = 20
        nbas_index = 5

        df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
        silim = ("GGA", funcs.gga_pch_winf_v2)
        acw = MOD_ISI_ACW()
        aci = get_interpolator(acw=acw, mode="M")

        with_singles = True

        def method1(mf):
            mymp = driver.UEGACMP2(mf)
            mymp.ac_interpolator = aci
            mymp.si_limit = silim
            mymp.df_codes = df_codes
            return mymp

        mf, ek, ex = driver.get_ueg_mf(
            nelecs[1], MAGIC_NUMBERS[nbas_index], 2.7, h0, False
        )
        mymp = method1(mf)
        mymp.verbose = 0
        ecorr, _ = mymp.kernel()
        ens = np.array([ek, ex, ecorr / nelecs[1]])
        print(ens, np.sum(ens))
        ref_ens = ens
        print()

        mf, ek, ex = driver.get_ueg_mf(
            nelecs[1], MAGIC_NUMBERS[nbas_index], 2.7,
            h0, False, beta=1000
        )
        mymp = method1(mf)
        mymp.verbose = 0
        mymp = make_ftmp2(mymp, beta=1000, particle_fix=None,
                          ecorr_method="zeroth_order",
                          with_singles=with_singles)
        ecorr, _ = mymp.kernel()
        ens = np.array([ek, ex, ecorr / nelecs[1]])
        assert_allclose(ens, ref_ens, atol=1e-4, rtol=0)

        mymp = method1(mf)
        mymp.verbose = 0
        mymp = make_ftmp2(mymp, beta=1000, particle_fix="dv2",
                          ecorr_method="finite_difference",
                          with_singles=with_singles)
        ecorr, _ = mymp.kernel()
        ens = np.array([ek, ex, ecorr / nelecs[1]])
        assert_allclose(ens, ref_ens, atol=1e-4, rtol=0)

        df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
        silim = ("GGA", funcs.gga_pch_winf_v2)
        acw = MOD_ISI_ACW()
        aci = get_interpolator(acw=acw, mode="M")

        method2 = {
            "name": "ac",
            "silim": silim,
            "df_codes": df_codes,
            "aci": aci,
        }

        ecorr_method = "finite_difference"
        for nelec in nelecs:
            mf, ek, ex = driver.get_ueg_mf(
                nelec, MAGIC_NUMBERS[nbas_index], 2.7,
                h0, False, beta=beta
            )
            mymp = method1(mf)
            mymp.verbose = 0
            # mymp = make_ftmp2(mymp, beta=beta, particle_fix=None,
            #                   ecorr_method="zeroth_order")
            mymp = make_ftmp2(mymp, beta=beta, particle_fix="dv2",
                              ecorr_method=ecorr_method,
                              with_singles=with_singles)
            ecorr, _ = mymp.kernel()
            ens = np.array([ek, ex, ecorr / nelec])
            echeck = ens.sum()
            print()
            print(ens, ens.sum(), mymp.e_free / nelec, mymp.e_tot / nelec,
                  mymp.e_zero / nelec)
            print()

            settings = {
                "nelec": nelec,
                "nbas_index": nbas_index,
                "ws_radius": 2.7,
                "h0": h0,
                "vcut": False,
                "beta": beta,
                "particle_fix": "dv2",
                "ecorr_method": ecorr_method,
                "with_singles": with_singles,
                "mp2_init": method1,
            }
            etest_ks = driver.run_ueg_calc(**settings).sum()
            settings.pop("particle_fix")
            settings.pop("ecorr_method")
            etest_gp = driver.run_ueg_calc(**settings).sum()
            assert_allclose(echeck, etest_ks, atol=1e-8, rtol=0)

            settings["method"] = method2
            settings.pop("mp2_init")
            res = fast_driver_ft.run_ueg_calc(**settings)
            assert_allclose(res[3], etest_gp, atol=1e-8, rtol=0)

            settings["particle_fix"] = "dv2"
            settings["ecorr_method"] = ecorr_method
            res = fast_driver_ft.run_ueg_calc(**settings)
            assert_allclose(res[3], etest_ks, atol=1e-8, rtol=0)

    def test_ft_mp2_kinetic(self):
        self.test_ft_mp2("ks")


if __name__ == "__main__":
    unittest.main()
