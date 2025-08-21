from pyscf.acmp.ueg import driver, fast_driver
from pyscf.acmp.mp2_numint import gga_hsq_sce_limit, mgga_hsq_sce_limit, \
    mgga_chi, lda_rho
from pyscf.acmp.ac_interpolators import ACInterpolator, ACMatrix, ACW
from itertools import product
import numpy as np



def print_errs(method1, method2):
    h0s = ["fock", "kinetic"]
    # rss = [0.01, 0.25, 0.5, 1.0, 1.3, 1.8, 4.5, 100]
    rss = [0.01, 0.4, 1.0, 2.3, 100]
    nelec_indices = [1, 2, 3]
    nbas_adds = [1, 2, 3, 4]
    vcuts = [False]
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
        if method2["name"] != "ac":
            fres = fast_driver.run_ueg_calc(**settings)
        else:
            fres = fast_driver.run_ueg_calc(**settings)
        fc = fast_driver.post_process(method2, fres)
        if method2["name"] != "ac":
            fk, fx = fres[:2]
        else:
            fk, fx = fres[:2].mean(-1)
        if np.isnan([k, x, c, fk, fx, fc]).any():
            print("ERROR", settings, k, x, c, fk, fx, fc)
            exit()
        results.append([k, x, c, fk, fx, fc])
        print(c, fc, c - fc, "STEP DONE\n\n\n")

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

    print("DONE\n\n\n\n")
    err_summary = []
    for res, err in zip([results[:, 0], results[:, 1], results[:, 2]],
                        [k_err, x_err, c_err]):
        err_summary.append((np.max(np.abs(err)), np.max(np.abs(err) / np.abs(res))))

    return results[:, :6].T, [k_err, x_err, c_err], err_summary


class MyACW(ACW):
    def get_winf_diff(self,winf, exx, a):
        ratio = winf / exx
        func = lambda x: np.log(1 + np.exp(a * (1 - x))) / np.log(1 + np.exp(a))
        if isinstance(winf, ACMatrix):
            val = ratio.apply(func)
        else:
            val = func(ratio)
        wdiff = winf - (exx * (1 - val))
        return wdiff

    def compute_cache(self, w_list):
        is_mat = isinstance(w_list[0], ACMatrix)
        if is_mat:
            for w in w_list:
                assert isinstance(w, ACMatrix)
        else:
            for w in w_list:
                assert isinstance(w, np.ndarray) and w.ndim == 1
        self.clear_cache()
        a, b, c, d, e, f, g = self.params
        w0, exx, chi, gga, rho, mgga = w_list
        mgga = mgga * -1
        rho = rho * -1
        # chi is 1 for UEG, 0 for 1-orbital
        chiterm = chi - 1  # 0 for UEG, -1 for 1-orbital
        chiterm = gga / mgga  # 0 for UEG, 1 for 1-orbital
        winf_denom = (1 + f * chiterm) * mgga + a * (1 + c * chiterm) * gga
        winf = rho / winf_denom
        winf = self.get_winf_diff(winf, exx, 10)
        ratio = (g + 1 - chiterm + 1e-10) * (winf / exx)
        term0 = w0 / winf
        term1 = b * ratio / winf**0.5 * (w0 / (1 + e * w0))
        term2 = d * ratio * (w0 / winf**0.5)
        self._cache["w0"] = w0
        self._cache["terms"] = (term0, term1, term2)

    def clear_cache(self):
        self._cache = {}

    def __call__(self, alpha):
        w0 = self._cache["w0"]
        term0, term1, term2 = self._cache["terms"]
        denom = 1 + term2 * alpha**0.5 + (term1 * alpha**0.5 + term0 * alpha)**2
        return alpha * w0 / denom**0.5


def main():
    def get_vmat(w_list):
        #return w_list[0]  # np.ones_like(w_list[0])
        oo_mp2, oo_screen = w_list
        a = 0.3
        tmp = (-a * oo_mp2)**0.5
        tmp = tmp / (1 + tmp)
        return tmp * oo_screen
    gap_model = (get_vmat, True)

    version = "a"

    if version == "k":
        k_method1 = lambda mf: driver.UEGKappaMP2(mf, kappa=1.5, damping="kappa")
        k_method2 = {"name": "kappa", "param": 1.5}
        res, errs, summary = print_errs(k_method1, k_method2)
    elif version == "l":
        l_method1 = lambda mf: driver.UEGLambdaMP2(mf, gap_mode="M", gap_model=gap_model)
        l_method2 = {"name": "lambda", "gap_model": gap_model, "df_codes": []}
        res, errs, summary = print_errs(l_method1, l_method2)
    else:
        params = [ 1.266e+00, 4.554e+00, -8.453e-01, 5.427e+00, 2.356e+00,
                  -1.601e-01, 2.703e-01]
        acw = MyACW(params)
        df_codes = [("MGGA", mgga_chi), ("GGA", gga_hsq_sce_limit),
                    ("LDA", lda_rho)]
        silim = ("MGGA", mgga_hsq_sce_limit)

        m_interp = ACInterpolator(512, "M", acw)
        e_interp = ACInterpolator(512, "E", acw)

        def get_acmp2(mf):
            acmp = driver.UEGACMP2(mf)
            acmp.df_codes = df_codes
            acmp.si_limit = silim
            acmp.ac_interpolator = e_interp
            return acmp

        a_method1 = get_acmp2
        a_method2 = {"name": "ac", "aci": e_interp, "df_codes": df_codes, "silim": silim}
        res, errs, summary = print_errs(a_method1, a_method2)

    print(errs[2])
    print(res[2])
    print(res[5])
    print()
    print(summary)


if __name__ == "__main__":
    main()

