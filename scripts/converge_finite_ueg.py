from pyscf.acmp.ueg import driver, fast_driver
from itertools import product
import numpy as np
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import MOD_ISI_ACW, get_interpolator
from pyscf.acmp.ueg.magic_numbers import MAGIC_NUMBERS
import sys


def run_acpt2(rs, nelec_index, nbas_index, mode="ac"):
    df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
    silim = ("GGA", funcs.gga_pch_winf_v2)
    acw = MOD_ISI_ACW()
    aci = get_interpolator(acw=acw, mode="E")
    if mode == "ac":
        method = {
            "name": "ac",
            "silim": silim,
            "df_codes": df_codes,
            "aci": aci,
        }
    else:
        method = {
            "name": "kappa",
            "param": 1.1,
        }
    settings = {}
    settings["h0"] = "kinetic"
    settings["ws_radius"] = rs
    settings["nelec_index"] = nelec_index
    settings["nbas_index"] = nelec_index + nbas_index
    settings["vcut"] = False
    settings["method"] = method
    fres = fast_driver.run_ueg_calc(**settings)
    fc = fast_driver.post_process(method, fres)
    fk, fx = fres[:2]
    fk = np.mean(fk)
    fx = np.mean(fx)
    return fk, fx, fc


def extrap(x, y):
    X = np.stack([x, np.ones_like(x)])
    M = X.dot(X.T)
    return np.linalg.solve(M, X.dot(y))[1]


if __name__ == "__main__":
    rs = float(sys.argv[1])
    print(MAGIC_NUMBERS.size)
    ecbss = []
    nelecs = []
    for n in range(50, 85, 5):
        b = n
        nelecs.append(MAGIC_NUMBERS[n])
        bs = []
        ecs = []
        while MAGIC_NUMBERS[b] < 7 * MAGIC_NUMBERS[n]:
            b += 1
        bs.append(b)
        while MAGIC_NUMBERS[b] < 10 * MAGIC_NUMBERS[n]:
            b += 1
        bs.append(b)
        while MAGIC_NUMBERS[b] < 13 * MAGIC_NUMBERS[n]:
            b += 1
        bs.append(b)
        while MAGIC_NUMBERS[b] < 16 * MAGIC_NUMBERS[n]:
            b += 1
        bs.append(b)
        while MAGIC_NUMBERS[b] < 19 * MAGIC_NUMBERS[n]:
            b += 1
        bs.append(b)
        for b0 in bs:
            ec = run_acpt2(rs, n, b0, mode="ac")[2]
            ecs.append(ec)
            print(MAGIC_NUMBERS[n], MAGIC_NUMBERS[b0], ec)
            print()
        sizes = np.array([MAGIC_NUMBERS[b] for b in bs])
        ec_cbs = extrap(1.0 / np.array(sizes), np.array(ecs))
        ecbss.append(ec_cbs)
        print()
        print(ec_cbs)
        print()
    nelecs = np.array(nelecs)
    print(extrap(1 / nelecs, ecbss))
    print(extrap(1 / nelecs**(2.0 / 3), ecbss))
    print(extrap(1 / nelecs**(1.0 / 3), ecbss))

