import numpy as np
import ctypes
import os
from numpy.polynomial.legendre import leggauss
import yaml
import sys
from math import ceil


import numpy as np
from pyscf.dft import radi
from numpy.polynomial.legendre import leggauss


GRID1 = False


def get_occ_grid(nl):
    if GRID1:
        rl, drl = leggauss(nl)
        rl = 0.5 * rl + 0.5
        drl *= 0.5
    else:
        xl = 0.5 + np.arange(nl)
        xl /= nl
        alpha = 10
        # alpha = 6
        rl = (1 - np.exp(-alpha * xl)) / (1 - np.exp(-alpha))
        drl = alpha * np.exp(-alpha * xl) / (1 - np.exp(-alpha)) / nl
    return rl, 4 * np.pi * rl * rl * drl


def get_q_grid(nu, rmax):
    ru, dru = radi.treutler_ahlrichs(nu, 1)
    ratio = rmax / np.max(ru)
    ru = ratio * ru
    dru = ratio * dru
    return ru, 4 * np.pi * ru * ru * dru


def get_linear_grid(rmin, rmax, ngpb):
    nrad = ceil((rmax - rmin) * ngpb)
    dr = (rmax - rmin) / nrad
    rs = np.arange(nrad, dtype=np.float64) * dr + dr * 0.5
    rs += rmin
    dv = 4 * np.pi * rs * rs * dr
    return rs, dv


def main():
    settings_file = sys.argv[1]

    with open(settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.CLoader)

    print("SETTINGS")
    print(yaml.dump(settings))

    ED_CONSTANT = 1.5 / (8 * np.pi**5)

    lpath = os.path.dirname(__file__)
    mylib = np.ctypeslib.load_library("libacfq.so", lpath)

    WS_RADIUS = settings["ws_radius"]
    KFERMI = (9 * np.pi / 4)**(1.0 / 3) / WS_RADIUS
    nsph = settings["nsph"]

    rl, dvl = get_occ_grid(settings["nl"])
    ru, dvu = get_q_grid(settings["nu"], settings["rmax"])
    num = 1 if GRID1 else 2

    rho = 3.0 / (4 * np.pi * WS_RADIUS**3)
    TEX = 0.3 * (3 * np.pi**2)**(2.0 / 3) * rho**(2.0 / 3)
    KEX = -0.75 * (3.0 / np.pi)**(1.0 / 3) * rho**(1.0 / 3)

    cost, wts = leggauss(nsph)
    sint = np.sqrt(1 - cost * cost)
    wt = 0.5 * wts

    fn = mylib.epsx_ueg_py
    fn.restype = ctypes.c_double
    exxlist = np.zeros(rl.size)
    for i, k in enumerate(rl):
        exx = fn(ctypes.c_double(k))
        exxlist[i] = 0.5 * exx * KFERMI

    dvl[:] *= 3 / (8 * np.pi)
    exxlist[:] *= -2
    exx = exxlist.dot(dvl)
    print(exx, KEX, dvl.sum())
    os.makedirs("results", exist_ok=True)
    with open(f"results/exx_{WS_RADIUS:.4f}.yaml", "w") as f:
        yaml.dump(
            {
                "settings": settings,
                "ekin": TEX,
                "dv": dvl,
                "rl": rl,
                "exxlist": exxlist,
                "exx": exx,
            }, f
        )


if __name__ == "__main__":
    main()
