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

    version = settings["style"] + "mp2"
    suffix = settings.get("suffix", "")
    assert suffix in ["", "2", "l", "x"]
    version = version + suffix
    if settings["h0"] == "fock":
        version = version + "hf"
    elif settings["h0"] != "kinetic":
        raise ValueError("h0 must be 'fock' or 'kinetic'")

    ED_CONSTANT = 1.5 / (8 * np.pi**5)

    lpath = os.path.dirname(__file__)
    mylib = np.ctypeslib.load_library("libacfq.so", lpath)

    WS_RADIUS = settings["ws_radius"]
    KFERMI = (9 * np.pi / 4)**(1.0 / 3) / WS_RADIUS
    PARAM = settings["param"]
    nsph = settings["nsph"]

    rl, dvl = get_occ_grid(settings["nl"])
    ru, dvu = get_q_grid(settings["nu"], settings["rmax"])
    if settings.get("rcut", None) is not None:
        cond = (ru < settings["rcut"])
        ru = ru[cond].copy()
        dvu = dvu[cond].copy()
    num = 1 if GRID1 else 2
    # np.save(f"g{num}r.npy", ru)
    # np.save(f"g{num}v.npy", dvu)

    rho = 3.0 / (4 * np.pi * WS_RADIUS**3)
    TEX = 0.3 * (3 * np.pi**2)**(2.0 / 3) * rho**(2.0 / 3)
    KEX = -0.75 * (3.0 / np.pi)**(1.0 / 3) * rho**(1.0 / 3)

    cost, wts = leggauss(nsph)
    sint = np.sqrt(1 - cost * cost)
    wt = 0.5 * wts

    fds = np.zeros((ru.size, rl.size))
    fxs = np.zeros((ru.size, rl.size))
    print("VERSION", version)
    if version == "mp2":
        fn = mylib.calculate_mp2
    elif version == "mp2hf":
        fn = mylib.calculate_mp2hf
    elif version == "kappamp2":
        fn = mylib.calculate_kappamp2
    elif version == "kappamp2hf":
        fn = mylib.calculate_kappamp2hf
    elif version == "lambdamp2":
        fn = mylib.calculate_lambdamp2
    elif version == "lambdamp22":
        fn = mylib.calculate_lambdamp22
    elif version == "lambdamp2l":
        fn = mylib.calculate_lambdamp2l
    elif version == "lambdamp2x":
        fn = mylib.calculate_lambdaxmp2
    elif version == "lambdamp2xhf":
        fn = mylib.calculate_lambdaxmp2hf
    elif version == "lambdamp2hf":
        fn = mylib.calculate_lambdamp2hf
    elif version == "taump2hf":
        fn = mylib.calculate_taump2hf
    else:
        raise ValueError
    fn.restype = ctypes.c_double
    for q in range(ru.size):
        if q % 100 == 0:
            print(q)
        fn(
            cost.ctypes.data_as(ctypes.c_void_p),
            sint.ctypes.data_as(ctypes.c_void_p),
            rl.ctypes.data_as(ctypes.c_void_p),
            wt.ctypes.data_as(ctypes.c_void_p),
            dvl.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(wt.size),
            ctypes.c_int(rl.size),
            ctypes.c_double(ru[q]),
            ctypes.c_double(PARAM),
            ctypes.c_double(KFERMI),
            fds[q].ctypes.data_as(ctypes.c_void_p),
            fxs[q].ctypes.data_as(ctypes.c_void_p),
        )
    ed = ED_CONSTANT * np.dot(dvu / ru**4, fds)
    ex = ED_CONSTANT * np.dot(0.5 * dvu / ru**2, fxs)
    et = ex - ed
    
    resdir = "res_{}_{:.3f}_{:.7f}.yaml".format(version, WS_RADIUS, PARAM)
    os.makedirs("results2", exist_ok=True)

    # adjust dvl so that it integrates to 1/2
    dvl[:] *= 3 / (8 * np.pi)
    print("THIS SHOULD BE 1", 2 * dvl.sum())
    ed[:] /= dvl
    ex[:] /= dvl
    et[:] /= dvl
    edtot = ed.dot(dvl)
    extot = ex.dot(dvl)
    ettot = et.dot(dvl)
    print(TEX, KEX, edtot, extot, ettot, TEX + KEX + ettot)

    if GRID1:
        num = 1
    else:
        num = 2
    # np.save(f"g{num}d.npy", fds)
    # np.save(f"g{num}x.npy", fxs)
    with open("results2/{}".format(resdir), "w") as f:
        yaml.dump(
            {
                "settings": settings,
                "rl": rl,
                "dv": dvl,
                "ekin": TEX,
                "exx": KEX,
                "edlist": ed,
                "ed": edtot,
                "exlist": ex,
                "ex": extot,
                "etlist": et,
                "et": ettot,
            }, f
        )


if __name__ == "__main__":
    main()
