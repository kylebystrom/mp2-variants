import numpy
import sys
from pyscf.acmp.mp2_numint import MP2NumInt
from pyscf.acmp.ueg import ueg
from pyscf.acmp.ueg.magic_numbers import MAGIC_NUMBERS
import os
import ctypes
import time
import sys
import yaml

lpath = os.path.dirname(__file__)
mylib = numpy.ctypeslib.load_library("libmp2.so", lpath)


DEFAULTS = {
    "ws_radius": 1.0,
    "nelec_index": 1,
    "nbas_index": 2,
    "vcut": False,
    "h0": "kinetic",
    "method": {"name": "kappa", "param": 1.1},
}

GLOBAL_NUMINT = MP2NumInt()


def fill_settings_(settings):
    for key, value in DEFAULTS.items():
        settings[key] = settings.get(key, value)


def kin(my_ueg, vecs):
    vecs = vecs.astype(numpy.int32)
    assert vecs.shape[-1] == 3
    assert vecs.flags.c_contiguous
    res = numpy.empty(vecs.size // 3, dtype=numpy.float64)
    mylib.get_kin(
        res.ctypes.data_as(ctypes.c_void_p),
        vecs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(res.size),
        ctypes.c_double(my_ueg._length),
    )
    res.shape = vecs.shape[:-1]
    return res


def get_k_potential(my_ueg, occ_gvecs, other_gvecs):
    occ_gvecs = occ_gvecs.astype(numpy.int32)
    other_gvecs = other_gvecs.astype(numpy.int32)
    assert occ_gvecs.shape[-1] == 3
    assert occ_gvecs.flags.c_contiguous
    assert other_gvecs.shape[-1] == 3
    assert other_gvecs.flags.c_contiguous
    nocc = occ_gvecs.size // 3
    nother = other_gvecs.size // 3
    vk = numpy.empty(nother, dtype=numpy.float64)
    args = [
        vk.ctypes.data_as(ctypes.c_void_p),
        occ_gvecs.ctypes.data_as(ctypes.c_void_p),
        other_gvecs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nocc),
        ctypes.c_int(nother),
        ctypes.c_double(my_ueg._length),
        ctypes.c_double(my_ueg._volume),
    ]
    if my_ueg.vcut:
        mylib.get_vk_vcut(*args)
    else:
        args.append(ctypes.c_double(my_ueg.madelung))
        mylib.get_vk_novcut(*args)
    return vk


def run_ueg_calc(**settings):
    fill_settings_(settings)
    STYLE = settings["method"].get("name", "kappa")
    nelec_index = settings["nelec_index"]
    nelec = 2 * MAGIC_NUMBERS[nelec_index]
    nbas = MAGIC_NUMBERS[settings["nbas_index"]]
    rs = settings["ws_radius"]

    t0 = time.monotonic()
    my_ueg = ueg.UEG(nelec, nbas, rs, verbose=False)
    if nbas != my_ueg.nbas:
        print("nbasis = %d is not a magic number. It has been increased to %d."%(nbas, my_ueg.nbas))
        nbas = my_ueg.nbas
    my_ueg.vcut = settings["vcut"]
    gvecs = numpy.asarray(my_ueg.rgvecs, order="C", dtype=numpy.float64)
    occ_gvecs = gvecs[:nelec//2]
    vir_gvecs = gvecs[nelec//2:]
    ekins_o = kin(my_ueg, occ_gvecs)
    ekins_v = kin(my_ueg, vir_gvecs)
    g_o = occ_gvecs.astype(numpy.int32)
    g_v = vir_gvecs.astype(numpy.int32)


    t1 = time.monotonic()
    if my_ueg.vcut:
        coulomb_ov = numpy.empty((ekins_o.size, ekins_v.size))
        mylib.get_coulomb_ov_cut(
            coulomb_ov.ctypes.data_as(ctypes.c_void_p),
            g_o.ctypes.data_as(ctypes.c_void_p),
            g_v.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ekins_o.size),
            ctypes.c_int(ekins_v.size),
            ctypes.c_double(my_ueg._length),
            ctypes.c_double(my_ueg._volume),
        )
    else:
        coulomb_ov = numpy.empty((ekins_o.size, ekins_v.size))
        mylib.get_coulomb_ov_nocut(
            coulomb_ov.ctypes.data_as(ctypes.c_void_p),
            g_o.ctypes.data_as(ctypes.c_void_p),
            g_v.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ekins_o.size),
            ctypes.c_int(ekins_v.size),
            ctypes.c_double(my_ueg._length),
            ctypes.c_double(my_ueg._volume),
        )
    eigk = -1 * get_k_potential(my_ueg, occ_gvecs, gvecs)
    t2 = time.monotonic()

    eigk_o = eigk[:my_ueg.nocc]
    eigk_v = eigk[my_ueg.nocc:]
    if settings["h0"] == "fock":
        eig_o = ekins_o + eigk_o
        eig_v = ekins_v + eigk_v
    elif settings["h0"] == "kinetic":
        eig_o = ekins_o.copy()
        eig_v = ekins_v.copy()
    else:
        raise ValueError("h0 must be 'fock' or 'kinetic'")
    mylib.get_ecorr_kappa_mp2.restype = ctypes.c_double
    assert coulomb_ov.flags.c_contiguous
    assert coulomb_ov.shape == (eig_o.size, eig_v.size)

    t3 = time.monotonic()
    if STYLE == "lambda":
        res_shape = (3,)
        if "gap_model" in settings["method"]:
            KAPPA = 0
            method = settings["method"]
            gap_model = method["gap_model"]
            from pyscf.acmp.ac_interpolators import ArtificialGapCalculator
            agc = ArtificialGapCalculator(mode="E", gap_model=gap_model)
            if agc.requires_mp2_mat:
                new_settings = settings.copy()
                new_settings["method"] = {"name": "ac"}
                wmat = run_ueg_calc(**new_settings)[-1]
            else:
                wmat = None
            df_codes = method["df_codes"]
            rho = 3.0 / (4 * numpy.pi * rs**3)
            if "omega_code" in method:
                omega = method["omega_code"]
            else:
                from pyscf.acmp.mp2_numint import lda_plasma_frequency
                omega = ("LDA", lda_plasma_frequency)
            df_codes = df_codes + [omega]
            rhovec = ueg.mgga_rho_vector(rho * numpy.ones_like(eig_o))
            w_list = [] if wmat is None else [wmat]
            for code in df_codes:
                if isinstance(code, tuple):
                    assert len(code) == 2
                    dtype, dffunc = code
                    if dtype == "LDA":
                        res = dffunc(rhovec[0])
                    elif dtype == "GGA":
                        res = dffunc(rhovec[:4])
                    else:
                        res = dffunc(rhovec[:])
                else:
                    if code == "HF":
                        res = 0.5 * eigk_o
                    else:
                        raise ValueError
                w_list.append(res)
            gaps = agc.compute_artificial_gap(w_list)
        else:
            KAPPA = settings["method"].get("param", 1.1)
            gaps = 0.5 * KAPPA * numpy.ones_like(eig_o)
        gaps = gaps.ctypes.data_as(ctypes.c_void_p)
    elif STYLE == "kappa":
        res_shape = (3,)
        KAPPA = settings["method"].get("param", 1.1)
        gaps = None
    elif STYLE == "ac":
        res_shape = (len(eig_o),)
    res = numpy.empty(res_shape, dtype=numpy.float64)
    args = [
        ctypes.c_int(eig_o.size),
        g_o.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(eig_v.size),
        g_v.ctypes.data_as(ctypes.c_void_p),
        # ctypes.c_double(KAPPA),
        coulomb_ov.ctypes.data_as(ctypes.c_void_p),
        eig_o.ctypes.data_as(ctypes.c_void_p),
        eig_v.ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        # gaps,
    ]
    if STYLE in ["lambda", "kappa"]:
        args.insert(4, ctypes.c_double(KAPPA))
        args.insert(len(args), gaps)
        fn = mylib.ecorr_kappa_mp2
    else:
        fn = mylib.ecorr_mp2_vec
    fn(*args)
    t4 = time.monotonic()
    print("TIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    rho = 3.0 / (4 * numpy.pi * rs**3)
    TEX = 0.3 * (3 * numpy.pi**2)**(2.0 / 3) * rho**(2.0 / 3)
    KEX = -0.75 * (3.0 / numpy.pi)**(1.0 / 3) * rho**(1.0 / 3)
    print("EXACT", TEX, KEX)

    if STYLE == "ac":
        e_kin = ekins_o
        e_exch = 0.5 * eigk_o
        dens = rho * numpy.ones_like(ekins_o)
        return numpy.stack([e_kin, e_exch, dens, res])
    else:
        e_kin = numpy.mean(ekins_o)
        e_exch = 0.5 * numpy.mean(eigk_o)
        return numpy.append([e_kin, e_exch], [res[-1] / nelec])


def post_process(method, ueg_result):
    if method["name"] in ["kappa", "lambda"]:
        return ueg_result[-1]
    elif method["name"] == "ac":
        aci = method["aci"]
        df_codes = method["df_codes"] + [method["silim"]]
        w_list = [ueg_result[-1], ueg_result[1]]
        rhovec = ueg.mgga_rho_vector(ueg_result[2])
        for dtype, dffunc in df_codes:
            if dtype == "LDA":
                res = dffunc(rhovec[0])
            elif dtype == "GGA":
                res = dffunc(rhovec[:4])
            else:
                res = dffunc(rhovec[:])
            w_list.append(res)
        w_list[-1] *= -1
        w_list[0] *= -1
        print("SUMS", [w.mean() for w in w_list])
        ec = aci([numpy.diag(w) for w in w_list])
        print("ECORR TOTAL", 2 * ec)
        return ec / w_list[0].size
    else:
        raise ValueError("Unsupported method")


def main():
    settings_file = sys.argv[1]
    with open(settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.CLoader)

    print("SETTINGS")
    print(yaml.dump(settings))
    KAPPA = settings["param"]
    STYLE = settings["style"]
    nelec_index = settings["nelec_index"]
    nelec = 2 * MAGIC_NUMBERS[nelec_index]
    rs = settings["ws_radius"]
    nbass = numpy.array(MAGIC_NUMBERS[nelec_index + 20 : : 10])
    nbass = nbass[nbass < settings["max_nbas"]]
    nbass = nbass[-settings["nsamp"]:]
    assert nelec < 2 * nbass[0]
    print("NELEC", nelec)

    energies = list()
    for nbas in nbass:
        energies.append(run_ueg_calc(nelec, nbas, rs, settings))

    for nbas, energy in zip(nbass, energies):
        print(nbas, energy)

    energies = numpy.array(energies)
    resdir = f"res_{STYLE}_{KAPPA}_{rs:.2f}_{nelec_index}"
    os.makedirs(resdir, exist_ok=True)
    numpy.save(f"{resdir}/nbas.npy", nbass)
    numpy.save(f"{resdir}/ekin.npy", energies[:, 0])
    numpy.save(f"{resdir}/exx.npy", energies[:, 1])
    numpy.save(f"{resdir}/ecd.npy", energies[:, 2])
    numpy.save(f"{resdir}/ecx.npy", energies[:, 3])
    numpy.save(f"{resdir}/ecorr.npy", energies[:, 4])


if __name__ == '__main__':
    main()

