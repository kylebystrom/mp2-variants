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


def get_k_potential(my_ueg, occ_gvecs, other_gvecs, occs=None):
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
    if not my_ueg.vcut:
        # args.append(ctypes.c_double(my_ueg.madelung))
        args.append(ctypes.c_double(0))
    if occs is None:
        if my_ueg.vcut:
            mylib.get_vk_vcut(*args)
        else:
            mylib.get_vk_novcut(*args)
    else:
        args.append(occs.ctypes.data_as(ctypes.c_void_p))
        if my_ueg.vcut:
            mylib.get_vk_vcut_ft(*args)
        else:
            mylib.get_vk_novcut_ft(*args)
    return vk


def get_ks_exx(my_ueg, occ_gvecs, occs):
    occ_gvecs = occ_gvecs.astype(numpy.int32)
    assert occ_gvecs.shape[-1] == 3
    assert occ_gvecs.flags.c_contiguous
    nocc = occ_gvecs.size // 3
    vk = numpy.empty(nocc, dtype=numpy.float64)
    args = [
        vk.ctypes.data_as(ctypes.c_void_p),
        occ_gvecs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nocc),
        ctypes.c_double(my_ueg._length),
        ctypes.c_double(my_ueg._volume),
        #ctypes.c_double(my_ueg.madelung),
        ctypes.c_double(0),
        occs.ctypes.data_as(ctypes.c_void_p),
    ]
    mylib.get_vk_ks_ft(*args)
    return -1 * vk.sum()


def get_e0(occs, nocc, eig_o):
    """
    eig_o is the zeroth-order Hamiltonian of the occupied orbitals
    """
    occs = occs[:nocc]
    n0 = occs.sum()
    return (eig_o * occs).sum() / n0


def get_gp1_v1(occs, nocc, eig_o, nelec0, beta, nbas, rs0, occ_tol,
               occ_gvecs, other_gvecs, apply_ks_dv):
    t0 = time.monotonic()
    tmp_ueg = ueg.LiteFTUEG(
        nelec0, beta, nbas, rs0, verbose=False, occ_tol=occ_tol
    )
    tmp_ueg.occs = occs
    tmp_ueg.nocc = nocc
    occs = occs[:nocc]
    n0 = occs.sum()
    t1 = time.monotonic()
    hcore = kin(tmp_ueg, occ_gvecs)
    t2 = time.monotonic()
    # veff = numpy.diag(tmp_ueg.get_veff())[:nocc]
    veff = -1 * get_k_potential(tmp_ueg, occ_gvecs, occ_gvecs, occs)[:nocc]
    exx_hf = (0.5 * occs * veff).sum() / n0
    exx_ks = get_ks_exx(tmp_ueg, occ_gvecs, occs) / n0

    t3 = time.monotonic()
    v1 = (hcore - eig_o)
    # NOTE account for Madelung
    v1 -= 0.5 * tmp_ueg.madelung
    e1 = ((v1 + 0.5 * veff) * occs).sum() / n0
    # e1 -= 0.5 * tmp_ueg.madelung * (occs**2).sum() / n0
    v1 += veff
    if apply_ks_dv:
        dndu = 2 * beta * occs * (1 - occs)
        dv_ks = (v1 * dndu[:nocc]).sum() / (dndu[:nocc].sum() + 1e-16)
        v1 -= dv_ks
    t4 = time.monotonic()
    # print("__TIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    return e1, v1, exx_hf, exx_ks


def get_gp2(STYLE, method, nelec0, beta, rs0, nbas, occs, nocc, nvir,
            eig_o, eig_v, g_o, g_v, coulomb_ov,
            v1=None):
    t0 = time.monotonic()
    if STYLE == "lambda":
        raise NotImplementedError
    elif STYLE == "kappa":
        raise NotImplementedError
    elif STYLE == "ac":
        res_shape = (len(eig_o),)
    res = numpy.zeros(res_shape, dtype=numpy.float64)
    f_o = occs[:nocc].copy()
    fm_v = 1 - occs[-nvir:]
    assert eig_o.size == g_o.shape[0] == f_o.size
    assert eig_v.size == g_v.shape[0] == fm_v.size
    assert coulomb_ov.shape == (eig_o.size, eig_v.size)
    for arr in [eig_o, f_o, fm_v, eig_v, eig_o, coulomb_ov, res]:
        assert arr.flags.c_contiguous
        assert arr.dtype == numpy.float64
    for arr in [g_o, g_v]:
        assert arr.flags.c_contiguous
        assert arr.dtype == numpy.int32
    args = [
        ctypes.c_int(eig_o.size),
        g_o.ctypes.data_as(ctypes.c_void_p),
        f_o.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(eig_v.size),
        g_v.ctypes.data_as(ctypes.c_void_p),
        fm_v.ctypes.data_as(ctypes.c_void_p),
        coulomb_ov.ctypes.data_as(ctypes.c_void_p),
        eig_o.ctypes.data_as(ctypes.c_void_p),
        eig_v.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(beta),
        res.ctypes.data_as(ctypes.c_void_p),
    ]
    t1 = time.monotonic()
    mylib.ecorr_mp2_vec_ft(*args)
    t2 = time.monotonic()
    if v1 is not None:
        vii = -2 * beta * v1 * v1 * (1 - occs[:nocc])
        res[:] += vii
    if STYLE == "ac":
        """
        my_ueg = ueg.FTUEG(nelec0, beta, nbas, rs0, verbose=False, occ_tol=0)
        my_ueg.occs = occs
        my_ueg.nocc = nocc
        #eigk0 = 0.5 * numpy.diag(my_ueg.get_veff())
        eigk = -0.5 * get_k_potential(my_ueg, g_o, g_o, occs)[:nocc]
        #print(eigk)
        #print(eigk0)
        res = post_process_ac(my_ueg, method, eigk, res)
        """
        t3 = time.monotonic()
        my_ueg = ueg.LiteFTUEG(nelec0, beta, nbas, rs0, verbose=False, occ_tol=0)
        my_ueg.occs = occs
        my_ueg.nocc = nocc
        eigk = -0.5 * get_k_potential(my_ueg, g_o, g_o, occs)[:nocc]
        if True:
            eigk[:] -= 0.5 * my_ueg.madelung * occs[:nocc]
        t4 = time.monotonic()
        res = post_process_ac(my_ueg, method, eigk, res, g_o=g_o)
        t5 = time.monotonic()
        print("__TIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    return res


def run_ueg_calc(**settings):
    t0 = time.monotonic()
    fill_settings_(settings)
    STYLE = settings["method"].get("name", "kappa")
    if "nelec" in settings:
        nelec = settings.pop("nelec")
    else:
        nelec_index = settings["nelec_index"]
        nelec = 2 * MAGIC_NUMBERS[nelec_index]

    nbas = MAGIC_NUMBERS[settings["nbas_index"]]
    rs = settings["ws_radius"]
    beta = settings["beta"]
    with_singles = settings.get("with_singles", True)
    occ_tol = settings.get("occ_tol", 0)
    ecorr_method = settings.get("ecorr_method", "zeroth_order")
    if ecorr_method not in ["zeroth_order", "finite_difference"]:
        raise NotImplementedError
    particle_fix = settings.get("particle_fix", None)
    if particle_fix not in [None, "dv2"]:
        raise NotImplementedError

    print("WS RADIUS", rs)
    my_ueg = ueg.FTUEG(nelec, beta, nbas, rs, verbose=False, occ_tol=occ_tol)
    if nbas != my_ueg.nbas:
        print("nbasis = %d is not a magic number. It has been increased to %d." % (nbas, my_ueg.nbas))
        nbas = my_ueg.nbas
    my_ueg.vcut = settings["vcut"]
    gvecs = numpy.asarray(my_ueg.rgvecs, order="C", dtype=numpy.float64)

    my_ueg.run_occ_scf(settings["h0"])
    nocc = my_ueg.nocc
    nvir = numpy.sum(1 - my_ueg.occs > my_ueg.occ_tol)

    occ_gvecs = gvecs[:nocc]
    vir_gvecs = gvecs[-nvir:]
    ekins_o = kin(my_ueg, occ_gvecs)
    ekins_v = kin(my_ueg, vir_gvecs)
    kin_en = (ekins_o * my_ueg.occs[:nocc]).sum() / my_ueg.occs[:nocc].sum()
    g_o = occ_gvecs.astype(numpy.int32)
    g_v = vir_gvecs.astype(numpy.int32)

    coulomb_ov = numpy.empty((ekins_o.size, ekins_v.size))
    args = [
        coulomb_ov.ctypes.data_as(ctypes.c_void_p),
        g_o.ctypes.data_as(ctypes.c_void_p),
        g_v.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ekins_o.size),
        ctypes.c_int(ekins_v.size),
        ctypes.c_double(my_ueg._length),
        ctypes.c_double(my_ueg._volume),
        # ctypes.c_double(my_ueg.madelung),
        ctypes.c_double(0),
    ]
    assert not my_ueg.vcut
    mylib.get_coulomb_ov_nocut_ft(*args)
    eigk = -1 * get_k_potential(my_ueg, occ_gvecs, gvecs, occs=my_ueg.occs)
    t1 = time.monotonic()

    eigk_o = eigk[:nocc]
    eigk_v = eigk[-nvir:]
    if settings["h0"] == "fock":
        eig_o = ekins_o + eigk_o
        eig_v = ekins_v + eigk_v
    elif settings["h0"] == "kinetic":
        eig_o = ekins_o.copy()
        eig_v = ekins_v.copy()
    elif settings["h0"] == "ks":
        eig_o = ekins_o + my_ueg.vxc_ks()
        eig_v = ekins_v + my_ueg.vxc_ks()
    else:
        raise ValueError("h0 must be 'fock', 'ks', or 'kinetic'")
    if nocc < nbas:
        eig_full = numpy.ascontiguousarray(
            numpy.append(eig_o, eig_v[-(nbas - nocc):])
        )
    else:
        eig_full = eig_o
    assert eig_full.size == nbas
    assert coulomb_ov.flags.c_contiguous
    assert coulomb_ov.shape == (eig_o.size, eig_v.size)

    t2 = time.monotonic()
    apply_ks_dv = (particle_fix == "dv2")
    e0 = get_e0(my_ueg.occs, nocc, eig_o)
    # e0 -= 0.5 * my_ueg.madelung  # TODO remove
    gp1, v1, exx_hf, exx_ks = get_gp1_v1(my_ueg.occs, nocc, eig_o,
                                         nelec, beta, nbas, rs, occ_tol,
                                         occ_gvecs, gvecs, apply_ks_dv)
    t3 = time.monotonic()
    dv = v1 if with_singles else None
    method = settings["method"]
    gp2 = get_gp2(STYLE, method, nelec, beta, rs, nbas,
                  my_ueg.occs, nocc, nvir,
                  eig_o, eig_v, g_o, g_v, coulomb_ov,
                  v1=dv)
    t4 = time.monotonic()

    if ecorr_method == "finite_difference":
        from pyscf.scf.addons import _fermi_smearing_occ

        dbeta = beta * 0.00001
        ress = []
        for _beta in [beta + 0.5 * dbeta, beta - 0.5 * dbeta]:
            _occs = _fermi_smearing_occ(my_ueg.mu, eig_full, 1.0 / _beta)
            _gp1, _v1, _exx_hf, _exx_ks = get_gp1_v1(
                _occs, nocc, eig_o, nelec, _beta,
                nbas, rs, occ_tol, occ_gvecs, gvecs, apply_ks_dv
            )
            _dv = _v1 if with_singles else None
            _gp2 = get_gp2(STYLE, method, nelec, _beta, rs, nbas,
                           _occs, nocc, nvir,
                           eig_o, eig_v, g_o, g_v, coulomb_ov,
                           v1=_dv)
            # The returned energies are per particle, but the number
            # of particles changes when beta changes and this is
            # important to account for in the finite difference
            # derivative
            ratio = 2 * _occs.sum() / nelec
            _gp1 *= ratio
            _gp2 *= ratio
            _exx_hf *= ratio
            _exx_ks *= ratio
            ress.append((_gp1, _gp2, _exx_hf, _exx_ks))
        e1 = gp1 + beta * (ress[0][0] - ress[1][0]) / dbeta
        e2 = gp2 + beta * (ress[0][1] - ress[1][1]) / dbeta
        dexx_hf = beta * (ress[0][2] - ress[1][2]) / dbeta
        dexx_ks = beta * (ress[0][3] - ress[1][3]) / dbeta
    else:
        e1 = gp1
        e2 = gp2
        dexx_hf = 0
        dexx_ks = 0
    t5 = time.monotonic()
    print("FT TIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)

    if STYLE == "ac":
        clipped_occs = numpy.clip(my_ueg.occs, 1e-200, 1)
        clipped_1occs = numpy.clip(1 - my_ueg.occs, 1e-200, 1)
        f0 = clipped_occs * numpy.log(clipped_occs)
        f0 += clipped_1occs * numpy.log(clipped_1occs)
        f0 = e0 + f0.sum() / (beta * my_ueg.occs.sum())
        e_free = (f0 + gp1 + gp2)
        e_tot = e0 + e1 + e2
        e_zero = 0.5 * (e_free + e_tot)
        e0_zero = 0.5 * (e0 + f0)
        e1_zero = e0_zero + 0.5 * (e1 + gp1)
        mterm1 = my_ueg.madelung * (my_ueg.occs**2).sum() / my_ueg.occs.sum()
        return numpy.array([e0, e1, e2, e_tot, e_free, e_zero,
                            kin_en, e0_zero, e1_zero, my_ueg.madelung,
                            exx_hf, exx_ks, dexx_hf, dexx_ks,
                            mterm1])
        """
        e_free = (e0 + gp1 + gp2)
        e_tot = e0 + e1 + e2
        e_zero = 0.5 * (e_free + e_tot)
        return numpy.array([e0, e1, e2, e_tot, e_free, e_zero])
        """
    else:
        raise NotImplementedError


def post_process_ac(my_ueg, method, w0, w0p, g_o=None):
    aci = method["aci"]
    df_codes = method["df_codes"] + [method["silim"]]
    w_list = [w0p.copy(), w0.copy()]
    rhovec = my_ueg.mgga_rho_vector(gvecs=g_o)
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
    w_list[1] *= -1
    nocc = my_ueg.nocc
    occs = my_ueg.occs[:nocc]
    for i, w in enumerate(w_list[2:]):
        w_list[i + 2] = w * numpy.ones_like(w_list[0])
    w_list = [w[:nocc] for w in w_list]
    ec = aci([numpy.diag(w) for w in w_list], occs=occs)
    ec = ec / occs.sum()
    return ec
