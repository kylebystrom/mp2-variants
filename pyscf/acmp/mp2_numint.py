import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import (NBINS, _scale_ao_sparse,
                              _dot_ao_ao_sparse, _tau_dot_sparse)
from pyscf.dft.libxc import eval_xc


CFC = 0.3 * (3 * numpy.pi**2)**(2.0 / 3)
LDAX_FACTOR = -3.0 / 4.0 * (3.0 / numpy.pi) ** (1.0 / 3)


def get_power_of_ws_radius_func(power, constant):
    pow3 = power / 3.0
    def _rs_function(rho):
        return constant * (3.0 / (4 * numpy.pi * rho))**pow3
    return ("LDA", _rs_function)


def _rs_function(rho, pow3):
    return (3.0 / (4 * numpy.pi * rho))**pow3


def ueg_hf_plasma_frequency(rho):
    wp2 = 4 * numpy.pi * rho * (1 + 0.029 / rho**(1.0 / 3))
    return numpy.sqrt(wp2)


def ueg_ks_plasma_frequency(rho):
    rs32 = _rs_function(rho, -0.5)
    rsi = _rs_function(rho, -0.33333333)
    return rs32 + 0.25 * rsi


def lda_plasma_helper(rho, prefac=None):
    if prefac is None:
        prefac = 1.0
    rsh = _rs_function(rho, -1.0 / 6)
    return prefac * (0.8 * rsh + 0.35)


def lda_plasma_frequency(rho, prefac=None):
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    return prefac * ueg_hf_plasma_frequency(rho)


def lda_ks_plasma_frequency(rho, prefac=None):
    if prefac is None:
        prefac = 1.0
    return prefac * ueg_ks_plasma_frequency(rho)


def gga_plasma_frequency(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    wp2 = 4 * numpy.pi * rho[0] * (1 + 0.029 / rho[0]**(1.0 / 3))
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    x = sigma / rho[0]**(8.0 / 3)
    wp2 /= 1 + 0.5 * x
    return prefac * numpy.sqrt(wp2)


def gga_ks_plasma_frequency(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = 1.0
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    x = sigma / rho[0]**(8.0 / 3)
    gga_fac = 1.0 / (1 + 0.35 * x)**0.5
    return prefac * ueg_ks_plasma_frequency(rho[0]) * gga_fac


def mgga_plasma_frequency(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    invm = (1 + 0.029 / rho[0]**(1.0 / 3))
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    tauw = sigma / (8 * rho[0])
    tau = rho[4]
    diff = numpy.maximum((tau - tauw) / CFC, 0)
    rho_eff = diff**0.6
    wp2 = 4 * numpy.pi * rho_eff * invm
    wp2[cond] = 0
    return prefac * numpy.sqrt(wp2)


def mgga_plasma_frequency2(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    invm = (1 + 0.029 / rho[0]**(1.0 / 3))
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    tauw = sigma / (8 * rho[0])
    tau = rho[4]
    diff = numpy.maximum((tau - tauw) / CFC, 0)
    rho_eff = diff / rho[0]**(2.0 / 3)
    wp2 = 4 * numpy.pi * rho_eff * invm
    wp2[cond] = 0
    return prefac * numpy.sqrt(wp2)


def mgga_plasma_frequency2(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = numpy.float64(3)**-0.5
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    tauw = sigma / (8 * rho[0])
    tau = rho[4]
    diff = numpy.maximum((tau - tauw) / CFC, 0)
    rho_eff = diff / rho[0]**(2.0 / 3)
    rs32 = _rs_function(rho_eff, -0.5)
    rsi = _rs_function(rho_eff, -0.33333333)
    wp = prefac * (rs32 + 0.5 * rsi)
    wp[cond] = 0
    return wp


def mgga_ks_plasma_frequency(rho, prefac=None):
    if prefac is None:
        prefac = 1.0
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    tauw = sigma / (8 * rho[0])
    tau = rho[4]
    tau0 = CFC * rho[0]**(5.0 / 3)
    chi = 2 * tau0**2
    chi /= tau0**2 + (tau - tauw)**2
    chi -= 1
    coeff = -0.2
    prefac *= 1 + coeff * chi
    x = sigma / rho[0]**(8.0 / 3)
    gga_fac = 1.0 / (1 + 0.4 * x)**0.5
    return prefac * ueg_ks_plasma_frequency(rho[0]) * gga_fac


def reduced_grad2(sigma, rho):
    return sigma / (4 * (3 * numpy.pi**2)**(2.0 / 3) * rho**(8.0 / 3))


def ef_kappa(mu, c, s2):
    return c / (1 + s2)


def wgea(mu, c, s2, sign=1):
    kappa = ef_kappa(mu, c, s2)
    return 1 + sign * (kappa - kappa / (1 + mu * s2 / kappa))


def gga_sce_limit(rho, prefac=None):
    pass


def lda_rho(rho, prefac=None):
    const = -1.44423075 * 0.3125**2 * 2 * CFC
    return const * rho


def mgga_hsq_sce_limit(rho, prefac=None):
    return mgga_hhh_sce_limit(rho)**2


def mgga_hhh_sce_limit(rho, prefac=None):
    return -0.3125 * numpy.sqrt(2 * rho[4] / (rho[0] + 1e-16))
    # -0.3125 * numpy.sqrt(2 * CFC * rho[0]**(5/3) / (rho[0] + 1e-8))


def mgga_hhh_sce_limit_chi(rho):
    return mgga_hhh_sce_limit(rho) * (1 - 0.5 * mgga_chi(rho))


def gga_hsq_sce_limit(rho, prefac=None):
    return gga_hhh_sce_limit(rho)**2


def gga_hhh_sce_limit(rho, prefac=None):
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    return -0.3125 * numpy.sqrt(sigma / (4 * rho[0]**2 + 1e-16))


"""
The next six functions are for computing the point-charge
plus continuum model of Seidl, Perdew, and Kurth
PRA 62, 012502 (2000).
"""

def lda_pc_model(rho):
    A = -0.9 * (4 * numpy.pi / 3)**(1.0 / 3)
    return A * rho**(1.0 / 3)


def gga_part_pc_model(rho):
    B = (3.0 / 350) * (4 * numpy.pi / 3)**(-1.0 / 3)
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    return B * sigma / (rho[0]**(7.0 / 3) + 1e-16)


def gga_full_pc_model(rho):
    return lda_pc_model(rho[0]) + gga_part_pc_model(rho)


def _gga_pch_term(lda, s2, mu, k):
    num = 1 + s2 * mu * (k + 1) / k
    den = 1 + s2 * mu / k
    return lda * num / den


def _gga_pch_term_v2(lda, s2, mu, k, fl=0):
    assert mu <= 0
    assert k <= 0
    mypow = 2
    #return lda * numpy.maximum(1 + mu * s2, 0)
    return lda * (fl + (1-fl) * numpy.exp(s2 * mu / (1 - fl) + s2**2 * k))
    num = 1.0
    den = 1 - s2 * mu / mypow
    return lda * num / den**mypow


def _gga_pch_term_v3(lda, s2, mu, k):
    assert mu >= 0 and k >= 0
    fac = 1 + k - k / (1 + mu * s2 / k)
    return lda * fac


def gga_pch_winf(rho):
    A = -0.9 * (4 * numpy.pi / 3)**(1.0 / 3)
    # mu = -3**(1.0 / 3) * (2 * numpy.pi)**(2.0 / 3) / 35
    sfac = 4 * (3 * numpy.pi**2)**(2.0 / 3)
    mu = (3.0 / 350) * (4 * numpy.pi / 3)**(-1.0 / 3) * sfac / A
    k = -7.11
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / sfac
    lda = A * rho[0]**(1.0 / 3)
    return _gga_pch_term(lda, s2, mu, k)


def gga_pch_winf_v2(rho):
    A = -0.9 * (4 * numpy.pi / 3)**(1.0 / 3)
    # mu = -3**(1.0 / 3) * (2 * numpy.pi)**(2.0 / 3) / 35
    sfac = 4 * (3 * numpy.pi**2)**(2.0 / 3)
    mu = (3.0 / 350) * (4 * numpy.pi / 3)**(-1.0 / 3) * sfac / A
    k = -7.11
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / sfac
    lda = A * rho[0]**(1.0 / 3)
    # return _gga_pch_term_v2(lda, s2, mu, 0.0, fl=0.35)
    return _gga_pch_term_v2(lda, s2, mu, 0.0, fl=0.5)
    # return _gga_pch_term_v3(lda, rho[0], s2, mu, fl=0.6)


def lda_pc_model_grad(rho):
    C = 0.5 * (3 * numpy.pi)**0.5
    return C * rho**0.5


def gga_part_pc_model_grad(rho):
    D = -0.02558
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    return D * sigma / (rho[0]**(13.0 / 6) + 1e-16)


def gga_full_pc_model_grad(rho):
    return lda_pc_model_grad(rho[0]) + gga_part_pc_model_grad(rho)


def gga_pch_winfp(rho):
    C = 0.5 * (3 * numpy.pi)**0.5
    mu = -0.7222
    sfac = 4 * (3 * numpy.pi**2)**(2.0 / 3)
    #mu = -0.02558 * sfac / C
    k = -99.11
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / sfac
    lda = C * rho[0]**0.5
    return _gga_pch_term(lda, s2, mu, k)


def gga_pch_winfp_v2(rho):
    A = -0.9 * (4 * numpy.pi / 3)**(1.0 / 3)
    C = 0.5 * (3 * numpy.pi)**0.5
    #return C / A * rho[0]**(1.0 / 6) * gga_pch_winf_v2(rho)
    mu = -0.7222
    sfac = 4 * (3 * numpy.pi**2)**(2.0 / 3)
    #mu = -0.02558 * sfac / C
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / sfac
    lda = C * rho[0]**0.5
    # return numpy.ones_like(lda) * 1e-6
    return _gga_pch_term_v2(lda, s2, mu, 0.0, fl=0.0)


def gga_pch_winf2_winfp_ratio(rho):
    A = -0.9 * (4 * numpy.pi / 3)**(1.0 / 3)
    C = 0.5 * (3 * numpy.pi)**0.5
    sfac = 4 * (3 * numpy.pi**2)**(2.0 / 3)
    mu = (3.0 / 350) * (4 * numpy.pi / 3)**(-1.0 / 3) * sfac / A
    mup = -0.7222
    mup = -0.638
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / sfac
    lda = A**2 / C * rho[0]**(1.0 / 6)
    # return lda * (1 + (2 * mu - mup) * s2)
    # return _gga_pch_term_v2(lda, s2, -0.5, 0.0, fl=0.2)
    return _gga_pch_term_v3(lda, s2, (2 * mu - mup), 0.2)


def mgga_ueg_sce_limit(rho, lda_const=1.44423075, prefac=None):
    dens = numpy.maximum(1e-8, rho[0])
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / 2**(2.0 / 3)
    s2 /= 4 * (3 * numpy.pi**2)**(2.0 / 3)
    return -lda_const * dens**(1.0 / 3)


def mgga_ueg_sce_limit_chi(rho):
    return mgga_ueg_sce_limit(rho) * mgga_chi(rho)


def mgga_sce_gea(rho, lda_const=1.44423075):
    dens = numpy.maximum(1e-8, rho[0])
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / 2**(2.0 / 3)
    s2 /= 4 * (3 * numpy.pi**2)**(2.0 / 3)
    const = 0.0053 * 4 * (3 * numpy.pi ** 2) ** (2.0 / 3) / lda_const
    return -lda_const * dens**(1.0 / 3) * (wgea(const, 20, s2, sign=-1) - 1)


# def mgga_sce_gea_chi(rho, lda_const=1.44423075)


def mgga_sce_limit(rho, prefac=None):
    rs_inv = ((4 * numpy.pi * rho[0]) / 3.0)**(1.0 / 3)
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    s2 = sigma / rho[0]**(8.0 / 3) / 2**(2.0 / 3)
    s2 /= 4 * (3 * numpy.pi**2)**(2.0 / 3)
    tauw = sigma / (8 * rho[0])
    tau = rho[4]
    tau0 = CFC * rho[0]**(5.0 / 3)
    # chi is 1 for 1e, 0 for UEG, -0 as alpha->inf
    mu = 0.21951
    kappa = 0.804
    # maxfac = 1.125
    # maxfac = 0.896
    # maxfac = -LDAX_FACTOR * (6.0 / (4 * numpy.pi))**(1.0 / 3)
    # maxfac = 1.05
    maxfac = 0.59
    gradfac = 0.59
    ldafac = 0.896
    mypow = 1
    chi = 2 * tau0**mypow
    chi /= tau0**mypow + numpy.maximum(tau - tauw, 0)**mypow
    chi -= 1
    fac = ldafac + (maxfac - ldafac) * chi
    fac += gradfac * (kappa - kappa / (1 + mu * s2 / kappa))
    fac *= rs_inv
    # dens = rho[0].copy()
    # dens[dens < 1e-6] = 0
    # return -0.3125 * numpy.sqrt(2 * tau / (rho[0] + 1e-8))
    # const = 0.0053 * 4 * (3 * numpy.pi ** 2) ** (2.0 / 3) / 1.44423
    # return -1.44423075 * dens**(1.0 / 3) * wgea(const, 20, s2, sign=-1)
    return -fac
    # return -2 * ldafac * rs_inv


def mgga_chi(rho):
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    tauw = sigma / (8 * rho[0] + 1e-8)
    tau = rho[4]
    tau0 = CFC * rho[0]**(5.0 / 3)
    taudiff = tau - tauw
    chi = 2 * taudiff * taudiff
    chi /= tau0 * tau0 + taudiff * taudiff + 1e-32
    return chi


def mgga_r2scan_strong_corr(rho, small_rho=1e-8):
    # scale the density down really small
    gamma = (rho[0] / small_rho)**(1.0 / 3)
    rho = rho.copy()
    # density is small_rho now
    rho[0] /= gamma**3
    rho[1:4] /= gamma**4
    rho[4] /= gamma**5
    exc = eval_xc("MGGA_X_R2SCAN,MGGA_C_R2SCAN", rho, deriv=0)[0]
    return exc * gamma


def nr_rmp2(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                omega = ni.eval_xc_eff(xc_code, rho, xctype=xctype,
                                       deriv=0)[0]
                if xctype == 'LDA':
                    den = rho * weight
                else:
                    den = rho[0] * weight
                nelec[i] += den.sum()
                wv = weight * omega
                excsum[i] += wv.sum()
                yield i, ao, mask, wv

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype in ['LDA', 'GGA', 'MGGA']:
        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            if ao.ndim == 3:
                ao = ao[0]
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def nr_ump2(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    dma, dmb = numint._format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset,nao,nao))

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)
                omega = ni.eval_xc_eff(xc_code, rho[0] + rho[1], xctype=xctype,
                                       deriv=0)[0]
                if xctype == 'LDA':
                    den_a = rho_a * weight
                    den_b = rho_b * weight
                else:
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += numpy.dot(den_a, omega)
                excsum[i] += numpy.dot(den_b, omega)
                wv = weight * omega
                yield i, ao, mask, wv

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype in ['LDA', 'GGA', 'MGGA']:
        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            if ao_deriv > 0:
                ao = ao[0]
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = numpy.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


PLASMA_FREQUENCY_MODELS = {
    "PLASMA_LDA_WP": ("LDA", lda_plasma_frequency),
    "PLASMA_GGA_WP": ("GGA", gga_ks_plasma_frequency),
    "PLASMA_MGGA_WP": ("MGGA", mgga_plasma_frequency),
}


class MP2NumIntMixin:
    def _xc_type(self, xc_code):
        if isinstance(xc_code, tuple):
            assert isinstance(xc_code[0], str)
            return xc_code[0]
        elif xc_code.startswith("PLASMA_"):
            return PLASMA_FREQUENCY_MODELS[xc_code][0]
        else:
            return self.libxc.xc_type(xc_code)

    def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None):
        if isinstance(xc_code, tuple):
            return xc_code[1](rho), None, None, None
        elif xc_code.startswith("PLASMA_"):
            return PLASMA_FREQUENCY_MODELS[xc_code][1](rho), None, None, None
        else:
            res = super().eval_xc_eff(xc_code, rho, deriv=deriv, omega=omega,
                                       xctype=xctype, verbose=verbose)
            return res


class MP2NumInt(MP2NumIntMixin, numint.NumInt):
    nr_rmp2 = nr_rmp2

    nr_ump2 = nr_ump2
