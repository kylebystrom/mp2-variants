import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import (NBINS, _scale_ao_sparse,
                              _dot_ao_ao_sparse, _tau_dot_sparse)


CFC = 0.3 * (3 * numpy.pi**2)**(2.0 / 3)


def lda_plasma_frequency(rho, prefac=None):
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    wp2 = 4 * numpy.pi * rho * (1 + 0.029 / rho**(1.0 / 3))
    return prefac * numpy.sqrt(wp2)


def gga_plasma_frequency(rho, prefac=None):
    cond = rho[0] < 1e-9
    rho[0, cond] = 1e-9
    if prefac is None:
        prefac = numpy.float64(2)**-0.5
    wp2 = 4 * numpy.pi * rho[0] * (1 + 0.029 / rho[0]**(1.0 / 3))
    sigma = numpy.einsum("xg,xg->g", rho[1:4], rho[1:4])
    x = sigma / rho[0]**(8.0 / 3)
    wp2 /= 1 + 0.4 * x
    return prefac * numpy.sqrt(wp2)


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
    "PLASMA_GGA_WP": ("GGA", gga_plasma_frequency),
    "PLASMA_MGGA_WP": ("MGGA", mgga_plasma_frequency2),
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
            return super().eval_xc_eff(xc_code, rho, deriv=deriv, omega=omega,
                                       xctype=xctype, verbose=verbose)


class MP2NumInt(MP2NumIntMixin, numint.NumInt):
    nr_rmp2 = nr_rmp2

    nr_ump2 = nr_ump2
