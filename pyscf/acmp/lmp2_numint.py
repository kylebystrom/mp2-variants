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
    if xc_code == "LDA_WP":
        xctype = 'LDA'  # TODO detect xctype
    elif xc_code == "GGA_WP":
        xctype = 'GGA'
    elif xc_code == "MGGA_WP":
        xctype = 'MGGA'
    else:
        raise NotImplementedError
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
                omega = ni.get_artificial_gap(xc_code, rho, xctype=xctype)
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
    if xc_code == "LDA_WP":
        xctype = 'LDA'  # TODO detect xctype
    elif xc_code == "GGA_WP":
        xctype = 'GGA'
    elif xc_code == "MGGA_WP":
        xctype = 'MGGA'
    else:
        raise NotImplementedError
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
                omega = ni.get_artificial_gap(xc_code, rho[0] + rho[1], xctype=xctype)
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


class LMP2NumInt(numint.NumInt):

    nr_rmp2 = nr_rmp2

    nr_ump2 = nr_ump2

    def get_artificial_gap(self, xc_code, rho, xctype='LDA'):
        if xc_code == "LDA_WP":
            assert xctype == "LDA"
            return lda_plasma_frequency(rho)
        elif xc_code == "GGA_WP":
            assert xctype == "GGA"
            return gga_plasma_frequency(rho)
        elif xc_code == "MGGA_WP":
            assert xctype == "MGGA"
            return mgga_plasma_frequency2(rho)
        else:
            raise NotImplementedError
