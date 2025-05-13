from pyscf.pbc.dft import numint
from pyscf.acmp.mp2_numint import MP2NumIntMixin, PLASMA_FREQUENCY_MODELS
import numpy
from pyscf.pbc.lib.kpts import KPoints


def nr_rmp2(ni, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
            kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_rks for functional {xc_code}')

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc
        deriv = 0
        vmat = [0]*nset
        v_hermi = 1  # the output matrix must be hermitian
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype).real
                omega = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[0]
                if xctype == 'LDA':
                    den = rho*weight
                else:
                    den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += den.dot(omega)
                wv = weight * omega
                wv = wv[None, :]
                if xctype != 'LDA':
                    if isinstance(ao_k1, list):
                        ao_k1 = [ao[0] for ao in ao_k1]
                    else:
                        ao_k1 = ao_k1[0]
                vmat[i] += ni._vxc_mat(cell, ao_k1, wv, mask, 'LDA',
                                       shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack(vmat)
        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat[0]
    else:
        nelec = excsum = vmat = 0
    return nelec, excsum, vmat


def nr_ump2(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=1,
            kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_uks for functional {xc_code}')

    dma, dmb = numint._format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi, False)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi, False)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    if xctype in ('LDA', 'GGA', 'MGGA'):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc
        deriv = 0
        vmata = [0]*nset
        vmatb = [0]*nset
        v_hermi = 1  # the output matrix must be hermitian
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype).real
                rho_b = make_rhob(i, ao_k2, mask, xctype).real
                rho = (rho_a, rho_b)
                omega = ni.eval_xc_eff(xc_code, rho[0] + rho[1], deriv,
                                       xctype=xctype)[0]
                if xctype == 'LDA':
                    dena = rho_a * weight
                    denb = rho_b * weight
                else:
                    dena = rho_a[0] * weight
                    denb = rho_b[0] * weight
                nelec[0,i] += dena.sum()
                nelec[1,i] += denb.sum()
                excsum[i] += dena.dot(omega)
                excsum[i] += denb.dot(omega)
                wv = weight * omega
                # TODO either there will eventually be spin-dependence here,
                # or the _vxcmat should only be called one for efficiency's sake.
                wv = numpy.stack([wv, wv])
                if xctype != 'LDA':
                    if isinstance(ao_k1, list):
                        ao_k1 = [ao[0] for ao in ao_k1]
                    else:
                        ao_k1 = ao_k1[0]
                vmata[i] += ni._vxc_mat(cell, ao_k1, wv[0], mask, 'LDA',
                                        shls_slice, ao_loc, v_hermi)
                vmatb[i] += ni._vxc_mat(cell, ao_k1, wv[1], mask, 'LDA',
                                        shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack([vmata, vmatb])
        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            nelec = nelec[:,0]
            excsum = excsum[0]
            vmat = vmat[:,0]
    else:
        nelec = excsum = vmat = 0
    return nelec, excsum, vmat


class MP2NumInt(MP2NumIntMixin, numint.NumInt):
    nr_rmp2 = nr_rmp2

    nr_ump2 = nr_ump2


class KMP2NumInt(MP2NumIntMixin, numint.KNumInt):
    nr_rmp2 = nr_rmp2

    nr_ump2 = nr_ump2
