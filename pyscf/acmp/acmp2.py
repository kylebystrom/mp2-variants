#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
kappa RMP2
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp.mp2 import MP2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    exx = -0.5 * mp._scf.get_k()
    exx_xo = lib.dot(exx, mo_coeff[:, :nocc])
    exx_oo = lib.dot(mo_coeff[:, :nocc].T, exx_xo)

    edi = numpy.zeros((nocc, nocc))
    exi = numpy.zeros((nocc, nocc))
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        # t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        ei = lib.direct_sum('jb+a->jba', eia, eia[i]) + 1e-10
        t2i = gi.conj() / ei
        edi[:] += lib.einsum('jab,kab->jk', t2i, gi) * 2
        exi[:] -= lib.einsum('jab,kba->jk', t2i, gi)
        if with_t2:
            t2[i] = t2i

    if True:
        exx_o = numpy.diag(exx_oo).real
        ecc_o = 2 * numpy.diag(edi + exi).real
        wr = ecc_o / exx_o
        w0 = ecc_o

        N = 4096
        alphas = numpy.linspace(0, 1, N + 1)[:-1] + 0.5 / N
        alphas = alphas[:, None]
        dalpha = 1.0 / N
        if 1:
            # interp = dalpha * alphas * w0 / (1 + alphas * wr)
            # interp = dalpha * alphas * w0 / (1 + alphas * wr) * 1 / (1 + alphas * numpy.sqrt(-w0 * wr))
            # interp = dalpha * alphas * w0 / (1 + 1.5 * (-alphas * w0 * wr)**0.5 + alphas * wr)
            # interp = dalpha * alphas * w0 / (1 + 1.5 * (-alphas**2 * w0 * wr)**0.25 + alphas * wr)
            # interp = 2 * dalpha * alphas * ecc_o / numpy.sqrt(1 + (alphas * 2 * ecc_o / exx_o)**2)
            bwrs = -alphas * alphas * w0 * wr
            awrs = alphas * wr
            winfs = -w0 / wr
            # interp = dalpha * alphas * w0 / ((1 + 0.582 * bwrs**0.25)**2 + alphas * wr)
            # wt = (1 + 0.5 * alphas) / (1 + 1.2 * bwrs**0.5)
            # interp = dalpha * alphas * w0 * wt / (1 + 1.14 * wt * awrs)
            # interp = dalpha * alphas * w0 / (numpy.exp(0.8 * bwrs**0.25 - 0.028 * awrs) + 1.25 * awrs)
            # a, b, c = 1.169, 1.346, 0.4201
            # wt = 1 + c * alphas / winfs
            # interp = dalpha * alphas * w0 * wt / (1 + b * bwrs**0.25 + a * awrs * wt)
            a, b, c = 1.136, 1.560, 0.2909
            a, b, c = 1.317, 1.285, 0.2690
            a, b, c = 1.116, 1.377, 0.5787
            a, b, c = 1.216, 1.177, 0.2569
            a, b, c = 1.246, 1.316, 0.3503
            a, b, c = 1.000, 1.318, 0.1789
            a, b, c = 3.159, 8.937, 0.1636
            # a, b, c = 5.300, 7.642, 0.05
            a, b, c = 0.1864, 1.479, 0.1685
            Alim = 1.0466
            # Alim = 1.0
            #wt = 1 + c * alphas / winfs
            #interp = dalpha * alphas * w0 * wt / (1 + wt**0.5 * b * bwrs**0.25 + a * awrs * wt)
            wt = 1 + c * alphas**0.5 / winfs + a * alphas**0.5 / winfs**0.5
            interp = dalpha * alphas * w0 * wt / (1 + b * bwrs**0.25 + Alim * awrs * wt)
            # interp = dalpha * alphas * w0 * wt / (1 + Alim * awrs * wt)
            
            # interp = dalpha * alphas * w0 * wt / (1 + a * awrs * wt)
            # interp = dalpha * alphas * w0 / (1 + awrs)
            # interp = dalpha * alphas * w0
            # interp = dalpha * alphas * w0 * wt / (1 + b * bwrs**0.5 + a * awrs * wt + (Alim * awrs * wt)**2)**0.5
            energy = interp.sum()
            print(w0, winfs, interp.sum(axis=0))
            print("SUMS", energy, exx_o.sum(), ecc_o.sum())
            print()
        else:
            a, b, c = 1.27, 0.0207, 1.963
            a, b, c = 1.175, 1, 1.359
            a, b, c = 1.05**2, 6.584, 6.047
            bwrs = -alphas**2 * wr * w0
            awrs = alphas * wr
            term = numpy.sqrt(c * bwrs / (1 + b * awrs**2) + (1 + a * awrs**2))
            wt = numpy.exp(-b * awrs)
            # term = 1 + c * bwrs**0.5 * wt + a * (1- wt) * awrs
            # term = 1 + c * bwrs**0.25 + a * awrs
            term = 1 + a * awrs
            # term = 1 + c * bwrs**0.6666666666 + a * awrs
            # term = ((1 + a * awrs)**2 + c * bwrs)**0.5
            # term = 1 + a * awrs
            interp = dalpha * alphas * w0 / term
            energy = interp.sum()
    else:
        w0 = 2 * (edi.real + exi.real)
        w0 = 0.5 * (w0 + w0.T)
        wr = numpy.linalg.solve(exx_oo.real, w0)
        wr = 0.5 * (wr + wr.T)
        w0 *= -1
        N = 4096
        alphas = numpy.linspace(0, 1, N + 1)[:-1] + 0.5 / N
        alphas = alphas[:, None]
        dalpha = 1.0 / N
        energy = 0
        idmat = numpy.identity(w0.shape[0])

        def matrix_pow(mat, mypow):
            eval, evec = numpy.linalg.eigh(mat)
            eval = numpy.maximum(eval, 0)
            eval = eval**mypow
            return (evec * eval).dot(evec.T)

        def matrix_exp(mat, expnt):
            eval, evec = numpy.linalg.eigh(mat)
            eval = numpy.exp(expnt * eval)
            return (evec * eval).dot(evec.T)

        for alpha in alphas:
            if 0:
                term = alpha * wr
            elif 0:
                a, b, c = 1.028, 2.972, 2.320
                a, b, c = 0.177, 0.500, 2.488
                a, b, c = 1.154, 0.428, 1.602
                term1 = matrix_pow(0.5 * (wr.dot(w0) + w0.dot(wr)), 0.5)
                term2 = matrix_pow(wr, 0.5)
                term3 = matrix_exp(wr, -b * alpha)
                term4 = term1.dot(term1)
                term4 = 0.5 * (term4 + term4.T)
                term = c * alpha * term4 + a * alpha * wr
                # term = a * alpha * wr
            else:
                eval1, evec1 = numpy.linalg.eigh(0.5 * (wr.dot(w0) + w0.dot(wr)))
                eval1 = numpy.maximum(eval1, 0)
                eval1 = numpy.sqrt(eval1)
                eval1 = numpy.sqrt(eval1)
                term1 = evec1.dot(eval1[:, None] * evec1.T)
                # term = 1.175 * alpha * wr
                term = 1.182 * alpha * wr
                # term += (1 + 0.528 * )
                # term = 1.36 * alpha**0.5 * term1 + 1.175 * alpha * wr
            term = alpha * numpy.linalg.solve(idmat + term, w0)
            energy -= dalpha * numpy.trace(term)

    edi = numpy.sum(numpy.diag(edi.real))
    exi = numpy.sum(numpy.diag(exi.real))
    emp2_ss = edi * 0.5 + exi
    emp2_os = edi * 0.5
    # emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    emp2 = lib.tag_array(energy, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    print(energy, emp2.real)

    return emp2.real, t2


class ACMP2(MP2):
    '''restricted kappa-MP2 with canonical HF
    '''

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, kappa=1.5):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.kappa = kappa

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('kappa-MP2 with kappa = %.3g', self.kappa)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E(SCS-%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot_scs, self.emp2_scs)
        log.info('E_corr(same-spin) = %.15g', self.e_corr_ss)
        log.info('E_corr(oppo-spin) = %.15g', self.e_corr_os)
        return self


ACRMP2 = ACMP2
