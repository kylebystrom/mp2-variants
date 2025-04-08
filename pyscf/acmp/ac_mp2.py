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
from pyscf.dft.gen_grid import Grids
from pyscf.acmp.mp2_numint import MP2NumInt

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def _get_ac_param_array(N):
    alphas = numpy.linspace(0, 1, N + 1)[:-1] + 0.5 / N
    alphas = alphas
    dalpha = 1.0 / N
    return alphas, dalpha


def _identity_like(m):
    return numpy.identity(m.shape[0])


def _acmp_matrix_pow(mat, mypow):
    eval, evec = numpy.linalg.eigh(mat)
    eval = numpy.maximum(eval, 0)
    eval = eval**mypow
    return (evec * eval).dot(evec.T)


def _acmp_matrix_exp(mat, expnt):
    eval, evec = numpy.linalg.eigh(mat)
    eval = numpy.exp(expnt * eval)
    return (evec * eval).dot(evec.T)


def get_acmp_si_limit(mp):
    if isinstance(mp.si_limit, str) and mp.si_limit == "HF":
        winf_xx = -0.5 * mp._scf.get_k()
    else:
        ni = MP2NumInt()
        grids = Grids(mp._scf.mol)
        grids.level = 3
        maxmem = mp._scf.mol.max_memory
        nelec, excsum, vmat = ni.nr_rmp2(mp._scf.mol, grids, mp.si_limit,
                                         mp._scf.make_rdm1(), relativity=0,
                                         hermi=1, max_memory=maxmem,
                                         verbose=None)
        winf_xx = vmat
    return winf_xx


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

    winf_xx = mp.get_acmp_si_limit()
    winf_xo = lib.dot(winf_xx, mo_coeff[:, :nocc])
    winf_oo = lib.dot(mo_coeff[:, :nocc].T, winf_xo)

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

    winf = -winf_oo.real
    w0 = -2 * (edi.real + exi.real)
    energy = mp.ac_interpolator(w0, winf)

    # TODO shouldn't set these to misleading values
    edi = energy
    exi = 0.0
    emp2_ss = edi * 0.5 + exi
    emp2_os = edi * 0.5
    emp2 = lib.tag_array(energy, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2


class _ACInterpolator():
    def __call__(self, w0, winf):
        raise NotImplementedError


class _NumACInterpolator(_ACInterpolator):
    def __init__(self, N, params):
        self._alphas, self._dalpha = _get_ac_param_array(N)
        self._params = params

    @property
    def dalpha(self):
        return self._dalpha

    @property
    def alphas(self):
        return self._alphas

    @property
    def params(self):
        return [p for p in self._params]


class _EigNumInterpolator(_NumACInterpolator):
    def interpolate(self, alphas, w0, wr):
        raise NotImplementedError

    def __call__(self, w0, winf):
        winf = numpy.diag(winf)
        w0 = numpy.diag(w0)
        wr = w0 / winf
        return -(self.dalpha * self.interpolate(self.alphas[:, None], w0, wr)).sum()
    

class BasicEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w0, wr):
        return alphas * w0 / (1 + alphas * wr)


class ScreenedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w0, wr):
        bwrs = alphas * alphas * w0 * wr
        awrs = alphas * wr
        return alphas * w0 / (1 + self.params[0] * bwrs**0.25 + awrs)
    

class ScreenedWeightedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w0, wr):
        a, b, c = self.params
        winfs = w0 / wr
        bwrs = alphas * alphas * w0 * wr
        awrs = alphas * wr
        wt = 1 + c * alphas**0.5 / winfs + a * alphas**0.5 / winfs**0.5
        return alphas * w0 * wt / (1 + b * bwrs**0.25 + awrs * wt)


class _MatNumInterpolator(_NumACInterpolator):
    def get_ac_term(self, alpha, idmat, w0, wr):
        raise NotImplementedError

    def __call__(self, w0, winf):
        w0 = 0.5 * (w0 + w0.T)
        wr = numpy.linalg.solve(winf, w0)
        wr = 0.5 * (wr + wr.T)
        energy = 0
        idmat = _identity_like(w0)
        for alpha in self.alphas:
            energy -= self.dalpha * numpy.trace(
                self.get_ac_term(alpha, idmat, w0, wr)
            )
        return energy
    

class BasicMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha, idmat, w0, wr):
        term = alpha * wr
        return alpha * numpy.linalg.solve(idmat + term, w0)


class ACMP2(MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.si_limit = "HF"
        self.ac_interpolator = None

    get_acmp_si_limit = get_acmp_si_limit

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('AC-MP2 with si_limit = %s', self.si_limit)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        return self


ACRMP2 = ACMP2
