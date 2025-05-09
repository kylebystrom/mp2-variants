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
from pyscf.acmp.mp2_numint import MP2NumInt, get_power_of_ws_radius_func

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
    exx_xx = -0.5 * mp._scf.get_k()
    if isinstance(mp.si_limit, str) and mp.si_limit == "HF":
        winf_xx = exx_xx.copy()
    else:
        ni = MP2NumInt()
        grids = Grids(mp._scf.mol)
        grids.level = 3
        maxmem = mp._scf.mol.max_memory
        nelec, excsum, vmat = ni.nr_rmp2(mp._scf.mol, grids, mp.si_limit,
                                         mp._scf.make_rdm1(), relativity=0,
                                         hermi=1, max_memory=maxmem,
                                         verbose=None)
        # Reference strong correlation limit to exx
        winf_xx = 2 * vmat - exx_xx
        # winf_xx = vmat
    return exx_xx, winf_xx


def get_artificial_gap(mp):
    ni = MP2NumInt()
    grids = Grids(mp._scf.mol)
    grids.level = 3
    maxmem = mp._scf.mol.max_memory
    nelec, excsum, vmat = ni.nr_rmp2(mp._scf.mol, grids, mp.agap_model,
                                     mp._scf.make_rdm1(), relativity=0,
                                     hermi=1, max_memory=maxmem,
                                     verbose=None)
    return vmat


def _acmp_ao2mo(mat_xx, coeff):
    mat_xo = lib.dot(mat_xx, coeff)
    return lib.dot(coeff.T, mat_xo)


def _get_corrected_winf(winf, exx):
    # Make sure winf_oo is positive
    d = 0.04
    prod = winf.dot(winf) + d * exx.dot(exx)
    return _acmp_matrix_pow(prod, 0.5)


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
    _eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    exx_xx, winf_xx = mp.get_acmp_si_limit()
    agap_xx = mp.get_artificial_gap()
    occ_coeff = mo_coeff[:, :nocc]
    exx_oo = _acmp_ao2mo(exx_xx, occ_coeff)
    winf_oo = _acmp_ao2mo(winf_xx, occ_coeff)
    agap_oo = _acmp_ao2mo(agap_xx, occ_coeff)
    numw = 3
    # TODO make this orbital-invariant
    print("AGAP", agap_oo, _eia[:, 0])
    eia_list = [
        _eia,
        _eia - 0.5 * numpy.diag(agap_oo)[:, None],
        _eia - numpy.diag(agap_oo)[:, None],
    ]

    w_list = [numpy.zeros((nocc, nocc)) for _ in range(numw)]
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        # t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        eia = eia_list[0]
        ei = lib.direct_sum('jb+a->jba', eia, eia[i]) + 1e-8
        t2i = gi.conj() / ei
        w_list[0][:] += lib.einsum('jab,kab->jk', t2i, gi).real * 2
        w_list[0][:] -= lib.einsum('jab,kba->jk', t2i, gi).real

        # ei = ei * ei
        # ei = -ei * numpy.log(-ei)
        # t2i = gi.conj() / ei * numpy.log(-ei)
        # t2i *= numpy.log(-ei)
        t2i /= ei
        w_list[2][:] += lib.einsum('jab,kab->jk', t2i, gi).real * 2
        w_list[2][:] -= lib.einsum('jab,kba->jk', t2i, gi).real

        #for w, eia in zip(w_list, eia_list):
        #    ei = lib.direct_sum('jb+a->jba', eia, eia[i]) + 1e-8
        #    t2i = gi.conj() / ei
        #    w[:] += lib.einsum('jab,kab->jk', t2i, gi).real * 2
        #    w[:] -= lib.einsum('jab,kba->jk', t2i, gi).real

    for w in w_list:
        w[:] *= -2
    exx = -exx_oo.real
    winf = -winf_oo.real
    winf = _get_corrected_winf(winf, exx)
    w_list.append(winf)
    w_list[1] = agap_oo
    print("WLIST", w_list)
    numpy.save("w_list.npy", numpy.array(w_list))
    energy = mp.ac_interpolator(w_list)

    # TODO shouldn't set these to misleading values
    edi = energy
    exi = 0.0
    emp2_ss = edi * 0.5 + exi
    emp2_os = edi * 0.5
    emp2 = lib.tag_array(energy, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2


class _ACInterpolator():
    def __call__(self, w_list):
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
    def interpolate(self, alphas, w_list):
        raise NotImplementedError

    def __call__(self, w_list):
        w_list = [numpy.diag(w) for w in w_list]
        return -(self.dalpha * self.interpolate(self.alphas[:, None], w_list)).sum()
    

class BasicEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        return alphas * w0 / (1 + alphas * wr)


class SquareEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5


class RegEigNumInterpolator(_EigNumInterpolator):
    def _interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        w1 = 2 * w_list[2] / w_list[1] - 1
        awrs = alphas * wr
        bwrs = alphas * w0 * wr
        scr = wr._mpac_erel**0.5
        print("SCR", scr)
        # a, b, c = 1.624e0, 1.593e0
        a, b, c = 8.474e+00, 7.766e+00, 9.116e+00
        a, b, c = 8.474e+00, 1.766e+00, 2.116e+00
        a, b, c = 8.474e+00, 0, 0
        a, b, c = 0.0, 1.684e+00, 0.0
        W1s = wr._mpac_erel
        #denom = (1 + awrs**2 + c * W1s * awrs**1.5)**0.5 + b * W1s * bwrs**0.5 + (1 + a * W1s * awrs)**0.5 - 1
        a, b, c = 1.474e+01, 7.348e+00, 5.000e-01
        denom = denom = (1 + awrs**2)**0.5 + (1 + b * W1s * bwrs**0.75 + a * W1s * awrs)**0.5 - 1
        return alphas * w0 / denom
    
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        w1 = w_list[1]
        print("HIHI")
        awrs = alphas * wr
        bwrs = alphas * w0**2 / w_list[-1]
        a, b, c = 6.363e+00, 1.070e-02, 8.462e+00
        a, b, c, = 5.822e+00, 2.003e-01, 7.681e+00
        # a, b, c = 4.000e+00, 2.302e+00, 2.209e+00
        # a, b, c = 2.991e+00, 2.30e+00, 1.214e+00
        # a, b, c = 0e+00, 0e+00, 1.214e+00
        # a, b, c = 7.526e-01, 1.682e-01, 2.406e+00
        # denom = (1 + awrs**2 + b * w1**2 * bwrs**0.5 + a * w1**2 * bwrs)**0.5
        # denom += c * w1 * bwrs**0.5
        
        # denom = (1 + awrs**2 + a * bwrs**0.5 + c * awrs * w1 / w0)**0.5 + b * bwrs**0.5 
        
        # denom = 1 + awrs + c * awrs * w1 / w0 + b * bwrs**0.5
        a, b, c = 2.601e+00, 2.443e+00, 1.270e-01
        mixer = 1 + a * (1.0 - numpy.exp(-b * bwrs**0.5))
        denom = mixer * (1 + c * (mixer - 1) * awrs**1.5 + awrs**2 / mixer**2)**0.5

        a, b, c = 4.836e+00, 3.632e+00, 3.155e+00
        mixer = 1 + (1.0 - numpy.exp(-b * bwrs**0.5))
        denom = mixer * (1 + a * awrs / mixer + c * (mixer - 1) * awrs**1.5 / mixer**1.5 + awrs**2 / mixer**2)**0.5
        denom = (mixer * mixer + a * (mixer - 1) + 2 * awrs + c * (mixer - 1) * awrs**1.5 + awrs**2)**0.5
        
        a, b, c = 2.7e+00, 9.463e+01, 0.000e+00
        denom = (mixer * mixer + 2 * a * mixer * awrs + c * (mixer - 1) * awrs**1.5 + awrs**2)**0.5

        a, b, c, d = 2.771e+00, 1.102e+00, 6.215e-01, 3.475e+00
        denom = (1.0 + awrs**2)**0.5 + c * (b * bwrs) / (1 + b * awrs**0.5 * w0) + d * (a * bwrs) / (1 + a * bwrs**0.75)

        return alphas * w0 / denom


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        winf = w0 / (-w_list[2] * 2)**0.5
        print(winf, w_list[-1], w0, w_list[2])
        wr = w0 / winf
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5
    

class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w_list[2]
        gap = w_list[1]
        w1 = w1 - w0 * numpy.log(gap)
        alpha = numpy.exp(w1 / w0) * gap
        winf = numpy.sqrt(0.5 * alpha * w0)
        print(gap, w0, w_list[2], winf, w_list[-1])
        wr = w0 / winf
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w0 / (-w_list[2] * 2)**0.5
        winf = w_list[-1]
        bwrs = alphas * w0 * w0 / winf
        a, b, c, d = 9.325e-01, 1.276e+00, 1.203e+00, 6.093e-01
        # a, b, c, d = 0, 0, 0, 6.093e-01
        hwrs = (
            c * bwrs**0.5 / (1 + a * w0**0.5)
            + d * bwrs**0.25 * w0**0.5 / (1 + b * w0**0.5)
        )
        hwrs *= (winf / w1)**4
        awrs = alphas * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        winf = w0 / (-w_list[2] * 2)**0.5
        return alphas * w0 / denom


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w0 / (-w_list[2] * 2)**0.5
        winf = w_list[-1]
        a, b, c, d = 4.627e-01, 1.378e+00, 7.555e-01, 4.852e-01
        bwrs = alphas * w0 * w0 / winf
        hwrs = c * bwrs**0.5 / (1 + a * w0) + d * bwrs**0.25 * w0**0.25 / (1 + b * w0**0.25)
        hwrs *= numpy.exp((winf / w1)**3 - 1)
        awrs = alphas * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        return alphas * w0 / denom


class ScreenedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        bwrs = alphas * w0 * wr
        awrs = alphas * wr
        a, b = 8.5, 6.4
        return alphas * w0 / (1 + b * bwrs**0.5 + a * awrs + awrs**2)**0.5


class ScreenedWeightedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        a, b, c = self.params
        winfs = w0 / wr
        bwrs = alphas * alphas * w0 * wr
        awrs = alphas * wr
        wt = 1 + c * alphas**0.5 / winfs + a * alphas**0.5 / winfs**0.5
        return alphas * w0 * wt / (1 + b * bwrs**0.25 + awrs * wt)


class BalancedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        bwrs = alphas * wr * w0
        awrs = alphas * wr
        a, b, c = 0.5048, 2.156, 0.5358
        a, b, c = 0, 0, 0
        mix = (1 + a * bwrs)**-0.25
        denom = 1 + b * bwrs**0.5 * mix + (c * awrs)**0.5 * (1 - mix) + awrs
        print((-self.dalpha * alphas * w0 / denom).sum(axis=0))
        return alphas * w0 / denom


class _MatNumInterpolator(_NumACInterpolator):
    def get_ac_term(self, alpha):
        raise NotImplementedError
    
    def _cache_intermediates(self, w_list):
        self._clear_cache()
        w0 = 0.5 * (w_list[0] + w_list[0].T)
        winf = w_list[-1]
        wr = numpy.linalg.solve(winf, w0)
        wr = 0.5 * (wr + wr.T)
        self._cache["w0"] = w0
        self._cache["wr"] = wr
        self._cache["id"] = _identity_like(w0)
    
    def _clear_cache(self):
        self._cache = {}

    def __call__(self, w_list):
        self._cache_intermediates(w_list)
        energy = 0
        for alpha in self.alphas:
            energy -= self.dalpha * numpy.trace(
                self.get_ac_term(alpha)
            )
        self._clear_cache()
        return energy
    

class BasicMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        term = alpha * wr
        return alpha * numpy.linalg.solve(idmat + term, w0)


class SquareMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        term = idmat + _acmp_matrix_pow(alpha * wr, 2)
        term = _acmp_matrix_pow(term, 0.5)
        return alpha * numpy.linalg.solve(term, w0)


class WinfMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        bwrs = w0.dot(wr)
        bwrs = 0.5 * alpha * (bwrs + bwrs.T)
        a = 1.67
        term = a * _acmp_matrix_pow(bwrs, 0.5) + alpha * wr
        return alpha * numpy.linalg.solve(idmat + term, w0)


class WinfMatNumInterpolator2(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        bwrs = _acmp_matrix_pow(w0, 0.5).dot(wr)
        bwrs = 0.5 * alpha * (bwrs + bwrs.T)
        a = 1.691
        b = 0.227
        term = a * _acmp_matrix_pow(bwrs, 0.25)
        term += b * _acmp_matrix_pow(bwrs, 0.50)
        term += alpha * wr
        return alpha * numpy.linalg.solve(idmat + term, w0)


class ACMP2(MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.si_limit = "HF"
        self.agap_model = get_power_of_ws_radius_func(-2, 1.0)
        self.ac_interpolator = None

    get_acmp_si_limit = get_acmp_si_limit

    get_artificial_gap = get_artificial_gap

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mp, mo_coeff=mo_coeff)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            mf = mp._scf.to_hf()
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)

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
