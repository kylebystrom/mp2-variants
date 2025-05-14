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

WITH_T2 = False
ACMP_PAIRED = 0
ACMP_D_ONLY = 1
ACMP_X_ONLY = 3
ACMP_D_AND_X = 2


def _identity_like(m):
    return numpy.identity(m.shape[0])


def _acmp_matrix_pow(mat, mypow):
    eval, evec = numpy.linalg.eigh(mat)
    eval = numpy.maximum(eval, 0)
    eval = eval**mypow
    return (evec * eval).dot(evec.T)


def _acmp_matrix_pow(mat, mypow):
    eval, evec = numpy.linalg.eig(mat)
    evec_inv = numpy.linalg.solve(evec, _identity_like(evec))
    # eval = numpy.maximum(eval, 0)
    eval = eval**mypow
    return (evec * eval).dot(evec_inv)


def get_acmp_si_limit(mp):
    exx_xx = -0.25 * mp._scf.get_k()
    if isinstance(mp.si_limit, str) and mp.si_limit == "HF":
        winf_xx = exx_xx.copy()
    else:
        ni = mp._numint
        grids = mp.grids
        maxmem = mp._scf.mol.max_memory
        nelec, excsum, vmat = ni.nr_rmp2(mp._scf.mol, grids, mp.si_limit,
                                         mp._scf.make_rdm1(), relativity=0,
                                         hermi=1, max_memory=maxmem,
                                         verbose=None)
        # Reference strong correlation limit to exx
        winf_xx = vmat - exx_xx
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
    mat_xx = numpy.ascontiguousarray(mat_xx)
    coeff = numpy.ascontiguousarray(coeff)
    mat_xo = lib.dot(mat_xx, coeff)
    return lib.dot(coeff.T.conj(), mat_xo)


def _get_corrected_winf(winf, exx):
    # Make sure winf_oo is positive
    d = 0.0001
    prod = winf.T.conj().dot(winf) + d * exx.T.conj().dot(exx)
    return _acmp_matrix_pow(prod, 0.5)


def concatentate_w(exx_oo, wlist_pt, wlist_df):
    for w in wlist_pt:
        w[:] *= -2
    exx = -exx_oo.real
    wlist_df = [-w.real for w in wlist_df]
    wlist_df[-1] = _get_corrected_winf(wlist_df[-1], exx)
    w_list = numpy.append(wlist_pt, wlist_df, axis=0)
    return w_list


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
        raise NotImplementedError
        # t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    exx_xx, winf_xx = mp.get_acmp_si_limit()
    occ_coeff = mo_coeff[:, :nocc]
    exx_oo = _acmp_ao2mo(exx_xx, occ_coeff)
    winf_oo = _acmp_ao2mo(winf_xx, occ_coeff)

    w_list = [numpy.zeros((nocc, nocc)) for _ in range(mp.get_pt_list_size())]
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        ei = lib.direct_sum('jb+a->jba', eia, eia[i]) - 1e-8
        mp.add_to_w_list_(w_list, gi, gi, ei, ACMP_PAIRED)

    print("TRACE", numpy.trace(w_list[0]), numpy.trace(w_list[1]), numpy.trace(winf_oo))
    w_list = concatentate_w(exx_oo, w_list, winf_oo[None, :, :])
    numpy.save("w_list.npy", numpy.array(w_list))
    energy = 2 * mp.ac_interpolator(w_list)

    # TODO shouldn't set these to misleading values
    edi = energy
    exi = 0.0
    emp2_ss = edi * 0.5 + exi
    emp2_os = edi * 0.5
    emp2 = lib.tag_array(energy, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2


def _contract_paired_(w, t2i, gd, gx, wt):
    w[:] += lib.einsum('jab,kab->jk', t2i, gd).real * wt
    w[:] -= lib.einsum('jab,kba->jk', t2i, gx).real * (0.5 * wt)


def _contract_d_only(w, t2i, gd, gx, wt):
    w[:] += numpy.einsum('iab,jab->ij', t2i, gd) * (0.5 * wt)


def _contract_d_x(w, t2i, gd, gx, wt):
    _contract_d_only(w, t2i, gd, gx, wt)
    w[:] -= numpy.einsum('iab,jba->ij', t2i, gx) * (0.5 * wt)


ACMP_CONTRACTTIONS = {
    ACMP_PAIRED: _contract_paired_,
    ACMP_D_ONLY: _contract_d_only,
    ACMP_D_AND_X: _contract_d_x
}


def _contract4_paired_(w, t2, gd, gx, wt):
    w[:] += lib.einsum('ikab,jkab->ij', t2, gd).real * wt
    w[:] -= lib.einsum('ikab,jkba->ij', t2, gx).real * (0.5 * wt)


def _contract4_d_only(w, t2, gd, gx, wt):
    w[:] += lib.einsum('ikab,jkab->ij', t2, gd) * (0.5 * wt)


def _contract4_x_only(w, t2, gd, gx, wt):
    w[:] -= lib.einsum('ikab,jkba->ij', t2, gx) * (0.5 * wt)


def _contract4_d_x(w, t2, gd, gx, wt):
    _contract4_d_only(w, t2, gd, gx, wt)
    _contract4_x_only(w, t2, gd, gx, wt)


ACMP_CONTRACTTIONS4 = {
    ACMP_PAIRED: _contract4_paired_,
    ACMP_D_ONLY: _contract4_d_only,
    ACMP_D_AND_X: _contract4_d_x,
    ACMP_X_ONLY: _contract4_x_only,
}


def add_to_w_list_(mp, w_list, gd, gx, ei, mode, wt=1.0):
    if gd.ndim == 3:
        assert gx.ndim == ei.ndim == 3
        contraction = ACMP_CONTRACTTIONS[mode]
    else:
        assert gd.ndim == gx.ndim == ei.ndim == 4
        contraction = ACMP_CONTRACTTIONS4[mode]
    if len(w_list) > 0:
        inv_ei = 1.0 / ei
        t2i = numpy.conj(gd * inv_ei)
        contraction(w_list[0], t2i, gd, gx, wt)
    if len(w_list) > 1:
        t2i *= inv_ei
        contraction(w_list[1], t2i, gd, gx, wt)
    if len(w_list) > 2:
        raise NotImplementedError


class ACMP2(MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.si_limit = "HF"
        self.ac_interpolator = None
        self._numint = MP2NumInt()
        self.grids = Grids(self._scf.mol)
        self.grids.level = 3

    def get_pt_list_size(self):
        return 2
    
    def get_df_list_size(self):
        return 1

    add_to_w_list_ = add_to_w_list_

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
        return kernel(self, mo_energy, mo_coeff, eris, with_t2=False)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('AC-MP2 with si_limit = %s', self.si_limit)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        return self


ACRMP2 = ACMP2
