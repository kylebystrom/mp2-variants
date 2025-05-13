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
UMP2 with spatial integrals
'''


import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.mp.ump2 import UMP2
from pyscf.acmp.ac_mp2 import concatentate_w, \
    ACMP_D_ONLY, ACMP_D_AND_X, _acmp_ao2mo, MP2NumInt, Grids, \
    add_to_w_list_

WITH_T2 = getattr(__config__, 'mp_ump2_with_t2', True)


def get_acmp_si_limit(mp):
    exx_sxx = -0.5 * mp._scf.get_k()
    if isinstance(mp.si_limit, str) and mp.si_limit == "HF":
        winf_sxx = exx_sxx.copy()
    else:
        ni = MP2NumInt()
        grids = Grids(mp._scf.mol)
        grids.level = 3
        maxmem = mp._scf.mol.max_memory
        nelec, excsum, vmat = ni.nr_ump2(mp._scf.mol, grids, mp.si_limit,
                                         mp._scf.make_rdm1(), relativity=0,
                                         hermi=1, max_memory=maxmem,
                                         verbose=None)
        # Reference strong correlation limit to exx
        winf_sxx = vmat - exx_sxx
        # winf_xx = vmat
    return exx_sxx, winf_sxx


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

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    if with_t2:
        raise NotImplementedError
        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
    else:
        t2 = None

    exx_sxx, winf_sxx = mp.get_acmp_si_limit()
    exxa_oo = _acmp_ao2mo(exx_sxx[0], mo_coeff[0][:, :nocca])
    winfa_oo = _acmp_ao2mo(winf_sxx[0], mo_coeff[0][:, :nocca])
    exxb_oo = _acmp_ao2mo(exx_sxx[1], mo_coeff[1][:, :noccb])
    winfb_oo = _acmp_ao2mo(winf_sxx[1], mo_coeff[1][:, :noccb])

    wa_list = [numpy.zeros_like(winfa_oo) for _ in range(mp.get_pt_list_size())]
    wb_list = [numpy.zeros_like(winfb_oo) for _ in range(mp.get_pt_list_size())]
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        mp.add_to_w_list_(wa_list, eris_ovov, eris_ovov, ei, ACMP_D_AND_X)
        #t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        #emp2a[:] += numpy.einsum('iab,jab->ij', t2i, eris_ovov) * .5
        #emp2a[:] -= numpy.einsum('iab,jba->ij', t2i, eris_ovov) * .5
        #if with_t2:
        #    t2aa[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
        eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        mp.add_to_w_list_(wb_list, eris_ovov, eris_ovov, ei, ACMP_D_ONLY)
        #t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        #emp2b[:] += numpy.einsum('IaB,JaB->IJ', t2i, eris_ovov) * 0.5
        #if with_t2:
        #    t2ab[i] = t2i

    for i in range(noccb):
        if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.OVOV[i]
        else:
            eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
        mp.add_to_w_list_(wb_list, eris_ovov, eris_ovov, ei, ACMP_D_AND_X)
        #t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
        #emp2b[:] += numpy.einsum('iab,jab->ij', t2i, eris_ovov) * .5
        #emp2b[:] -= numpy.einsum('iab,jba->ij', t2i, eris_ovov) * .5
        #if with_t2:
        #    t2bb[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[:, :, i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[:, i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nocca,nvira,nvirb)
        ei = lib.direct_sum('b+ia->iab', eia_b[i], eia_a)
        mp.add_to_w_list_(wa_list, eris_ovov, eris_ovov, ei, ACMP_D_ONLY)
        #t2i = eris_ovov.conj()/lib.direct_sum('b+ia->iab', eia_b[i], eia_a)
        #emp2a[:] += numpy.einsum('iaB,jaB->ij', t2i, eris_ovov) * 0.5

    wa_list = concatentate_w(exxa_oo, wa_list, winfa_oo[None, :, :])
    wb_list = concatentate_w(exxb_oo, wb_list, winfb_oo[None, :, :])
    energy = mp.ac_interpolator(wa_list)
    energy += mp.ac_interpolator(wb_list)

    # TODO don't assign misleading values to ss and os
    emp2 = lib.tag_array(energy, e_corr_ss=0, e_corr_os=energy)

    return emp2, t2


class ACUMP2(UMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.si_limit = "HF"
        self.ac_interpolator = None

    get_acmp_si_limit = get_acmp_si_limit

    add_to_w_list_ = add_to_w_list_

    def get_pt_list_size(self):
        return 2
    
    def get_df_list_size(self):
        return 1
    
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
