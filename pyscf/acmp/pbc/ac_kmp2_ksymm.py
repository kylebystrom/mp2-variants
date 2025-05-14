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
#
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.mp import kmp2, kmp2_ksymm
from pyscf.acmp.pbc import ac_kmp2
from pyscf.acmp.ac_mp2 import concatentate_w, ACMP_PAIRED, _acmp_ao2mo
from pyscf.pbc import df
from pyscf.pbc.lib import kpts as libkpts


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', False)

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    if with_t2:
        return kernel_with_t2(mp, mo_energy, mo_coeff, verbose, with_t2)
    else:
        t2 = None

    t0 = (logger.process_clock(), logger.perf_counter())
    nmo = mp.nmo
    nocc = mp.nocc
    nocc_list = mp.get_nocc(per_kpoint=True)
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts

    with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)
    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = mp._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s1')
    _, igroup = np.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = igroup.ravel()
    igroup = list(igroup) + [len(kijab)]

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp)

    emp2_ss = emp2_os = 0.
    energy = 0.
    nao2mo = 0
    icount = 0
    exx_k_xx, winf_k_xx = mp.get_acmp_si_limit()
    exx_k_xx = mp._scf.kpts.transform_dm(exx_k_xx)
    winf_k_xx = mp._scf.kpts.transform_dm(winf_k_xx)
    wlist_k = [None] * kd.nkpts_ibz
    ibz2bz = [None] * kd.nkpts_ibz
    for i in range(len(igroup)-1):
        istart = igroup[i]
        iend = igroup[i+1]
        kab = []
        for j in range(istart, iend):
            a, b = kijab[j][2:]
            kab.append([a, b])
            kab.append([b, a])
        kab = np.unique(np.asarray(kab), axis=0)

        ki = kijab[istart][0]
        kj = kijab[istart][1]
        kpts_i = kd.kpts[ki]
        kpts_j = kd.kpts[kj]
        orbo_i = mo_coeff[ki][:,:nocc]
        orbo_j = mo_coeff[kj][:,:nocc]
        ki_ibz = kd.bz2ibz[ki]

        # TODO lib.dot if possible
        my_nocc = nocc_list[ki]
        if wlist_k[ki_ibz] is None:
            ibz2bz[ki_ibz] = ki
            wlist_k[ki_ibz] = [np.zeros((my_nocc, my_nocc), dtype=np.complex128)
                               for _ in range(mp.get_pt_list_size())]
        # occ_coeff = mo_coeff[ki][:, :my_nocc]
        # exx_oo = _acmp_ao2mo(exx_k_xx[ki], occ_coeff)
        # winf_oo = _acmp_ao2mo(winf_k_xx[ki], occ_coeff)
        # winf = winf_oo
        # print(ki, i, igroup[i], igroup[i+1])

        for (ka, kb) in kab:
            if with_df_ints:
                oovv_ij[ka] = (1./nkpts) * einsum(
                    "Lia,Ljb->iajb",
                    Lov[ki, ka],
                    Lov[kj, kb]
                ).transpose(0,2,1,3)
            else:
                kpts_a = kd.kpts[ka]
                kpts_b = kd.kpts[kb]
                orbv_a = mo_coeff[ka][:,nocc:]
                orbv_b = mo_coeff[kb][:,nocc:]
                oovv_ij[ka] = fao2mo(
                    (orbo_i,orbv_a,orbo_j,orbv_b),
                    (kpts_i,kpts_a,kpts_j,kpts_b),
                    compact=False
                ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            nao2mo += 1

        for j in range(istart, iend):
            ka = kijab[j][2]
            kb = kijab[j][3]
            # Remove zero/padded elements from denominator
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            # t2_ijab = np.conj(oovv_ij[ka]/eijab)
            idx_ibz = k4_bz2ibz[ki*nkpts**2 + kj*nkpts + ka]
            assert(icount == idx_ibz)
            wt = weight[idx_ibz] * nkpts**3
            mp.add_to_w_list_(wlist_k[ki_ibz], oovv_ij[ka], oovv_ij[kb], eijab, ACMP_PAIRED, wt=wt)
            icount += 1

    for ki_ibz, w_list in enumerate(wlist_k):
        ki = ibz2bz[ki_ibz]
        wt = kd.weights_ibz[ki_ibz]
        occ_coeff = mo_coeff[ki][:, :my_nocc]
        exx_oo = _acmp_ao2mo(exx_k_xx[ki], occ_coeff)
        winf_oo = _acmp_ao2mo(winf_k_xx[ki], occ_coeff)
        winf = winf_oo
        for w in w_list:
            w[:] *= 1.0 / (wt * nkpts)
        w_list = concatentate_w(exx_oo, w_list, winf[None, :, :])
        # energy += 2 * mp.ac_interpolator(w_list).real * weight[idx_ibz] * nkpts**3
        energy += 2 * mp.ac_interpolator(w_list).real * wt * nkpts

    emp2_ss /= nkpts
    emp2_os /= nkpts
    energy /= nkpts
    # emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    emp2 = lib.tag_array(energy, e_corr_ss=0, e_corr_os=energy)
    assert(icount == len(kijab))
    logger.debug(mp, "Number of ao2mo transformations performed in KMP2: %d", nao2mo)
    logger.timer(mp, 'KMP2', *t0)
    return emp2, t2


def kernel_with_t2(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    #we need almost all t2 for computing rdm, so simply use kmp2 without symmetry
    kd = mp.kpts
    mp.kpts = kd.kpts
    emp2, t2 = ac_kmp2.kernel(mp, mo_energy, mo_coeff, verbose, with_t2)
    mp.kpts = kd
    return emp2, t2


class KsymAdaptedACKMP2(ac_kmp2.ACKMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
        if isinstance(self.kpts, libkpts.KPoints):
            self._mo_energy_ibz = mf.mo_energy
            self._mo_occ_ibz = mo_occ
            self._mo_coeff_ibz = mo_coeff

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mp, mo_coeff=mo_coeff)
        else:
            kpts = mp._scf.kpts
            mf = mp._scf.to_hf()
            dm = mp._scf.make_rdm1(mo_coeff, mp._mo_occ_ibz)
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        assert mo_coeff is None and mo_energy is None, "No SCF implemented"
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self.get_e_hf(mo_coeff=self._mo_coeff_ibz)

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    make_rdm1 = kmp2_ksymm.make_rdm1

    def make_rdm2(self):
        raise NotImplementedError
