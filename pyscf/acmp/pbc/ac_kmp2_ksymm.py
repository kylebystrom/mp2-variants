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

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s2')
    _, igroup = np.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = igroup.ravel()
    igroup = list(igroup) + [len(kijab)]

    emp2_ss = emp2_os = 0.
    energy = 0.
    nao2mo = 0
    icount = 0
    winf_k_xx = mp.get_acmp_si_limit()
    winf_k_xx = mp._scf.kpts.transform_dm(winf_k_xx)
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

        winf_xx = -winf_k_xx[ki]
        # TODO lib.dot if possible
        my_nocc = nocc_list[ki]
        winf_xo = np.dot(winf_xx, mo_coeff[ki][:, :my_nocc])
        winf_oo = np.dot(mo_coeff[ki][:, :my_nocc].T.conj(), winf_xo)
        winf = winf_oo.real
        print(ki)
        
        for (ka, kb) in kab:
            kpts_a = kd.kpts[ka]
            kpts_b = kd.kpts[kb]
            orbv_a = mo_coeff[ka][:,nocc:]
            orbv_b = mo_coeff[kb][:,nocc:]
            oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                 (kpts_i,kpts_a,kpts_j,kpts_b),
                                 compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
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
            t2_ijab = np.conj(oovv_ij[ka]/eijab)
            idx_ibz = k4_bz2ibz[ki*nkpts**2 + kj*nkpts + ka]
            assert(icount == idx_ibz)
            edi = einsum('ikab,jkab', t2_ijab, oovv_ij[ka]).real * 2
            exi = -einsum('ikab,jkba', t2_ijab, oovv_ij[kb]).real
            # emp2_ss += (edi*0.5 + exi) * weight[idx_ibz] * nkpts**3
            # emp2_os += edi*0.5 * weight[idx_ibz] * nkpts**3
            # TODO is this the right weighting of everything?
            w0 = -2 * (edi.real + exi.real)
            w0 = w0[:my_nocc, :my_nocc]
            energy += mp.ac_interpolator(w0, winf) * weight[idx_ibz] * nkpts**3
            icount += 1

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
    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

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
