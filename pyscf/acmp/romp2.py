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


import numpy, scipy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import mp2, ump2
from pyscf.ao2mo import _ao2mo
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_ump2_with_t2', True)


# This is restricted open-shell MP2 based on
# Knowles, Andrews, Amos, Handy, and Pople, Chemical Physics Letters 1991, 186, 2,3.

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    with_t2 = False  # TODO allow t2
    mo_energy, dfock, transforms, mo_coeff = mp._get_coefs()
    mo_energy = None
    mo_coeff = None

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    mo_energy = eris.mo_energy
    mo_coeff = eris.mo_coeff
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mo_coeff[0].shape[-1], mo_coeff[1].shape[-1]
    nvira, nvirb = nmoa-nocca, nmob-noccb

    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    e_singles = numpy.einsum("ia,ia->", dfock[0], dfock[0].conj() / eia_a)
    e_singles += numpy.einsum("ia,ia->", dfock[1], dfock[1].conj() / eia_b)

    if with_t2:
        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
    else:
        t2 = None

    emp2_ss = emp2_os = 0.0
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2aa[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
        eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        emp2_os += numpy.einsum('JaB,JaB', t2i, eris_ovov)
        if with_t2:
            t2ab[i] = t2i

    for i in range(noccb):
        if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.OVOV[i]
        else:
            eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2bb[i] = t2i - t2i.transpose(0,2,1)

    emp2_ss = emp2_ss.real + e_singles.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


def _acmp_ao2mo(mat_xx, coeff):
    mat_xx = numpy.ascontiguousarray(mat_xx)
    coeff = numpy.ascontiguousarray(coeff)
    mat_xo = lib.dot(mat_xx, coeff)
    return lib.dot(coeff.T.conj(), mat_xo)


class ROMP2(ump2.UMP2):

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return ump2._make_eris(self, mo_coeff, verbose=self.verbose)

    get_frozen_mask = ump2.get_frozen_mask

    get_nocc = ump2.get_nocc

    @property
    def nocc(self):
        if self._nocc is None:
            return self.mol.nelec
        else:
            return self._nocc
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    def _get_coefs(self):
        romf = self._scf
        umf = romf.to_uhf()
        umf.converged = romf.converged
        self._scf = umf
        self.mo_occ = umf.mo_occ
        moidx = self.get_frozen_mask()
        romo_coeff_a = romf.mo_coeff
        romo_coeff_b = romf.mo_coeff
        occ_a = umf.mo_occ[0]
        occ_b = umf.mo_occ[1]
        assert numpy.logical_or(occ_a == 0, occ_a == 1).all()
        assert numpy.logical_or(occ_b == 0, occ_b == 1).all()
        occ_a = occ_a > 0
        vir_a = numpy.logical_not(occ_a)
        occ_b = occ_b > 0
        vir_b = numpy.logical_not(occ_b)
        dm1 = umf.make_rdm1(
            mo_coeff=numpy.stack([romo_coeff_a, romo_coeff_b]),
            mo_occ=[occ_a, occ_b],
        )
        fock = umf.get_hcore(self.mol) + umf.get_veff(self.mol, dm1)
        dfock = [_acmp_ao2mo(fock[0], romo_coeff_a),
                 _acmp_ao2mo(fock[1], romo_coeff_b)]
        oe_a, ot_a = scipy.linalg.eigh(dfock[0][occ_a][:, occ_a])
        oe_b, ot_b = scipy.linalg.eigh(dfock[1][occ_b][:, occ_b])
        ve_a, vt_a = scipy.linalg.eigh(dfock[0][vir_a][:, vir_a])
        ve_b, vt_b = scipy.linalg.eigh(dfock[1][vir_b][:, vir_b])
        occ_act_a = numpy.logical_and(moidx[0], occ_a)
        occ_act_b = numpy.logical_and(moidx[1], occ_b)
        vir_act_a = numpy.logical_and(moidx[0], vir_a)
        vir_act_b = numpy.logical_and(moidx[1], vir_b)
        dfock = [dfock[0][occ_act_a][:, vir_act_a], dfock[1][occ_act_b][:, vir_act_b]]
        moe_a = numpy.append(oe_a, ve_a)
        moe_b = numpy.append(oe_b, ve_b)

        mc_a = numpy.append(
            romo_coeff_a[:, occ_a].dot(ot_a),
            romo_coeff_a[:, vir_a].dot(vt_a),
            axis=1,
        )
        mc_b = numpy.append(
            romo_coeff_b[:, occ_b].dot(ot_b),
            romo_coeff_b[:, vir_b].dot(vt_b),
            axis=1,
        )
        mo_coeff = numpy.stack([mc_a, mc_b])

        nmoa, nocca = occ_a.size, occ_a.sum()
        nmob, noccb = occ_b.size, occ_b.sum()
        occ_a = numpy.zeros(nmoa)
        occ_a[:nocca] = 1.0
        occ_b = numpy.zeros(nmob)
        occ_b[:noccb] = 1.0

        self._scf.mo_coeff[0] = mo_coeff[0]
        self._scf.mo_coeff[1] = mo_coeff[1]
        self._scf.mo_energy[0] = moe_a
        self._scf.mo_energy[1] = moe_b
        self.mo_coeff = self._scf.mo_coeff
        self.mo_occ = self._scf.mo_occ
        self.mo_energy = self._scf.mo_energy

        return (
            [moe_a, moe_b],
            dfock,
            [ot_a, ot_b, vt_a, vt_b],
            self.mo_coeff,
        )

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()

        log = logger.new_logger(self)

        cput0 = cput1 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        cput1 = log.timer('ehf', *cput1)

        if eris is None:
            eris = None  # self.ao2mo(mo_coeff)
        else:
            import warnings
            warnings.warn("ROMP2 assumes the eris are rotated to the ROHF mo_coeff")

        cput1 = log.timer('ao2mo', *cput1)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError
            #self.converged, self.e_corr, self.t2 = _iterative_kernel(self, eris)

        cput1 = log.timer('kernel', *cput1)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        log.timer(self.__class__.__name__, *cput0)

        self._finalize()
        return self.e_corr, self.t2
    
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

