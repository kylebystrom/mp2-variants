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

    if mo_coeff is None: mo_coeff = mp.mo_coeff
    with_t2 = False  # TODO allow t2
    mo_energy, dfock, transforms = mp._get_coefs()

    nocca, noccb = mp.nocc  # TODO get_nocc()
    nmoa, nmob = mo_coeff.shape[-1], mo_coeff.shape[-1]
    nvira, nvirb = nmoa-nocca, nmob-noccb

    assert mo_coeff is not None  # TODO allow mo_coeff to be None
    mc_a = numpy.append(
        mo_coeff[:, :nocca].dot(transforms[0]),
        mo_coeff[:, nocca:].dot(transforms[2]),
        axis=1,
    )
    mc_b = numpy.append(
        mo_coeff[:, :noccb].dot(transforms[1]),
        mo_coeff[:, noccb:].dot(transforms[3]),
        axis=1,
    )
    mo_coeff = numpy.stack([mc_a, mc_b])
    

    if eris is None:
        mp.mo_occ = mp._split_mo_occ
        mp._scf.mo_coeff = mo_coeff
        mp._scf.mo_energy = mo_energy
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

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

    @property
    def nocc(self):
        return self.mol.nelec
    @nocc.setter
    def nocc(self, n):
        raise NotImplementedError

    def _get_coefs(self):
        nmo = self.mo_coeff.shape[-1]
        umf = self._scf = self._scf.to_uhf()
        nocc_a = umf.nelec[0]
        nocc_b = umf.nelec[1]
        occ_a = numpy.zeros(nmo)
        occ_a[:nocc_a] = 1
        occ_b = numpy.zeros(nmo)
        occ_b[:nocc_b] = 1
        dm1 = umf.make_rdm1(
            mo_coeff=numpy.stack([self.mo_coeff, self.mo_coeff]),
            mo_occ=[occ_a, occ_b],
        )
        fock = umf.get_hcore(self.mol) + umf.get_veff(self.mol, dm1)
        dfock = [_acmp_ao2mo(fock[0], self.mo_coeff),
                 _acmp_ao2mo(fock[1], self.mo_coeff)]
        oe_a, ot_a = scipy.linalg.eigh(dfock[0][:nocc_a, :nocc_a])
        oe_b, ot_b = scipy.linalg.eigh(dfock[1][:nocc_b, :nocc_b])
        ve_a, vt_a = scipy.linalg.eigh(dfock[0][nocc_a:, nocc_a:])
        ve_b, vt_b = scipy.linalg.eigh(dfock[1][nocc_b:, nocc_b:])
        dfock[0][:nocc_a, :nocc_a] = 0
        dfock[0][nocc_a:, nocc_a:] = 0
        dfock[1][:nocc_b, :nocc_b] = 0
        dfock[1][nocc_b:, nocc_b:] = 0
        moe_a = numpy.append(oe_a, ve_a)
        moe_b = numpy.append(oe_b, ve_b)
        self._split_mo_occ = numpy.stack([occ_a, occ_b])
        return (
            [moe_a, moe_b],
            [dfock[0][:nocc_a, nocc_a:], dfock[1][:nocc_b, nocc_b:]],
            [ot_a, ot_b, vt_a, vt_b],
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

