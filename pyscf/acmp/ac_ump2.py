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
from pyscf.acmp.ac_mp2 import get_acmp_si_limit

WITH_T2 = getattr(__config__, 'mp_ump2_with_t2', True)

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
        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
    else:
        t2 = None

    winf_xx = get_acmp_si_limit(mp)
    print(mo_coeff[0].shape, winf_xx.shape)
    winf_xo = lib.dot(winf_xx[0], mo_coeff[0][:, :nocca])
    winfa_oo = lib.dot(mo_coeff[0][:, :nocca].T, winf_xo)
    winf_xo = lib.dot(winf_xx[1], mo_coeff[1][:, :noccb])
    winfb_oo = lib.dot(mo_coeff[1][:, :noccb].T, winf_xo)

    emp2a = numpy.zeros_like(winfa_oo)
    emp2b = numpy.zeros_like(winfb_oo)
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        emp2a[:] += numpy.einsum('iab,jab->ij', t2i, eris_ovov) * .5
        emp2a -= numpy.einsum('iab,jba->ij', t2i, eris_ovov) * .5
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
        emp2b[:] += numpy.einsum('IaB,JaB->IJ', t2i, eris_ovov) * 0.5
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
        emp2b[:] += numpy.einsum('iab,jab->ij', t2i, eris_ovov) * .5
        emp2b[:] -= numpy.einsum('iab,jba->ij', t2i, eris_ovov) * .5
        if with_t2:
            t2bb[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[:, i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[:, i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nocca,nvira,nvirb)
        t2i = eris_ovov.conj()/lib.direct_sum('b+ia->iab', eia_b[i], eia_a)
        emp2a[:] += numpy.einsum('iaB,jaB->ij', t2i, eris_ovov) * 0.5

    energy = mp.ac_interpolator(-2 * emp2a.real, -winfa_oo.real)
    energy += mp.ac_interpolator(-2 * emp2b.real, -winfb_oo.real)

    # TODO don't assign misleading values to ss and os
    emp2 = lib.tag_array(energy, e_corr_ss=0, e_corr_os=energy)

    return emp2, t2


class ACUMP2(UMP2):
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
