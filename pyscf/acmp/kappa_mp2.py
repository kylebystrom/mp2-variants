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

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    maxi = -1e10
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        # t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        ei = lib.direct_sum('jb+a->jba', eia, eia[i])
        maxi = max(numpy.max(ei), maxi)
        t2i = gi.conj()/ei * mp.get_damping_factor(ei)
        edi = numpy.einsum('jab,jab', t2i, gi) * 2
        exi = -numpy.einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i
    print("MAXI", maxi, (1 - numpy.exp(mp.kappa*maxi))**2)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2


class KappaMP2Mixin:
    _keys = {'kappa', 'damping'}

    def __init__(self, kappa, damping):
        self.kappa = kappa
        self.damping = damping

    def get_damping_factor(self, delta):
        # Note delta must be non-positive!
        # assumes we have e_occ - e_vir for delta
        if self.damping == "kappa":
            term = (1 - numpy.exp(self.kappa * delta))
            term[:] *= term
            return term
        elif self.damping == "sigma":
            return 1 - numpy.exp(-self.kappa * delta * delta)
        else:
            raise ValueError("Unrecognized damping: {}".format(self.damping))


class KappaMP2(KappaMP2Mixin, MP2):
    '''restricted kappa-MP2 with canonical HF
    '''
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 kappa=1.5, damping="kappa"):
        MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        KappaMP2Mixin.__init__(self, kappa, damping)

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


KappaRMP2 = KappaMP2
