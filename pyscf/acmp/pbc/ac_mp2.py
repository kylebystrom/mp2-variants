#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

from pyscf.mp import mp2
from pyscf.acmp import ac_mp2
from pyscf.mp import ump2
from pyscf.acmp import ac_ump2
from pyscf.acmp.pbc import mp2_numint
from pyscf.pbc.dft.gen_grid import BeckeGrids


class ACRMP2(ac_mp2.ACMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        ac_mp2.ACMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self._numint = mp2_numint.MP2NumInt()
        self.grids = BeckeGrids(self._scf.mol)
        self.grids.level = 3

    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        eris = mp2._make_eris(self, mo_coeff, ao2mofn, self.verbose)
        return eris

class ACUMP2(ac_ump2.ACUMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        ac_ump2.ACUMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self._numint = mp2_numint.MP2NumInt()
        self.grids = BeckeGrids(self._scf.mol)
        self.grids.level = 3

    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        eris = ump2._make_eris(self, mo_coeff, ao2mofn, self.verbose)
        return eris

def _gen_ao2mofn(mf):
    with_df = mf.with_df
    kpt = mf.kpt
    def ao2mofn(mo_coeff):
        return with_df.ao2mo(mo_coeff, kpt, compact=False)
    return ao2mofn
