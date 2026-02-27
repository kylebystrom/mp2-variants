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

from pyscf import lib
from pyscf.acmp import ft_mp2
from pyscf.pbc.dft.gen_grid import BeckeGrids


def make_ftmp2(mymp, beta=1, mu0=None, occ_tol=0, mu_tol=1e-6,
               particle_fix=None, ecorr_method="analytical",
               max_mu_steps=50, with_singles=True):
    kwargs = dict(
        beta=beta, mu0=mu0, occ_tol=occ_tol,
        mu_tol=mu_tol, particle_fix=particle_fix,
        ecorr_method=ecorr_method, max_mu_steps=max_mu_steps,
        with_singles=with_singles
    )
    if isinstance(mymp, FTMP2Mixin):
        FTMP2Mixin.__init__(mymp, **kwargs)
        return mymp

    if hasattr(mymp, "ac_interpolator"):
        mixin_cls = FTACMP2Mixin
    else:
        mixin_cls = FTMP2Mixin

    return lib.set_class(mixin_cls(mymp, **kwargs),
                         (mixin_cls, mymp.__class__))


def _ao2mo(mp, mo_coeff=None, mu=None, beta=None):
        ao2mofn = _gen_ao2mofn(mp._scf)
        return ft_mp2._make_eris(mp, mo_coeff, ao2mofn=ao2mofn,
                                 mu=mu, beta=beta,
                                 verbose=mp.verbose)


class FTMP2Mixin(ft_mp2.FTMP2Mixin):
    ao2mo = _ao2mo


class FTACMP2Mixin(ft_mp2.FTACMP2Mixin):
    ao2mo = _ao2mo


def _gen_ao2mofn(mf):
    with_df = mf.with_df
    kpt = mf.kpt
    def ao2mofn(mo_coeff):
        return with_df.ao2mo(mo_coeff, kpt, compact=False)
    return ao2mofn
