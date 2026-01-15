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
Finite-Temperature RMP2
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp.mp2 import MP2, MP2Base, _ChemistsERIs, _mem_usage, _ao2mo_ovov
from pyscf import ao2mo
from pyscf.scf.addons import _smearing_optimize, _fermi_smearing_occ

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def gc_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, mu=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff, mu)

    if mo_energy is None:
        mo_energy = eris.mo_energy
    
    occs = _fermi_smearing_occ(mu, mo_energy, 1.0 / mp.beta)

    nocc = eris.nocc
    nvir = eris.nvir
    eia = mo_energy[:nocc,None] - mo_energy[None,-nvir:]

    v1 = mp._get_v1(mo_energy, eris.mo_coeff, occs)
    v1ia = v1[:nocc,-nvir:]
    v1ia[:] *= v1ia.conj()

    #ediff = mo_energy - mu
    #nn_all = numpy.exp(0.5 * mp.beta * ediff)
    #dn_all = dn_all + numpy.exp(0.5 * mp.beta * ediff)
    #dn_occ = 1.0 / dn_all[:nocc]
    #dn_vir = 1.0 / dn_all[-nvir:]
    dn_ia = occs[:nocc, None] * (1 - occs[None, -nvir:])

    anom_ia = mo_energy[None, -nvir:] * occs[None, -nvir:]
    anom_ia = anom_ia - mo_energy[:nocc,None] * (1 - occs[:nocc, None])
    anom_ia[:] *= mp.beta

    a2_ia = -occs[None, -nvir:] + (1 - occs[:nocc, None])
    a2_ia[:] *= mp.beta

    a3_ia = occs[None, -nvir:] * (1 - occs[None, -nvir:])
    a3_ia = a3_ia + (1 - occs[:nocc, None]) * occs[:nocc, None]
    a3_ia[:] *= -1 * mp.beta ** 2

    inve = eia.copy()
    inve[numpy.abs(inve) < 1e-10] = -2 / mp.beta
    inve[:] = 1.0 / inve

    if mp.ensemble is None:
        e1 = (2 * inve * v1ia * dn_ia).sum()
        n1 = 0
        n2 = 0
    elif mp.ensemble in ["gc", "c_scf"]:
        e1 = (2 * inve * v1ia * dn_ia * (2 + anom_ia)).sum()
        n1 = (-inve * v1ia * dn_ia * a2_ia).sum()
        n2 = (-inve * v1ia * dn_ia * (a2_ia * a2_ia + a3_ia)).sum()
    elif mp.ensemble == "c_pt2":
        raise NotImplementedError
    else:
        raise ValueError("Unsupported thermal ensemble")

    with_t2 = False  # TODO
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    small_gap = 1e-9
    n0_count = 0
    nval_count = n1
    dnval_count = n2
    print("MU", mu)
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        n0_count += 2 * occs[i]
        dnval_count += 2 * mp.beta * occs[i] * (1 - occs[i])
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        ei = lib.direct_sum('jb+a->jba', eia, eia[i])
        
        #ei[numpy.abs(ei) < small_gap] = small_gap
        #ei = (1 - numpy.exp(0.5 * mp.beta * ei)) / ei
        #ei *= lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
        #t2i = gi.conj() * ei
        
        if mp.ensemble is None:
            ei[numpy.abs(ei) < small_gap] = -2 / mp.beta
            ei[:] = lib.einsum('jb,a->jba', dn_ia, dn_ia[i]) / ei
        elif mp.ensemble in ["gc", "c_scf"]:
            cond = numpy.abs(ei) < small_gap
            ai = lib.direct_sum('jb+a->jba', anom_ia, anom_ia[i])
            a2i = lib.direct_sum('jb+a->jba', a2_ia, a2_ia[i])
            a3i = lib.direct_sum('jb+a->jba', a3_ia, a3_ia[i])
            ei[cond] = 1
            ni = -1.0 / ei
            ni[cond] = 0.5 * mp.beta
            ni[:] *= a2i * lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
            ni2 = -1.0 / ei
            ni2[cond] = 0.5 * mp.beta
            ni2[:] *= (a2i * a2i + a3i) * lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
            ei[:] = (1 + ai) / ei
            ei[cond] = -mp.beta * (1 + 0.5 * ai[cond])
            ei[:] *= lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
            n2i = gi.conj() * ni
            ndi = numpy.einsum('jab,jab', n2i, gi) * 2
            nxi = -numpy.einsum('jab,jba', n2i, gi)
            nval_count += ndi + nxi
            n2i2 = gi.conj() * ni2
            ndi = numpy.einsum('jab,jab', n2i2, gi) * 2
            nxi = -numpy.einsum('jab,jba', n2i2, gi)
            dnval_count += ndi + nxi
        elif mp.ensemble == "c_pt2":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported thermal ensemble")
        t2i = gi.conj() * ei

        edi = numpy.einsum('jab,jab', t2i, gi) * 2
        exi = -numpy.einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i

    print("NVAL", nval_count, n0_count, nval_count + n0_count, dnval_count)
    emp2_ss = emp2_ss.real + e1
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, nval_count + n0_count, dnval_count


def _occ_helper(x, order):
    expx = numpy.exp(x)
    if order == 0:
        return 1 / (1 + expx)
    elif order == 1:
        return -1 * expx / (1 + expx)**2
    elif order == 2:
        return expx * (expx - 1) / (1 + expx)**3
    else:
        raise ValueError


def _fermi_dirac(e, mu, beta):
    return _occ_helper(beta * (e - mu), 0)


def _dfdu_fermi_dirac(e, mu, beta):
    return -beta * _occ_helper(beta * (e - mu), 1)


def _d2fdu2_fermi_dirac(e, mu, beta):
    return beta * beta * _occ_helper(beta * (e - mu), 2)


def get_nocc_nvir_nval(mp, mu):
    # occ = _fermi_dirac(mp._scf.mo_energy, mu, mp.beta)
    occ = _fermi_smearing_occ(mu, mp._scf.mo_energy, 1.0 / mp.beta)
    nval = numpy.sum(occ)
    # Note: Using the Fermi-Dirac occupations,
    # not the mp.mo_occ
    occ_idx = occ > mp.occ_tol
    vir_idx = occ < 1.0 - mp.occ_tol
    if mp.frozen is None:
        nocc = numpy.count_nonzero(occ_idx)
        nvir = numpy.count_nonzero(vir_idx)
        assert (nocc > 0)
    elif isinstance(mp.frozen, (int, numpy.integer)):
        occ_idx[:mp.frozen] = False
        nocc = numpy.count_nonzero(occ_idx) - mp.frozen
        nvir = numpy.count_nonzero(vir_idx)
        nval -= numpy.sum(occ[:mp.frozen])
        assert (nocc > 0)
    elif hasattr(mp.frozen, '__len__'):
        occ_idx[list(mp.frozen)] = False
        vir_idx[list(mp.frozen)] = False
        nocc = numpy.count_nonzero(occ_idx)
        nvir = numpy.count_nonzero(vir_idx)
        nval -= numpy.sum(occ[list(mp.frozen)])
        assert (nocc > 0)
    else:
        raise NotImplementedError
    return nocc, nvir, nval


def get_correct_nval(mp):
    """
    Get the correct number of valence electrons.
    """
    ntot = mp.mol.nelectron
    if mp.frozen is None:
        return ntot
    elif isinstance(mp.frozen, (int, numpy.integer)):
        nval = ntot - numpy.sum(mp.mo_occ[:mp.frozen])
        assert nval > 0
        return nval
    elif hasattr(mp.frozen, '__len__'):
        nval = ntot - numpy.sum(mp.mo_occ[list(mp.frozen)])
        assert nval > 0
        return nval
    else:
        raise NotImplementedError



class FTMP2Mixin:
    _keys = {"mu_algo", "occ_tol", "mu_tol"}

    def __init__(self, beta=1, mu_algo="mu0", mu0=None,
                 occ_tol=1e-9, mu_tol=1e-6, ensemble=None):
        """
        beta (float): 1 / (k_B * T) in atomic units. Default is equivalent to
            a temperature of ~316 K.
        mu_algo (str) is the method for determining the chemical potential.
            "mu0": Use the initial guess mu0 as the chemical potential.
                   if mu0 is None, the chemical potential that results
                   in <N>=N for the zeroth-order Hamiltonian is used.
            "search": Do a Newton algorithm search for the chemical potential
                such that <N>^(0) + <N>^(1) + <N>^(2) = N. Requires multiple
                iterations of calling the MP2 kernel. If provided, mu0 is
                used as the initial guess for mu.
            "pt2": Expand mu to preserve <N>=N to second order. The formula
                for the energy is more involved, but does not require
                multiple iterations like "search".
        mu0 (float): Initial guess for mu. If not provided, it is set to the
            value that results in <N>=N for the zeroth-order Hamiltonian.
        occ_tol (float): Orbitals with mo_occ < occ_tol are considered fully
            virtual, and orbitals with mo_occ > 1 - occ_tol are considered
            fully occupied.
        mu_tol (float): Only used for mu_algo="search". The tolerance for
            converging mu with the Newton algorithm.
        ensemble (str, None): Thermal ensemble for the finite temperature
            effects. For 
        """
        self.beta = beta
        if mu_algo not in ["mu0", "search", "pt2"]:
            raise ValueError("Unsupported mu_algo={}".format(mu_algo))
        self.mu_algo = mu_algo
        self.mu0 = mu0
        self.occ_tol = occ_tol
        self.mu_tol = mu_tol
        self.ensemble = ensemble
        self._nocc = None

    @property
    def nocc(self):
        return self._nocc
    @nocc.setter
    def nocc(self, n):
        raise ValueError("Cannot set nocc for FT-MP2")

    def _get_v1(self, mo_energy, mo_coeff, ac_occ):
        """
        Get the single-particle term of the Hamiltonian perturbation.
        """
        ac_occ = 2 * ac_occ
        if self.frozen is None:
            mo_occ = ac_occ.copy()
        elif isinstance(self.frozen, (int, numpy.integer)):
            mo_occ = self._scf.mo_occ.copy()
            mo_occ[self.frozen:] = ac_occ
        elif hasattr(self.frozen, '__len__'):
            mask = self.get_frozen_mask()
            mo_occ = self._scf.mo_occ.copy()
            mo_occ[mask] = ac_occ
        else:
            raise NotImplementedError
        dm = self._scf.make_rdm1(self._scf.mo_coeff, mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf = vj - 0.5 * vk
        fockao = self._scf.get_fock(vhf=vhf, dm=dm)
        fock = mo_coeff.conj().T.dot(fockao).dot(mo_coeff)
        return fock - numpy.diag(mo_energy)

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

        if self.mu0 is None:
            mu = self._init_smearing()
        else:
            mu = self.mu0

        if eris is None:
            eris = self.ao2mo(mo_coeff, mu)

        cput1 = log.timer('ao2mo', *cput1)

        if self._scf.converged:
            if self.mu_algo != "mu0":
                raise NotImplementedError
            self.e_corr, n, dn = gc_kernel(
                self,
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                eris=eris,
                mu=mu,
            )
            if self.ensemble == "c_scf":
                delta = n - get_correct_nval(self)
                while abs(delta) > self.mu_tol:
                    mu = mu - delta / dn
                    self.e_corr, n, dn = gc_kernel(
                        self,
                        mo_energy=mo_energy,
                        mo_coeff=mo_coeff,
                        eris=eris,
                        mu=mu,
                    )
                    delta = n - get_correct_nval(self)
        else:
            raise NotImplementedError("Non-canonical FT-MP2")

        cput1 = log.timer('kernel', *cput1)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        log.timer(self.__class__.__name__, *cput0)

        self._finalize()
        return self.e_corr, self.t2

    def get_nocc_nvir_nval(self, mu):
        raise NotImplementedError
    
    def _init_smearing(self):
        assert self._scf.converged
        nelec = get_correct_nval(self)
        mu, occs = _smearing_optimize(_fermi_smearing_occ, self._scf.mo_energy,
                                      nelec // 2, 1.0 / self.beta)
        return mu.item()

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mo_coeff=mo_coeff)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            mf = mp._scf.to_hf()
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)


class FTMP2(FTMP2Mixin, MP2Base):
    '''restricted kappa-MP2 with canonical HF
    '''
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 beta=20, mu_algo="mu0", mu0=None,
                 occ_tol=1e-7, mu_tol=1e-6):
        MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        FTMP2Mixin.__init__(self, beta, mu_algo, mu0, occ_tol, mu_tol)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        #return gc_kernel(self, mo_energy, mo_coeff, eris, with_t2)
        raise NotImplementedError

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('FT-MP2 with beta = %.3g', self.beta)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E(SCS-%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot_scs, self.emp2_scs)
        log.info('E_corr(same-spin) = %.15g', self.e_corr_ss)
        log.info('E_corr(oppo-spin) = %.15g', self.e_corr_os)
        return self

    def ao2mo(self, mo_coeff=None, mu=None):
        return _make_eris(self, mo_coeff, mu=mu, verbose=self.verbose)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nocc_nvir_nval = get_nocc_nvir_nval


FTRMP2 = FTMP2


def _make_eris(mp, mo_coeff=None, ao2mofn=None, mu=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc, nvir, nval = mp.get_nocc_nvir_nval(mu)
    eris.nocc = nocc
    eris.nvir = nvir
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:, :nocc], order='F')
    cv = numpy.asarray(mo_coeff[:, -nvir:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        #log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
        #         '3-tensor integrals.\n'
        #         'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        eris.ovov = _ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    log.timer('Integral transformation', *time0)
    return eris
