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
from pyscf.acmp.ac_mp2 import ACMP_PAIRED, concatenate_w, ACMP2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def gc_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, mu=None,
              return_osmi=False, with_singles=True, beta=None):
    small_gap = 1e-10
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if beta is None:
        beta = mp.beta

    if eris is None:
        eris = mp.ao2mo(mo_coeff, mu, beta=beta)

    if mo_energy is None:
        mo_energy = eris.mo_energy
    
    occs = _fermi_smearing_occ(mu, mo_energy, 1.0 / beta)

    nocc = eris.nocc
    nvir = eris.nvir
    eia = mo_energy[:nocc,None] - mo_energy[None,-nvir:]

    e0, de0, v1 = mp._get_v1(mo_energy, eris.mo_coeff, occs)
    dvdu, d2vdu2, dvdeu = mp._get_dv1(mo_energy, eris.mo_coeff, occs, beta=beta)
    v1ia = v1[:nocc,-nvir:]
    v1ii = numpy.diag(v1)[:nocc]
    dvdu_ii = numpy.diag(dvdu)[:nocc]
    dvdu = dvdu[:nocc,-nvir:]
    d2vdu2 = d2vdu2[:nocc,-nvir:]
    dvdeu = dvdeu[:nocc,-nvir:]
    v1ia2 = v1ia * v1ia.conj()

    dn_ia = occs[:nocc, None] * (1 - occs[None, -nvir:])
    if return_osmi:
        dn0_ia = numpy.ones_like(occs[:nocc, None]) * (1 - occs[None, -nvir:])
    else:
        dn0_ia = dn_ia

    # Multiplier for the anomalous energy term
    emul_ia = mo_energy[None, -nvir:] * occs[None, -nvir:]
    emul_ia = emul_ia - mo_energy[:nocc,None] * (1 - occs[:nocc, None])
    emul_ia[:] *= beta

    # Multiplier for the anomalous particle number term
    nterm_ia = -occs[None, -nvir:] + (1 - occs[:nocc, None])
    nterm_ia[:] *= beta

    # Multiplier for the derivative of the anomalous particle number term 
    dnterm_ia = occs[None, -nvir:] * (1 - occs[None, -nvir:])
    dnterm_ia = dnterm_ia + (1 - occs[:nocc, None]) * occs[:nocc, None]
    dnterm_ia[:] *= -1 * beta ** 2

    def _get_ei_helper(gap, get_tderiv=False):
        cond = gap < 0
        cond2 = numpy.abs(gap) > small_gap
        expei = numpy.exp(-beta * numpy.abs(gap))
        ei = numpy.empty_like(gap)
        ei[:] = -0.5 * beta
        ei[cond2] = 1.0 / gap[cond2]
        if get_tderiv:
            tmp1 = 2 * beta * gap + (beta * gap)**3
            tmp2 = (beta * gap)**2 - 2
            dexpei = numpy.where(
                cond,
                tmp1 * expei * (1 - 2 * expei - expei * expei) + tmp2 * expei * (1 - expei) * (1 + expei * expei),
                tmp1 * expei * (expei * expei - 2 * expei - 1) + tmp2 * (expei - 1) * (1 + expei * expei),
            ) / (1 + expei * expei)**2
        expei[:] = numpy.where(
            cond,
            expei * (1 - expei),
            (expei - 1)
        ) / (1 + expei * expei)
        ei[cond2] += (
            (1.0 / beta)
            * (2 + beta**2 * gap * gap) * expei
            / (gap * gap)
        )[cond2]
        if get_tderiv:
            dei = numpy.empty_like(gap)
            dei[:] = -0.5 * beta
            dei[cond2] = dexpei[cond2] / (beta * gap[cond2] * gap[cond2])
            return ei, dei
        return ei

    fancy = True
    if fancy:
        inve, dinve = _get_ei_helper(eia, True)
    else:
        inve = eia.copy()
        inve[numpy.abs(inve) < 1e-10] = -2 / beta
        inve[:] = 1.0 / inve

    if return_osmi:
        w_list = [numpy.zeros((nocc, nocc)) for _ in range(mp.get_pt_list_size())]
        if mp.ensemble is not None and mp.ensemble != "fd":
            err = "OSMI only supports ensemble=None or 'fd'"
            raise ValueError(err)
    if with_singles:
        if mp.ensemble is None or mp.ensemble == "fd":
            e1 = (2 * inve * v1ia2 * dn_ia).sum()
            e1 += de0
            n1 = 0
            n2 = 0
        elif mp.ensemble in ["gc", "c_scf"]:
            if fancy:
                dndu = beta * occs[:nocc] * (1 - occs[:nocc])
                #iocc = numpy.argmax(dndu)
                #occs_fd = occs.copy()
                #occs_fd[iocc] += 1e-5
                #e01, de01, v11 = mp._get_v1(mo_energy, eris.mo_coeff, occs_fd)
                #v11 = (de01 - de0) / 1e-5
                #print("V11", de0, v11, v1[iocc, iocc])
                #print(mo_energy[:nocc], numpy.diag(v1ia).real, dndu)
                #print(mo_energy[:nocc] * numpy.diag(v1ia).real, dndu)

                # factor of 2 for spin
                e1 = (2 * inve * v1ia2 * dn_ia * (1 + emul_ia)).sum()
                e1 += (2 * dinve * v1ia2 * dn_ia).sum()
                e1 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdeu)).sum()
                e1 -= (2 * mo_energy[:nocc] * v1ii.real * dndu).sum()
                e1 += de0
                n1 = (-2 * inve * v1ia2 * dn_ia * nterm_ia).sum()
                n1 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdu)).sum()
                n1 -= (2 * v1ii.real * dndu).sum()
                n2 = (-2 * inve * v1ia2 * dn_ia * (nterm_ia * nterm_ia + dnterm_ia)).sum()
                n2 += (-8 * inve * dn_ia * nterm_ia * numpy.real(v1ia.conj() * dvdu)).sum()
                n2 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * d2vdu2)).sum()
                n2 += (-4 * inve * dn_ia * numpy.real(dvdu.conj() * dvdu)).sum()
                n2 -= (2 * dvdu_ii.real * dndu).sum()
                n2 -= (2 * v1ii.real * dndu * beta * (1 - 2 * occs[:nocc])).sum()
            else:
                # factor of 2 for spin
                e1 = (2 * inve * v1ia2 * dn_ia * (1 + emul_ia)).sum()
                # another factor of 2 for d/du (v^2)
                e1 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdeu)).sum()
                e1 += de0
                dndu = beta * occs[:nocc] * (1 - occs[:nocc])
                e1 -= (2 * mo_energy[:nocc] * numpy.diag(v1ia).real * dndu).sum()
                n1 = (-2 * inve * v1ia2 * dn_ia * nterm_ia).sum()
                n1 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdu)).sum()
                n1 -= (2 * numpy.diag(v1ia).real * dndu).sum()
                n2 = (-2 * inve * v1ia2 * dn_ia * (nterm_ia * nterm_ia + dnterm_ia)).sum()
                n2 += (-8 * inve * dn_ia * nterm_ia * numpy.real(v1ia.conj() * dvdu)).sum()
                n2 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * d2vdu2)).sum()
                n2 += (-4 * inve * dn_ia * numpy.real(dvdu.conj() * dvdu)).sum()
                n2 -= (2 * numpy.diag(dvdu) * dndu).sum()
                n2 -= (2 * numpy.diag(v1ia).real * dndu * beta * (1 - 2 * occs[:nocc])).sum()
        elif mp.ensemble == "c_pt2":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported thermal ensemble")
    else:
        e1 = n1 = n2 = 0
    print("E1", e1)

    with_t2 = False  # TODO
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    n0_count = 0
    nval_count = n1
    dnval_count = n2

    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        n0_count += 2 * occs[i]
        dnval_count += 2 * beta * occs[i] * (1 - occs[i])
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        ei = lib.direct_sum('jb+a->jba', eia, eia[i])

        if mp.ensemble is None or mp.ensemble == "fd":
            if not return_osmi:
                ei[numpy.abs(ei) < small_gap] = -2 / beta
                ei[:] = lib.einsum('jb,a->jba', dn_ia, dn_ia[i]) / ei
            else:
                ei = _get_ei_helper(ei)
                ei[:] *= lib.einsum('jb,a->jba', dn0_ia, dn_ia[i])
        elif mp.ensemble in ["gc", "c_scf"]:
            ai = lib.direct_sum('jb+a->jba', emul_ia, emul_ia[i])
            a2i = lib.direct_sum('jb+a->jba', nterm_ia, nterm_ia[i])
            a3i = lib.direct_sum('jb+a->jba', dnterm_ia, dnterm_ia[i])
            # two equivalent ways of getting these terms
            if False:
                occi = lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
                ei, dei = _get_ei_helper(ei, True)
                print(ei.shape, dei.shape, occi.shape, ai.shape, a2i.shape, a3i.shape)
                ni = -ei * a2i * occi
                ni2 = -ei * (a2i * a2i + a3i) * occi
                ei[:] *= (1 + ai)
                ei[:] += dei
                ei[:] *= occi
            else:
                cond = numpy.abs(ei) < small_gap
                ei[cond] = 1
                ni = -1.0 / ei
                ni[cond] = 0.5 * beta
                ni[:] *= a2i * lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
                ni2 = -1.0 / ei
                ni2[cond] = 0.5 * beta
                ni2[:] *= (a2i * a2i + a3i) * lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
                ei[:] = (1 + ai) / ei
                ei[cond] = -beta * (1 + 0.5 * ai[cond])
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

        if return_osmi:
            mp.add_to_w_list_(w_list, gi, gi, ei, ACMP_PAIRED, invert_ei=False)
        else:
            t2i = gi.conj() * ei
            edi = numpy.einsum('jab,jab', t2i, gi) * 2
            exi = -numpy.einsum('jab,jba', t2i, gi)
            emp2_ss += edi*0.5 + exi
            emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i
    print("NVAL", nval_count)

    if return_osmi:
        return w_list, occs[:nocc]
    else:
        emp2_ss = emp2_ss.real + e1.real
        emp2_os = emp2_os.real
        emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os, e0=e0)
        return emp2, nval_count + n0_count, dnval_count


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


def get_nocc_nvir_nval(mp, mu, beta):
    occ = _fermi_smearing_occ(mu, mp._scf.mo_energy, 1.0 / beta)
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
    _keys = {"mu_algo", "occ_tol", "mu_tol", "mu0", "beta",
             "ensemble", "max_mu_steps"}

    def __init__(self, beta=1, mu_algo="mu0", mu0=None,
                 occ_tol=0, mu_tol=1e-6, ensemble=None):
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
        self.max_mu_steps = 50

    @property
    def nocc(self):
        return self._nocc
    @nocc.setter
    def nocc(self, n):
        raise ValueError("Cannot set nocc for FT-MP2")

    def _get_e0(self, mo_energy, mo_coeff, ac_occ):
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

        if not hasattr(self._scf, "to_hf"):
            mf = self._scf
        else:
            mf = self._scf.to_hf()
        # HF energy of the current occupations
        # Equal to the sum of eqs 55 and 56 of Santra & Schirmer
        e0 = mf.energy_tot(dm=dm, vhf=vhf)
        print("EDIFF", e0 - self.e_hf)
        # de0 = e0 - self.e_hf

        # TODO need to modify this for frozen core
        de0 = e0 - mo_energy.dot(ac_occ)

        fockao = mf.get_fock(vhf=vhf, dm=dm)
        fock = mo_coeff.conj().T.dot(fockao).dot(mo_coeff)
        return mo_energy.dot(ac_occ) - self.e_hf, de0, fock - numpy.diag(mo_energy)

    def _get_dv1(self, mo_energy, mo_coeff, ac_occ, beta):
        assert beta is not None
        mo_occ = ac_occ * (1 - ac_occ) * beta
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf1 = vj - 0.5 * vk
        vhf1 = mo_coeff.conj().T.dot(vhf1).dot(mo_coeff)

        mo_occ *= beta * (1 - 2 * ac_occ)
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf2 = vj - 0.5 * vk
        vhf2 = mo_coeff.conj().T.dot(vhf2).dot(mo_coeff)

        mo_occ = ac_occ * (1 - ac_occ) * beta * mo_energy
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf3 = vj - 0.5 * vk
        vhf3 = mo_coeff.conj().T.dot(vhf3).dot(mo_coeff)
        return vhf1, vhf2, vhf3

    gc_kernel = gc_kernel

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
            eris = self.ao2mo(mo_coeff, mu, beta=self.beta)

        cput1 = log.timer('ao2mo', *cput1)

        if self._scf.converged:
            if self.mu_algo != "mu0":
                raise NotImplementedError
            self.e_corr, n, dn = self.gc_kernel(
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                eris=eris,
                mu=mu,
                beta=self.beta
            )
            if self.ensemble == "c_scf":
                delta = n - get_correct_nval(self)
                for step in range(self.max_mu_steps):
                    mu = mu - delta / dn
                    self.e_corr, n, dn = self.gc_kernel(
                        mo_energy=mo_energy,
                        mo_coeff=mo_coeff,
                        eris=eris,
                        mu=mu,
                        beta=self.beta
                    )
                    delta = n - get_correct_nval(self)
                    if abs(delta) < self.mu_tol:
                        break
                else:
                    raise RuntimeError("Chemical potential not converged!")
                self.e_corr += self.e_corr.e0
            elif self.ensemble == "fd":
                mu_delta = 1e-4
                beta_delta = self.beta * 0.00001
                ep_mu = self.gc_kernel(
                    mo_energy=mo_energy,
                    mo_coeff=mo_coeff,
                    eris=eris,
                    mu=mu + 0.5 * mu_delta,
                    beta=self.beta,
                )[0]
                em_mu = self.gc_kernel(
                    mo_energy=mo_energy,
                    mo_coeff=mo_coeff,
                    eris=eris,
                    mu=mu - 0.5 * mu_delta,
                    beta=self.beta,
                )[0]
                ep_beta = self.gc_kernel(
                    mo_energy=mo_energy,
                    mo_coeff=mo_coeff,
                    eris=eris,
                    mu=mu,
                    beta=self.beta + 0.5 * beta_delta
                )[0]
                em_beta = self.gc_kernel(
                    mo_energy=mo_energy,
                    mo_coeff=mo_coeff,
                    eris=eris,
                    mu=mu,
                    beta=self.beta - 0.5 * beta_delta
                )[0]
                nelec = (em_mu - ep_mu) / mu_delta
                print("DN", nelec)
                self.f_corr = self.e_corr
                print("ENTROPY", self.beta * (ep_beta - em_beta) / beta_delta)
                print("CHEME", mu * nelec)
                self.e_corr = (
                    self.f_corr
                    + self.beta * (ep_beta - em_beta) / beta_delta
                    + mu * nelec
                )
                self.e0_corr = 0.5 * (self.e_corr + self.f_corr)

                self.e_corr += self.f_corr.e0
                self.e0_corr += self.f_corr.e0
                self.f_corr += self.f_corr.e0
                print("ENERGIES", self.e_corr, self.f_corr, self.e0_corr, self.f_corr + mu * nelec)
            else:
                self.e_corr += self.e_corr.e0
        else:
            raise NotImplementedError("Non-canonical FT-MP2")

        self.mu_opt = mu

        cput1 = log.timer('kernel', *cput1)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        log.timer(self.__class__.__name__, *cput0)

        self._finalize()
        return self.e_corr, self.t2

    def get_nocc_nvir_nval(self, mu, beta):
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

    # full density matrix for RHF
    def _make_rdm1_for_dv(self, mo_coeff, mo_occ):
        '''One-particle density matrix in AO representation

        Args:
            mo_coeff : 2D ndarray
                Orbital coefficients. Each column is one orbital.
            mo_occ : 1D ndarray
                Occupancy
        Returns:
            One-particle density matrix, 2D ndarray
        '''
        dm = (mo_coeff*mo_occ).dot(mo_coeff.conj().T)
        return dm


class FTMP2(FTMP2Mixin, MP2Base):
    '''restricted kappa-MP2 with canonical HF
    '''
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 beta=20, mu_algo="mu0", mu0=None,
                 occ_tol=0, mu_tol=1e-6, ensemble=None):
        MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        FTMP2Mixin.__init__(self, beta, mu_algo, mu0, occ_tol, mu_tol,
                            ensemble)

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

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        #return gc_kernel(self, mo_energy, mo_coeff, eris, with_t2)
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None, mu=None, beta=None):
        return _make_eris(self, mo_coeff, mu=mu, beta=beta,
                          verbose=self.verbose)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nocc_nvir_nval = get_nocc_nvir_nval


FTRMP2 = FTMP2

class FTACMP2(FTMP2Mixin, ACMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 beta=20, mu_algo="mu0", mu0=None,
                 occ_tol=1e-10, mu_tol=1e-6, ensemble=None):
        ACMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        FTMP2Mixin.__init__(self, beta, mu_algo, mu0, occ_tol, mu_tol,
                            ensemble)

    def gc_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, mu=None,
                  with_singles=True, beta=None):
        if eris is None:
            eris = mp.ao2mo(mo_coeff, beta=beta)
        w_list, occs = gc_kernel(mp, mo_energy, mo_coeff, eris, mu,
                                 return_osmi=True, with_singles=False,
                                 beta=beta)
        if mo_coeff is None:
            mo_coeff = eris.mo_coeff[:, :eris.nocc]

        wlist_df = mp.get_acmp_df_wlist(mo_coeff)
        w_list = concatenate_w(w_list, wlist_df)
        lnocc = numpy.log(numpy.clip(occs, 1e-16, 1))
        wt_mat = numpy.exp(-0.25 * (lnocc - lnocc[:, None])**2)
        w_list = [((w + w.T) * 0.5 * wt_mat) for w in w_list]
        energy = 2 * mp.ac_interpolator(w_list, occs=occs)

        # TODO shouldn't set these to misleading values
        edi = energy
        exi = 0.0
        emp2_ss = edi * 0.5 + exi
        emp2_os = edi * 0.5
        # TODO e0 correction
        emp2 = lib.tag_array(energy, e_corr_ss=emp2_ss, e_corr_os=emp2_os, e0=0)
        mp.acmp_wlist = w_list

        return emp2.real, None, None

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('FT-AC-MP2 with si_limit = %s', self.si_limit)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        #return gc_kernel(self, mo_energy, mo_coeff, eris, with_t2)
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None, mu=None, beta=None):
        return _make_eris(self, mo_coeff, mu=mu, beta=beta,
                          verbose=self.verbose)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nocc_nvir_nval = get_nocc_nvir_nval


def _make_eris(mp, mo_coeff=None, ao2mofn=None, mu=None, verbose=None,
               beta=None):
    assert mu is not None
    assert beta is not None
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc, nvir, nval = mp.get_nocc_nvir_nval(mu, beta)
    nvir = max(nvir, 1)
    nocc = max(nocc, 1)
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
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv), compact=False)

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        #log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
        #         '3-tensor integrals.\n'
        #         'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv), compact=False)

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        eris.ovov = _ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    log.timer('Integral transformation', *time0)
    return eris
