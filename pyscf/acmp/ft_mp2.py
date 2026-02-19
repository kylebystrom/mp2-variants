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


def gc_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, mu=None,
              task="gp", with_singles=True, beta=None,
              smooth_edep=True):
    """
    Args:
        mo_energy: Zeroth-order Hamiltonian eigenvalues
        mo_coeff: Zeroth-order Hamiltonian eigenvectors
        eris: Electron repulsion integrals object
        mu: Chemical potential of the system. If mu is a float,
            it is the chemical potential. If mu is a 3-tuple,
            mu[i] is the i-th order perturbation expansion of
            the chemical potential.
        beta: Inverse temperature of the system
        with_singles: Compute singles contribution to second-order
            energy term
        smooth_edep: Whether to use a continuous and differentiable
            definition of the inverse energy gap. Can increase
            compute cost somewhat but is required for the OSMI
            task.

    Returns:
        result (dict): A dictionary of the results. Included
            values depends on the task, as described below.

    Note: with_singles refers to the singles contribution to the
        second-order energy term. For the first-order energy term,
        both singles and doubles contributions are computed.

    Note: All tasks return at least "e0", "f0", "n0". Additional
        result keys returned for each task are listed below.

    Tasks:
        "gp": Compute the first and second-order contributions
            to the grand potential.
            extra result keys: "gp1", "gp2"
        "opt": Compute the first and second-order contributions
            to the grand potential, particle number, and derivative
            of the particle number with respect to the chemical
            potential. Useful for finding the chemical potential
            that preserves the particle number.
            extra result keys: "gp1", "gp2", "n1", "n2", "dn0", "dn1", "dn2"
        "all": Compute the first and second-order contributions
            to the grand potential, particle number, and and
            internal energy. (The free energy can be extracted
            from these terms as well.)
            extra result keys: "gp1", "gp2", "n1", "n2", "e1", "e2"
        "mu": Compute the second-order perturbation expansion
            of the chemical potential.
            extra result keys: "mu1", "mu2", "n1", "n2", "dn0", "dn1", "dn2"
        "osmi": Compute the first-order contribution the
            grand potential and the OSMI W_0' matrix contribution
            to the grand potential.
            extra result keys: "gp1", "w_list"

    result keys:
        "dn0": Derivative of n0 wrt chemical potential
        "dn1": Derivative of n1 wrt chemical potential
        "dn2": Derivative of n2 wrt chemical potential
        "e0": Zeroth-order contribution to the internal energy, i.e.
            the sum of mo_energy * mo_occ
        "e1": First-order contribution to the internal energy.
        "e2": First-order contribution to the internal energy.
        "f0": Zeroth-order contribution to the free energy, i.e.
            e0 + the entropy term of the occupations.
        "gp1": First-order contribution to the grand potential.
        "gp2": Second-order contribution to the grand potential.
        "mu1": First-order expansion term of the chemical potential
        "mu2": Second-order expansion term of the chemical potential
        "n0": Zeroth-order particle number, i.e. sum(mo_occ)
        "n1": First-order particle number contribution
        "n2": Second-order particle number contribution
        "occs": Occupation numbers computed using the mu passed to this
            function, i.e. the zeroth-order chemical potential mu^{(0)}
        "w_list": W matrices for OSMI calculations. Contains second-order
            contributions to the grand potential, decomposed into a matrix.
    """
    all_tasks = ["gp", "opt", "all", "mu", "osmi"]
    results_keys = ["occs", "e0", "f0", "n0", "mu0"]
    extra_keys = {
        "gp": ["gp1", "gp2"],
        "opt": ["gp1", "gp2", "n1", "n2", "dn0", "dn1", "dn2"],
        "all": ["gp1", "gp2", "n1", "n2", "e1", "e2"],
        "mu": ["mu1", "mu2", "n1", "n2", "dn0", "dn1", "dn2"],
        "osmi": ["gp1", "gp2", "w_list"],
    }
    results_keys = results_keys + extra_keys[task]

    if isinstance(mu, tuple):
        assert len(mu) == 3
        mu, mu1, mu2 = mu
    else:
        mu1 = None
        mu2 = None

    if task not in all_tasks:
        raise ValueError("Unsupported FT-PT2 task")

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

    if False: # mu1 is not None:
        dmoe = mo_energy - mu
    else:
        dmoe = mo_energy

    occs = _fermi_smearing_occ(mu, mo_energy, 1.0 / beta)

    nocc = eris.nocc
    nvir = eris.nvir
    eia = mo_energy[:nocc, None] - mo_energy[None, -nvir:]

    clipped_occs = numpy.clip(occs, 1e-200, 1)
    clipped_1occs = numpy.clip(1 - occs, 1e-200, 1)
    e0, gp1, v1 = mp._get_v1(mo_energy, eris.mo_coeff, occs)
    f0 = clipped_occs * numpy.log(clipped_occs)
    f0 += clipped_1occs * numpy.log(clipped_1occs)
    f0 = e0 + 2 / beta * f0.sum()
    n0 = 2 * numpy.sum(occs)
    results = {"e0": e0, "f0": f0, "n0": n0, "occs": occs, "mu0": mu}
    if "dn0" in results_keys:
        results["dn0"] = 2 * beta * numpy.sum(occs * (1 - occs))
    if task == "mu":
        ddn0 = 2 * beta**2 * numpy.sum(occs * (1 - occs) * (1 - 2 * occs))

    if with_singles:
        v1ia = v1[:nocc, -nvir:]
        v1ia2 = v1ia * v1ia.conj()

    if task in ["opt", "all", "mu"]:
        dvdu, d2vdu2, dvdeu = mp._get_dv1(
            dmoe, eris.mo_coeff, occs, beta=beta
        )
        dvdu_ii = numpy.diag(dvdu)[:nocc]
        dvdu = dvdu[:nocc, -nvir:]
        d2vdu2 = d2vdu2[:nocc, -nvir:]
        dvdeu = dvdeu[:nocc, -nvir:]

    v1ii = numpy.diag(v1)[:nocc]
    dndu = beta * occs[:nocc] * (1 - occs[:nocc])
    if "gp1" in results_keys:
        results["gp1"] = gp1
        if mu1 is not None:
            results["gp1"] -= mu1 * n0
    if "e1" in results_keys:
        results["e1"] = gp1 - (2 * mo_energy[:nocc] * v1ii.real * dndu).sum()
        if mu1 is not None:
            results["e1"] += mu1 * (2 * mo_energy[:nocc] * dndu).sum()
    if "n1" in results_keys:
        results["n1"] = (-2 * v1ii.real * dndu).sum()
        #if mu1 is not None:
        #    results["n1"] += (2 * dndu).sum() * mu1
    if "dn1" in results_keys:
        results["dn1"] = (-2 * dvdu_ii.real * dndu).sum()
        results["dn1"] -= (2 * v1ii.real * dndu * beta * (1 - 2 * occs[:nocc])).sum()
        #if mu1 is not None:
        #    raise NotImplementedError

    dn_ia = occs[:nocc, None] * (1 - occs[None, -nvir:])
    if task == "osmi":
        dn0_ia = numpy.ones_like(occs[:nocc, None]) * (1 - occs[None, -nvir:])
    else:
        dn0_ia = None

    if task == "all":
        # Multiplier for the anomalous energy term
        emul_ia = dmoe[None, -nvir:] * occs[None, -nvir:]
        emul_ia = emul_ia - dmoe[:nocc, None] * (1 - occs[:nocc, None])
        emul_ia[:] *= beta

    if task in ["opt", "all", "mu"]:
        # Multiplier for the anomalous particle number term
        nterm_ia = -occs[None, -nvir:] + (1 - occs[:nocc, None])
        nterm_ia[:] *= beta

    if task in ["opt", "mu"]:
        # Multiplier for the derivative of the anomalous particle number term 
        dnterm_ia = occs[None, -nvir:] * (1 - occs[None, -nvir:])
        dnterm_ia = dnterm_ia + (1 - occs[:nocc, None]) * occs[:nocc, None]
        dnterm_ia[:] *= -1 * beta ** 2

    if smooth_edep:
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
                    (tmp1 * expei * (1 - 2 * expei - expei * expei)
                     + tmp2 * expei * (1 - expei) * (1 + expei * expei)),
                    (tmp1 * expei * (expei * expei - 2 * expei - 1)
                     + tmp2 * (expei - 1) * (1 + expei * expei)),
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
    else:
        def _get_ei_helper(gap, get_tderiv=False):
            cond2 = numpy.abs(gap) > small_gap
            ei = numpy.empty_like(gap)
            ei[:] = -0.5 * beta
            ei[cond2] = 1.0 / gap[cond2]
            if get_tderiv:
                dei = numpy.zeros_like(gap)
                dei[numpy.logical_not(cond2)] = -0.5 * beta
                return ei, dei
            return ei

    inve, dinve = _get_ei_helper(eia, True)

    if task == "osmi":
        w_list = [numpy.zeros((nocc, nocc)) for _ in range(mp.get_pt_list_size())]

    if with_singles:
        # factors of 2 for spin
        if "w_list" in results_keys:
            results["gp2"] = 0.0
            tmp = numpy.sqrt(-inve * dn0_ia) * v1ia
            w_list[0][:] = -lib.einsum("ia,ja->ij", tmp, tmp.conj())
            w_list[0][:] = w_list[0] + w_list[0].T.conj()
        elif "gp2" in results_keys:
            results["gp2"] = (2 * inve * v1ia2 * dn_ia).sum()
        if "e2" in results_keys:
            e2 = (2 * inve * v1ia2 * dn_ia * (1 + emul_ia)).sum()
            e2 += (2 * dinve * v1ia2 * dn_ia).sum()
            e2 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdeu)).sum()
            results["e2"] = e2
        if "n2" in results_keys:
            n2 = (-2 * inve * v1ia2 * dn_ia * nterm_ia).sum()
            n2 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * dvdu)).sum()
            results["n2"] = n2
        if "dn2" in results_keys:
            n2 = (-2 * inve * v1ia2 * dn_ia * (nterm_ia * nterm_ia + dnterm_ia)).sum()
            n2 += (-8 * inve * dn_ia * nterm_ia * numpy.real(v1ia.conj() * dvdu)).sum()
            n2 += (-4 * inve * dn_ia * numpy.real(v1ia.conj() * d2vdu2)).sum()
            n2 += (-4 * inve * dn_ia * numpy.real(dvdu.conj() * dvdu)).sum()
            results["dn2"] = n2
    else:
        if "gp2" in results_keys:
            results["gp2"] = 0.0
        if "e2" in results_keys:
            results["e2"] = 0.0
        if "n2" in results_keys:
            results["n2"] = 0.0
        if "dn2" in results_keys:
            results["dn2"] = 0.0
    if mu2 is not None:
        if "gp2" in results_keys:
            # factor of 2 for nelec, factor of 1 for second deriv taylor
            results["gp2"] -= mu1**2 * dndu.sum()
            results["gp2"] -= mu2 * n0
            # the term in parentheses is -1 * n1 but we might not be computing
            # it earlier, depending on the task
            results["gp2"] += mu1 * (2 * v1ii.real * dndu).sum()
        if "e2" in results_keys:
            results["e2"] += 2 * mu1 * (2 * v1ii.real * dndu).sum()
            results["e2"] -= 2 * mu1**2 * dndu.sum()
            dndudb = dndu * dmoe[:nocc] * (1 - 2 * occs[:nocc])
            results["e2"] -= 2 * beta * mu1 * (v1ii.real * dndudb).sum()
            results["e2"] -= 2 * (mu1 * dndu * dmoe[:nocc] * dvdu_ii.real).sum()
            # factor or 2 and factor 1/2 cancel again here
            results["e2"] += beta * mu1**2 * dndudb.sum()
            results["e2"] += 2 * mu2 * (dmoe[:nocc] * dndu).sum()

    e2_ss = e2_os = 0
    gp2_ss = gp2_os = 0
    nval_count = dnval_count = 0

    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
        gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
        ei = lib.direct_sum('jb+a->jba', eia, eia[i])

        if task == "osmi":
            ei = _get_ei_helper(ei)
            ei[:] *= lib.einsum('jb,a->jba', dn0_ia, dn_ia[i])
            mp.add_to_w_list_(w_list, gi, gi, ei, ACMP_PAIRED, invert_ei=False)
        else:
            occi = lib.einsum('jb,a->jba', dn_ia, dn_ia[i])
            if task == "all":
                ei, dei = _get_ei_helper(ei, True)
                dei *= occi
            else:
                ei = _get_ei_helper(ei)
                dei = None
            ei[:] *= occi
            if "gp2" in results_keys:
                t2i = gi.conj() * ei
                edi = numpy.einsum('jab,jab', t2i, gi) * 2
                exi = -numpy.einsum('jab,jba', t2i, gi)
                gp2_ss += edi*0.5 + exi
                gp2_os += edi*0.5
            if "n2" in results_keys:
                a2i = lib.direct_sum('jb+a->jba', nterm_ia, nterm_ia[i])
                ni = -ei * a2i
                ni = gi.conj() * ni
                ndi = numpy.einsum('jab,jab', ni, gi) * 2
                nxi = -numpy.einsum('jab,jba', ni, gi)
                nval_count += ndi + nxi
            if "dn2" in results_keys:
                a3i = lib.direct_sum('jb+a->jba', dnterm_ia, dnterm_ia[i])
                ni = -ei * (a2i * a2i + a3i)
                ni = gi.conj() * ni
                ndi = numpy.einsum('jab,jab', ni, gi) * 2
                nxi = -numpy.einsum('jab,jba', ni, gi)
                dnval_count += ndi + nxi
            if "e2" in results_keys:
                ai = lib.direct_sum('jb+a->jba', emul_ia, emul_ia[i])
                ei[:] *= (1 + ai)
                ei[:] += dei
                t2i = gi.conj() * ei
                edi = numpy.einsum('jab,jab', t2i, gi) * 2
                exi = -numpy.einsum('jab,jba', t2i, gi)
                e2_ss += edi*0.5 + exi
                e2_os += edi*0.5

    # These results might already have singles contributions,
    # so we have to add to them. If with_singles=False,
    # they are set to 0 above.
    if "gp2" in results_keys:
        results["gp2"] = lib.tag_array(
            gp2_ss + gp2_os + results["gp2"],
            gp_corr_ss=gp2_ss,
            gp_corr_os=gp2_os,
        )
    if "n2" in results_keys:
        results["n2"] += nval_count
    if "dn2" in results_keys:
        results["dn2"] += dnval_count
    if "e2" in results_keys:
        results["e2"] = lib.tag_array(
            e2_ss + e2_os + results["e2"],
            e_corr_ss=e2_ss,
            e_corr_os=e2_os,
        )
    if "w_list" in results_keys:
        results["w_list"] = [0.5 * (w + w.conj().T) for w in w_list]

    if task == "mu":
        # Regularize the denominator to stop things from exploding
        # in the zero-T limit.
        results["mu1"] = -1 * results["n1"] / (results["dn0"] + 1e-16)
        results["mu2"] = results["n2"]
        results["mu2"] += 0.5 * results["mu1"]**2 * ddn0
        results["mu2"] += results["mu1"] * results["dn1"]
        results["mu2"] *= -1 / (results["dn0"] + 1e-16)

    return results


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

    __name_mixin__ = "FTPT2-"

    _keys = {"mu_algo", "occ_tol", "mu_tol", "mu0", "beta",
             "particle_fix", "ecorr_method", "max_mu_steps",
             "with_singles", "mu_opt"}

    def __init__(self, mymp, beta=1, mu0=None, occ_tol=0, mu_tol=1e-6,
                 particle_fix=None, ecorr_method="analytical",
                 max_mu_steps=50, with_singles=True):
        """
        beta (float): 1 / (k_B * T) in atomic units.
        mu0 (float): Initial guess for mu. If not provided, it is set to the
            value that results in <N>=N for the zeroth-order Hamiltonian.
        occ_tol (float): Orbitals with mo_occ < occ_tol are considered fully
            virtual, and orbitals with mo_occ > 1 - occ_tol are considered
            fully occupied.
        mu_tol (float): Only used for mu_algo="search". The tolerance for
            converging mu with the Newton algorithm.
        particle_fix (str) is the method for fixing the number of particles in
            the system. In the grand canonical ensemble at finite temperature,
            PT2 can change the expectation value of the number of electrons,
            which may be undesirale. The options are:
                None: Use the initial guess mu0 as the chemical potential,
                    and do not fix the number of electrons
                "pt": Expand the chemical potential in orders of the
                    perturbation strength and solve for the chemical
                    potential that preserves the electron number
                    to second order.
                "dv": Expand the effective potential in order of the
                    perturbation strength and solve for the effective
                    potential that preserves the electron number
                    to second order.
                "iter": Iteratively solve for the chemical potential for
                    which the expectation value of the particle number
                    at the PT2 level is equal to the number of electrons
                    in the underlying mean-field calculation.
                "iter_fd": Same as "iter", but use finite-difference
                    to compute the particle number and is mu derivative
                    at each step. Useful for testing and for if the underlying
                    PT2 method does not support explicit particle number
                    computation.
        ecorr_method (str) is the method for computing the correlation
            energy from the correlation grand potential. The FT-PT2 formalism
            computes the correlation grand potential Omega_c, which is related
            to the correlation energy by
            E_c = Omega_c + T S_c + mu N_c
                = Omega_c + beta (dOmega_c / dbeta) - mu (dOmega_c / dmu)
            The means that computing E_c requires differentiating Omega_c
            with respect to beta and mu. This variable determines how these
            derivatives are handled.
            The default is "exact", with the options being
                "zeroth_order": Do not differentiate Omega_c with respect
                    to beta and mu. NOTE: This is a coarse approximation
                    and can be drastically different than the analytical
                    approach.
                "first_order": (TODO NOT IMPLEMENTED) Compute the Omega_c
                    derivatives only for the first-order term Omega^{(1)}.
                    NOTE: This is a coarse approximation and can be
                    drastically different than the analytical approach.
                "analytical": Explicitly compute the Omega_c derivatives
                    to second order, i.e. provide the "exact" PT2 internal
                    correlation energy within the grand canonical formalism.
                "finite_difference": Same as analytical, but perform the
                    derivatives of the first and second order terms using
                    a finite difference stencil.
                "1ana_2fd": (TODO NOT IMPLEMENTED)
                    Same as analytical, but perform the derivatives
                    of Omega^{(1)} analytically and the derivatives
                    of Omega^{(2)} with finite difference.
        with_singles (bool): Whether to include single-excitation contributions
            in the second-order perturbation terms.
        """
        self.__dict__.update(mymp.__dict__)
        self.beta = beta
        self.mu0 = mu0
        self.occ_tol = occ_tol
        self.mu_tol = mu_tol
        if particle_fix not in [None, "pt", "iter", "iter_fd"]:
            raise ValueError("Unsupported particle_fix={}".format(particle_fix))
        self.particle_fix = particle_fix
        if ecorr_method not in ["zeroth_order", "first_order", "analytical",
                                "finite_difference", "1ana_2fd"]:
            raise ValueError("Unsupported ecorr_method={}".format(ecorr_method))
        self.ecorr_method = ecorr_method
        self.with_singles = with_singles
        self._nocc = None
        self.mu_opt = None
        self.max_mu_steps = max_mu_steps

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

        if not hasattr(self._scf, "to_hf"):
            mf = self._scf
        else:
            mf = self._scf.to_hf()
        # HF energy of the current occupations
        # Equal to the sum of eqs 55 and 56 of Santra & Schirmer
        e0 = mf.energy_tot(dm=dm, vhf=vhf)

        # TODO need to modify this for frozen core
        de0 = e0 - mo_energy.dot(ac_occ)

        fockao = mf.get_fock(vhf=vhf, dm=dm)
        fock = mo_coeff.conj().T.dot(fockao).dot(mo_coeff)
        return mo_energy.dot(ac_occ), de0, fock - numpy.diag(mo_energy)

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

    @property
    def results(self):
        res = {
            "e_corr": self.e_corr,
            "e_tot": self.e_tot,
            "e_free": self.e_free,
            "e_zero": self.e_zero,
        }
        res.update(self._results)
        return res

    @property
    def e_corr(self):
        return self._e_tot - self.e_hf

    @property
    def e_tot(self):
        return self._e_tot

    @property
    def e_free(self):
        return self._f_tot

    @property
    def e_zero(self):
        return 0.5 * (self._e_tot + self._f_tot)

    @property
    def emp2_scs(self):
        raise NotImplementedError

    @property
    def e_tot_scs(self):
        raise NotImplementedError

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

        def _call_kernel(mu, beta, task):
            return self.gc_kernel(
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                eris=eris,
                mu=mu,
                beta=beta,
                with_singles=self.with_singles,
                task=task,
            )

        if not self._scf.converged:
            raise NotImplementedError("Non-canonical FT-MP2")

        if self.particle_fix == "pt":
            results = _call_kernel(mu, self.beta, "mu")
            mu = (results["mu0"], results["mu1"], results["mu2"])
        elif self.particle_fix in ["iter", "iter_fd"]:
            # TODO might be able to get away with only re-computing
            # ao2mo when a certain threshold change in occupations
            # is detected.
            use_fd = (self.particle_fix == "iter_fd")
            for step in range(self.max_mu_steps):
                if use_fd:
                    mu_delta = max(1e-3 / self.beta, 1e-7)
                    results = _call_kernel(mu, self.beta, "gp")
                    gp0 = results["gp1"] + results["gp2"]
                    resp = _call_kernel(mu + 0.5 * mu_delta, self.beta, "gp")
                    gpp = resp["gp1"] + resp["gp2"]
                    resm = _call_kernel(mu - 0.5 * mu_delta, self.beta, "gp")
                    gpm = resm["gp1"] + resm["gp2"]
                    n = results["n0"] + (gpm - gpp) / mu_delta
                    dn = (resp["n0"] - resm["n0"]) / mu_delta
                    dn += -4 * (gpm - 2 * gp0 + gpp) / mu_delta**2
                else:
                    results = _call_kernel(mu, self.beta, "opt")
                    n = results["n0"] + results["n1"] + results["n2"]
                    dn = results["dn0"] + results["dn1"] + results["dn2"]
                delta = n - get_correct_nval(self)
                if abs(delta) < self.mu_tol:
                    break
                else:
                    mu = mu - delta / dn
                    eris = self.ao2mo(mo_coeff, mu, beta=self.beta)
            else:
                raise RuntimeError("Chemical potential not converged!")
        else:
            # keep chemical potential fixed as the initial mu
            assert self.particle_fix is None
            results = {}

        if self.ecorr_method == "zeroth_order":
            if "gp1" not in results or "gp2" not in results:
                results.update(_call_kernel(mu, self.beta, "gp"))
            self._e_tot = results["e0"] + results["gp1"] + results["gp2"]
            # add zeroth-order entropic term to get free energy
            self._f_tot = self._e_tot + results["f0"] - results["e0"]
        elif self.ecorr_method == "first_order":
            raise NotImplementedError
        elif self.ecorr_method == "analytical":
            results.update(_call_kernel(mu, self.beta, "all"))
            self._f_tot = results["f0"] + results["gp1"] + results["gp2"]
            if isinstance(mu, tuple):
                # 1st and 2nd-order mu * N contribution
                # N is fixed to n0 to second order
                self._f_tot += (results["mu1"] + results["mu2"]) * results["n0"]
            else:
                # 1st and 2nd-order mu * N contribution
                # mu is fixed, n1 and n2 contribute to the expansion
                self._f_tot += mu * (results["n1"] + results["n2"])
            self._e_tot = results["e0"] + results["e1"] + results["e2"]
        elif self.ecorr_method == "finite_difference":
            if "gp1" not in results or "gp2" not in results:
                results.update(_call_kernel(mu, self.beta, "gp"))

            if isinstance(mu, tuple):
                # particle number is fixed by mu is not
                nelec = results["n0"]
                mu1_term = nelec * results["mu1"]
                mu2_term = nelec * results["mu2"]
            else:
                # Differentiate GP wrt mu to get particle number term
                mu_delta = max(1e-4 / self.beta, 1e-7)
                mu_delta = 1e-3 / self.beta
                res = _call_kernel(mu + 0.5 * mu_delta, self.beta, "gp")
                ep1_mu = res["gp1"]
                ep2_mu = res["gp2"]
                res = _call_kernel(mu - 0.5 * mu_delta, self.beta, "gp")
                em1_mu = res["gp1"]
                em2_mu = res["gp2"]
                nelec1 = (em1_mu - ep1_mu) / mu_delta
                nelec2 = (em2_mu - ep2_mu) / mu_delta
                mu1_term = nelec1 * mu
                mu2_term = nelec2 * mu

            # Differentiate GP wrt beta to get entropy term.
            # We compute this derivative with mu fixed, so if
            # particle_fix="pt", we still get the right derivative
            # for computing the entropy term.
            beta_delta = self.beta * 0.00001
            res = _call_kernel(mu, self.beta + 0.5 * beta_delta, "gp")
            ep1_beta = res["gp1"]
            ep2_beta = res["gp2"]
            res = _call_kernel(mu, self.beta - 0.5 * beta_delta, "gp")
            em1_beta = res["gp1"]
            em2_beta = res["gp2"]
            s1_term = self.beta * (ep1_beta - em1_beta) / beta_delta
            s2_term = self.beta * (ep2_beta - em2_beta) / beta_delta

            results["e1"] = results["gp1"] + mu1_term + s1_term
            results["e2"] = results["gp2"] + mu2_term + s2_term

            # Set the energies from the finite difference terms
            self._e_tot = results["e0"] + results["e1"] + results["e2"]
            self._f_tot = (
                results["f0"] + mu1_term + mu2_term
                + results["gp1"] + results["gp2"]
            )
        elif self.ecorr_method == "1ana_2fd":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported ecorr_method")

        self.mu_opt = mu

        cput1 = log.timer('kernel', *cput1)

        log.timer(self.__class__.__name__, *cput0)

        self._results = results
        self._finalize()
        return self.e_corr, self.results

    get_nocc_nvir_nval = get_nocc_nvir_nval

    def _init_smearing(self):
        assert self._scf.converged
        nelec = get_correct_nval(self)
        mu, occs = _smearing_optimize(_fermi_smearing_occ, self._scf.mo_energy,
                                      nelec // 2, 1.0 / self.beta)
        if isinstance(mu, float):
            return mu
        return mu.item()

    def get_e_hf(mp, mo_coeff=None):
        """
        Note: This should always be the total INTERNAL energy, not
        the free energy or zero-T extrapolated energy, even if
        smearing is used. The use of get_e_hf assumes this is
        internal energy and adds the smearing entropy term to
        get the free energy and zero-T extrapolated energy.
        """
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            if mo_coeff is None:
                mo_coeff = mp.mo_coeff
            if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
                return mp._scf.e_tot
            else:
                raise NotImplementedError("Non-canonical")
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

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('FT-MP2 with beta = %.3g', self.beta)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('F(%s) = %.15g  E0(extrapolated) = %.15g',
                 self.__class__.__name__, self.e_free, self.e_zero)
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


class FTACMP2Mixin(FTMP2Mixin):
    def gc_kernel(mp, mo_energy=None, mo_coeff=None, eris=None, mu=None,
                  task="gp", with_singles=True, beta=None,
                  smooth_edep=True):
        if task not in ["mu", "gp", "osmi", "opt"]:
            raise ValueError("Unsupported kernel task for FT-ACMP2")
        if task == "gp":
            task = "osmi"

        if eris is None:
            eris = mp.ao2mo(mo_coeff, beta=beta)
        kwargs = dict(
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            eris=eris,
            mu=mu,
            with_singles=with_singles,
            beta=beta,
            smooth_edep=smooth_edep,
        )
        if task == "mu":
            kwargs["task"] = "mu"
            return gc_kernel(mp, **kwargs)
        elif task == "opt":
            kwargs["task"] = "opt"
            return gc_kernel(mp, **kwargs)
        else:
            kwargs["task"] = "osmi"
            results = gc_kernel(mp, **kwargs)
        w_list = results["w_list"]
        occs = results["occs"][:eris.nocc]

        if mo_coeff is None:
            mo_coeff = eris.mo_coeff[:, :eris.nocc]

        wlist_df = mp.get_acmp_df_wlist(mo_coeff)
        w_list = concatenate_w(w_list, wlist_df)
        lnocc = numpy.log(numpy.clip(occs, 1e-16, 1))
        wt_mat = numpy.exp(-0.25 * (lnocc - lnocc[:, None])**2)
        w_list = [((w + w.T) * 0.5 * wt_mat) for w in w_list]
        print(results["gp2"], 2 * mp.ac_interpolator(w_list, occs=occs))
        results["gp2"] += 2 * mp.ac_interpolator(w_list, occs=occs)
        mp.acmp_wlist = w_list

        # results should contain gp1 and gp2 as predicted by OSMI
        return results


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
