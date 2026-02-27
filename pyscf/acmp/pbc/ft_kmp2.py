from pyscf.acmp import ft_mp2
from pyscf.acmp.pbc import ac_kmp2
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.scf.addons import _smearing_optimize, _fermi_smearing_occ
import numpy as np


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


def get_ft_occ_terms(task, beta, nocc, nvir, occi, moei, occa, moea):
    if task == "osmi":
        dn_ia = np.ones_like(occi[:nocc, None]) * (1 - occa[None, -nvir:])
    else:
        dn_ia = occi[:nocc, None] * (1 - occa[None, -nvir:])

    if task == "all":
        # Multiplier for the anomalous energy term
        emul_ia = moea[None, -nvir:] * occa[None, -nvir:]
        emul_ia = emul_ia - moei[:nocc, None] * (1 - occi[:nocc, None])
        emul_ia[:] *= beta
    else:
        emul_ia = None

    if task in ["opt", "all", "mu"]:
        # Multiplier for the anomalous particle number term
        nterm_ia = -occa[None, -nvir:] + (1 - occi[:nocc, None])
        nterm_ia[:] *= beta
    else:
        nterm_ia = None

    if task in ["opt", "mu"]:
        # Multiplier for the derivative of the anomalous particle number term 
        dnterm_ia = occa[None, -nvir:] * (1 - occa[None, -nvir:])
        dnterm_ia = dnterm_ia + (1 - occi[:nocc, None]) * occi[:nocc, None]
        dnterm_ia[:] *= -1 * beta ** 2
    else:
        dnterm_ia = None

    return dn_ia, emul_ia, nterm_ia, dnterm_ia


def gc_kernel(mp, mo_energy, mo_coeff, eris=None, mu=None,
              task="gp", with_singles=True, beta=None,
              smooth_edep=True, dv=None):
    """
    See the docs of the non-pbc version for details
    eris is a misnomer, it actually contains a nocc_list,
    nvir_list, nmo_list tuple so that it can stay constant
    between iterations and avoid discontinuous changes
    in nocc
    """
    if isinstance(mu, tuple):
        assert len(mu) == 3
        mu, mu1, mu2 = mu
    else:
        mu1 = None
        mu2 = None

    if beta is None:
        beta = mp.beta

    # TODO this should not be None
    if mo_energy is None:
        mo_energy = mp.mo_energy
    
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff

    nkpts = mp.nkpts
    small_gap = 1e-10
    occs_k = []
    if eris is None:
        nocc_list = []
        nvir_list = []
        nmo_list = []
    else:
        nocc_list, nvir_list, nmo_list = eris
    for k in range(nkpts):
        occs_k.append(_fermi_smearing_occ(mu, mo_energy[k], 1.0 / beta))
        if eris is None:
            nmo_list.append(mo_energy[k].size)
            nocc_list.append((occs_k[k] > mp.occ_tol).sum())
            nvir_list.append(((1 - occs_k[k]) > mp.occ_tol).sum())
    for _nvir in nvir_list:
        assert _nvir > 0

    with_df_ints = ac_kmp2._mp2_check_mem(
        mp, nkpts, np.max(nocc_list), np.max(nvir_list)
    )

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    oovv_ij = np.empty((nkpts), dtype=object)
    mo_e_o = [mo_energy[k][:nocc_list[k]] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][-nvir_list[k]:] for k in range(nkpts)]

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = _init_mp_df_eris(mp, nocc_list, nvir_list, nmo_list)
    else:
        raise NotImplementedError

    results = ft_mp2.initialize_ft_results(task)
    _get_ei_helper = ft_mp2.construct_ei_helper(
        beta, smooth_edep, small_gap, ("e2" in results)
    )
    results["occs"] = occs_k
    if task == "osmi":
        results["w_list"] = [[
            np.zeros((nocc_list[k], nocc_list[k]))
            for _ in range(mp.get_pt_list_size())
        ] for k in range(nkpts)]

    e0, gp1, v1 = mp._get_v1(mo_energy, mo_coeff, occs_k)
    results["e0"] = e0
    results["f0"] = e0  # entropy term added below
    results["mu0"] = mu
    if dv is not None:
        for k in range(nkpts):
            v1[k][:] += dv * np.identity(v1[k].shape[-1])
    
    n0_k = []
    dn0_k = []
    for k in range(nkpts):
        results["f0"] += ft_mp2.ft_entropy_0term(beta, occs_k[k])
        n0_k.append(2 * np.sum(occs_k[k]))
        if "dn0" in results:
            dn0_k.append(2 * beta * np.sum(occs_k[k] * (1 - occs_k[k])))
    results["n0"] = np.sum(n0_k)
    if "dn0" in results:
        results["dn0"] = np.sum(dn0_k)

    if task in ["opt", "all", "mu"]:
        dvdu_k, d2vdu2_k, dvdeu_k = mp._get_dv1(
            mo_energy, mo_coeff, occs_k, beta=beta
        )
        vterms_k = [[
            np.diag(dvdu_k[k])[:nocc_list[k]],
            dvdu_k[k][:nocc_list[k], -nvir_list[k]:],
            d2vdu2_k[k][:nocc_list[k], -nvir_list[k]:],
            dvdeu_k[k][:nocc_list[k], -nvir_list[k]:],
        ] for k in range(nkpts)]
    else:
        vterms_k = [[None] * 4] * nkpts

    muterms = [mu, mu1, mu2]
    for k in range(nkpts):
        ft_mp2.calculate_ft_mu_contribs_(
            results, beta, gp1, occs_k[k],
            mo_energy[k], nocc_list[k], n0_k[k],
            v1[k], muterms, vterms_k[k][0]
        )
        eia = mo_energy[k][:nocc_list[k], None]
        eia = eia - mo_energy[k][None, -nvir_list[k]:]
        eterms = _get_ei_helper(eia)
        dn_ia, dn0_ia, dterms = ft_mp2.get_ft_occ_terms(
            task, beta, occs_k[k], mo_energy[k], nocc_list[k], nvir_list[k]
        )
        occterms = [nocc_list[k], nvir_list[k], dn_ia, dn0_ia]
        if with_singles:
            ft_mp2.calculate_ft_singles_(
                results, v1[k], occterms, eterms, dterms, vterms_k[k][1:]
            )

    if task == "osmi":
        full_w_list = results["w_list"]

    e2_ss = e2_os = gp2_ss = gp2_os = 0
    scaled_kpts = mp._scf.cell.get_scaled_kpts(mp._scf.kpts)

    for ki in range(nkpts):
        if task == "osmi":
            results["w_list"] = full_w_list[k]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                # (ia|jb)
                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts) * lib.einsum(
                        "Lia,Ljb->iajb",
                        Lov[ki, ka],
                        Lov[kj, kb],
                    ).transpose(0,2,1,3)
                else:
                    raise NotImplementedError
                    orbo_i = mo_coeff[ki][:,:nocc]
                    orbo_j = mo_coeff[kj][:,:nocc]
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = fao2mo(
                        (orbo_i, orbv_a, orbo_j, orbv_b),
                        (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                        compact=False
                    ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                kpts = mp._scf.kpts
                diff = scaled_kpts[ki] + scaled_kpts[kj] - scaled_kpts[ka] - scaled_kpts[kb]
                diff = diff % 1
                diff[diff > 0.5] -= 1
                if np.linalg.norm(diff) > 1e-8:
                    raise ValueError("Not k-convserving!", kpts[ki], kpts[kj], kpts[ka], kpts[kb], diff)

                # Remove zero/padded elements from denominator
                eia = (mo_e_o[ki][:,None] - mo_e_v[ka])
                ejb = (mo_e_o[kj][:,None] - mo_e_v[kb])

                occi = occs_k[ki]
                occa = occs_k[ka]
                occj = occs_k[kj]
                occb = occs_k[kb]
                moei = mo_energy[ki]
                moea = mo_energy[ka]
                moej = mo_energy[kj]
                moeb = mo_energy[kb]
                dn_ia, emul_ia, nterm_ia, dnterm_ia = get_ft_occ_terms(
                    task, beta, nocc_list[k], nvir_list[k], occi, moei, occa, moea
                )
                _task = "gp" if task == "osmi" else task
                dn_jb, emul_jb, nterm_jb, dnterm_jb = get_ft_occ_terms(
                    _task, beta, nocc_list[k], nvir_list[k], occj, moej, occb, moeb
                )
                occ_ijab = lib.einsum('ia,jb->ijab', dn_ia, dn_jb)
                dterms = [
                    (emul_ia, emul_jb),
                    (nterm_ia, nterm_jb),
                    (dnterm_ia, dnterm_jb),
                ]

                eijab = lib.direct_sum('ia,jb->ijab', eia, ejb)
                eijab, deijab = _get_ei_helper(eijab)
                eijab *= occ_ijab
                if deijab is not None:
                    deijab[:] *= occ_ijab

                sterms = ft_mp2.add_ft_pt2_terms_(
                    mp, results, task, eijab, deijab,
                    (oovv_ij[ka], oovv_ij[kb]), dterms
                )
                e2_ss += sterms[0]
                e2_os += sterms[1]
                gp2_ss += sterms[2]
                gp2_os += sterms[3]
        if task == "osmi":
            full_w_list[k] = [0.5 * (w + w.conj().T) for w in results["w_list"]]

    if task == "mu":
        ddn0 = 0
        for k in range(nkpts):
            ddn0 += 2 * beta**2 * np.sum(
                occs_k[k] * (1 - occs_k[k]) * (1 - 2 * occs_k[k])
            )
    else:
        ddn0 = None
    if "w_list" in results:
        results["w_list"] = full_w_list
    ft_mp2.finalize_ft_results_(
        results, task, ddn0, e2_ss, e2_os, gp2_ss, gp2_os
    )
    for k in results:
        if k not in ["w_list", "occs"]:
            results[k] /= mp.nkpts
            if np.abs(results[k].imag) > 1e-10:
                logger.warn("Imaginary energy term in FT-MP2! {} {}".format(k, results[k]))
            results[k] = results[k].real

    return results


def _init_mp_df_eris(mp, nocc_list, nvir_list, nmo_list):
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2) -- A KMP2 instance

    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts, nkpts, naux, nocc, nvir)
    """
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(mp.stdout, mp.verbose)

    if mp._scf.with_df._cderi is None:
        mp._scf.with_df.build()

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nao = cell.nao_nr()

    # TODO account for mp.frozen
    mo_coeff = mp.mo_coeff  # _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    kpts = mp.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Lov = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())

    with df.CDERIArray(mp._scf.with_df._cderi) as cderi_array:
        tao = []
        ao_loc = None
        for ki in range(nkpts):
            for kj in range(nkpts):
                bra_start = 0
                bra_end = nocc_list[ki]
                ket_end = nmo_list[ki] + nmo_list[kj]
                ket_start = ket_end - nvir_list[kj]
                Lpq_ao = cderi_array[ki,kj]

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
                Lov[ki, kj] = out.reshape(-1, nocc_list[ki], nvir_list[kj])

    log.timer_debug1("transforming DF-MP2 integrals", *cput0)

    return Lov


class _PBC4FT_Mixin:
    def _get_v1(self, mo_energy, mo_coeff, ac_occ):
        ac_occ = [2 * o for o in ac_occ]
        if self.frozen is None:
            mo_occ = [o.copy() for o in ac_occ]
        elif isinstance(self.frozen, (int, np.integer)):
            mo_occ = [o.copy() for o in self._scf.mo_occ]
            for k in range(len(mo_occ)):
                mo_occ[k][self.frozen:] = ac_occ[k]
        elif hasattr(self.frozen, '__len__'):
            mask = self.get_frozen_mask()
            mo_occ = [o.copy() for o in self._scf.mo_occ]
            for k in range(len(mo_occ)):
                mo_occ[k][mask] = ac_occ
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
        de0 = e0

        fockao = mf.get_fock(vhf=vhf, dm=dm)
        dfock = [
            coeff.conj().T.dot(fock).dot(coeff)
            for coeff, fock in zip(mo_coeff, fockao)
        ]
        e0 = 0
        for k in range(self.nkpts):
            e0 += mo_energy[k].dot(ac_occ[k])
            dfock[k] -= np.diag(mo_energy[k])
        return e0, de0 - e0 / self.nkpts, dfock
    
    def _get_dv1(self, mo_energy, mo_coeff, ac_occ, beta):
        assert beta is not None
        mo_occ = [o * (1 - o) * beta for o in ac_occ]
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf1 = [
            coeff.conj().T.dot(j - 0.5 * k).dot(coeff)
            for coeff, j, k in zip(mo_coeff, vj, vk)
        ]

        for k in range(self.nkpts):
            mo_occ[k] *= beta * (1 - 2 * ac_occ[k])
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf2 = [
            coeff.conj().T.dot(j - 0.5 * k).dot(coeff)
            for coeff, j, k in zip(mo_coeff, vj, vk)
        ]

        for k in range(self.nkpts):
            mo_occ[k] = ac_occ[k] * (1 - ac_occ[k]) * beta * mo_energy[k]
        dm = self._make_rdm1_for_dv(mo_coeff, 2 * mo_occ)
        vj, vk = self._scf.get_jk(self.mol, dm)
        vhf3 = [
            coeff.conj().T.dot(j - 0.5 * k).dot(coeff)
            for coeff, j, k in zip(mo_coeff, vj, vk)
        ]
        return vhf1, vhf2, vhf3

    gc_kernel = gc_kernel

    def _init_smearing(self):
        assert self._scf.converged
        nelec = ft_mp2.get_correct_nval(self)
        mo_es = np.hstack(self._scf.mo_energy)
        mu, occs = _smearing_optimize(_fermi_smearing_occ, mo_es,
                                      self.nkpts * nelec // 2, 1.0 / self.beta)
        if isinstance(mu, float):
            return mu
        return mu.item()

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
        dm = [
            (coeff*occ).dot(coeff.conj().T)
            for coeff, occ in zip(mo_coeff, mo_occ)
        ]
        return dm

    def ao2mo(self, mo_coeff=None, mu=None, beta=None):
        if beta is None:
            beta = self.beta
        if mu is None:
            raise ValueError
        occs_k = []
        nocc_list = []
        nvir_list = []
        nmo_list = []
        # TODO account for frozen orbs etc.
        mo_energy = self.mo_energy
        for k in range(self.nkpts):
            occs_k.append(_fermi_smearing_occ(mu, mo_energy[k], 1.0 / beta))
            nmo_list.append(mo_energy[k].size)
            nocc_list.append((occs_k[k] > self.occ_tol).sum())
            nvir_list.append(((1 - occs_k[k]) > self.occ_tol).sum())
        return nocc_list, nvir_list, nmo_list


class FTMP2Mixin(_PBC4FT_Mixin, ft_mp2.FTMP2Mixin):
    pass


class FTACMP2Mixin(_PBC4FT_Mixin, ft_mp2.FTACMP2Mixin):
    pass
