from pyscf.acmp.pbc.kappa_kmp2 import KappaKMP2, WITH_T2, logger, LARGE_DENOM, \
    einsum, lib, df
from pyscf.acmp.pbc import kappa_kmp2
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp import kmp2_ksymm
import numpy as np


def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    if with_t2:
        return kernel_with_t2(mp, mo_energy, mo_coeff, verbose,
                              with_t2)
    else:
        t2 = None

    t0 = (logger.process_clock(), logger.perf_counter())
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts

    with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)
    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = mp._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s1')
    _, igroup = np.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = igroup.ravel()
    igroup = list(igroup) + [len(kijab)]

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp)

    emp2_ss = emp2_os = 0.
    nao2mo = 0
    icount = 0
    for i in range(len(igroup)-1):
        istart = igroup[i]
        iend = igroup[i+1]
        kab = []
        for j in range(istart, iend):
            a, b = kijab[j][2:]
            kab.append([a, b])
            kab.append([b, a])
        kab = np.unique(np.asarray(kab), axis=0)

        ki = kijab[istart][0]
        kj = kijab[istart][1]
        kpts_i = kd.kpts[ki]
        kpts_j = kd.kpts[kj]
        orbo_i = mo_coeff[ki][:,:nocc]
        orbo_j = mo_coeff[kj][:,:nocc]

        for (ka, kb) in kab:
            if with_df_ints:
                oovv_ij[ka] = (1./nkpts) * einsum(
                    "Lia,Ljb->iajb",
                    Lov[ki, ka],
                    Lov[kj, kb]
                ).transpose(0,2,1,3)
            else:
                kpts_a = kd.kpts[ka]
                kpts_b = kd.kpts[kb]
                orbv_a = mo_coeff[ka][:,nocc:]
                orbv_b = mo_coeff[kb][:,nocc:]
                oovv_ij[ka] = fao2mo(
                    (orbo_i,orbv_a,orbo_j,orbv_b),
                    (kpts_i,kpts_a,kpts_j,kpts_b),
                    compact=False
                ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            nao2mo += 1

        for j in range(istart, iend):
            ka = kijab[j][2]
            kb = kijab[j][3]
            # Remove zero/padded elements from denominator
            eia = -LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = -LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2_ijab = np.conj(oovv_ij[ka]/eijab)
            t2_ijab[:] *= mp.get_damping_factor(eijab)
            idx_ibz = k4_bz2ibz[ki*nkpts**2 + kj*nkpts + ka]
            assert(icount == idx_ibz)
            edi = einsum('ijab,ijab', t2_ijab, oovv_ij[ka]).real * 2
            exi = -einsum('ijab,ijba', t2_ijab, oovv_ij[kb]).real
            emp2_ss += (edi*0.5 + exi) * weight[idx_ibz] * nkpts**3
            emp2_os += edi*0.5 * weight[idx_ibz] * nkpts**3
            icount += 1

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    assert(icount == len(kijab))
    logger.debug(mp, "Number of ao2mo transformations performed in KMP2: %d", nao2mo)
    logger.timer(mp, 'KMP2', *t0)
    return emp2, t2


def kernel_with_t2(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    #we need almost all t2 for computing rdm, so simply use kmp2 without symmetry
    kd = mp.kpts
    mp.kpts = kd.kpts
    emp2, t2 = kappa_kmp2.kernel(mp, mo_energy, mo_coeff, verbose, with_t2)
    mp.kpts = kd
    return emp2, t2


class KsymAdaptedKappaKMP2(KappaKMP2):
    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

        print("KMP2 KERNEL", kernel)
        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    make_rdm1 = kmp2_ksymm.make_rdm1

    def make_rdm2(self):
        raise NotImplementedError
