from pyscf.pbc.mp.kmp2 import logger, WITH_T2, lib, df, padding_k_idx, \
    _init_mp_df_eris, einsum, LARGE_DENOM, KMP2, _add_padding
from pyscf.acmp.kappa_mp2 import KappaMP2Mixin
import numpy as np


def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    """Computes k-point RMP2 energy.

    Args:
        mp (KMP2): an instance of KMP2
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to WITH_T2 (=True).

    Returns:
        KMP2 energy and t2 amplitudes (=None if with_t2 is False)
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mp.dump_flags()
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

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
    if with_t2:
        mem_usage += (nkpts**3 * (nocc * nvir)**2) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = _init_mp_df_eris(mp)

    emp2_ss = emp2_os = 0.
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                # (ia|jb)
                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts) * einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
                else:
                    orbo_i = mo_coeff[ki][:,:nocc]
                    orbo_j = mo_coeff[kj][:,:nocc]
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                         (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]),
                                         compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]

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
                if with_t2:
                    t2[ki, kj, ka] = t2_ijab
                edi = einsum('ijab,ijab', t2_ijab, oovv_ij[ka]).real * 2
                exi = -einsum('ijab,ijba', t2_ijab, oovv_ij[kb]).real
                emp2_ss += edi*0.5 + exi
                emp2_os += edi*0.5

    log.timer("KMP2", *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class KappaKMP2(KappaMP2Mixin, KMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 kappa=1.5, damping='kappa'):
        KMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        KappaMP2Mixin.__init__(self, kappa, damping)

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        mo_coeff, mo_energy = _add_padding(self, mo_coeff, mo_energy)

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

KappaKRMP2 = KappaKMP2
