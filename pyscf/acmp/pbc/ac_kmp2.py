from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.pbc.df import df
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib import kpts as libkpts
from pyscf.lib.parameters import LARGE_DENOM
from pyscf import __config__
from pyscf.pbc.mp import kmp2
import numpy as np
from pyscf.acmp.ac_mp2 import _acmp_ao2mo, concatentate_w, add_to_w_list_, ACMP_PAIRED, ACMP_D_ONLY, ACMP_X_ONLY
from pyscf.acmp.pbc.mp2_numint import KMP2NumInt
from pyscf.pbc.dft.gen_grid import BeckeGrids


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


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
    nocc_list = mp.get_nocc(per_kpoint=True)
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
    # if with_t2:
    #     mem_usage += (nkpts**3 * (nocc * nvir)**2) * 16 / 1e6
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
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    if with_t2:
        raise NotImplementedError
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp)

    exx_k_xx, winf_k_xx = mp.get_acmp_si_limit()

    energy = 0.
    for ki in range(nkpts):
        # print("KINDEX", ki, mp._scf.kpts[ki], len(winf_k_xx))
        # winf_xx = -winf_k_xx[ki]
        # TODO lib.dot if possible
        my_nocc = nocc_list[ki]
        w_list = [np.zeros((my_nocc, my_nocc), dtype=np.complex128)
                  for _ in range(mp.get_pt_list_size())]
        occ_coeff = mo_coeff[ki][:, :my_nocc]
        exx_oo = _acmp_ao2mo(exx_k_xx[ki], occ_coeff)
        winf_oo = _acmp_ao2mo(winf_k_xx[ki], occ_coeff)
        winf = winf_oo
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                # (ia|jb)
                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts) * einsum(
                        "Lia,Ljb->iajb",
                        Lov[ki, ka],
                        Lov[kj, kb]
                    ).transpose(0,2,1,3)
                else:
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
                kb = kconserv[ki,ka,kj]

                # Remove zero/padded elements from denominator
                eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                mp.add_to_w_list_(w_list, oovv_ij[ka], oovv_ij[kb], eijab, ACMP_PAIRED)
        w_list = concatentate_w(exx_oo, w_list, winf[None, :, :])
        energy += 2 * mp.ac_interpolator(w_list).real
        # energy += mp.ac_interpolator(w0, winf)

    log.timer("KMP2", *cput0)

    energy /= nkpts
    emp2 = lib.tag_array(energy, e_corr_ss=0, e_corr_os=energy)

    return emp2, t2


def get_acmp_si_limit(mp):
    exx_xx = -0.25 * mp._scf.get_k()
    if isinstance(mp.si_limit, str) and mp.si_limit == "HF":
        winf_k_xx = exx_xx.copy()
    else:
        ni = mp._numint
        grids = mp.grids
        maxmem = mp._scf.mol.max_memory
        mf = mp._scf
        if isinstance(mf.kpts, np.ndarray):
            kpts = mf.kpts
            kpts_band = mf.kpts
            dm = mf.make_rdm1()
        else:
            kpts = mf.kpts.kpts
            kpts_band = mf.kpts.kpts_ibz
            dm = mf.kpts.transform_dm(mf.make_rdm1())
        nelec, excsum, vmat = ni.nr_rmp2(mf.mol, grids, mp.si_limit,
                                         dm, relativity=0,
                                         kpts=kpts, kpts_band=kpts_band,
                                         hermi=1, max_memory=maxmem,
                                         verbose=None)
        winf_k_xx = vmat - exx_xx
    return exx_xx, winf_k_xx


class ACKMP2(kmp2.KMP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.ac_interpolator = None
        self.si_limit = "HF"
        self._numint = KMP2NumInt()
        self.grids = BeckeGrids(self._scf.mol)
        self.grids.level = 3

    get_acmp_si_limit = get_acmp_si_limit

    add_to_w_list_ = add_to_w_list_

    def get_pt_list_size(self):
        return 2
    
    def get_df_list_size(self):
        return 1

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mp, mo_coeff=mo_coeff)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            mf = mp._scf.to_hf()
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        with_t2 = False
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

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2
