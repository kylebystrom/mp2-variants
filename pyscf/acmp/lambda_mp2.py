from pyscf.mp.mp2 import MP2, logger
from pyscf.acmp.mp2_numint import MP2NumInt
from pyscf.acmp.ac_mp2 import matrix_kernel, get_acmp_df_mat, add_to_w_list_
from pyscf.acmp.ac_interpolators import ArtificialGapCalculator
from pyscf.dft import gen_grid
from pyscf import __config__
import numpy

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def _lmp_matrix_pow(mat, mypow):
    eval, evec = numpy.linalg.eigh(mat)
    eval = numpy.maximum(eval, 0)
    eval = eval**mypow
    return (evec * eval).dot(evec.T)


class LambdaMP2(MP2):
    _keys = {
        'omega_code', 'grids'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 gap_model=None, gap_mode=None, df_codes=None):
        super(LambdaMP2, self).__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        from pyscf.acmp.mp2_numint import lda_plasma_frequency
        self.omega_code = ("LDA", lda_plasma_frequency)
        self._numint = MP2NumInt()
        self.grids = gen_grid.Grids(self.mol)
        self.gap_model = gap_model
        self.gap_mode = gap_mode
        self.df_codes = df_codes
        if self.df_codes is None:
            self.df_codes = []
        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)

    add_to_w_list_ = add_to_w_list_

    get_acmp_df_mat = get_acmp_df_mat

    def get_acmp_df_wlist(self, mo_coeff, wmat=None):
        wlist_df = [] if wmat is None else [w for w in wmat]
        for df_code in self.df_codes + [self.omega_code]:
            wlist_df.append(self.get_acmp_df_mat(df_code, mo_coeff))
        return wlist_df
    
    def get_pt_list_size(self):
        return 1

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mo_coeff=mo_coeff)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            mf = mp._scf.to_hf()
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()

        if mo_energy is not None or mo_coeff is not None or eris is not None:
            raise NotImplementedError("Passing mo/eris directly to kernel")

        agc = ArtificialGapCalculator(mode=self.gap_mode, gap_model=self.gap_model)

        if agc.requires_mp2_mat:
            # get the correlation energy matrix, which is twice the
            # correlation energy per PARTICLE (not per orbital).
            # So for closed-shell, tr(wmat) = e_corr (factor of 2 needed
            # for electrons per orbital is canceled by factor of half
            # to get from <W> to E_corr)
            wmat = matrix_kernel(self, mo_energy, mo_coeff, eris, False)[0]
            wmat = [w + w.T.conj() for w in wmat]
        else:
            wmat = None

        mf = self._scf
        occ = self.mo_occ > 1e-20  # TODO use occdrop
        oo_vmat = agc.compute_artificial_gap(
            self.get_acmp_df_wlist(self.mo_coeff, wmat=wmat)
        )
        if oo_vmat.ndim == 1:
            oo_vmat = numpy.diag(oo_vmat)

        oo_fock = numpy.diag(mf.mo_energy[occ]) - oo_vmat
        oo_energy, oo_transform = mf.eig(oo_fock, numpy.identity(oo_fock.shape[0]))
        self.mo_coeff[:, occ] = self.mo_coeff[:, occ].dot(oo_transform)
        old_energy = mf.mo_energy[occ].copy()
        # set to new eigenvalues for the kernel
        mf.mo_energy[occ] = oo_energy
        res = super(LambdaMP2, self).kernel(
            mo_energy=mo_energy, mo_coeff=mo_coeff, eris=eris, with_t2=with_t2
        )
        # set back to the old eigenvalues
        mf.mo_energy[occ] = old_energy
        return res
