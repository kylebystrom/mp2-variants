from pyscf.mp.ump2 import UMP2, logger
from pyscf.acmp.mp2_numint import MP2NumInt
from pyscf.acmp.ac_ump2 import matrix_kernel, get_acmp_df_mat, add_to_w_list_
from pyscf.acmp.ac_interpolators import ArtificialGapCalculator
from pyscf.dft import gen_grid
from pyscf import __config__
import numpy

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


class LambdaUMP2(UMP2):

    _keys = {
        'omega_code', 'grids'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 gap_model=None, gap_mode=None, df_codes=None):
        super(LambdaUMP2, self).__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        self.omega_code = "PLASMA_LDA_WP"
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

    def get_acmp_df_wlist(self, mo_coeff, wlist_a=None, wlist_b=None):
        wlist_df_a = [] if wlist_a is None else [w for w in wlist_a]
        wlist_df_b = [] if wlist_b is None else [w for w in wlist_b]
        for df_code in self.df_codes + [self.omega_code]:
            wa, wb = self.get_acmp_df_mat(df_code, mo_coeff)
            wlist_df_a.append(wa)
            wlist_df_b.append(wb)
        return wlist_df_a, wlist_df_b

    def get_pt_list_size(self):
        return 1

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
            wa_list, wb_list = matrix_kernel(self, mo_energy, mo_coeff, eris, False)[0]
            wa_list = [w + w.T.conj() for w in wa_list]
            wb_list = [w + w.T.conj() for w in wb_list]
        else:
            wa_list, wb_list = None, None

        mf = self._scf
        occa = self.mo_occ[0] > 1e-20  # TODO use occdrop
        occb = self.mo_occ[1] > 1e-20  # TODO use occdrop
        wa_list, wb_list = self.get_acmp_df_wlist(
            self.mo_coeff, wlist_a=wa_list, wlist_b=wb_list
        )
        oo_vmat_a = agc.compute_artificial_gap(wa_list)
        oo_vmat_b = agc.compute_artificial_gap(wb_list)

        oo_fock_a = numpy.diag(mf.mo_energy[0, occa]) - oo_vmat_a
        oo_fock_b = numpy.diag(mf.mo_energy[1, occb]) - oo_vmat_b
        oo_energy, oo_transform = mf.eig((oo_fock_a, oo_fock_b),
                                         numpy.identity(oo_fock_a.shape[0]))
        ooe_a, ooe_b = oo_energy
        oot_a, oot_b = oo_transform
        self.mo_coeff[0][:, occa] = self.mo_coeff[0][:, occa].dot(oot_a)
        self.mo_coeff[1][:, occb] = self.mo_coeff[1][:, occb].dot(oot_b)
        old_energy_a = mf.mo_energy[0, occa].copy()
        old_energy_b = mf.mo_energy[1, occb].copy()
        mf.mo_energy[0, occa] = ooe_a
        mf.mo_energy[1, occb] = ooe_b
        res = super(LambdaUMP2, self).kernel(
            mo_energy=mo_energy, mo_coeff=mo_coeff, eris=eris, with_t2=with_t2
        )
        mf.mo_energy[0, occa] = old_energy_a
        mf.mo_energy[1, occb] = old_energy_b
        return res
