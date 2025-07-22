from pyscf.mp.mp2 import MP2, logger
from pyscf.acmp.mp2_numint import MP2NumInt, mgga_sce_limit
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

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super(LambdaMP2, self).__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        self.omega_code = "PLASMA_LDA_WP"
        self._numint = MP2NumInt()
        self.grids = gen_grid.Grids(self.mol)
        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)

    def get_1e_vmat(self, code):
        mf = self._scf
        if code == "__HF__":
            return 0.5 * mf.get_k()
        mol = mf.mol
        dm = mf.make_rdm1()
        nelec, excsum, vmat = self._numint.nr_rmp2(mol, self.grids, code, dm)
        return vmat

    def get_artificial_gap_simple(self):
        return self.get_1e_vmat(self.omega_code)

    def get_artificial_gap_with_exx(self):
        kmat = 0.5 * self.get_1e_vmat("__HF__")
        sce = self.get_1e_vmat(("MGGA", mgga_sce_limit))
        mul = self.get_1e_vmat(self.omega_code)
        return kmat, sce, mul

    def get_artificial_gap(self):
        if False:
            return self.get_artificial_gap_simple()
        else:
            return self.get_artificial_gap_with_exx()

    def get_e_hf(mp, mo_coeff=None):
        if not hasattr(mp._scf, "to_hf"):
            # This is HF object
            return super().get_e_hf(mo_coeff=mo_coeff)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            mf = mp._scf.to_hf()
            vhf = mf.get_veff(mf.mol, dm)
            return mf.energy_tot(dm=dm, vhf=vhf)
    
    def _get_vmat(self, mol, dm):
        nelec, excsum, vmat = self._numint.nr_rmp2(
            mol, self.grids, self.omega_code, dm
        )

    def _get_kmat(self, mf):
        mf.get_k()
    
    # get_artificial_gap = get_artificial_gap_simple
    get_artificial_gap = get_artificial_gap_with_exx

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

        mf = self._scf
        occ = self.mo_occ > 1e-20  # TODO use occdrop
        omo_coeff = self.mo_coeff[:, occ]

        vmat = self.get_artificial_gap(mf)
        if isinstance(vmat, tuple):
            kmat, sce, mul = vmat
            oo_kmat = omo_coeff.T.dot(kmat.dot(omo_coeff))
            oo_sce = omo_coeff.T.dot(sce.dot(omo_coeff))
            oo_mul = omo_coeff.T.dot(mul.dot(omo_coeff))
            print("HI", numpy.diag(oo_kmat), numpy.diag(oo_sce))
            # oo_vmat = -oo_mul.dot(oo_sce)
            oo_sce = (-oo_sce - oo_kmat)
            oo_sce = oo_sce.dot(oo_sce) + 0.0001 * oo_kmat.dot(oo_kmat)
            oo_sce = _lmp_matrix_pow(oo_sce, 0.5)
            oo_vmat = oo_mul.dot(oo_sce)
            oo_vmat = 0.5 * (oo_vmat + oo_vmat.T)
        else:
            oo_vmat = omo_coeff.T.dot(vmat.dot(omo_coeff))
        print("VMAT", numpy.diag(oo_vmat))

        oo_fock = numpy.diag(mf.mo_energy[occ]) - oo_vmat
        oo_energy, oo_transform = mf.eig(oo_fock, numpy.identity(oo_fock.shape[0]))
        self.mo_coeff[:, occ] = self.mo_coeff[:, occ].dot(oo_transform)
        mf.mo_energy[occ] = oo_energy

        return super(LambdaMP2, self).kernel(
            mo_energy=mo_energy, mo_coeff=mo_coeff, eris=eris, with_t2=with_t2
        )
