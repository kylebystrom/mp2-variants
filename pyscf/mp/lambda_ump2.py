from pyscf.mp.ump2 import UMP2, logger
from pyscf.dft.lmp2_numint import LMP2NumInt
from pyscf.dft import gen_grid
from pyscf import __config__
import numpy

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


class LambdaUMP2(UMP2):

    _keys = {
        'omega_code', 'grids'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super(LambdaUMP2, self).__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        self.omega_code = "LDA_WP"
        self._numint = LMP2NumInt()
        self.grids = gen_grid.Grids(self.mol)
        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)

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
        mol = mf.mol
        s1e = mf.get_ovlp(mol)
        dm = mf.make_rdm1()
        h1e = mf.get_hcore(mol)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)

        nelec, excsum, vmat = self._numint.nr_ulmp2(
            mol, self.grids, self.omega_code, dm
        )
        fock -= vmat

        occa = self.mo_occ[0] > 1e-20  # TODO use occdrop
        occb = self.mo_occ[1] > 1e-20  # TODO use occdrop
        print(self.mo_coeff.shape, vmat.shape)
        omoa = self.mo_coeff[0][:, occa]
        omob = self.mo_coeff[1][:, occb]
        oo_va = omoa.T.dot(vmat.dot(omoa))
        oo_vb = omob.T.dot(vmat.dot(omob))
        oo_fa = numpy.diag(mf.mo_energy[0, occa]) - oo_va
        oo_fb = numpy.diag(mf.mo_energy[0, occb]) - oo_vb
        print(oo_va.shape, oo_fa.shape)
        print(oo_vb.shape, oo_fb.shape)
        import scipy.linalg
        oo_ea, oo_ta = scipy.linalg.eigh(oo_fa)
        oo_eb, oo_tb = scipy.linalg.eigh(oo_fb)
        self.mo_coeff[0][:, occa] = self.mo_coeff[0][:, occa].dot(oo_ta)
        self.mo_coeff[1][:, occb] = self.mo_coeff[1][:, occb].dot(oo_tb)
        mo_eca = self.mo_coeff[0][:, occa].T.dot(fock.dot(self.mo_coeff[0][:, occa]))
        print(mo_eca.shape, self.mo_coeff[0][:, occa].shape, fock.shape)
        mo_eca = numpy.diag(self.mo_coeff[0][:, occa].T.dot(
            fock[0].dot(self.mo_coeff[0][:, occa])))
        mo_ecb = numpy.diag(self.mo_coeff[1][:, occb].T.dot(
            fock[1].dot(self.mo_coeff[1][:, occb])))
        print(nelec, excsum, oo_ea, oo_eb, mo_eca, mo_ecb)
        mf.mo_energy[0, occa] = oo_ea
        mf.mo_energy[1, occb] = oo_eb

        return super(LambdaUMP2, self).kernel(
            mo_energy=mo_energy, mo_coeff=mo_coeff, eris=eris, with_t2=with_t2
        )
