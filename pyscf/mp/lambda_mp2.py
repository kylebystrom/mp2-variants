from pyscf.mp.mp2 import MP2, logger
from pyscf.dft.lmp2_numint import LMP2NumInt
from pyscf.dft import gen_grid
from pyscf import __config__
import numpy

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


class LambdaMP2(MP2):
    _keys = {
        'omega_code', 'grids'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super(LambdaMP2, self).__init__(
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
        k1e = mol.intor_symmetric('int1e_kin')
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)

        nelec, excsum, vmat = self._numint.nr_rlmp2(
            mol, self.grids, self.omega_code, dm
        )
        fock -= vmat

        print("KINETIC", numpy.diag(self.mo_coeff.T.dot(k1e.dot(self.mo_coeff))))

        occ = self.mo_occ > 1e-20  # TODO use occdrop
        omo_coeff = self.mo_coeff[:, occ]
        oo_vmat = omo_coeff.T.dot(vmat.dot(omo_coeff))
        oo_fock = numpy.diag(mf.mo_energy[occ]) - oo_vmat
        #oo_vmat = numpy.diag(numpy.diag(
        #    omo_coeff.T.dot(vmat.dot(omo_coeff))
        #))
        #elumo = numpy.min(mf.mo_energy[numpy.logical_not(occ)])
        #oo_fock = numpy.diag(numpy.minimum(
        #    mf.mo_energy[occ] - elumo, -1 * numpy.diag(oo_vmat)
        #) + elumo)
        oo_energy, oo_transform = mf.eig(oo_fock, numpy.identity(oo_fock.shape[0]))
        self.mo_coeff[:, occ] = self.mo_coeff[:, occ].dot(oo_transform)
        mo_energy_check = self.mo_coeff[:, occ].T.dot(fock.dot(self.mo_coeff[:, occ]))
        mo_energy_check = numpy.diag(mo_energy_check)
        # print(nelec, excsum, oo_energy, mo_energy_check)
        lumo = numpy.min(mf.mo_energy[numpy.logical_not(occ)])
        # print("BEFORE", lumo - numpy.max(mf.mo_energy[occ]))
        mf.mo_energy[occ] = oo_energy
        # print("AFTER", lumo - numpy.max(mf.mo_energy[occ]))


        return super(LambdaMP2, self).kernel(
            mo_energy=mo_energy, mo_coeff=mo_coeff, eris=eris, with_t2=with_t2
        )
