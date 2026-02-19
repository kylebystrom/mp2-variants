from pyscf.mp.mp2 import MP2


class PT2(MP2):
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
