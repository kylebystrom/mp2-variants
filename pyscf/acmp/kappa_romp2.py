from pyscf.acmp.kappa_mp2 import KappaMP2Mixin, WITH_T2
from pyscf.acmp.romp2 import ROMP2
from pyscf.lib import logger
from pyscf import lib
import numpy


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    with_t2 = False  # TODO allow t2
    mo_energy, dfock, transforms, mo_coeff = mp._get_coefs()
    mo_energy = None
    mo_coeff = None
    
    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    mo_energy = eris.mo_energy
    mo_coeff = eris.mo_coeff
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mo_coeff[0].shape[-1], mo_coeff[1].shape[-1]
    nvira, nvirb = nmoa-nocca, nmob-noccb

    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]
    
    e_singles = numpy.einsum("ia,ia->", dfock[0], dfock[0].conj() / eia_a)
    e_singles += numpy.einsum("ia,ia->", dfock[1], dfock[1].conj() / eia_b)

    # TODO add option to regularize the singles
    #fac_a = mp.get_damping_factor(2 * eia_a) / eia_a
    #fac_b = mp.get_damping_factor(2 * eia_b) / eia_b
    #e_singles = numpy.einsum("ia,ia->", dfock[0], dfock[0].conj() * fac_a)
    #e_singles += numpy.einsum("ia,ia->", dfock[1], dfock[1].conj() * fac_b)
    e_singles = numpy.einsum("ia,ia->", dfock[0], dfock[0].conj() / eia_a)
    e_singles += numpy.einsum("ia,ia->", dfock[1], dfock[1].conj() / eia_b)

    if with_t2:
        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
    else:
        t2 = None

    emp2_ss = emp2_os = 0.0
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        t2i = eris_ovov.conj() / ei * mp.get_damping_factor(ei)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2aa[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
        eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        t2i = eris_ovov.conj() / ei * mp.get_damping_factor(ei)
        emp2_os += numpy.einsum('JaB,JaB', t2i, eris_ovov)
        if with_t2:
            t2ab[i] = t2i

    for i in range(noccb):
        if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.OVOV[i]
        else:
            eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
        ei = lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
        t2i = eris_ovov.conj() / ei * mp.get_damping_factor(ei)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2bb[i] = t2i - t2i.transpose(0,2,1)

    emp2_ss = emp2_ss.real + e_singles.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class KappaROMP2(KappaMP2Mixin, ROMP2):
    '''restricted kappa-MP2 with canonical HF
    '''
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 kappa=1.5, damping="kappa"):
        ROMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        KappaMP2Mixin.__init__(self, kappa, damping)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('kappa-MP2 with kappa = %.3g', self.kappa)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E(SCS-%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot_scs, self.emp2_scs)
        log.info('E_corr(same-spin) = %.15g', self.e_corr_ss)
        log.info('E_corr(oppo-spin) = %.15g', self.e_corr_os)
        return self
