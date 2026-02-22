'''
Helper functions for the uniform electron gas
'''

import numpy as np
import time
import math
import sys


def sorter_function(inarr):
    val = inarr[ 0 ]*inarr[ 0 ] + inarr[ 1 ]*inarr[ 1 ] + inarr[ 2 ]*inarr[ 2 ]
    return val


def change_basis_2el_complex(g,C):
    """Change basis for 2-el integrals in phys notation with complex coefficients.
    """
    g = np.tensordot(C,g,axes=[0,3]).transpose(1,2,3,0)
    g = np.tensordot(C,g,axes=[0,2]).transpose(1,2,0,3)
    g = np.tensordot(C.conj(),g,axes=[0,1]).transpose(1,0,2,3)
    g = np.tensordot(C.conj(),g,axes=[0,0])
    return g


def mgga_rho_vector(density):
    CFC = 0.3 * (3 * np.pi**2)**(2.0 / 3)
    zero = 0 * density
    return np.array([density, zero, zero, zero, CFC * density**(5.0 / 3)])


class UEG(object):
    def __init__(self, nelec, nbasis, rs, verbose=False):
        self.nelec = nelec
        # Limited to closed shell 
        if ((self.nelec % 2) != 0):
            sys.exit("Need an even number of electrons!")
        self.nocc = self.nelec // 2
        self.nbas = nbasis
        self.dim = 3
        self.rs = rs
        self._ulim = 0
        self._llim = 0
        self._rgvecs = np.zeros(0)
        self._scf_en = 0.0
        self.verbose = verbose

        self._volume = self.nelec * (4.0/3.0) * np.pi * self.rs**3
        self._length = self._volume ** (1. / 3.)
        self.madelung = 2.83729747948149 / self._length

        self.vcut = False

        self.create_gvecs()
        if self.verbose:
            self.print_info()

    def create_gvecs(self):
        converged = False
        icur = int(math.ceil((3. / (4. * np.pi) * self.nbas) ** (1. / 3.)))
        Gvecs = []
        while not converged:
            curbas = 0
            self.gnorm = []
            if self.verbose:
                print("...finding gvecs in sphere ix = [ %4d ]" % icur)
            for ix in range(-icur, icur+1):
                ix2 = ix*ix
                for iy in range(-icur, icur+1):
                    iy2 = iy*iy
                    for iz in range(-icur, icur+1):
                        iz2 = iz*iz
                        Gvecs.append(ix)
                        Gvecs.append(iy)
                        Gvecs.append(iz)
                        self.gnorm.append(ix2 + iy2 + iz2)
                        curbas = curbas + 1
            self.gnorm = sorted(self.gnorm)
            if icur < np.sqrt(self.gnorm[self.nbas - 1]):
                converged = False
                icur = icur + 1
            else:
                converged = True
        upper_limit = self.nbas - 1
        same_value = True
        while same_value:
            upper_limit = upper_limit + 1
            if (self.gnorm[upper_limit] - self.gnorm[upper_limit - 1] > 0):
                same_value = False
            else:
                same_value = True

        lower_limit = self.nbas
        same_value = True
        while same_value:
            lower_limit = lower_limit - 1
            if (self.gnorm[lower_limit] - self.gnorm[lower_limit - 1] > 0):
                same_value = False
            else:
                same_value = True

        self.ulim = upper_limit
        self.llim = lower_limit
        outbas = self.ulim
        self.nbas = outbas
        self.rgvecs = np.zeros(( outbas, 3 ))
        count = 0
        maxnorm = self.gnorm[outbas - 1]
        for i in range(0, int(len(Gvecs)/3)):
            ix = Gvecs[ 3 * i + 0 ]
            iy = Gvecs[ 3 * i + 1 ]
            iz = Gvecs[ 3 * i + 2 ]
            if (ix*ix + iy*iy + iz*iz <= maxnorm ):
                self.rgvecs[ count ][ 0 ] = ix
                self.rgvecs[ count ][ 1 ] = iy
                self.rgvecs[ count ][ 2 ] = iz
                count = count + 1
        zrgvecs = self.rgvecs
        zrgvecs = sorted( zrgvecs, key=sorter_function )
        self.rgvecs = np.array( zrgvecs ).tolist()

    def print_k(self):
        twopidl = 2.0 * np.pi / self._length
        for i in range(0,len(self.gnorm)):
            print("K%3d : %14.8f" % (i,twopidl*np.sqrt(self.gnorm[ i ])))

    def kin(self, p, q):
        twopdg = 2. * np.pi / self._length
        twopdgsq = twopdg * twopdg
        kin = 0.0
        if p == q:
            px = self.rgvecs[p][0]
            py = self.rgvecs[p][1]
            pz = self.rgvecs[p][2]
            inorm = px*px + py*py + pz*pz
            kin = 0.5 * twopdgsq * inorm
        return kin

    def eri(self, p, q, r, s):
        pqx = self.rgvecs[p][0] - self.rgvecs[q][0]
        pqy = self.rgvecs[p][1] - self.rgvecs[q][1]
        pqz = self.rgvecs[p][2] - self.rgvecs[q][2]
        srx = self.rgvecs[s][0] - self.rgvecs[r][0]
        sry = self.rgvecs[s][1] - self.rgvecs[r][1]
        srz = self.rgvecs[s][2] - self.rgvecs[r][2]
        normsq = pqx*pqx + pqy*pqy + pqz*pqz

        integral = 0.0
        if (pqx == srx and pqy == sry and pqz == srz):
            G = 2*np.pi/self._length * np.sqrt(normsq)
            integral = self.v(G)
        return integral

    def kohnsham(self):
        ks = np.zeros(self.nbas, )
        for p in range(0, self.nbas):
            ks[p] = self.kin(p, p) + self.vxc_lda()
        return ks

    def vxc_lda(self):
        rs = self.rs
        density = 1.0 / (4./3 * np.pi * rs**3)

        A = 0.0311
        B = -0.048
        C = 0.0020
        D = -0.0116
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334

        vx = -(3.0*density/np.pi)**(1./3)
        if rs > 1:
            vc = gamma/(1+beta1*np.sqrt(rs)+beta2*rs)*(
                      (1+7./6*beta1*np.sqrt(rs)+4./3*beta2*rs)
                    / (1+beta1*np.sqrt(rs)+beta2*rs) )
        else:
            vc = A*np.log(rs) + B - A/3. + 2./3*C*rs*np.log(rs) + (2*D-C)*rs/3

        return vx + vc

    def mgga_rho_vector(self):
        rs = self.rs
        density = 1.0 / (4./3 * np.pi * rs**3)
        return mgga_rho_vector(density)[:, None]

    #def energy(self):
    #    energy = 0.0
    #    for p in range( 0, self.nocc ):
    #        energy = energy + 2. * self.kin( p, p )
    #        for q in range( 0, self.nocc ):
    #            energy = energy - self.eri( q, p, p, q )
    #    self.scf_en = energy
    #    return energy

    def get_hcore(self):
        h = np.zeros((self.nbas, self.nbas))
        for p in range(self.nbas):
            h[p, p] = self.kin(p, p)
            #if p < self.nocc:
            #    h[p,p] -= self.madelung
        return h

    def get_veff(self):
        # UEG doesn't have a Hartree energy due to positive background
        veff = np.zeros((self.nbas, self.nbas))
        for p in range(self.nbas):
            vk = 0.0
            for i in range(self.nocc):
                vk += self.eri(p, i, i, p)
            veff[p, p] = -vk
        return veff

    def get_fock(self):
        vcoul = np.zeros((self.nbas, self.nbas))
        hcore = self.get_hcore()
        for p in range(self.nbas):
            twoel = 0.0
            for i in range(self.nocc):
                twoel -= self.eri(p, i, i, p)
            vcoul[p, p] = twoel
        return hcore + vcoul

    def Umat(self):
        '''The transformation matrix from complex PWs to real cos/sin orbitals'''
        gvecs = np.array(self.rgvecs)
        Umat = np.zeros((self.nbas, self.nbas), dtype=complex)
        basis = 0
        for gi,g in enumerate(gvecs):
            mgi = np.linalg.norm(-g-gvecs,axis=1).argmin()
            if gi == mgi:
                Umat[gi,basis] = 1.0
            elif gi < mgi:
                Umat[gi,basis] = 1.0/np.sqrt(2.0)
                Umat[mgi,basis] = 1.0/np.sqrt(2.0)
            else:
                Umat[gi,basis] = 1j/np.sqrt(2.0)
                Umat[mgi,basis] = -1j/np.sqrt(2.0)
            basis += 1
        return Umat

    def eri_chem_real(self):
        '''Generate ERIs in the cos/sin basis in chemists notation'''
        Umat = self.Umat()
        # changing basis in fock does nothing
        eri = self.eri_full_fast()
        eri = np.tensordot(Umat.conj(),eri,axes=[0,0])
        eri = np.tensordot(Umat,eri,axes=[0,1]).transpose(1,0,2,3)
        eri = np.tensordot(Umat.conj(),eri,axes=[0,2]).transpose(1,2,0,3)
        eri = np.tensordot(Umat,eri,axes=[0,3]).transpose(1,2,3,0).real

        return eri

    def eri_full_fast(self):
        '''Generate ERIs in the PW basis in chemists notation'''
        outvec = np.zeros((self.nbas, self.nbas,
                           self.nbas, self.nbas))
        for p in range( 0, len( self.rgvecs ) ):
            px = self.rgvecs[ p ][ 0 ]
            py = self.rgvecs[ p ][ 1 ]
            pz = self.rgvecs[ p ][ 2 ]
            for q in range( p, len( self.rgvecs ) ):
                pqx = px - self.rgvecs[ q ][ 0 ]
                pqy = py - self.rgvecs[ q ][ 1 ]
                pqz = pz - self.rgvecs[ q ][ 2 ]
                normsq = pqx*pqx + pqy*pqy + pqz*pqz
                G = 2*np.pi/self._length * np.sqrt(normsq)
                integral = self.v(G)
                for r in range( p, len( self.rgvecs ) ):
                    rx = self.rgvecs[ r ][ 0 ]
                    ry = self.rgvecs[ r ][ 1 ]
                    rz = self.rgvecs[ r ][ 2 ]
                    for s in range( 0, len( self.rgvecs ) ):
                        srx = self.rgvecs[ s ][ 0 ] - rx
                        sry = self.rgvecs[ s ][ 1 ] - ry
                        srz = self.rgvecs[ s ][ 2 ] - rz
                        if( pqx == srx and
                            pqy == sry and
                            pqz == srz ):
                            outvec[ p, q, r, s ] = integral
                            outvec[ q, p, s, r ] = integral
                            outvec[ r, s, p, q ] = integral
                            outvec[ s, r, q, p ] = integral

        return outvec

    def v(self, G):
        '''Coulomb kernel'''
        if self.vcut:
            R = self._length / 2
            G += 1e-8
            integral = 4 * np.pi / (self._volume * G * G) * (1 - np.cos(G * R))
        else:
            if np.isclose(G, 0):
                integral = self.madelung
            else:
                integral = 4 * np.pi / (self._volume*G*G)
        return integral

    def print_info(self):
        print("Uniform Electron Gas Parameters")
        print(" - Dimension of System       = %14d " % self.dim)
        print(" - Number of Electrons       = %14d " % self.nelec)
        print(" - Madelung constant         = %14.8f " % self.madelung)
        print(" - Volume of Box             = %14.8f " % self._volume)
        print(" - Length of Box             = %14.8f " % self._length)
        print(" - Number of Basis Functions = %14d " % self.nbas)


class FTUEG(UEG):
    def __init__(self, nelec, beta, nbasis, rs, verbose=False, occ_tol=0.0):
        # NOTE: nelec can be a float
        self.nelec = nelec
        self.beta = beta
        self.occs = None
        self.nocc = None
        self.mu = None
        self.occ_tol = occ_tol
        self.nbas = nbasis
        self.dim = 3
        self.rs = rs
        self._ulim = 0
        self._llim = 0
        self._rgvecs = np.zeros(0)
        self._scf_en = 0.0
        self.verbose = verbose

        self._volume = self.nelec * (4.0/3.0) * np.pi * self.rs**3
        self._length = self._volume ** (1. / 3.)
        self.madelung = 2.83729747948149 / self._length

        self.vcut = False

        self.create_gvecs()
        if self.verbose:
            self.print_info()

    def run_occ_scf(self, h0, occ0=None):
        from pyscf.scf.addons import _fermi_smearing_occ, _smearing_optimize

        def opt_occs(mo_es, nocc, sigma):
            return _smearing_optimize(_fermi_smearing_occ, mo_es, nocc, sigma)

        if occ0 is None:
            mo_es = np.diag(self.get_hcore())
            self.mu, self.occs = opt_occs(mo_es, self.nelec / 2, 1.0 / self.beta)
        else:
            assert self.mu is not None
            self.occs = occ0
        self.nocc = np.sum(self.occs > self.occ_tol)

        if h0 == "fock":
            for step in range(50):
                mo_es = np.diag(self.get_hcore() + self.get_veff())
                old_occs = self.occs.copy()
                self.mu, self.occs = opt_occs(
                    mo_es, self.nelec / 2, 1.0 / self.beta
                )
                self.nocc = np.sum(self.occs > self.occ_tol)
                err = np.sum(np.abs(old_occs - self.occs))
                if err < 1e-10:
                    break
            else:
                raise RuntimeError("Failed to converge occupations")

    def get_veff(self):
        # UEG doesn't have a Hartree energy due to positive background
        veff = np.zeros((self.nbas, self.nbas))
        for p in range(self.nbas):
            vk = 0.0
            for i in range(self.nocc):
                vk += self.occs[i] * self.eri(p, i, i, p)
            veff[p, p] = -vk
        return veff

    def get_fock(self):
        vcoul = np.zeros((self.nbas, self.nbas))
        hcore = self.get_hcore()
        for p in range(self.nbas):
            twoel = 0.0
            for i in range(self.nocc):
                twoel -= self.occs[i] * self.eri(p, i, i, p)
            vcoul[p, p] = twoel
        return hcore + vcoul

    def mgga_rho_vector(self):
        rs = self.rs
        density = 1.0 / (4./3 * np.pi * rs**3)
        ek = 0
        for p in range(self.nocc):
            ek += 2 * self.occs[p] * self.kin(p, p)
        ek /= self.nelec
        # now ek is kinetic energy per electron
        # multiply by density to get KE
        ek *= density
        zero = 0 * density
        return np.array([density, zero, zero, zero, ek])[:, None]

    def print_info(self):
        print("Uniform Electron Gas Parameters")
        print(" - Dimension of System       = %14d " % self.dim)
        print(" - Number of Electrons       = %14.8f " % self.nelec)
        print(" - Madelung constant         = %14.8f " % self.madelung)
        print(" - Volume of Box             = %14.8f " % self._volume)
        print(" - Length of Box             = %14.8f " % self._length)
        print(" - Number of Basis Functions = %14d " % self.nbas)
