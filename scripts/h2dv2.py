from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.acmp.ac_mp2 import ACMP2, BasicEigNumInterpolator, \
        BasicMatNumInterpolator
import numpy as np
import matplotlib.pyplot as plt
from pyscf.cc import CCSD
import matplotlib


matplotlib.use("QtAgg")


BASIS = "def2-tzvp"

at = gto.M(atom="H", basis=BASIS, spin=1)
mf = scf.UHF(at)
mf.kernel()
e0 = 2 * mf.e_tot
exx_at = 0.5 * (mf.get_k() * mf.make_rdm1()).sum()


def get_mol(r):
    return gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=BASIS)


from pyscf.dft.libxc import eval_xc

silim = ("GGA", lambda rho: 0.94 * eval_xc("GGA_X_PBE", rho, deriv=0)[0])


Ninterp = 4096
eigint = BasicEigNumInterpolator(Ninterp, [])
matint = BasicMatNumInterpolator(Ninterp, [])

dm0 = None
t1, t2 = None, None
def run_calc(r):
    global dm0, t1, t2
    mol = get_mol(r)
    mf = scf.RKS(mol)
    mf.xc = "PBE"
    mf.kernel(dm0=dm0)
    dm0 = mf.make_rdm1()
    myhf = scf.RHF(mol)
    ehf = myhf.energy_tot(mf.make_rdm1())
    eo = np.max(mf.mo_energy[mf.mo_occ > 1e-10])
    eu = np.min(mf.mo_energy[mf.mo_occ <= 1e-10])
    j, k = mf.get_jk()
    dm = mf.make_rdm1()
    exx = 0.25 * (dm * k).sum()
    mymp = MP2(mf)
    acmp = ACMP2(mf)
    acmp.ac_interpolator = matint
    acmp.si_limit = "GGA_X_PBE"
    # acmp.si_limit = silim
    mymp.kernel()
    acmp.kernel()
    try:
        mycc = CCSD(mf.to_hf())
        ecorr, t1, t2 = mycc.kernel(t1=t1, t2=t2)
        ecc = mycc.e_tot
    except np.linalg.LinAlgError:
        ecorr, ecc, t1, t2 = 0, e0, None, None
    return mf.e_tot, ehf, acmp.e_corr + ehf, ecc, eu - eo


rs = np.linspace(0.5, 7.0, 50)
ehfs = []
emps = []
eacs = []
eccs = []
gaps = []
rs = np.linspace(0.5, 7.0, 100)
rs = 0.5 * 10**np.linspace(0, 2, 100)
rs = np.append(rs, [200, 500, 1000])
for r in rs:
    ehf, emp, eac, ecc, gap = run_calc(r)
    ehfs.append(ehf - e0)
    emps.append(emp - e0)
    eacs.append(eac - e0)
    eccs.append(ecc - e0)
    gaps.append(gap)

plt.semilogx(rs, ehfs, label="HF")
plt.semilogx(rs, emps, label="MP2")
plt.semilogx(rs, eacs, label="ACMP2")
plt.semilogx(rs, eccs, label="CCSD")
plt.semilogx(rs, gaps, label="GAP")
plt.legend()
plt.title("H2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

