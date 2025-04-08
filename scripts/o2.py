from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.mp.ump2 import UMP2
from pyscf.acmp.ac_mp2 import ACMP2, BasicEigNumInterpolator, \
        BasicMatNumInterpolator
from pyscf.acmp.ac_ump2 import ACUMP2
import numpy as np
import matplotlib.pyplot as plt
from pyscf.cc import UCCSD
import matplotlib


matplotlib.use("QtAgg")


BASIS = "def2-tzvp"


Ninterp = 4096
eigint = BasicEigNumInterpolator(Ninterp, [])
matint = BasicMatNumInterpolator(Ninterp, [])


atom = gto.M(atom="O", basis=BASIS, spin=2)


def get_mol(r):
    return gto.M(atom=f"O 0 0 0; O 0 0 {r}", basis=BASIS, spin=2)

mf = scf.UHF(atom)
ehf0 = 2 * mf.kernel()

mymp = UMP2(mf)
mymp.kernel()
emp0 = 2 * mymp.e_tot

myac = ACUMP2(mf)
myac.si_limit = "GGA_X_PBE"
myac.ac_interpolator = matint
myac.kernel()
eac0 = 2 * myac.e_tot


def run_calc(r):
    mol = get_mol(r)
    mf = scf.UKS(mol)
    mf.xc = "HF"
    mf.kernel()
    ehf = scf.UHF(mol).kernel(mf.make_rdm1())
    eo = np.max(mf.mo_energy[mf.mo_occ > 1e-10])
    eu = np.min(mf.mo_energy[mf.mo_occ <= 1e-10])
    j, k = mf.get_jk()
    dm = mf.make_rdm1()
    exx = 0.25 * (dm * k).sum()
    mymp = UMP2(mf)
    acmp = ACUMP2(mf)
    acmp.si_limit = "GGA_X_PBE"
    acmp.ac_interpolator = matint
    mymp.kernel()
    acmp.kernel()
    mycc = UCCSD(mf.to_hf())
    if False:
        mycc.kernel()
        ecc = mycc.e_tot
    else:
        ecc = acmp.e_tot
    print("LOOK", mf.e_tot, mymp.e_tot, acmp.e_tot, ecc)
    print()
    return mf.e_tot, mymp.e_corr + ehf, acmp.e_corr + ehf, ecc


rs = np.linspace(0.9, 4.5, 30)
# rs = np.append(rs, np.linspace(3, 13, 11))
ehfs = []
emps = []
eacs = []
eccs = []
for r in rs:
    print("BOND LENGTH", r)
    ehf, emp, eac, ecc = run_calc(r)
    ehfs.append(ehf - ehf0)
    emps.append(emp - emp0)
    eacs.append(eac - eac0)
    eccs.append(ecc - eac0)


plt.plot(rs, ehfs, label="HF")
plt.plot(rs, emps, label="MP2")
plt.plot(rs, eacs, label="ACMP2")
plt.plot(rs, eccs, label="CCSD")
plt.legend()
plt.title("N2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

