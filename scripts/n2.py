from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.mp.acmp2 import ACMP2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pyscf.cc import CCSD


BASIS = "def2-tzvp"


def get_mol(r):
    return gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis=BASIS)


def run_calc(r):
    mol = get_mol(r)
    mf = scf.RHF(mol)
    mf.kernel()
    eo = np.max(mf.mo_energy[mf.mo_occ > 1e-10])
    eu = np.min(mf.mo_energy[mf.mo_occ <= 1e-10])
    j, k = mf.get_jk()
    dm = mf.make_rdm1()
    exx = 0.25 * (dm * k).sum()
    mymp = MP2(mf)
    acmp = ACMP2(mf)
    mymp.kernel()
    acmp.kernel()
    mycc = CCSD(mf)
    if False:
        mycc.kernel()
        ecc = mycc.e_tot
    else:
        ecc = acmp.e_tot
    return mf.e_tot, mymp.e_tot, acmp.e_tot, ecc


rs = np.linspace(0.9, 3.5, 20)
# rs = np.append(rs, np.linspace(3, 13, 11))
ehfs = []
emps = []
eacs = []
eccs = []
for r in rs:
    print("BOND LENGTH", r)
    ehf, emp, eac, ecc = run_calc(r)
    ehfs.append(ehf)
    emps.append(emp)
    eacs.append(eac)
    eccs.append(ecc)


plt.plot(rs, ehfs, label="HF")
plt.plot(rs, emps, label="MP2")
plt.plot(rs, eacs, label="ACMP2")
plt.plot(rs, eccs, label="CCSD")
plt.legend()
plt.title("N2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

