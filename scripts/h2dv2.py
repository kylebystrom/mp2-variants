from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.mp.acmp2 import ACMP2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pyscf.cc import CCSD


BASIS = "def2-tzvp"

at = gto.M(atom="H", basis=BASIS, spin=1)
mf = scf.UHF(at)
mf.kernel()
e0 = 2 * mf.e_tot
exx_at = 0.5 * (mf.get_k() * mf.make_rdm1()).sum()


def get_mol(r):
    return gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=BASIS)


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
    mycc.kernel()
    return mf.e_tot, mymp.e_tot, acmp.e_tot, mycc.e_tot


rs = np.linspace(0.5, 7.0, 100)
ehfs = []
emps = []
eacs = []
eccs = []
for r in rs:
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
plt.title("H2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

