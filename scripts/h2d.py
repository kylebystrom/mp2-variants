from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.acmp.ac_mp2 import ACMP2
import numpy as np
import matplotlib.pyplot as plt
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
    # mymp = MP2(mf)
    mymp = ACMP2(mf)
    mymp.kernel()
    mycc = CCSD(mf)
    mycc.kernel()
    return exx, eu - eo, mf.e_tot, mymp.e_corr, mycc.e_tot


rs = np.linspace(0.6, 7.0, 40)
exxs = []
gaps = []
ens = []
corrs = []
eccs = []
for r in rs:
    exx, gap, e_hf, e_corr, ecc = run_calc(r)
    exxs.append(exx)
    gaps.append(gap)
    ens.append(e_corr + e_hf - e0)
    corrs.append(e_corr)
    eccs.append(ecc - e_hf)

print("HI", exx_at)
plt.plot(rs, exxs, label="EXX")
plt.plot(rs, gaps, label="GAP")
plt.plot(rs, ens, label="MP2")
plt.plot(rs, corrs, label="CORR")
plt.plot(rs, eccs, label="CCSD")
plt.legend()
plt.show()

