from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.mp.ump2 import UMP2
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.acmp.ac_ump2 import ACUMP2
import numpy as np
import matplotlib.pyplot as plt
from pyscf.cc import CCSD, UCCSD
import matplotlib
from pyscf.fci import FCI
from pyscf.acmp.mp2_numint import mgga_sce_limit, mgga_chi


matplotlib.use("QtAgg")


BASIS = "cc-pvdz"
Ninterp = 4096
eigint = get_interpolator("E", name="screen")
matint = get_interpolator("M", name="basic")

silim = ("MGGA", mgga_sce_limit)
df_codes = [("MGGA", mgga_chi)]


def get_mol(r):
    return gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis=BASIS, max_memory=100000)


def read_n2_ref():
    with open("n2_ref_data.txt", "r") as f:
        lines = f.readlines()[4:]
    n2_ref_rs = []
    n2_ref_ens = []
    for l in lines:
        r, e, _ = l.split()
        n2_ref_rs.append(float(r) * 0.529177)
        n2_ref_ens.append(-float(e) - 108)
    for i in range(len(n2_ref_ens)):
        n2_ref_ens[i] -= n2_ref_ens[-1]
    return n2_ref_rs[:-1], n2_ref_ens[:-1]

r_exs, e_exs = read_n2_ref()

t1 = t2 = dm00 = None
def run_methods(mol):
    global dm00, t1, t2
    if mol.spin == 0:
        mf = scf.RKS(mol).newton()
    else:
        mf = scf.UKS(mol).newton()
    mf.xc = "PBE"
    mf.grids.level = 4
    mf.kernel(dm0=dm00)
    dm00 = mf.make_rdm1()
    if mol.spin == 0:
        myhf = scf.RHF(mol)
    else:
        myhf = scf.UHF(mol)
    ehf = myhf.energy_tot(dm00)
    myhf.kernel()
    if mol.spin == 0:
        mymp = ACMP2(mf)
        acmp = ACMP2(mf)
    else:
        mymp = ACUMP2(mf)
        acmp = ACUMP2(mf)
    mymp.si_limit = silim
    mymp.df_codes = df_codes
    mymp.ac_interpolator = matint
    acmp.si_limit = silim
    acmp.df_codes = df_codes
    acmp.ac_interpolator = eigint
    mymp.kernel()
    acmp.kernel()
    if mol.spin == 0:
        # mycc = FCI(myhf)
        mycc = CCSD(mf.to_hf())
    else:
        # mycc = FCI(myhf)
        mycc = UCCSD(mf.to_hf())
    if False:
        # mycc.kernel()
        _, t1, t2 = mycc.kernel(t1=t1, t2=t2)
        ecc = mycc.e_tot
        #etrip = mycc.ccsd_t()
        #print("TRIPLES", r, etrip)
    else:
        ecc = myhf.e_tot
    print("LOOK", mf.e_tot, mymp.e_tot, acmp.e_tot, ecc)
    print()
    return mf.e_tot, mymp.e_corr + ehf, acmp.e_tot, ecc


atm = gto.M(atom="N", basis=BASIS, spin=3, max_memory=100000)
e0_pbe, e0_acmp, e0_screen, e0_cc = run_methods(atm)

atm = gto.M(atom="N", basis=BASIS, spin=-3, max_memory=100000)
e0_pbe, e0_acmp, e0_screenm, e0_cc = run_methods(atm)
print(e0_screenm, e0_screen)

t1 = t2 = dm00 = None
def run_calc(r):
    mol = get_mol(r)
    return run_methods(mol)

rs = np.linspace(0.9, 6.0, 50)
rs = np.array(r_exs)
# rs = np.append(rs, np.linspace(3, 13, 11))
ehfs = []
emps = []
eacs = []
eccs = []
for r in rs:
    print("BOND LENGTH", r)
    ehf, emp, eac, ecc = run_calc(r)
    ehfs.append(ehf - 2 * e0_pbe)
    emps.append(emp - 2 * e0_acmp)
    eacs.append(eac - 2 * e0_screen)
    eccs.append(ecc - 2 * e0_cc)


plt.plot(rs, ehfs, label="HF")
plt.plot(rs, emps, label="ACMP2, w/o screen")
plt.plot(rs, eacs, label="ACMP2, w/screen")
plt.plot(rs, eccs, label="CCSD")
plt.plot(r_exs, e_exs, label="Ref")
plt.legend()
plt.title("N2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

