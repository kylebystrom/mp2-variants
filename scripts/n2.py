from pyscf import scf, gto
from pyscf.mp.mp2 import MP2
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ac_interpolators import BasicEigNumInterpolator, \
        BasicMatNumInterpolator, WinfMatNumInterpolator, \
        BalancedEigNumInterpolator, SquareEigNumInterpolator, \
        ScreenedEigNumInterpolator, \
        RegEigNumInterpolator, ExtractEigNumInterpolator
import numpy as np
import matplotlib.pyplot as plt
from pyscf.cc import CCSD
import matplotlib


matplotlib.use("QtAgg")


BASIS = "def2-tzvp"


Ninterp = 4096
matint = SquareEigNumInterpolator(Ninterp, [])
#eigint = ScreenedEigNumInterpolator(Ninterp, [])
eigint = RegEigNumInterpolator(Ninterp, [])
eigint = ExtractEigNumInterpolator(Ninterp, [])
# matint = BasicMatNumInterpolator(Ninterp, [])
#matint = SquareMatNumInterpolator(Ninterp, [])


def get_mol(r):
    return gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis=BASIS)


t1 = t2 = dm00 = None
def run_calc(r):
    global dm00, t1, t2
    mol = get_mol(r)
    mf = scf.RKS(mol).newton()
    mf.xc = "PBE"
    mf.grids.level = 4
    mf.kernel(dm0=dm00)
    dm00 = mf.make_rdm1()
    myhf = scf.RHF(mol)
    ehf = myhf.energy_tot(dm00)
    myhf.kernel()
    eu = np.min(mf.mo_energy[mf.mo_occ <= 1e-10])
    j, k = mf.get_jk()
    dm = mf.make_rdm1()
    exx = 0.25 * (dm * k).sum()
    #mymp = MP2(mf)
    mymp = ACMP2(mf)
    mymp.si_limit = "1.05*GGA_X_PBE"
    mymp.ac_interpolator = matint
    acmp = ACMP2(mf)
    acmp.si_limit = "1.05*GGA_X_PBE"
    acmp.ac_interpolator = eigint
    mymp.kernel()
    acmp.kernel()
    mycc = CCSD(myhf)
    if True:
        _, t1, t2 = mycc.kernel(t1=t1, t2=t2)
        ecc = mycc.e_tot
        #etrip = mycc.ccsd_t()
        #print("TRIPLES", r, etrip)
    else:
        ecc = acmp.e_tot
    print("LOOK", mf.e_tot, mymp.e_tot, acmp.e_tot, ecc)
    print()
    return mf.e_tot, mymp.e_corr + ehf, acmp.e_tot, ecc


rs = np.linspace(0.9, 4.0, 30)
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
plt.plot(rs, emps, label="ACMP2, w/o screen")
plt.plot(rs, eacs, label="ACMP2, w/screen")
plt.plot(rs, eccs, label="CCSD")
plt.legend()
plt.title("N2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

