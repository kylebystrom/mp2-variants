from pyscf import scf, gto
from pyscf.mp.ump2 import UMP2
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ac_interpolators import BasicEigNumInterpolator, \
        BasicMatNumInterpolator, WinfMatNumInterpolator, \
        WinfMatNumInterpolator2, BalancedEigNumInterpolator, \
        SquareMatNumInterpolator, ScreenedEigNumInterpolator, \
        RegEigNumInterpolator, ExtractEigNumInterpolator
from pyscf.acmp.ac_ump2 import ACUMP2
import numpy as np
import matplotlib.pyplot as plt
from pyscf.cc import CCSD
import matplotlib
from pyscf.gw.rpa import RPA
import os
import shutil

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
#myint = BasicEigNumInterpolator(Ninterp, [])
#eigint = ScreenedEigNumInterpolator(Ninterp, [])
# eigint = RegEigNumInterpolator(Ninterp, [])
eigint = ExtractEigNumInterpolator(Ninterp, [])
matint = BasicMatNumInterpolator(Ninterp, [])
matint = SquareMatNumInterpolator(Ninterp, [])

dm0 = None
t1, t2 = None, None
def run_calc(r):
    global dm0, t1, t2
    print("RADIUS", r)
    mol = get_mol(r)
    mf = scf.UKS(mol).newton()
    mf.xc = "PBE"
    mf.grids.level = 4
    mf.kernel(dm0=dm0)
    dm0 = mf.make_rdm1()
    myhf = scf.UHF(mol)
    ehf = myhf.energy_tot(mf.make_rdm1())
    eo = np.max(mf.mo_energy[mf.mo_occ > 1e-10])
    eu = np.min(mf.mo_energy[mf.mo_occ <= 1e-10])
    j, k = mf.get_jk()
    dm = mf.make_rdm1()
    exx = 0.25 * (dm * k).sum()
    mymp = ACUMP2(mf)
    mymp.ac_interpolator = matint
    mymp.si_limit = "1.24*GGA_X_PBE"
    acmp = ACUMP2(mf)
    acmp.ac_interpolator = eigint
    acmp.si_limit = "1.24*GGA_X_PBE"
    # acmp.si_limit = silim
    mymp.kernel()
    acmp.kernel()
    try:
        mycc = CCSD(mf.to_hf())
        ecorr, t1, t2 = mycc.kernel(t1=t1, t2=t2)
        ecc = mycc.e_tot
    except np.linalg.LinAlgError:
        ecorr, ecc, t1, t2 = 0, e0, None, None
    print("DIFF", np.linalg.norm(dm0[0] - dm0[1]))
    print(mymp.e_tot, acmp.e_tot, ehf, e0)
    print()
    shutil.copyfile("w_list.npy", f"{r:.3f}_wlist.npy")
    return mf.e_tot, mymp.e_tot, acmp.e_tot, ecc, eu - eo


ehfs = []
emps = []
eacs = []
eccs = []
gaps = []
rs = np.linspace(0.5, 7.0, 50)
for r in rs:
    ehf, emp, eac, ecc, gap = run_calc(r)
    ehfs.append(ehf - e0)
    emps.append(emp - e0)
    eacs.append(eac - e0)
    eccs.append(ecc - e0)
    gaps.append(gap - e0)

plt.semilogx(rs, ehfs, label="PBE")
plt.semilogx(rs, emps, label="ACMP2, w/o screen")
plt.semilogx(rs, eacs, label="ACMP2, w/screen")
plt.semilogx(rs, eccs, label="CCSD")
plt.semilogx(rs, gaps, label="RPA")
plt.legend()
plt.title("H2 dissociation Curve")
plt.xlabel("Bond Length (Angstrom)")
plt.show()

