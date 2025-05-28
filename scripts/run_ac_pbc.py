from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ac_interpolators import ExtractEigNumInterpolator, \
    ScreenedMatNumInterpolator, BasicEigNumInterpolator, \
    BasicMatNumInterpolator, SquareMatNumInterpolator
from pyscf.acmp.pbc.ac_mp2 import ACRMP2
from pyscf.acmp.ac_ump2 import ACUMP2
from pyscf.acmp.pbc.ac_kmp2 import ACKMP2
from pyscf.acmp.pbc.ac_kmp2_ksymm import KsymAdaptedACKMP2 as SymACKMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KMP2
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pgto, scf as pscf, dft as pdft
import numpy as np
import time
from pyscf.acmp.mp2_numint import mgga_sce_limit
from pyscf import cc


basis = "def2-tzvp"
mol_strings = [
    "H 0 0 -0.35; H 0 0 0.35", 
    "H 0 0 -0.55; F 0 0 0.55",
    "F 0 0 -0.70; F 0 0 0.70",
]


# eigint = ExtractEigNumInterpolator(512, [])
eigint = ScreenedMatNumInterpolator(512, [])

# eigint = BasicEigNumInterpolator(512, [])
# eigint = SquareMatNumInterpolator(512, [])

def set_ac_(acmp):
    acmp.ac_interpolator = eigint
    # acmp.si_limit = "1.1*GGA_X_PBE"
    acmp.si_limit = ("MGGA", mgga_sce_limit)


iso_ens = []
uks_ens = []
pbc_ens = []
kpt_ens = []
sym_ens = []
ccd_ens = []
cct_ens = []
nk = [2, 2, 2]
for atom in mol_strings:
    print()
    mol = gto.M(atom=atom, basis=basis, verbose=3)
    cell = pgto.M(a=np.eye(3)*10, atom=atom, basis=basis, verbose=3)
    
    mf = dft.RKS(mol).density_fit()
    mf.xc = "PBE"
    mf.kernel()
    iso_mp = ACMP2(mf)
    set_ac_(iso_mp)
    iso_mp.kernel()

    mf = dft.UKS(mol).density_fit()
    mf.xc = "PBE"
    mf.kernel()
    uks_mp = ACUMP2(mf)
    set_ac_(uks_mp)
    uks_mp.kernel()

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    
    pmf = pdft.RKS(cell).density_fit()
    pmf.xc = "PBE"
    pmf.kernel()
    pbc_mp = ACRMP2(pmf)
    set_ac_(pbc_mp)
    pbc_mp.kernel()

    kpts = cell.make_kpts(nk)
    kmf = pdft.KRKS(cell, kpts)
    kmf.xc = "1.00*GGA_X_PBE+1.00*GGA_C_PBE+0.00*HF"
    kmf = kmf.density_fit()
    kmf.xc = "PBE"
    kmf.kernel()
    t0 = time.monotonic()
    kpt_mp = ACKMP2(kmf)
    set_ac_(kpt_mp)
    kpt_mp.kernel()
    t1 = time.monotonic()

    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()
    kpts = cell.make_kpts(
        nk,
        space_group_symmetry=True,
        time_reversal_symmetry=True,
    )
    kmf = pdft.KRKS(cell, kpts)
    kmf.xc = "1.00*GGA_X_PBE+1.00*GGA_C_PBE+0.00*HF"
    kmf = kmf.density_fit()
    # kmf.xc = "PBE"
    kmf.kernel()
    t2 = time.monotonic()
    sym_mp = SymACKMP2(kmf)
    set_ac_(sym_mp)
    sym_mp.kernel()
    t3 = time.monotonic()

    print("MP2 times", t1 - t0, t3 - t2)

    iso_ens.append(iso_mp.e_tot)
    uks_ens.append(uks_mp.e_tot)
    pbc_ens.append(pbc_mp.e_tot)
    kpt_ens.append(kpt_mp.e_tot)
    sym_ens.append(sym_mp.e_tot)
    ccd_ens.append(mycc.e_tot)
    cct_ens.append(mycc.e_tot + mycc.ccsd_t())
print()
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(iso_ens))
print(rxn_energy(uks_ens))
print(rxn_energy(pbc_ens))
print(rxn_energy(kpt_ens))
print(rxn_energy(sym_ens))
print(rxn_energy(ccd_ens))
print(rxn_energy(cct_ens))

