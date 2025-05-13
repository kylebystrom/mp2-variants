from pyscf.acmp.ac_mp2 import ACMP2, ExtractEigNumInterpolator, \
    ScreenedMatNumInterpolator, BasicEigNumInterpolator, \
    BasicMatNumInterpolator
from pyscf.acmp.pbc.ac_mp2 import ACRMP2
from pyscf.acmp.pbc.ac_kmp2 import ACKMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KMP2
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pgto, scf as pscf, dft as pdft
import numpy as np


basis = "def2-tzvp"
mol_strings = [
    "H 0 0 0; H 0 0 0.7", 
    "H 0 0 0; F 0 0 1.1",
    "F 0 0 0; F 0 0 1.4",
]


# eigint = ExtractEigNumInterpolator(512, [])
eigint = ScreenedMatNumInterpolator(512, [])

eigint = BasicEigNumInterpolator(512, [])
# eigint = BasicMatNumInterpolator(512, [])

def set_ac_(acmp):
    acmp.ac_interpolator = eigint
    acmp.si_limit = "1.1*GGA_X_PBE"


iso_ens = []
pbc_ens = []
kpt_ens = []
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
    
    pmf = pdft.RKS(cell).density_fit()
    pmf.xc = "PBE"
    pmf.kernel()
    pbc_mp = ACRMP2(pmf)
    set_ac_(pbc_mp)
    pbc_mp.kernel()

    kpts = cell.make_kpts([2, 2, 2])
    kmf = pdft.KRKS(cell, kpts)
    kmf.xc = "1.00*GGA_X_PBE+1.00*GGA_C_PBE+0.00*HF"
    kmf = kmf.density_fit()
    kmf.xc = "PBE"
    kmf.kernel()
    kpt_mp = ACKMP2(kmf)
    set_ac_(kpt_mp)
    kpt_mp.kernel()

    iso_ens.append(iso_mp.e_tot)
    pbc_ens.append(pbc_mp.e_tot)
    kpt_ens.append(kpt_mp.e_tot)
print()
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(iso_ens))
print(rxn_energy(pbc_ens))
print(rxn_energy(kpt_ens))

