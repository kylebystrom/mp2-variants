from pyscf.acmp.ac_mp2 import ACMP2, BasicEigNumInterpolator, \
        BasicMatNumInterpolator
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.acmp.pbc.ac_mp2 import ACRMP2, ACUMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np


basis = "def2-tzvp"
mol_strings = [
    "H 0 0 0; H 0 0 0.7", 
    "H 0 0 0; F 0 0 1.1",
    "F 0 0 0; F 0 0 1.4",
]

Ninterp = 4096
eigint = BasicEigNumInterpolator(Ninterp, [])
matint = BasicMatNumInterpolator(Ninterp, [])

mp_ens = []
rmp_ens = []
ump_ens = []
cc_ens = []
cct_ens = []
for atom in mol_strings:
    print()
    mol = gto.M(a=np.eye(3)*6, atom=atom, basis=basis, verbose=5)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mymp = RMP2(mf)
    mymp.kernel()
    myrmp = ACRMP2(mf)
    myrmp.ac_interpolator = matint
    myrmp.si_limit = "GGA_X_PBE"
    myrmp.kernel()

    umf = scf.UHF(mol).density_fit()
    umf.kernel()
    myump = ACUMP2(umf)
    myump.ac_interpolator = matint
    myump.si_limit = "GGA_X_PBE"
    myump.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    etrip = mycc.ccsd_t()
    mp_ens.append(mymp.e_tot)
    rmp_ens.append(myrmp.e_tot)
    ump_ens.append(myump.e_tot)
    cc_ens.append(mycc.e_tot)
    cct_ens.append(mycc.e_tot + etrip)
print()
print(cc_ens)
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(mp_ens))
print()
print(rxn_energy(rmp_ens))
print()
print(rxn_energy(ump_ens))
print()
print(rxn_energy(cc_ens))
print()
print(rxn_energy(cct_ens))
print()

