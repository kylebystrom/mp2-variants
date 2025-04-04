from pyscf.acmp.pbc.kappa_mp2 import KappaRMP2
from pyscf.pbc.mp.mp2 import RMP2
# from pyscf.pbc.mp.kappa_ump2 import KappaUMP2, UMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np


basis = "def2-tzvp"
mol_strings = [
    "H 0 0 0; H 0 0 0.7", 
    "H 0 0 0; F 0 0 1.1",
    "F 0 0 0; F 0 0 1.4",
]


mp_ens = []
lmp_ens = []
cc_ens = []
cct_ens = []
for atom in mol_strings:
    print()
    mol = gto.M(a=np.eye(3)*6, atom=atom, basis=basis, verbose=5)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mymp = RMP2(mf)
    mymp.kernel()
    mylmp = KappaRMP2(mf, kappa=1.2)
    mylmp.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    etrip = mycc.ccsd_t()
    mp_ens.append(mymp.e_tot)
    lmp_ens.append(mylmp.e_tot)
    cc_ens.append(mycc.e_tot)
    cct_ens.append(mycc.e_tot + etrip)
print()
print(cc_ens)
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(mp_ens))
print(rxn_energy(lmp_ens))
print(rxn_energy(cc_ens))
print(rxn_energy(cct_ens))

