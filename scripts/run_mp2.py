from pyscf.mp.lmp2 import LambdaMP2, MP2
from pyscf.mp.ulmp2 import LambdaUMP2, UMP2
from pyscf import gto, scf
from pyscf import cc

basis = "aug-cc-pvtz"
atom = "He"

mol = gto.M(atom=atom, basis=basis, spin=0)

mf = scf.RHF(mol)
mf.kernel()

mymp = MP2(mf)
mymp.kernel()

mylmp = LambdaMP2(mf)
mylmp.kernel()

mf = scf.UHF(mol)
mf.kernel()

mymp = UMP2(mf)
mymp.kernel()

mylmp = LambdaUMP2(mf)
mylmp.kernel()

mp_ens = []
lmp_ens = []
cc_ens = []
cct_ens = []

mol_strings = [
    "H 0 0 0; H 0 0 0.7", 
    "H 0 0 0; F 0 0 1.1",
    "F 0 0 0; F 0 0 1.4",
]

for atom in mol_strings:
    print()
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    mymp = RMP2(mf)
    mymp.kernel()
    mylmp = LambdaRMP2(mf)
    mylmp.omega_code = "MGGA_WP"
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

