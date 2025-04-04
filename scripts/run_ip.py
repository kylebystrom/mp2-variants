from pyscf.acmp.lambda_mp2 import LambdaMP2, MP2
from pyscf.acmp.lambda_ump2 import LambdaUMP2, UMP2
from pyscf import gto, scf
from pyscf import cc

basis = "aug-cc-pvtz"
mf_ens = []
mp_ens = []
lmp_ens = []
cc_ens = []
cct_ens = []
for charge, spin in [(0, 2), (1, 3)]:
    mol = gto.M(atom="O", basis=basis, charge=charge, spin=spin)
    mf = scf.UHF(mol)
    mf.kernel()
    mf_ens.append(mf.e_tot)

    mymp = UMP2(mf)
    mymp.kernel()
    mp_ens.append(mymp.e_tot)

    mylmp = LambdaUMP2(mf)
    mylmp.omega_code = "MGGA_WP"    
    mylmp.kernel()
    lmp_ens.append(mylmp.e_tot)
    
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    etrip = mycc.ccsd_t()
    cc_ens.append(mycc.e_tot)
    cct_ens.append(mycc.e_tot + etrip)
    print()

for ens in [mf_ens, mp_ens, lmp_ens, cc_ens, cct_ens]:
    print(ens[1] - ens[0])

