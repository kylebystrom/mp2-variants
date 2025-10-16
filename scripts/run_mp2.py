from pyscf.acmp.lambda_mp2 import LambdaMP2, MP2
from pyscf.acmp.lambda_ump2 import LambdaUMP2, UMP2
from pyscf.acmp.kappa_mp2 import KappaMP2
from pyscf.acmp.mp2_numint import lda_plasma_helper, mgga_sce_limit
from pyscf import gto, scf
from pyscf import cc

# basis = "aug-cc-pvtz"
basis = "def2-tzvp"
atom = "He"

mol = gto.M(atom=atom, basis=basis, spin=0)

OMEGA = ("LDA", lda_plasma_helper)
# OMEGA = "PLASMA_LDA_WP"

mf = scf.RHF(mol)
mf.kernel()

mymp = MP2(mf)
mymp.kernel()

mylmp = LambdaMP2(mf)
mylmp.omega_code = "PLASMA_LDA_WP"
mylmp.kernel()

mf = scf.UHF(mol)
mf.kernel()

ks = scf.UKS(mol)
ks.xc = "PBE"
ks.kernel()

mymp = UMP2(mf)
mymp.kernel()

# mylmp = LambdaUMP2(mf)
mylmp = LambdaUMP2(ks)
mylmp.omega_code = "PLASMA_LDA_WP"
mylmp.kernel()

mp_ens = []
lmp_ens = []
kmp_ens = []
cc_ens = []
cct_ens = []

mol_strings = [
    "H 0 0 0; H 0 0 0.7", 
    "H 0 0 0; F 0 0 1.1",
    "F 0 0 0; F 0 0 1.4",
]
import numpy
def get_vmat(w_list):
    oo_kmat, oo_mul, oo_sce = w_list
    oo_kmat = -1 * oo_kmat
    oo_sce = (-1 * oo_sce - oo_kmat)
    print(numpy.max(oo_sce._data), numpy.min(oo_sce._data))
    print(numpy.max(oo_kmat._data), numpy.min(oo_kmat._data))
    # oo_sce = oo_sce.dot(oo_sce) + 0.0001 * oo_kmat.dot(oo_kmat)
    oo_sce = oo_sce * oo_sce + 0.0001 * oo_kmat * oo_kmat
    # oo_sce = _lmp_matrix_pow(oo_sce, 0.5)
    oo_sce = oo_sce**0.5
    oo_vmat = oo_mul * oo_sce
    # oo_vmat = oo_mul.dot(oo_sce)
    # oo_vmat = 0.5 * (oo_vmat + oo_vmat.T)
    return oo_vmat

def get_vmat2(w_list):
    oo_mp2, oo_screen = w_list
    a = 0.3
    tmp = (-a * oo_mp2)**0.5
    # goes -> 0 for large gap/small corr
    # goes -> inf for small gap/large corr
    tmp = tmp / (1 + tmp)
    return tmp * oo_screen


for atom in mol_strings:
    print()
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    ks = scf.RKS(mol)
    ks.xc = "PBE"
    ks.kernel()
    mymp = MP2(mf)
    mymp.kernel()

    mylmp = LambdaMP2(mf, gap_model=(get_vmat2, True))
    mylmp.omega_code = "PLASMA_GGA_WP"
    # mylmp = LambdaMP2(mf, gap_model=get_vmat)

    # mylmp = LambdaMP2(mf)

    # mylmp.df_codes = ["HF", OMEGA]
    # mylmp = LambdaMP2(ks)
    # mylmp.omega_code = ("MGGA", mgga_sce_limit)

    # mylmp.omega_code = OMEGA
    mylmp.kernel()
    mf = scf.RHF(mol)
    mf.kernel()
    mykmp = KappaMP2(mf, kappa=1.1)
    mykmp.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    etrip = mycc.ccsd_t()
    mp_ens.append(mymp.e_tot)
    lmp_ens.append(mylmp.e_tot)
    kmp_ens.append(mykmp.e_tot)
    cc_ens.append(mycc.e_tot)
    cct_ens.append(mycc.e_tot + etrip)
print()
print(cc_ens)
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(mp_ens))
print(rxn_energy(lmp_ens))
print(rxn_energy(kmp_ens))
print(rxn_energy(cc_ens))
print(rxn_energy(cct_ens))

