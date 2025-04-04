from pyscf.acmp.acmp2 import MP2, ACMP2
from pyscf import gto, scf, dft
from pyscf import cc

basis = "aug-cc-pvtz"
for atom in ["He", "Ne", "Ar", "Kr"]:
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()

    ks = dft.RKS(mol)
    ks.xc = "LDA,"
    print("X only", ks.energy_tot(mf.make_rdm1()))
    ks.xc = "LDA,LDA_C_PW_MOD"
    print("LDA", ks.energy_tot(mf.make_rdm1()))
    ks.xc = "LDA,GGA_C_PBE"
    print("PBE", ks.energy_tot(mf.make_rdm1()))
    ks.xc = "HF"
    print("HF", ks.energy_tot(mf.make_rdm1()))
    ks.xc = ""
    print("Hartree", ks.energy_tot(mf.make_rdm1()))

    mymp = MP2(mf)
    mymp.kernel()

    acmp = ACMP2(mf)
    acmp.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()
    print()

