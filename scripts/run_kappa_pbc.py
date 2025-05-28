from pyscf.acmp.kappa_mp2 import KappaMP2
from pyscf.acmp.pbc.kappa_mp2 import KappaRMP2
from pyscf.acmp.pbc.kappa_kmp2 import KappaKMP2
from pyscf.acmp.pbc.kappa_kmp2_ksymm import KsymAdaptedKappaKMP2 as SymKappaKMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KMP2
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pgto, scf as pscf, dft as pdft
import numpy as np
import time


basis = "def2-tzvp"
mol_strings = [
    "H 0 0 -0.35; H 0 0 0.35", 
    "H 0 0 -0.55; F 0 0 0.55",
    "F 0 0 -0.70; F 0 0 0.70",
]


iso_ens = []
pbc_ens = []
kpt_ens = []
sym_ens = []
KAPPA = 1.1
WITH_T2 = False
nk = [3, 2, 2]
for atom in mol_strings:
    print()
    mol = gto.M(atom=atom, basis=basis, verbose=3)
    cell = pgto.M(a=np.eye(3)*10, atom=atom, basis=basis, verbose=3)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    iso_mp = KappaMP2(mf, kappa=KAPPA)
    iso_mp.kernel(with_t2=WITH_T2)

    pmf = pscf.RHF(cell).density_fit()
    pmf.kernel()
    pbc_mp = KappaRMP2(pmf, kappa=KAPPA)
    pbc_mp.kernel(with_t2=WITH_T2)

    kpts = cell.make_kpts(nk)
    kmf = pscf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()
    t0 = time.monotonic()
    kpt_mp = KappaKMP2(kmf, kappa=KAPPA)
    kpt_mp.kernel(with_t2=WITH_T2)
    t1 = time.monotonic()

    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()
    kpts = cell.make_kpts(
        nk,
        space_group_symmetry=True,
        time_reversal_symmetry=True,
    )
    kmf = pscf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()
    t2 = time.monotonic()
    sym_mp = SymKappaKMP2(kmf, kappa=KAPPA)
    sym_mp.kernel(with_t2=WITH_T2)
    t3 = time.monotonic()

    print("MP2 times", t1 - t0, t3 - t2)

    iso_ens.append(iso_mp.e_tot)
    pbc_ens.append(pbc_mp.e_tot)
    kpt_ens.append(kpt_mp.e_tot)
    sym_ens.append(sym_mp.e_tot)
print()
def rxn_energy(lst):
    return 2 * lst[1] - lst[0] - lst[2]

print(rxn_energy(iso_ens))
print(rxn_energy(pbc_ens))
print(rxn_energy(kpt_ens))
print(rxn_energy(sym_ens))

