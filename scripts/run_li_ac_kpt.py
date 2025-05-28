from pyscf.acmp.pbc.kappa_mp2 import KappaRMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KRMP2
# from pyscf.pbc.mp.kappa_ump2 import KappaUMP2, UMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np
import ase.io
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
from pyscf.gto.basis import parse_nwchem
from pyscf.pbc import mp


n = 1
nk = 5


def get_basis(atoms):
    basname = 'cc-pvdz'
    fbas = '../ccgto/basis/gth-hf-rev/%s-lc.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis


struct = ase.io.read("Li.cif")
struct = struct * [n, n, n]
atom = ase_atoms_to_pyscf(struct)


cell = gto.M(
    a=struct.cell,
    atom=atom,
    basis=get_basis(['Li']),
    pseudo='gth-hf-rev',
    verbose=4,
    max_memory=64000,
    space_group_symmetry=True,
    symmorphic=False,
)

kpts = cell.make_kpts(
    [nk, nk, nk],
    space_group_symmetry=True,
    time_reversal_symmetry=True,
)
kmf = scf.KRHF(cell, kpts)
kmf = kmf.density_fit()
ehf = kmf.kernel()

import time
t0 = time.monotonic()
mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)
t1 = time.monotonic()
print("MP2 time", t1 - t0)

#exit()
#mykmp = KappaRMP2(mf)
#mykmp.kernel()

#print(mymp.e_tot, mykmp.e_tot)

