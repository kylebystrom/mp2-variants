from pyscf.pbc.mp.kappa_mp2 import KappaRMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KRMP2
# from pyscf.pbc.mp.kappa_ump2 import KappaUMP2, UMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np
import ase.io
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf


n = 1
nk = 8

struct = ase.io.read("Li.cif")
struct = struct * [n, n, n]
atom = ase_atoms_to_pyscf(struct)


cell = gto.M(
    a=struct.cell,
    atom=atom,
    #basis="def2-svp",
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4,
)

kpts = cell.make_kpts([nk, nk, nk])
kmf = scf.KRHF(cell)
kmf.kpts = kpts
kmf = kmf.density_fit()
ehf = kmf.kernel()

mypt = KRMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

exit()
mykmp = KappaRMP2(mf)
mykmp.kernel()

print(mymp.e_tot, mykmp.e_tot)

