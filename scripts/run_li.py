from pyscf.pbc.mp.kappa_mp2 import KappaRMP2
from pyscf.pbc.mp.mp2 import RMP2
# from pyscf.pbc.mp.kappa_ump2 import KappaUMP2, UMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np
import ase.io
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf


n = 2

struct = ase.io.read("Li.cif")
struct = struct * [n, n, n]
atom = ase_atoms_to_pyscf(struct)


cell = gto.M(
    a=struct.cell,
    atom=atom,
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4,
)

mf = scf.RHF(cell)
mf.kernel()

mymp = RMP2(mf)
mymp.kernel()

mykmp = KappaRMP2(mf)
mykmp.kernel()

print(mymp.e_tot, mykmp.e_tot)

