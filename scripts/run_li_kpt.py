from pyscf.acmp.pbc.kappa_mp2 import KappaRMP2
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.mp.kmp2 import KMP2
from pyscf.pbc.mp.kmp2_ksymm import KsymAdaptedKMP2
from pyscf.acmp.pbc.kappa_kmp2 import KappaKMP2
from pyscf.acmp.pbc.kappa_kmp2_ksymm import KsymAdaptedKappaKMP2
# from pyscf.pbc.mp.kappa_ump2 import KappaUMP2, UMP2
from pyscf.pbc import gto, scf
from pyscf.pbc import cc
import numpy as np
import ase.io
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
from pyscf.gto.basis import parse_nwchem
from pyscf.pbc import mp
from pyscf.scf.addons import remove_linear_dep_
from pyscf.scf.diis import CDIIS, EDIIS, ADIIS


n = 1
nk = 4


def get_basis(atoms):
    basname = 'cc-pvdz'
    fbas = '../ccgto/basis/gth-hf-rev/%s-lc.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis


#struct = ase.io.read("Li.cif")
#struct = struct * [n, n, n]
#atom = ase_atoms_to_pyscf(struct)

cell = gto.Cell(
    #a=struct.cell,
    #atom=atom,
    basis=get_basis(['Li']),
    pseudo='gth-hf-rev',
    #atom = '''Si  0.0000000000 0.0000000000 0.0000000000
    #          Si  1.3467560987 1.3467560987 1.3467560987''',
    #a = np.asarray([[0.0, 2.6935121974, 2.6935121974],
    #                [2.6935121974, 0.0, 2.6935121974],
    #                [2.6935121974, 2.6935121974, 0.0]]),
    #basis='gth-szv',
    #pseudo='gth-pade',
    verbose=4,
    max_memory=64000,
    #space_group_symmetry=True,
    #symmorphic=False,
)
cell.fromfile('Li.poscar')
cell.mesh = [24, 24, 24]
cell.build()

kpts = cell.make_kpts(
    [nk, nk, nk],
    #space_group_symmetry=True,
    #time_reversal_symmetry=True,
)
kmf = scf.KRHF(cell, kpts)
# kmf.xc = "PBE"
kmf.diis_start_cycle = 3
kmf.diis_space = 12
kmf.diis_damp = 0.8
kmf = remove_linear_dep_(kmf.density_fit())
ehf = kmf.kernel()
exit()

mypt = KsymAdaptedKMP2(kmf)
# mypt = KMP2(kmf)
mypt.kernel(with_t2=False)

mykmp = KsymAdaptedKappaKMP2(kmf, kappa=1.1)
# mykmp = KappaKMP2(kmf, kappa=1.1)
mykmp.kernel(with_t2=False)

print(mypt.e_tot, mykmp.e_tot)

