from pyscf.pbc import gto, scf, mp
from pyscf.acmp.pbc import kappa_kmp2
from pyscf import df

from pyscf.gto.basis import parse_nwchem


import sys


NKPT = int(sys.argv[1])
BASIS = sys.argv[2]


def get_basis(atoms):
    basname = 'dz'
    fbas = '../2022_data_for_paper_cc_al_li/basis_sets/%s.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis

cell = gto.Cell()
cell.fromfile('Li.345.poscar')
cell.pseudo = 'gth-hf-rev'
cell.basis = get_basis(['Li'])  # 'gth-szv'
cell.verbose = 4
cell.space_group_symmetry = True
cell.symmorphic = False
cell.max_memory = 200000
cell.exp_to_discard = 0.1
cell.build()

kpts = cell.make_kpts(
    [5, 5, 5],
    # space_group_symmetry=True,
    # time_reversal_symmetry=True,
    scaled_center=[1.0/6, 1.0/6, 0.5],
    # scaled_center=[-0.25, -0.25, -0.25],
    # scaled_center=[0.5, 0.5, 0.5],
)
kmf = scf.KHF(cell, kpts).density_fit()
# kmf = scf.KRKS(cell, kpts).density_fit()
# kmf.xc = "PBE"
ehf = kmf.kernel()

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

mypt = kappa_kmp2.KappaKMP2(kmf, kappa=1000)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

mypt = kappa_kmp2.KappaKMP2(kmf, kappa=1.5)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

