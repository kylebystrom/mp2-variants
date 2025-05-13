from pyscf.pbc import gto, scf, mp
from pyscf.gto.basis import parse_nwchem


def get_basis(atoms):
    basname = 'cc-pvdz'
    fbas = '../ccgto/basis/gth-hf-rev/%s-lc.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis

cell = gto.Cell()
cell.verbose = 4
cell.fromfile('Li.poscar')
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.basis = get_basis(['Li'])
cell.max_memory = 100000
# cell.space_group_symmetry = True
# cell.symmorphic = False
cell.build()

kpts = cell.make_kpts(
    [2, 2, 2],
    #space_group_symmetry=True,
    #time_reversal_symmetry=True,
)
kmf = scf.KRHF(cell).density_fit()
# kmf = scf.KRHF(cell).density_fit().newton()
# kmf = scf.KRHF(cell, kpts)
kmf.kpts = kpts
ehf = kmf.kernel()

print(kmf.mo_energy)

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)
