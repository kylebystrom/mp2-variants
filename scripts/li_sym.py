from pyscf.pbc import gto, scf
from pyscf.pbc.gto.cell import fromfile

a, atom = fromfile("Li.poscar")
# Uncomment in new version to trigger warning and OOM error
atom = atom.replace("1.71965623", "1.71966")

cell = gto.Cell(
    a=a,
    atom=atom,
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4,
    space_group_symmetry=True,
    symmorphic=False,
)
cell.build()


kpts = cell.make_kpts(
    [2, 2, 2],
    space_group_symmetry=True,
    time_reversal_symmetry=True,
)
kmf = scf.KRHF(cell, kpts).density_fit()
ehf = kmf.kernel()

