from pyscf.pbc import gto, scf, mp


cell = gto.Cell()
cell.fromfile('Li.poscar')
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.verbose = 6
# cell.space_group_symmetry = True
# cell.symmorphic = False
cell.build()

kpts = cell.make_kpts(
    [6, 6, 6],
    #space_group_symmetry=True,
    #time_reversal_symmetry=True,
)
# kmf = scf.KRHF(cell).density_fit()
kmf = scf.KRHF(cell).density_fit()
# kmf = scf.KRHF(cell, kpts)
kmf.kpts = kpts
ehf = kmf.kernel()

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)
