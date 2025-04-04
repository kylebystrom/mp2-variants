from pyscf.pbc import gto, scf, mp


cell = gto.Cell()
cell.fromfile('Li.poscar')
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.verbose = 5
cell.max_memory = 100000
# cell.space_group_symmetry = True
# cell.symmorphic = False
cell.build()

kpts = cell.make_kpts(
    [4, 4, 4],
    #space_group_symmetry=True,
    #time_reversal_symmetry=True,
)
# kmf = scf.KRHF(cell).density_fit()
# kmf = scf.KRHF(cell).density_fit()
kmf = scf.KRKS(cell).density_fit()
kmf.xc = "PBE"
# kmf = scf.KRHF(cell, kpts)
kmf.kpts = kpts
ehf = kmf.kernel()
print(kmf.grids)
exit()

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

