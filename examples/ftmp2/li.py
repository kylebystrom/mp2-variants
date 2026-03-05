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
from pyscf.acmp.pbc.ft_kmp2 import make_ftmp2
from pyscf.acmp.pbc.ac_kmp2 import ACKMP2
from pyscf.pbc.scf.addons import smearing
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import MOD_ISI_ACW, MMISI_ACW, get_interpolator
import sys


n = 1
nk = int(sys.argv[1])


def get_basis(atoms):
    basname = 'cc-pvdz'
    fbas = '../ccgto/basis/gth-hf-rev/%s-lc.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis


struct = ase.io.read("Li.cif")
struct = struct * [n, n, n]
atom = ase_atoms_to_pyscf(struct)

#df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
#silim = ("GGA", funcs.gga_pch_winf_v2)
df_codes = [("MGGA", funcs.mgga_epc_winfp)]
silim = ("MGGA", funcs.mgga_epc_winf)
# acw = MOD_ISI_ACW()
acw = MMISI_ACW([0.085])
interpolator = get_interpolator(acw=acw, mode="M")

cell = gto.M(
    a=struct.cell,
    atom=atom,
    basis=get_basis(['Li']),
    pseudo='gth-hf-rev',
    verbose=4,
    max_memory=64000,
    #space_group_symmetry=True,
    #symmorphic=False,
    precision=1e-11,
)

kpts = cell.make_kpts(
    [nk, nk, nk],
    # [2, 1, 1],
    #space_group_symmetry=True,
    #time_reversal_symmetry=True,
)
beta = 1000
kmf = smearing(scf.KRKS(cell, kpts, xc="PBE"), sigma=1/beta)
kmf = kmf.density_fit()
ehf = kmf.kernel()

import time
t0 = time.monotonic()
mypt = ACKMP2(kmf)
mypt.ac_interpolator = interpolator
mypt.df_codes = df_codes
mypt.si_limit = silim
mypt = make_ftmp2(
    mypt,
    beta=beta,
    ecorr_method="finite_difference",
    particle_fix="ks",
    with_singles=True,
)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)
t1 = time.monotonic()
print("MP2 time", t1 - t0)

print(mypt.e_hf)

#exit()
#mykmp = KappaRMP2(mf)
#mykmp.kernel()

#print(mymp.e_tot, mykmp.e_tot)

