from pyscf.pbc import gto, scf, mp
from pyscf import df
import numpy as np
from pyscf.gto.basis import parse_nwchem
from pyscf.pbc.gto.cell import fromfile

# from pyscf.acmp.ac_interpolators import ScreenedMatNumInterpolator as Interp
# from pyscf.acmp.ac_interpolators import SquareMatNumInterpolator as Interp
# from pyscf.acmp.ac_interpolators import BasicEigNumInterpolator as Interp
from pyscf.acmp.ac_interpolators import BasicMatNumInterpolator as Interp
from pyscf.acmp.pbc.ac_kmp2 import ACKMP2
from pyscf.acmp.mp2_numint import mgga_sce_limit

from pyscf import scf as iscf

import sys
import yaml


NKPT = int(sys.argv[1])
BASIS = sys.argv[2]
LATTICE = float(sys.argv[3])


def get_basis(atoms):
    basname = BASIS
    fbas = '../2022_data_for_paper_cc_al_li/basis_sets/%s.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis


cell = gto.Cell()
a, atom = fromfile('Li.345.poscar', None)
_atoms = []
_atom = []
fac = LATTICE / 3.45
for i, at in enumerate(atom.split()):
    if i % 4 == 0:
        symb = at
    else:
        _atom.append(float(at) * fac)
    if i % 4 == 3:
        _atoms.append((symb, _atom))
    _atom = []
atom = _atoms
a = np.array([float(b) for b in a.split()]).reshape(3, 3)
a[:] *= fac
cell.a = a
cell.set_geom_(atom, unit='Angstrom', inplace=True)
cell.pseudo = 'gth-pade'
cell.basis = get_basis(['Li'])
cell.verbose = 4
cell.space_group_symmetry = True
cell.symmorphic = False
cell.max_memory = 200000
# cell.exp_to_discard = 0.1
cell.build()

kpts = cell.make_kpts(
    [NKPT, NKPT, NKPT],
    scaled_center=[1.0/6, 1.0/6, 0.5],
)
kmf = scf.KRKS(cell, kpts).density_fit() # .newton()
kmf = scf.addons.smearing_(kmf, sigma=.0003, method='gauss')
kmf.xc = "PBE"
# kmf.level_shift = 0.1
# iscf.addons.dynamic_level_shift_(kmf, factor=0.5)
kmf.conv_tol = 1e-7
kmf.conv_tol_grad = 2e-3
ehf = kmf.kernel()

kmf.smearing_method = False
kmf.mo_occ = kmf.get_occ()

mypt = mp.KMP2(kmf)
mypt.kernel(with_t2=False)
e_mp2 = mypt.e_corr
print("KMP2 energy (per unit cell) =", e_mp2)

mypt = ACKMP2(kmf)
mypt.ac_interpolator = Interp(512, [])
mypt.si_limit = ("MGGA", mgga_sce_limit)
mypt.kernel(with_t2=False)
print("Kappa-KMP2 energy (per unit cell) =", mypt.e_tot)
e_acmp2 = mypt.e_tot
ecorr = mypt.e_corr
ehf_nscf = e_acmp2 - ecorr

dat = {
    "converged": kmf.converged,
    "et_ks": ehf.item(),
    "et_hf": ehf_nscf.item(),
    "ec_acmp2": ecorr,
}

fname = f"ac_results/{NKPT}_{BASIS}_{LATTICE:.2f}.yaml"
with open(fname, 'w') as f:
    yaml.dump(dat, f)

