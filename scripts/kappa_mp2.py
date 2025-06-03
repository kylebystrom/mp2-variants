from pyscf.pbc import gto, scf, mp
from pyscf.acmp.pbc import kappa_kmp2
from pyscf import df
import numpy as np
from pyscf.gto.basis import parse_nwchem
from pyscf.pbc.gto.cell import fromfile

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
print(_atoms)
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
kmf = scf.KHF(cell, kpts).density_fit()
kmf.conv_tol = 1e-7
kmf.conv_tol_grad = 2e-3
ehf = kmf.kernel()

mypt = mp.KMP2(kmf)
mypt.kernel(with_t2=False)
e_mp2 = mypt.e_tot
print("KMP2 energy (per unit cell) =", mypt.e_tot)

# mypt = kappa_kmp2.KappaKMP2(kmf, kappa=1000)
# mypt.kernel(with_t2=False)
# print("KMP2 energy (per unit cell) =", mypt.e_tot)

mypt = kappa_kmp2.KappaKMP2(kmf, kappa=1.5)
mypt.kernel(with_t2=False)
e15 = mypt.e_tot

mypt.kappa = 1.1
mypt.kernel(with_t2=False)
e11 = mypt.e_tot

mypt.kappa = 0.4
mypt.damping = "sigma"
mypt.kernel(with_t2=False)
esigma = mypt.e_tot

dat = {
    "converged": kmf.converged,
    "et_hf": ehf.item(),
    "ec_mp2": (e_mp2 - ehf).item(),
    "ec_kappa_1.5": (e15 - ehf).item(),
    "ec_kappa_1.1": (e11 - ehf).item(),
    "ec_sigma_0.4": (esigma - ehf).item(),
}

fname = f"kappa_results/{NKPT}_{BASIS}_{LATTICE:.2f}.yaml"
with open(fname, 'w') as f:
    yaml.dump(dat, f)

