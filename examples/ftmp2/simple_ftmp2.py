from pyscf.mp.mp2 import MP2
from pyscf.acmp.ft_mp2 import FTMP2
from pyscf import gto, scf
from pyscf.scf.addons import smearing


basis = "def2-tzvp"


def get_h2(r):
    return gto.M(atom="H 0 0 0; H 0 0 {}".format(r), basis=basis)


def get_n2(r):
    return gto.M(atom="N 0 0 0; N 0 0 {}".format(r), basis=basis)


def make_scf(mol):
    # return scf.RHF(mol)
    ks = scf.RKS(mol, xc="PBE")
    ks.grids.level = 5
    return ks

h2 = get_h2(5.0)

mf = make_scf(h2)
mf.kernel()

beta = 60.0

ftmf = smearing(make_scf(h2), sigma=1.0 / beta)
ftmf.kernel()

pt = MP2(mf)
pt.kernel()
ftpt = FTMP2(mf, beta=beta)
ftpt.kernel()
ftpt = FTMP2(ftmf, beta=beta)
ftpt.kernel()

