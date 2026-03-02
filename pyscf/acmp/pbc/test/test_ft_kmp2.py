from pyscf.acmp.ac_interpolators import MOD_ISI_ACW, get_interpolator
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc import scf as pdft
from pyscf.acmp import ac_mp2, ft_mp2
from pyscf.acmp.pbc import ac_mp2 as pbc_ac
from pyscf.acmp.pbc import ac_kmp2 as pbc_ack
from pyscf.acmp.pbc import ft_mp2 as pbc_ft
from pyscf.acmp.pbc import ft_kmp2 as pbc_ftk
from pyscf.mp.mp2 import MP2
from pyscf.pbc.mp.mp2 import RMP2 as PMP2
from pyscf.pbc.mp.kmp2 import KMP2
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.scf.addons import smearing
import unittest
from numpy.testing import assert_allclose
import numpy as np


basis = "def2-svp"
mol_strings = [
    "H 0 0 -0.35; H 0 0 0.35", 
    "H 0 0 -0.55; F 0 0 0.55",
    "F 0 0 -0.70; F 0 0 0.70",
]


df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
silim = ("GGA", funcs.gga_pch_winf_v2)
acw = MOD_ISI_ACW()
aci = get_interpolator(acw=acw, mode="E")

def set_ac_(acmp):
    acmp.ac_interpolator = aci
    acmp.df_codes = df_codes
    acmp.si_limit = silim

ftk = dict(
    beta=30,
    ecorr_method="analytical",
    particle_fix="pt",
)

nk = [3, 1, 1]
for atom in mol_strings:
    ens = []
    print()
    mol = gto.M(atom=atom, basis=basis, verbose=3)
    cell = pgto.M(a=np.eye(3)*10, atom=atom, basis=basis, verbose=3)
    
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    iso_mp = MP2(mf)
    iso_mp.kernel()
    ens.append(mf.e_tot)
    ens.append(iso_mp.e_tot)
    
    pmf = pscf.RHF(cell).density_fit()
    pmf.kernel()
    pbc_mp = PMP2(pmf)
    pbc_mp.kernel()
    ens.append(pmf.e_tot)
    ens.append(pbc_mp.e_tot)

    kpts = cell.make_kpts(nk)
    kmf = pscf.KRHF(cell, kpts).density_fit()
    kmf.kernel()
    kpt_mp = KMP2(kmf)
    kpt_mp.kernel()
    ens.append(kmf.e_tot)
    ens.append(kpt_mp.e_tot)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    iso_mp = ft_mp2.make_ftmp2(MP2(mf), **ftk)
    iso_mp.kernel()
    ens.append(mf.e_tot)
    ens.append(iso_mp.e_zero)
    
    pmf = pscf.RHF(cell).density_fit()
    pmf.kernel()
    pbc_mp = pbc_ft.make_ftmp2(PMP2(pmf), **ftk)
    pbc_mp.kernel()
    ens.append(pmf.e_tot)
    ens.append(pbc_mp.e_zero)

    kpts = cell.make_kpts(nk)
    kmf = pscf.KRHF(cell, kpts).density_fit()
    kmf.kernel()
    kpt_mp = pbc_ftk.make_ftmp2(KMP2(kmf), **ftk)
    kpt_mp.kernel()
    ens.append(kmf.e_tot)
    ens.append(kpt_mp.e_zero)
    print()
    ens = [e.item() for e in ens]
    print("ENS", ens[:len(ens)//2])
    print("ENS", ens[len(ens)//2:])
    resk = kpt_mp.results
    resp = pbc_mp.results
    resp.pop('occs')
    resk.pop('occs')
    print(resp)
    print(resk)
    print()
