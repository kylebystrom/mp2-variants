from pyscf.acmp.ac_interpolators import BasicACW, MOD_ISI_ACW, PURE_MOD_ISI_ACW
from pyscf import gto, scf, dft
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ft_mp2 import FTMP2, FTACMP2
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.scf.addons import smearing


mol = gto.M(atom="F 0 0 0; F 0 0 1.1", basis="def2-tzvp")
mf = dft.RKS(mol, xc="PBE").newton()
mf.grids.level = 5
mf.kernel()

df_codes = [("GGA", funcs.gga_pch_winfp_v2)]
silim = ("GGA", funcs.gga_pch_winf_v2)
# acw = MOD_ISI_ACW()
acw = BasicACW()
interpolator = get_interpolator(acw=acw, mode="M")
mymp = ACMP2(mf)
if False:
    mymp = mymp.set_frozen()
mymp.ac_interpolator = interpolator
mymp.df_codes = df_codes
mymp.si_limit = silim
mymp.kernel(with_t2=False)

beta = 100.0

ftmf = smearing(dft.RKS(mol, xc="PBE"), sigma=1.0 / beta)
ftmf.kernel()

mymp = FTACMP2(ftmf, beta=beta)
mymp.ac_interpolator = interpolator
mymp.df_codes = df_codes
mymp.si_limit = silim
mymp.kernel()

mymp = FTMP2(ftmf, beta=beta, ensemble=None)
mymp.kernel()

