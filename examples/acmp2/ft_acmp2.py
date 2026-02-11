from pyscf.acmp.ac_interpolators import BasicACW, MOD_ISI_ACW, PURE_MOD_ISI_ACW
from pyscf import gto, scf, dft
from pyscf.acmp.ac_mp2 import ACMP2
from pyscf.acmp.ft_mp2 import FTMP2, FTACMP2
from pyscf.acmp import mp2_numint as funcs
from pyscf.acmp.ac_interpolators import get_interpolator
from pyscf.scf.addons import smearing

for i in range(8):
    print()

mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-tzvp")
mf = dft.RKS(mol, xc="PBE").newton()
mf.kernel()
mymp = FTMP2(mf, beta=10000)
mymp.kernel()

print()
print()

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

beta = 40.0

ftmf = smearing(dft.RKS(mol, xc="PBE"), sigma=1.0 / beta)
ftmf.kernel()
print("MF ENS", ftmf.e_tot, ftmf.e_free, ftmf.e_zero)

mymp = FTACMP2(ftmf, beta=beta)
mymp.ac_interpolator = interpolator
mymp.df_codes = df_codes
mymp.si_limit = silim
mymp.kernel()

mymp = FTACMP2(ftmf, beta=beta, ensemble="fd")
mymp.ac_interpolator = interpolator
mymp.df_codes = df_codes
mymp.si_limit = silim
mymp.kernel()

mymp = FTMP2(ftmf, beta=beta, ensemble="gc")
mymp.kernel()

mymp = FTMP2(ftmf, beta=beta, ensemble="fd")
mymp.kernel()
exit()
f0 = mymp.f_corr
e0 = mymp.e_corr
mu = mymp.mu_opt

delta = 0.000001
mymp = FTMP2(ftmf, beta=beta*(1+0.5*delta), mu0=mu)
mymp.kernel()
f1 = mymp.e_corr

mymp = FTMP2(ftmf, beta=beta*(1-0.5*delta), mu0=mu)
mymp.kernel()
f2 = mymp.e_corr

mymp = FTMP2(ftmf, beta=beta, mu0=mu+0.5*delta)
mymp.kernel()
f3 = mymp.e_corr

mymp = FTMP2(ftmf, beta=beta, mu0=mu-0.5*delta)
mymp.kernel()
f4 = mymp.e_corr

print("FINAL ENS", e0, f0 + (f1-f2)/delta + mu*(f4-f3)/delta)
# print("FINAL ENS", e0, f0 + mu*(f4-f3)/delta)

mymp.ensemble = None
mymp.kernel()

