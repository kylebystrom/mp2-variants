import numpy
from pyscf import gto, scf, ao2mo
from pyscf.acmp.ueg import ueg
from pyscf.acmp.ueg.magic_numbers import MAGIC_NUMBERS
from pyscf.acmp.kappa_mp2 import KappaMP2
from pyscf.acmp.lambda_mp2 import LambdaMP2
from pyscf.acmp.ac_mp2 import ACMP2

INCORE_ERIS = True


DEFAULTS = {
    "ws_radius": 1.0,
    "nelec_index": 1,
    "nbas_index": 2,
    "vcut": False,
    "h0": "kinetic",
    "mp2_init": lambda mf: KappaMP2(mf, kappa=1.1, damping="kappa"),
}


def get_1e_vmat_ueg(mp, code, mo_coeff=None):
    mf = mp._scf
    nocc = int(numpy.rint(numpy.sum(mf.mo_occ))) // 2
    if code == "__HF__":
        return 0.5 * mf.get_k()[:nocc, :nocc]
    elif code == "HF":
        return -0.25 * mf.get_k()[:nocc, :nocc]
    if isinstance(code, tuple):
        xctype = code[0]
    else:
        xctype = mp._numint._xc_type(code)
    rho = mf.mgga_rho_vector()
    if xctype == "LDA":
        rho = rho[0]
    elif xctype == "GGA":
        rho = rho[:4]
    v = mp._numint.eval_xc_eff(code, rho, deriv=0, xctype=xctype)[0]
    # density functional is always constant and diagonal
    # for the UEG
    return v.item() * numpy.identity(nocc)


class UEGKappaMP2(KappaMP2):
    pass


class UEGLambdaMP2(LambdaMP2):
    get_acmp_df_mat = get_1e_vmat_ueg


class UEGACMP2(ACMP2):
    get_acmp_df_mat = get_1e_vmat_ueg


def fill_settings_(settings):
    for key, value in DEFAULTS.items():
        settings[key] = settings.get(key, value)


def get_ueg_mf(nelec, nbas, rs, h0, vcut):
    mol = gto.M()
    mol.nelectron = nelec
    mf = scf.RHF(mol)

    my_ueg = ueg.UEG(nelec, nbas, rs, verbose=True)
    print("Madelung energy =", my_ueg.madelung)
    my_ueg.vcut = vcut
    print("Using vcut =", my_ueg.vcut)

    # overwrite the mean-field methods in order to use custom integrals
    hcore = my_ueg.get_hcore()
    nocc = nelec // 2
    ek = numpy.mean(numpy.diag(hcore)[:nocc])
    veff = my_ueg.get_veff()
    # For closed-shell, veff is 0.5 * k
    kmat = 2 * veff
    ex = 0.5 * numpy.mean(numpy.diag(veff)[:nocc])
    if h0 == "kinetic":
        veff[:] = 0.0
    elif h0 == "fock":
        pass
    else:
        raise ValueError("Unsupported h0")
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: numpy.eye(nbas)
    mf.get_k = lambda *args, **kwargs: kmat
    mf.mgga_rho_vector = lambda *args: my_ueg.mgga_rho_vector()
    # need to overwrite get_veff because UEG has no hartree energy
    # also, this is simpler than overwriting get_jk()
    mf.get_veff = lambda *args: veff

    # this stores N^4 _eri in memory, which is limiting
    if INCORE_ERIS:
        mf._eri = ao2mo.restore(8, my_ueg.eri_chem_real(), nbas)
    else:
        raise NotImplementedError

    mf.verbose = 0
    mf.init_guess = '1e'
    mf.scf()
    return mf, ek, ex


def run_ueg_calc(**settings):
    fill_settings_(settings)
    nelec = 2 * MAGIC_NUMBERS[settings["nelec_index"]]
    nbas = MAGIC_NUMBERS[settings["nbas_index"]]
    mf, ek, ex = get_ueg_mf(
        nelec, nbas, settings["ws_radius"], settings["h0"], settings["vcut"]
    )
    mymp = settings["mp2_init"](mf)
    mymp.verbose = 0
    ecorr, _ = mymp.kernel()
    return numpy.array([ek, ex, ecorr / nelec])


def run_ueg_loop(nelec_index, nbas_indices, rs, mp2_init):
    settings = {
        "nelec_index": nelec_index,
        "ws_radius": rs,
        "mp2_init": mp2_init,
    }
    energies = list()
    for nbas_index in nbas_indices:
        settings["nbas_index"] = nbas_index
        energies.append(run_ueg_calc(**settings))
    return energies

