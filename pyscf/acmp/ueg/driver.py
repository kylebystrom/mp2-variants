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
    nocc = mp.nocc
    if nocc is None:
        if mo_coeff is not None:
            nocc = mo_coeff.shape[-1]
        else:
            nocc = mf.mo_occ.shape[-1]
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


def get_ueg_mf(nelec, nbas, rs, h0, vcut, beta=None):
    mol = gto.M()
    mol.nelectron = nelec
    mf = scf.RHF(mol)
    # mf.conv_tol = 1e-20
    # mf.conv_tol_grad = 1e-20

    def _check_conv(envs):
        is_conv = numpy.max(numpy.abs(envs["dm"] - envs["dm_last"])) < 1e-10
        is_conv = is_conv and abs(envs["e_tot"] - envs["last_hf_e"]) < envs["conv_tol"]
        is_conv = is_conv and envs["norm_gorb"] < envs["conv_tol_grad"]
        return is_conv

    mf.check_convergence = _check_conv
    mf.direct_scf = False
    mf.diis_start_cycle = 1000

    if beta is None:
        my_ueg = ueg.UEG(nelec, nbas, rs, verbose=True)
        print("Madelung energy =", my_ueg.madelung)
        my_ueg.vcut = vcut
        print("Using vcut =", my_ueg.vcut)
        nocc = nelec // 2
        occs = numpy.ones(nocc)
    else:
        from pyscf.scf.addons import smearing

        mf = smearing(mf, sigma=1.0 / beta)
        my_ueg = ueg.FTUEG(nelec, beta, nbas, rs, verbose=True)
        print("Madelung energy =", my_ueg.madelung)
        my_ueg.vcut = vcut
        print("Using vcut =", my_ueg.vcut)
        my_ueg.run_occ_scf(h0)
        nocc = my_ueg.nocc
        occs = my_ueg.occs[:nocc]

    # overwrite the mean-field methods in order to use custom integrals
    hcore = my_ueg.get_hcore()
    veff = my_ueg.get_veff()
    # For closed-shell, veff is -0.5 * k
    kmat = -2 * veff
    assert (kmat >= 0).all()
    jmat = numpy.zeros_like(kmat)
    if h0 == "kinetic":
        veff[:] = 0.0
    elif h0 == "fock":
        pass
    else:
        raise ValueError("Unsupported h0")
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: numpy.eye(nbas)

    def _check_inputs(hermi, omega):
        if hermi != 1:
            raise NotImplementedError
        if omega is not None:
            raise NotImplementedError

    # need to overwrite get_veff because UEG has no hartree energy
    # Also need to override get_k and get_jk for ACMP2 and FTMP2.
    if beta is None:
        mf.get_k = lambda *args, **kwargs: kmat
        mf.get_jk = lambda *args, **kwargs: (jmat, kmat)
        mf.get_veff = lambda *args, **kwargs: veff
    else:
        def get_k(mol=None, dm=None, hermi=1, omega=None):
            _check_inputs(hermi, omega)
            if dm is None:
                occs = my_ueg.occs
            else:
                occs = numpy.diag(dm) / 2
            my_ueg.occs = occs
            my_ueg.nocc = numpy.sum(my_ueg.occs > my_ueg.occ_tol)
            return -2 * my_ueg.get_veff()

        def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                   omega=None):
            _check_inputs(hermi, omega)
            if with_j:
                j = jmat
            else:
                j = None
            if with_k:
                k = get_k(dm=dm)
            else:
                k = None
            return j, k

        def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            _check_inputs(hermi, None)
            if h0 == "kinetic":
                return jmat
            else:
                return -0.5 * get_k(dm=dm)

        mf.get_k = get_k
        mf.get_jk = get_jk
        mf.get_veff = get_veff

    mf.get_j = lambda *args, **kwargs: jmat
    mf.mgga_rho_vector = lambda *args, **kwargs: my_ueg.mgga_rho_vector()
    mf.get_mu = lambda *args, **kwargs: my_ueg.mu

    # this stores N^4 _eri in memory, which is limiting
    if INCORE_ERIS:
        mf._eri = ao2mo.restore(8, my_ueg.eri_chem_real(), nbas)
    else:
        raise NotImplementedError

    mf.verbose = 0
    mf.init_guess = '1e'
    mf.scf()
    ek = (occs * numpy.diag(hcore)[:nocc]).sum() / occs.sum()
    ex = -0.25 * (occs * numpy.diag(kmat)[:nocc]).sum() / occs.sum()
    mf.e_tot = (ek + ex) * nelec
    occs_test = mf.mo_occ[:nocc] / 2
    assert numpy.linalg.norm(occs_test - occs) < 1e-8
    assert numpy.max(numpy.abs(occs_test - occs)) < 1e-9
    if beta is not None:
        my_ueg.occs = mf.mo_occ[:nocc] / 2
        my_ueg.nocc = numpy.sum(my_ueg.occs > my_ueg.occ_tol)
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
