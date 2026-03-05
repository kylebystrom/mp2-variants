import numpy as np


def _get_ac_param_array(N):
    alphas = np.linspace(0, 1, N + 1)[:-1] + 0.5 / N
    alphas = alphas
    dalpha = 1.0 / N
    return alphas, dalpha


def _identity_like(m):
    return np.identity(m.shape[0])


def _acmp_matrix_pow(mat, mypow):
    eval, evec = np.linalg.eigh(mat)
    eval = np.maximum(eval, 0)
    eval = eval**mypow
    return (evec * eval).dot(evec.T.conj())


def _acmp_matrix_exp(mat, expnt=1.0):
    eval, evec = np.linalg.eigh(mat)
    eval = np.exp(expnt * eval)
    return (evec * eval).dot(evec.T.conj())


def _acmp_matrix_pow(mat, mypow):
    eval, evec = np.linalg.eig(mat)
    evec_inv = np.linalg.solve(evec, _identity_like(evec))
    # eval = np.maximum(eval, 0)
    # eval = np.minimum(eval, 1e10)
    eval = eval**mypow
    return (evec * eval).dot(evec_inv)


def _acmp_matrix_exp(mat, expnt=1.0):
    eval, evec = np.linalg.eig(mat)
    evec_inv = np.linalg.solve(evec, _identity_like(evec))
    eval = np.exp(expnt * eval)
    return (evec * eval).dot(evec_inv)


class ACMatrix:
    """
    Hermitian matrix object supporting some basic operations.
    All operations assume that the matrix is positive semi-definite,
    and all operations (other than subtraction) preserve positive
    semi-definite-ness
    """

    def __init__(self, arr):
        if np.isnan(arr).any():
            raise ValueError("array must be finite to make ACMatrix")
        self._data = arr.copy()
        self._data[:] += self._data.T.conj()
        self._data[:] *= 0.5

    @property
    def N(self):
        return self._data.shape[0]

    @property
    def shape(self):
        return self._data.shape

    def get_data_view(self):
        """Return data without copying. It is immutable,
        to avoid data corruption"""
        res = self._data.view()
        res.flags.writeable = False
        return res

    def __mul__(self, other):
        if isinstance(other, ACMatrix):
            tmp = (other**0.5)._data
            return ACMatrix(tmp.dot(self._data).dot(tmp))
        else:
            return ACMatrix(self._data * other)

    def __rmul__(self, other):
        if isinstance(other, ACMatrix):
            return other * self
        else:
            return ACMatrix(self._data * other)

    def __add__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(self._data + other._data)
        elif isinstance(other, np.ndarray):
            assert self._data.shape == other.shape
            return ACMatrix(self._data + other)
        else:
            return ACMatrix(self._data + other * np.identity(self.N))

    __radd__ = __add__

    def __truediv__(self, other):
        if isinstance(other, ACMatrix):
            tmp = other**0.5
            res = np.linalg.solve(tmp._data, self._data)
            res = np.linalg.solve(tmp._data, res.T.conj())
            # res = np.linalg.solve(other._data, self._data)
            # res = _acmp_matrix_pow(res.dot(res.T), 0.5)
            return ACMatrix(res)
            # return ACMatrix(np.linalg.solve(other._data, self._data))
        else:
            return ACMatrix(self._data / other)

    def __sub__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(self._data - other._data)
        else:
            return ACMatrix(self._data - other * np.identity(self.N))

    def __rsub__(self, other):
        res = self.__sub__(other)
        res._data[:] *= -1
        return res

    def __pow__(self, power):
        return ACMatrix(_acmp_matrix_pow(self._data, power))

    def exp(self):
        return ACMatrix(_acmp_matrix_exp(self._data))

    def apply(self, func):
        """
        Take an arbitrary function of the matrix by the spectral approach
        (take its eigenvalues and apply the function to the eigenvalues).
        """
        evals, evecs = np.linalg.eigh(self._data)
        evals = func(evals)
        return ACMatrix((evecs * evals).dot(evecs.T.conj()))

    def __repr__(self):
        return repr(self._data).replace("array", "ACMat")

    def __str__(self):
        return str(self._data)


def get_interpolator(
        mode="M", name=None, acw=None, params=None, nalpha=None
):
    if name is not None:
        if name == "basic":
            acw = BasicACW()
        elif name == "square":
            acw = SquareACW()
        elif name == "screen":
            acw = ScreenACW(params=params)
        else:
            raise ValueError
    elif acw is None:
        acw = BasicACW()
    return ACInterpolator(nalpha=nalpha, mode=mode, acw=acw)


def _get_corrected_winf(winf, exx):
    # Make sure winf_oo is positive
    d = 0.0001
    prod = (winf - exx)**2 + d * exx**2
    return prod**0.5   # _acmp_matrix_pow(prod, 0.5)


def _check_fmt(w_list):
    is_mat = isinstance(w_list[0], ACMatrix)
    if is_mat:
        for w in w_list:
            assert isinstance(w, ACMatrix)
    else:
        for w in w_list:
            assert isinstance(w, np.ndarray) and w.ndim == 1


class ACW:
    """
    Adiabatic Connection function
    """
    def __init__(self, params=None):
        if params is None:
            params = []
        self._params = params
        self._cache = {}

    @property
    def params(self):
        return [p for p in self._params]

    @property
    def analytical_result(self):
        res = self._cache.get("__ANALYTICAL__", None)
        if isinstance(res, ACMatrix):
            res = np.trace(res._data)
        elif isinstance(res, np.ndarray):
            res = np.sum(res)
        return res

    def compute_cache(self, w_list):
        _check_fmt(w_list)
        self.clear_cache()
        # winf = _get_corrected_winf(w_list[-1], w_list[1])
        winf = get_damped_winf_diff(w_list[-1], w_list[1], 8)
        self._cache["w0"] = w_list[0]
        self._cache["wr"] = w_list[0] / winf

    def clear_cache(self):
        self._cache = {}

    def __call__(self, alpha):
        raise NotImplementedError


class BasicACW(ACW):
    def __call__(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        return (alpha * w0) / (1 + alpha * wr)


class SquareACW(ACW):
    def __call__(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        term = (alpha * wr)**2
        return (alpha * w0) / (1 + term)**0.5


def _ac_exp(a):
    if isinstance(a, ACMatrix):
        return a.exp()
    else:
        return np.exp(a)


class ScreenACW(ACW):
    def __init__(self, params=None):
        if params is None:
            params = [4.627e-01, 1.378e+00, 7.555e-01, 4.852e-01]
        super().__init__(params=params)

    def compute_cache(self, w_list):
        self.clear_cache()
        winf = _get_corrected_winf(w_list[-1], w_list[2])
        self._cache["w0"] = w_list[0]
        self._cache["winf"] = winf
        self._cache["w1"] = w_list[0] / (-2 * w_list[1])**0.5

    def __call__(self, alpha):
        w0, winf, w1 = self._cache["w0"], self._cache["winf"], self._cache["w1"]
        a, b, c, d = self.params
        bwrs = alpha * w0 * w0 / winf
        hwrs = c * bwrs**0.5 / (1 + a * w0) + d * bwrs**0.25 * w0**0.25 / (1 + b * w0**0.25)
        hwrs = hwrs * _ac_exp((winf / w1)**3 - 1)
        awrs = alpha * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        return alpha * w0 / denom


def _apply_func(func, mat):
    if isinstance(mat, ACMatrix):
        val = mat.apply(func)
    else:
        val = func(mat)
    return val


def get_damped_winf_diff(winf, exx, a):
    ratio = winf / exx + 1e-10
    func = lambda x: np.log(1 + np.exp(a * (1 - x))) / np.log(1 + np.exp(a))
    if isinstance(winf, ACMatrix):
        val = ratio.apply(func)
    else:
        val = func(ratio)
    wdiff = winf * (1 - ratio**-1 * (1 - val))
    return wdiff


def get_damped_winf_diff_grad(winf, exx, a):
    ratio = winf / exx
    func = lambda x: np.log(1 + np.exp(a * (1 - x))) / np.log(1 + np.exp(a))
    dfunc = lambda x: (
        a * (np.exp(a * (1 - x)) + 1e-14)
        / (np.log(1 + np.exp(a * (1 - x)) + 1e-14)
            * np.log(1 + np.exp(a)))
    )
    if isinstance(winf, ACMatrix):
        val = ratio.apply(func)
        dval = ratio.apply(dfunc)
    else:
        val = func(ratio)
        dval = dfunc(ratio)
    dwinf = winf / winf - exx * dval / exx
    dexx = val - 1 + exx * dval * winf / exx**2
    return dwinf, dexx


class SPL_ACW(ACW):
    def __init__(self, params=None):
        super().__init__(params=params)

    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0 = w_list[:2]
        winf = w_list[-1]
        winf_eff = get_damped_winf_diff(winf, w0, 5)
        self._cache["terms"] = (winf_eff, winf_eff, 2 * w0p / winf_eff, 0)
        func = lambda x: 1 - (np.sqrt(1 + 2 * (x + 1e-10)) - 1) / (x + 1e-10)
        res = _apply_func(func, w0p / winf_eff)
        analytical = winf_eff * res
        self._cache["__ANALYTICAL__"] = analytical

    def __call__(self, alpha):
        w, x, y, z = self._cache["terms"]
        return w - x / ((1 + y * alpha)**0.5 + z)


def corr_eigvals_(acm):
    mat = acm._data
    eval, evec = np.linalg.eigh(mat)
    eval = np.abs(eval)
    acm._data = (evec * eval).dot(evec.T.conj())


def corr_eigvals2_(acm, tol=1e-16):
    mat = acm._data
    eval, evec = np.linalg.eigh(mat)
    eval = np.maximum(eval, tol)
    acm._data = (evec * eval).dot(evec.T.conj())


class ISI_ACW(SPL_ACW):
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp = w_list[:3]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            assert (w0p >= 0).all()
            assert (winf >= 0).all()
            assert (w0 >= 0).all()
            assert (winfp >= 0).all()
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
        winf_eff = get_damped_winf_diff(winf, w0, 5)
        x = 2 * w0p
        y = winfp
        z = winf_eff
        tmp1 = (y**2)
        tmp3 = x**2 * tmp1
        tmp2 = z**2
        tmp1 = x * tmp1
        xp = tmp1 / tmp2
        yp = tmp3 / (tmp2**2)
        zp = xp / z - 1
        # corr_eigvals_(zp)
        self._cache["terms"] = (winf_eff, xp, yp, zp)


class PURE_ISI_ACW(SPL_ACW):
    # numerical issues, should only be used with mode="E"
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp = w_list[:3]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            pass
        else:
            raise NotImplementedError
        winf_eff = winf - w0
        x = 2 * w0p
        y = winfp
        z = winf_eff
        tmp1 = (y**2)
        tmp3 = x**2 * tmp1
        tmp2 = z**2
        tmp1 = x * tmp1
        xp = tmp1 / tmp2
        yp = tmp3 / (tmp2**2)
        zp = xp / z - 1
        self._cache["terms"] = (winf_eff, xp, yp, zp)


class MOD_ISI_ACW(SPL_ACW):
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp = w_list[:3]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            w0p = np.clip(w0p, 0, 1e100)
            winf = np.clip(winf, 0, 1e100)
            w0 = np.clip(w0, 0, 1e100)
            winfp = np.clip(winfp, 0, 1e100)
            assert (w0p >= -1e-14).all()
            assert (winf >= -1e-14).all()
            assert (w0 >= -1e-14).all()
            assert (winfp >= -1e-14).all()
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
        winf_eff = get_damped_winf_diff(winf, w0, 8)
        self._cache["wterms"] = (w0, w0p, winf, winfp, winf_eff)
        self._cache["terms"] = (w0p, w0p * winfp / winf_eff**2, w0p / winf_eff)

    def __call__(self, alpha):
        x, y, z = self._cache["terms"]
        return alpha * x / (1 + alpha**0.5 * y + alpha * z)


class MMISI_ACW(MOD_ISI_ACW):
    def compute_cache(self, w_list):
        super().compute_cache(w_list)
        w0, w0p, winf, winfp, winf_eff = self._cache["wterms"]
        if len(self._params) > 0:
            self._mix = self._params[0]
        else:
            self._mix = 1.0
        self._cache["rem"] = ((w0p / winf_eff) * winfp**2 - winf_eff**2) / (w0p + 1e-10)**2

    def __call__(self, alpha):
        x, y, z = self._cache["terms"]
        d = 1 + alpha**0.5 * y + alpha * z
        r = self._cache["rem"]
        func = lambda x: self._mix * (2 / (1 + np.exp(x / self._mix)) - 1)
        res = _apply_func(func, r * d * (2 / (alpha + 1e-10)**2))
        return (alpha * x / d) * (1 + res)


class MOD_ISI_ACW_MAT(SPL_ACW):
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp = w_list[:3]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            raise NotImplementedError
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
            w0p = w0p._data
            winf = winf._data
            w0 = w0._data
            winfp = winfp._data


class PURE_MOD_ISI_ACW(SPL_ACW):
    # numerical issues, should only be used with mode="E"
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp = w_list[:3]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            assert (w0p >= -1e-14).all()
            assert (winf >= -1e-14).all()
            assert (w0 >= -1e-14).all()
            assert (winfp >= -1e-14).all()
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
        winf_eff = winf - w0
        self._cache["terms"] = (w0p, w0p * winfp / winf_eff**2, w0p / winf_eff)

    def __call__(self, alpha):
        x, y, z = self._cache["terms"]
        return alpha * x / (1 + alpha**0.5 * y + alpha * z)


class ISI3_ACW(ACW):
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp, winfpp = w_list[:4]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            assert (w0p >= 0).all()
            assert (winf >= 0).all()
            assert (w0 >= 0).all()
            assert (winfp >= 0).all()
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
        winf_eff = get_damped_winf_diff(winf, w0, 8)
        z = w0p**2 / winf_eff**2
        y = winfp / winf_eff
        y = 2 * y * z
        x = winfpp + 3 * winfp**2 * (winf_eff**-1 - winf**-1)
        x = x * z / winf_eff
        self._cache["terms"] = (w0p, x, y, z)

    def __call__(self, alpha):
        w, x, y, z = self._cache["terms"]
        return alpha * w / (1 + alpha * x + alpha**1.5 * y + alpha**2 * z)**0.5


class ISI30_ACW(ACW):
    def compute_cache(self, w_list):
        # note w0 is exx, w0p is 2*EMP2
        self.clear_cache()
        w0p, w0, winfp, winfpp = w_list[:4]
        winf = w_list[-1]
        if isinstance(w0p, np.ndarray):
            assert (w0p >= 0).all()
            assert (winf >= 0).all()
            assert (w0 >= 0).all()
            assert (winfp >= 0).all()
        else:
            assert not np.isnan(w0p._data).any()
            assert not np.isnan(winf._data).any()
            assert not np.isnan(w0._data).any()
            assert not np.isnan(winfp._data).any()
            for acm in [w0p, winf, w0, winfp]:
                corr_eigvals_(acm)
        winf_eff = get_damped_winf_diff(winf, w0, 8)
        z = w0p**2 / winf_eff**2
        y = winfp / winf_eff
        y = 2 * y * z
        x = winfpp # + 3 * (winf_eff**-1 - winf**-1) * winfp**2
        x = x * w0p / winf_eff**3
        self._cache["terms"] = (w0p, x, y, z)

    def __call__(self, alpha):
        w, x, y, z = self._cache["terms"]
        return alpha * w / (1 + alpha * x + alpha**1.5 * y + alpha**2 * z)**0.5


class NLANE_ACW(ACW):
    def compute_cache(self, w_list):
        self.clear_cache()
        w0p, w0, w1 = w_list
        w1 = w1 * -1
        def func_getter(lam):
            def func(alpha):
                sgn = np.sign(alpha)
                alpha = np.abs(alpha)
                # avoid nan
                alpha[sgn == 0] = 1e-12
                if np.isnan(alpha).any():
                    raise ValueError
                # https://github.com/dkhan42/nLanE-DH/blob/main/nLanE.py
                c = np.sqrt(9*alpha**2 - 16*np.sqrt(2)*alpha + 12*alpha +4) - alpha + 2
                c = c/(4*alpha)
                if np.isnan(c).any():
                    raise ValueError
                num = (1 - np.sqrt(lam+1)/(c*lam + 1))
                res = num / (c - 0.5) / alpha
                if np.isnan(res).any():
                    raise ValueError
                return res  # sgn * res
            return func
        if isinstance(w0p, ACMatrix):
            corr_eigvals2_(w0p, tol=1e-8)
            corr_eigvals_(w1)
            corr_eigvals_(w0)
        else:
            raise NotImplementedError
        weff = w1 - w0
        corr_eigvals_(weff)
        corr_eigvals2_(weff, tol=1e-12)
        alpha = weff / w0p
        corr_eigvals_(alpha)
        corr_eigvals2_(alpha, tol=1e-12)
        self._cache["weff"] = weff
        self._cache["alpha"] = alpha
        self._cache["getter"] = func_getter

    def __call__(self, alpha):
        func = self._cache["getter"](alpha)
        return self._cache["weff"] * self._cache["alpha"].apply(func)


class MOD_NLANE_ACW(ACW):
    def compute_cache(self, w_list):
        self.clear_cache()
        w0p, w0, w1 = w_list
        w1 = w1 * -1
        def func_getter(lam):
            def func(alpha):
                c = 0.5 + 1 / alpha
                sgn = np.sign(alpha)
                alpha = np.abs(alpha)
                alpha[sgn == 0] = 1e-12
                return (1 - np.sqrt(lam+1)/(c*lam + 1))
            return func
        if isinstance(w0p, ACMatrix):
            corr_eigvals2_(w0p, tol=1e-8)
            corr_eigvals_(w1)
            corr_eigvals_(w0)
        else:
            raise NotImplementedError
        weff = w1 - w0
        corr_eigvals_(weff)
        corr_eigvals2_(weff, tol=1e-12)
        alpha = weff / w0p
        self._cache["weff"] = weff
        self._cache["alpha"] = alpha
        self._cache["getter"] = func_getter

    def __call__(self, alpha):
        func = self._cache["getter"](alpha)
        return self._cache["weff"] * self._cache["alpha"].apply(func)


class ACInterpolator:
    def __init__(self, nalpha=None, mode=None, acw=None):
        if nalpha is None:
            nalpha = 512
        if mode is None:
            mode = "M"
        else:
            mode = mode.upper()[0]
        assert mode in ["M", "E"]
        if acw is None:
            acw = BasicACW()
        self.mode = mode
        self._acw = acw
        self._alphas, self._dalpha = _get_ac_param_array(nalpha)

    @property
    def dalpha(self):
        return self._dalpha

    @property
    def alphas(self):
        return self._alphas

    @property
    def params(self):
        return self._acw.params

    def __call__(self, w_list, occs=None):
        if occs is None:
            occs = np.ones(w_list[0].shape[0])
        if self.mode == "E":
            w_list = [np.diag(w) for w in w_list]
        else:
            w_list = [ACMatrix(w) for w in w_list]
        self._acw.compute_cache(w_list)
        if self._acw.analytical_result is not None:
            return -1 * self._acw.analytical_result
        if self.mode == "E":
            res = -(self.dalpha * self._acw(self.alphas[:, None])).sum(0)
            return res.dot(occs)
        else:
            energy = 0
            for alpha in self.alphas:
                acterm = self._acw(alpha).get_data_view()
                #energy -= self.dalpha * np.diag(acterm).dot(occs)
                energy -= self.dalpha * np.diag(acterm).real#.dot(occs)
            energy = energy.dot(occs)
            return energy


class BaseGapModel:
    """
    lambda-MP2 artificial gap function
    """
    def __init__(self, params=None, needs_mp2_mat=False):
        if params is None:
            params = []
        self._params = params
        self._needs_mp2 = needs_mp2_mat

    @property
    def needs_mp2_mat(self):
        return self._needs_mp2

    def __call__(self, w_list):
        raise NotImplementedError


class AnyGapModel(BaseGapModel):
    def __init__(self, call=None, needs_mp2_mat=False):
        self._call = call
        super().__init__(needs_mp2_mat=needs_mp2_mat)

    def __call__(self, w_list):
        return self._call(w_list)


class BasicGapModel(BaseGapModel):
    def __init__(self, needs_mp2_mat=False):
        super().__init__(params=None, needs_mp2_mat=needs_mp2_mat)
    
    def __call__(self, w_list):
        return w_list[0]


class ArtificialGapCalculator:
    def __init__(self, mode=None, gap_model=None):
        if mode is None:
            mode = "M"
        else:
            mode = "E"
        assert mode in ["M", "E"]
        if gap_model is None:
            gap_model = BasicGapModel()
        elif not isinstance(gap_model, BaseGapModel):
            if isinstance(gap_model, tuple):
                assert len(gap_model) == 2
                call, needs_mp2 = gap_model
            else:
                call = gap_model
                needs_mp2 = False
            assert callable(call)
            gap_model = AnyGapModel(call, needs_mp2)
        self.gap_model = gap_model
        self.mode = mode
    
    @property
    def requires_mp2_mat(self):
        return self.gap_model.needs_mp2_mat
    
    def compute_artificial_gap(self, w_list):
        if self.mode == "E":
            _w_list = w_list
            w_list = []
            for w in _w_list:
                if w.ndim == 1:
                    w_list.append(w)
                else:
                    w_list.append(np.diag(w))
            w_list = [w for w in w_list]
        else:
            w_list = [ACMatrix(w) for w in w_list]
        result = self.gap_model(w_list)
        if self.mode == "E":
            return result
        else:
            return result._data
