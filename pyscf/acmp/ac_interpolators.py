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
    return (evec * eval).dot(evec.T)


def _acmp_matrix_exp(mat, expnt=1.0):
    eval, evec = np.linalg.eigh(mat)
    eval = np.exp(expnt * eval)
    return (evec * eval).dot(evec.T)


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
            res = np.linalg.solve(tmp._data, res.T)
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
        return ACMatrix((evecs * evals).dot(evecs.T))

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

    def compute_cache(self, w_list):
        _check_fmt(w_list)
        self.clear_cache()
        winf = _get_corrected_winf(w_list[-1], w_list[1])
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

    def __call__(self, w_list):
        if self.mode == "E":
            w_list = [np.diag(w) for w in w_list]
        else:
            w_list = [ACMatrix(w) for w in w_list]
        self._acw.compute_cache(w_list)
        if self.mode == "E":
            return -(self.dalpha * self._acw(self.alphas[:, None])).sum()
        else:
            energy = 0
            for alpha in self.alphas:
                acterm = self._acw(alpha).get_data_view()
                energy -= self.dalpha * np.trace(acterm)
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
