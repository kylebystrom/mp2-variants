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
    eval = np.maximum(eval, 0)
    eval = np.minimum(eval, 1e10)
    eval = eval**mypow
    return (evec * eval).dot(evec_inv)


def _acmp_matrix_exp(mat, expnt=1.0):
    eval, evec = np.linalg.eig(mat)
    evec_inv = np.linalg.solve(evec, _identity_like(evec))
    eval = np.exp(expnt * eval)
    return (evec * eval).dot(evec_inv)


class ACMatrix:

    def __init__(self, arr):
        self._data = arr

    def __mul__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(self._data.dot(other._data))
        else:
            return ACMatrix(self._data * other)

    def __add__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(self._data + other._data)
        else:
            return ACMatrix(self._data + other)

    def __div__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(np.linalg.solve(other._data, self._data))
        else:
            return ACMatrix(self._data / other)

    def __sub__(self, other):
        if isinstance(other, ACMatrix):
            return ACMatrix(self._data - other._data)
        else:
            return ACMatrix(self._data - other)

    def __pow__(self, power):
        return ACMatrix(_acmp_matrix_pow(self._data, power))


class _ACInterpolator():
    def __call__(self, w_list):
        raise NotImplementedError


class _NumACInterpolator(_ACInterpolator):
    def __init__(self, N, params):
        self._alphas, self._dalpha = _get_ac_param_array(N)
        self._params = params

    @property
    def dalpha(self):
        return self._dalpha

    @property
    def alphas(self):
        return self._alphas

    @property
    def params(self):
        return [p for p in self._params]


class _EigNumInterpolator(_NumACInterpolator):
    def interpolate(self, alphas, w_list):
        raise NotImplementedError

    def __call__(self, w_list):
        w_list = [np.diag(w) for w in w_list]
        return -(self.dalpha * self.interpolate(self.alphas[:, None], w_list)).sum()
    

class BasicEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        return alphas * w0 / (1 + alphas * wr)


class SquareEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5


class RegEigNumInterpolator(_EigNumInterpolator):
    def _interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        w1 = 2 * w_list[2] / w_list[1] - 1
        awrs = alphas * wr
        bwrs = alphas * w0 * wr
        scr = wr._mpac_erel**0.5
        # a, b, c = 1.624e0, 1.593e0
        a, b, c = 8.474e+00, 7.766e+00, 9.116e+00
        a, b, c = 8.474e+00, 1.766e+00, 2.116e+00
        a, b, c = 8.474e+00, 0, 0
        a, b, c = 0.0, 1.684e+00, 0.0
        W1s = wr._mpac_erel
        #denom = (1 + awrs**2 + c * W1s * awrs**1.5)**0.5 + b * W1s * bwrs**0.5 + (1 + a * W1s * awrs)**0.5 - 1
        a, b, c = 1.474e+01, 7.348e+00, 5.000e-01
        denom = denom = (1 + awrs**2)**0.5 + (1 + b * W1s * bwrs**0.75 + a * W1s * awrs)**0.5 - 1
        return alphas * w0 / denom
    
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        w1 = w_list[1]
        awrs = alphas * wr
        bwrs = alphas * w0**2 / w_list[-1]
        a, b, c = 6.363e+00, 1.070e-02, 8.462e+00
        a, b, c, = 5.822e+00, 2.003e-01, 7.681e+00
        # a, b, c = 4.000e+00, 2.302e+00, 2.209e+00
        # a, b, c = 2.991e+00, 2.30e+00, 1.214e+00
        # a, b, c = 0e+00, 0e+00, 1.214e+00
        # a, b, c = 7.526e-01, 1.682e-01, 2.406e+00
        # denom = (1 + awrs**2 + b * w1**2 * bwrs**0.5 + a * w1**2 * bwrs)**0.5
        # denom += c * w1 * bwrs**0.5
        
        # denom = (1 + awrs**2 + a * bwrs**0.5 + c * awrs * w1 / w0)**0.5 + b * bwrs**0.5 
        
        # denom = 1 + awrs + c * awrs * w1 / w0 + b * bwrs**0.5
        a, b, c = 2.601e+00, 2.443e+00, 1.270e-01
        mixer = 1 + a * (1.0 - np.exp(-b * bwrs**0.5))
        denom = mixer * (1 + c * (mixer - 1) * awrs**1.5 + awrs**2 / mixer**2)**0.5

        a, b, c = 4.836e+00, 3.632e+00, 3.155e+00
        mixer = 1 + (1.0 - np.exp(-b * bwrs**0.5))
        denom = mixer * (1 + a * awrs / mixer + c * (mixer - 1) * awrs**1.5 / mixer**1.5 + awrs**2 / mixer**2)**0.5
        denom = (mixer * mixer + a * (mixer - 1) + 2 * awrs + c * (mixer - 1) * awrs**1.5 + awrs**2)**0.5
        
        a, b, c = 2.7e+00, 9.463e+01, 0.000e+00
        denom = (mixer * mixer + 2 * a * mixer * awrs + c * (mixer - 1) * awrs**1.5 + awrs**2)**0.5

        a, b, c, d = 2.771e+00, 1.102e+00, 6.215e-01, 3.475e+00
        denom = (1.0 + awrs**2)**0.5 + c * (b * bwrs) / (1 + b * awrs**0.5 * w0) + d * (a * bwrs) / (1 + a * bwrs**0.75)

        return alphas * w0 / denom


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        winf = w0 / (-w_list[2] * 2)**0.5
        wr = w0 / winf
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5
    

class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w_list[2]
        gap = w_list[1]
        w1 = w1 - w0 * np.log(gap)
        alpha = np.exp(w1 / w0) * gap
        winf = np.sqrt(0.5 * alpha * w0)
        wr = w0 / winf
        return alphas * w0 / (1 + alphas**2 * wr**2)**0.5


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w0 / (-w_list[2] * 2)**0.5
        winf = w_list[-1]
        bwrs = alphas * w0 * w0 / winf
        a, b, c, d = 9.325e-01, 1.276e+00, 1.203e+00, 6.093e-01
        # a, b, c, d = 0, 0, 0, 6.093e-01
        hwrs = (
            c * bwrs**0.5 / (1 + a * w0**0.5)
            + d * bwrs**0.25 * w0**0.5 / (1 + b * w0**0.5)
        )
        hwrs *= (winf / w1)**4
        awrs = alphas * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        winf = w0 / (-w_list[2] * 2)**0.5
        return alphas * w0 / denom


class ExtractEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w0 / (-w_list[1] * 2)**0.5
        winf = w_list[-1]
        a, b, c, d = 4.627e-01, 1.378e+00, 7.555e-01, 4.852e-01
        bwrs = alphas * w0 * w0 / winf
        hwrs = c * bwrs**0.5 / (1 + a * w0) + d * bwrs**0.25 * w0**0.25 / (1 + b * w0**0.25)
        hwrs *= np.exp((winf / w1)**3 - 1)
        awrs = alphas * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        return alphas * w0 / denom


class _ScreenedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        bwrs = alphas * w0 * wr
        awrs = alphas * wr
        a, b = 8.5, 6.4
        return alphas * w0 / (1 + b * bwrs**0.5 + a * awrs + awrs**2)**0.5


class ScreenedWeightedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        a, b, c = self.params
        winfs = w0 / wr
        bwrs = alphas * alphas * w0 * wr
        awrs = alphas * wr
        wt = 1 + c * alphas**0.5 / winfs + a * alphas**0.5 / winfs**0.5
        return alphas * w0 * wt / (1 + b * bwrs**0.25 + awrs * wt)


class BalancedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        wr = w0 / w_list[-1]
        bwrs = alphas * wr * w0
        awrs = alphas * wr
        a, b, c = 0.5048, 2.156, 0.5358
        a, b, c = 0, 0, 0
        mix = (1 + a * bwrs)**-0.25
        denom = 1 + b * bwrs**0.5 * mix + (c * awrs)**0.5 * (1 - mix) + awrs
        return alphas * w0 / denom


class ScreenedEigNumInterpolator(_EigNumInterpolator):
    def interpolate(self, alphas, w_list):
        W0s = w_list[0]
        Winfs = w_list[-1]
        exxs = w_list[-3]
        ratio = w_list[-2]**0 * (Winfs / exxs)**2
        aow = alphas / Winfs
        term = W0s * aow
        a, b, c = 3.232e+00, 5.150e+00, 2.099e+00
        mix = W0s * aow / (b * aow**0.5 + c)
        mix = ratio / (ratio + mix)
        term += a * mix * W0s * aow**0.5
        denom = np.sqrt(1 + term * term)
        return alphas * W0s / denom


class _MatNumInterpolator(_NumACInterpolator):
    def get_ac_term(self, alpha):
        raise NotImplementedError
    
    def _cache_intermediates(self, w_list):
        self._clear_cache()
        w0 = 0.5 * (w_list[0] + w_list[0].T.conj())
        winf = w_list[-1]
        wr = np.linalg.solve(winf, w0)
        wr = 0.5 * (wr + wr.T.conj())
        self._cache["w0"] = w0
        self._cache["wr"] = wr
        self._cache["id"] = _identity_like(w0)
    
    def _clear_cache(self):
        self._cache = {}

    def __call__(self, w_list):
        self._cache_intermediates(w_list)
        energy = 0
        for alpha in self.alphas:
            energy -= self.dalpha * np.trace(
                self.get_ac_term(alpha)
            )
        self._clear_cache()
        return energy

class ScreenedMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        hwrs = alpha**0.5 * self._cache["hwca"] + alpha**0.25 * self._cache["hwdb"]
        hwrs += alpha * self._cache["aw"]
        hwrs = 0.5 * (hwrs + hwrs.T)
        denom = self._cache["id"] + _acmp_matrix_pow(hwrs, 2)
        denom = _acmp_matrix_pow(denom, 0.5)
        return alpha * np.linalg.solve(denom, self._cache["w0"]).real

    def interpolate(self, alphas, w_list):
        w0 = w_list[0]
        w1 = w0 / (-w_list[1] * 2)**0.5
        winf = w_list[-1]
        a, b, c, d = 4.627e-01, 1.378e+00, 7.555e-01, 4.852e-01
        bwrs = alphas * w0 * w0 / winf
        hwrs = c * bwrs**0.5 / (1 + a * w0) + d * bwrs**0.25 * w0**0.25 / (1 + b * w0**0.25)
        hwrs *= np.exp((winf / w1)**3 - 1)
        awrs = alphas * w0 / winf
        denom = (1 + (awrs + hwrs)**2)**0.5
        return alphas * w0 / denom

    def _cache_intermediates(self, w_list):
        self._clear_cache()
        a, b, c, d = 4.627e-01, 1.378e+00, 7.555e-01, 4.852e-01
        w0 = 0.5 * (w_list[0] + w_list[0].T)
        w1 = -1 * (w_list[1] + w_list[1].T)
        winf = 0.5 * (w_list[-1] + w_list[-1].T)
        w1 = _acmp_matrix_pow(w1, -0.5)
        w1 = w0.dot(w1)
        if w0.size == 1:
            print(w0, w1, winf)
        idm = _identity_like(w0)
        da = _acmp_matrix_pow(idm + a * w0, -1)
        db = _acmp_matrix_pow(idm + b * _acmp_matrix_pow(w0, 0.25), -1)
        wm1 = _acmp_matrix_pow(winf, -0.25)
        dc = c * w0.dot(wm1).dot(wm1)
        dd = d * _acmp_matrix_pow(w0, 0.75).dot(wm1)
        hw = _acmp_matrix_pow(winf, 3).dot(_acmp_matrix_pow(w1, -3))
        hw = _acmp_matrix_exp(0.5 * (hw + hw.T) - idm)
        aw = w0.dot(_acmp_matrix_pow(wm1, 4))
        self._cache["aw"] = 0.5 * (aw + aw.T)
        self._cache["id"] = idm
        self._cache["w0"] = w0
        tmp = hw.dot(dc).dot(da)
        self._cache["hwca"] = 0.5 * (tmp + tmp.T)
        tmp = hw.dot(dd).dot(db)
        self._cache["hwdb"] = 0.5 * (tmp + tmp.T)


class BasicMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        term = alpha * wr
        return alpha * np.linalg.solve(idmat + term, w0)


class SquareMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        term = idmat + _acmp_matrix_pow(alpha * wr, 2)
        term = _acmp_matrix_pow(term, 0.5)
        return alpha * np.linalg.solve(term, w0)


class WinfMatNumInterpolator(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        bwrs = w0.dot(wr)
        bwrs = 0.5 * alpha * (bwrs + bwrs.T)
        a = 1.67
        term = a * _acmp_matrix_pow(bwrs, 0.5) + alpha * wr
        return alpha * np.linalg.solve(idmat + term, w0)


class WinfMatNumInterpolator2(_MatNumInterpolator):
    def get_ac_term(self, alpha):
        w0 = self._cache["w0"]
        wr = self._cache["wr"]
        idmat = self._cache["id"]
        bwrs = _acmp_matrix_pow(w0, 0.5).dot(wr)
        bwrs = 0.5 * alpha * (bwrs + bwrs.T)
        a = 1.691
        b = 0.227
        term = a * _acmp_matrix_pow(bwrs, 0.25)
        term += b * _acmp_matrix_pow(bwrs, 0.50)
        term += alpha * wr
        return alpha * np.linalg.solve(idmat + term, w0)