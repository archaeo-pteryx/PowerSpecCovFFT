import numpy as np


class PowerLawDecomp:

    def __init__(self, nu, kmin, kmax, nmax):
        self.nu = nu
        self.kmin = kmin
        self.kmax = kmax
        self.nmax = nmax
        self.kn = np.geomspace(kmin, kmax, nmax)
        self.eta_m = 2 * np.pi / (nmax / (nmax-1) * np.log(kmax / kmin)) * (np.arange(nmax + 1) - nmax // 2)
        self.nu_m = self.nu + self.eta_m * 1j
        self.kn_tile = np.tile(self.kn, (len(self.nu_m), 1))
        self.nu_m_tile = np.tile(self.nu_m, (len(self.kn), 1)).T

    def compute(self, func, kwarg={}):
        fn_biased = func(self.kn, **kwarg) * (self.kn / self.kmin)**(-self.nu)
        c_m = np.fft.fft(fn_biased) / self.nmax
        c_m_sym = self.kmin**(-self.nu_m) * np.array([c_m[int(self.nmax // 2 - i)].conj() if i < self.nmax // 2 else c_m[int(i - self.nmax // 2)] for i in range(self.nmax + 1)])
        c_m_sym[0] = c_m_sym[0] / 2
        c_m_sym[-1] = c_m_sym[-1] / 2
        self.c_m = c_m_sym

        self.c_m_tile = np.tile(self.c_m, (len(self.kn), 1)).T
        self.func_q = self.c_m_tile * self.kn_tile**self.nu_m_tile
        self.func_rec = np.sum(self.func_q, axis=0)
