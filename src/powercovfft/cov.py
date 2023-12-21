import numpy as np
from scipy.integrate import quad

from power_law_decomp import PowerLawDecomp
import utils


class PowerSpecCovFFT:

    def __init__(self):
        print('Loading master integrals...')
        self.get_master_int = utils.MasterIntegral()

        print('Loading coefficient functions...')
        self.set_coeff_func()

        print('Loading integral formulae...')
        self.set_integral_formulae()

        print('done.')

    def set_coeff_func(self, names=['T2211_1','T2211_2','T_SN_B_2','T_SN_P']):
        self.coeff_func = {}
        self.set_ab = set()
        for name in names:
            self.coeff_func[name] = utils.CovCoeff(name)
            self.set_ab = self.set_ab | self.coeff_func[name].set_ab

    def set_integral_formulae(self, names=['T3111','T_SN_B_1']):
        self.cov_integral = {}
        for name in names:
            self.cov_integral[name] = utils.CovIntegral(name)

    # for direct integration
    def set_integrand_formulae(self, names=['T2211_1','T2211_2','T_SN_B_2','T_SN_P']):
        self.cov_integrand = {}
        for name in names:
            self.cov_integrand[name] = utils.CovIntegrand(name)

    def set_power_law_decomp(self, config):
        nu = config['nu']
        kmin = config['kmin']
        kmax = config['kmax']
        nmax = config['nmax']
        self.decomp = PowerLawDecomp(nu, kmin, kmax, nmax)

    def set_pk_lin(self, get_pk_lin):
        self.get_pk_lin = get_pk_lin
        self.decomp.compute(self.get_pk_lin)

    def set_params(self, vol, ndens, fgrowth, bias):
        self.vol = vol
        self.ndens = ndens
        self.fgrowth = fgrowth
        self.bias = bias

    def calc_master_integral(self, k1, k2, z_switch=0.1):
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        eta_m = np.transpose(np.tile(self.decomp.eta_m, (len(k1),len(k2),1)), (2,0,1))
        k1_tile = np.tile(k1, (len(eta_m),1,1))
        k2_tile = np.tile(k2, (len(eta_m),1,1))

        self.master_int = {}
        for (a,b) in sorted(self.set_ab):
            a_m = 0.5 * (a - self.decomp.nu - eta_m * 1j)
            self.master_int[(a,b)] = self.get_master_int(a_m, b, k1_tile, k2_tile, z_switch=z_switch)

    def calc_base_integral(self):
        self.base_int = {}
        for (a,b) in sorted(self.set_ab):
            self.base_int[(a,b)] = np.tensordot(self.decomp.c_m, self.master_int[(a,b)], axes=([0],[0])).real

    def get_cov_T0_term(self, l1, l2, k1, k2, name):
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        terms = []
        for (a,b) in sorted(self.set_ab):
            if not (l1,l2,a,b) in self.coeff_func[name].expr.keys(): continue
            coeff = self.coeff_func[name](a, b, l1, l2, k1, k2, self.fgrowth, self.bias)
            term = coeff * self.base_int[(a,b)]
            terms.append(term)
        res = np.sum(terms, axis=0)

        if len(k1) == 1 or len(k2) == 1:
            res = np.ravel(res)
        if len(k1) == 1 and len(k2) == 1:
            res = res[0]
        return res

    def get_cov_T0_term_direct(self, l1, l2, k1, k2, name, epsrel=1e-8, limit=10000):
        integrand = lambda mu: self.cov_integrand[name](mu, l1, l2, k1, k2, self.fgrowth, self.bias) * self.get_pk_lin(np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu))
        res = quad(integrand, -1, 1, epsrel=epsrel, limit=limit)[0] / 2
        return res

    def get_cov_T2211(self, l1, l2, k1, k2):
        # T0 contribution from T2211 ("snake") term, whose integration is done using FFTLog-based method.
        term1 = 8 * self.get_pk_lin(k1)**2 * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_1')
        term2 = 8 * self.get_pk_lin(k2)**2 * self.get_cov_T0_term(l1, l2, k2, k1, name='T2211_1')
        term3 = 16 * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_2')
        return term1 + term2 + term3

    def get_cov_T3111(self, l1, l2, k1, k2):
        # T0 contribution from T3111 ("star") term, whose integration is done analytically.
        term1 = self.get_pk_lin(k1)**2 * self.get_pk_lin(k2) * self.cov_integral['T3111'](l1, l2, k1, k2, self.fgrowth, self.bias)
        term2 = self.get_pk_lin(k2)**2 * self.get_pk_lin(k1) * self.cov_integral['T3111'](l1, l2, k2, k1, self.fgrowth, self.bias)
        return term1 + term2

    def get_cov_T_SN_B(self, l1, l2, k1, k2):
        term1 = 8 * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.cov_integral['T_SN_B_1'](l1, l2, k1, k2, self.fgrowth, self.bias)
        term2 = 8 * self.get_pk_lin(k1) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_2')
        term3 = 8 * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k2, k1, name='T_SN_B_2')
        return (term1 + term2 + term3) / self.ndens

    def get_cov_T_SN_P(self, l1, l2, k1, k2):
        term = 2 * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_P')
        return term / self.ndens**2

    # main function to compute T0 part Eq. (7)
    def get_cov_T0(self, l1, l2, k1, k2):
        term_2211 = self.get_cov_T2211(l1, l2, k1, k2)
        term_3111 = self.get_cov_T3111(l1, l2, k1, k2)
        cov = (term_2211 + term_3111) / self.vol
        return cov

    # main function to compute shot-noise part Eq. (C4)
    def get_cov_T0_SN(self, l1, l2, k1, k2):
        term_B = self.get_cov_T_SN_B(l1, l2, k1, k2)
        term_P = self.get_cov_T_SN_P(l1, l2, k1, k2)
        cov = (term_B + term_P) / self.vol
        return cov

    def get_cov_T0_direct(self, l1, l2, k1, k2, epsrel=1e-8, limit=10000):
        term_2211 = quad(self.get_T2211_integrand, -1, 1, args=(l1,l2,k1,k2), epsrel=epsrel, limit=limit)[0] / 2
        term_3111 = self.get_cov_T3111(l1, l2, k1, k2)
        cov = (term_2211 + term_3111) / self.vol
        return cov

    def get_cov_T0_SN_direct(self, l1, l2, k1, k2, epsrel=1e-8, limit=10000):
        term_SN_B1 = 8 / self.ndens * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.cov_integral['T_SN_B_1'](l1, l2, k1, k2, self.fgrowth, self.bias)
        term_SN_B2_P = quad(self.get_T0_SN_integrand, -1, 1, args=(l1,l2,k1,k2), epsrel=epsrel, limit=limit)[0] / 2
        cov = (term_SN_B1 + term_SN_B2_P) / self.vol
        return cov

    def get_T2211_integrand(self, mu12, l1, l2, k1, k2):
        # T0 integrand from T2211 ("snake") terms
        term1 = 8 * self.get_pk_lin(k1)**2 * self.cov_integrand['T2211_1'](mu12, l1, l2, k1, k2, self.fgrowth, self.bias)
        term2 = 8 * self.get_pk_lin(k2)**2 * self.cov_integrand['T2211_1'](mu12, l1, l2, k2, k1, self.fgrowth, self.bias)
        term3 = 16 * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.cov_integrand['T2211_2'](mu12, l1, l2, k1, k2, self.fgrowth, self.bias)

        k12 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
        integrand = self.get_pk_lin(k12) * (term1 + term2 + term3)
        return integrand

    def get_T0_SN_integrand(self, mu12, l1, l2, k1, k2):
        # T0 integrand from shot-noise terms
        term1 = 8 / self.ndens * self.get_pk_lin(k1) * self.cov_integrand['T_SN_B_2'](mu12, l1, l2, k1, k2, self.fgrowth, self.bias)
        term2 = 8 / self.ndens * self.get_pk_lin(k2) * self.cov_integrand['T_SN_B_2'](mu12, l1, l2, k2, k1, self.fgrowth, self.bias)
        term3 = 2 / self.ndens**2 * self.cov_integrand['T_SN_P'](mu12, l1, l2, k1, k2, self.fgrowth, self.bias)

        k12 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
        integrand = self.get_pk_lin(k12) * (term1 + term2 + term3)
        return integrand
