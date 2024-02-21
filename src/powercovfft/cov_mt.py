import numpy as np
from scipy.integrate import quad

from .cov import PowerSpecCovFFT
from . import utils_mt

def dkron(a, b):
    return 1 if a == b else 0

class PowerSpecCovFFTMultiTracer(PowerSpecCovFFT):

    def set_coeff_func(self, names=['T2211_1','T2211_2','T_SN_B_2','T_SN_P']):
        self.coeff_func = {}
        self.set_ab = set()
        for name in names:
            self.coeff_func[name] = utils_mt.CovCoeff(name + '_mt')
            self.set_ab = self.set_ab | self.coeff_func[name].set_ab

    def set_integral_formulae(self, names=['T3111','T_SN_B_1']):
        self.cov_integral = {}
        for name in names:
            self.cov_integral[name] = utils_mt.CovIntegral(name + '_mt')

    # for direct integration
    def set_integrand_formulae(self, names=['T2211_1','T2211_2','T_SN_B_2','T_SN_P']):
        self.cov_integrand = {}
        for name in names:
            self.cov_integrand[name] = utils_mt.CovIntegrand(name + '_mt')

    def set_params(self, vol, fgrowth, tracer_name_list, bias_list, ndens_list):
        self.vol = vol
        self.fgrowth = fgrowth
        self.tracer_names = {'a': tracer_name_list[0], 'b': tracer_name_list[1], 'c': tracer_name_list[2], 'd': tracer_name_list[3]}
        self.bias = {'a': bias_list[0], 'b': bias_list[1], 'c': bias_list[2], 'd': bias_list[3]}
        self.ndens = {'a': ndens_list[0], 'b': ndens_list[1], 'c': ndens_list[2], 'd': ndens_list[3]}

    def _get_bias_dict(self, indices=['a','b','c','d']):
        bias_names = ['b1', 'b2', 'bG2', 'b3', 'bG3', 'bdG2', 'bGamma3']
        bias_dict = {bias_name: [self.bias[index][bias_name] for index in indices] for bias_name in bias_names}
        return bias_dict

    def get_cov_T0_term(self, l1, l2, k1, k2, name, indices=['a','b','c','d']):
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        bias_dict = self._get_bias_dict(indices)

        if name in self.coeff_func.keys():
            terms = []
            for (a,b) in sorted(self.set_ab):
                if not (l1,l2,a,b) in self.coeff_func[name].expr.keys(): continue
                coeff = self.coeff_func[name](a, b, l1, l2, k1, k2, self.fgrowth, bias_dict)
                term = coeff * self.base_int[(a,b)]
                terms.append(term)
            res = np.sum(terms, axis=0)

        elif name in self.cov_integral.keys():
            res = self.cov_integral[name](l1, l2, k1, k2, self.fgrowth, bias_dict)

        else:
            raise ValueError('The term name %s is invalid.' % (name))
        
        if len(k1) == 1 or len(k2) == 1:
            res = np.ravel(res)
        if len(k1) == 1 and len(k2) == 1:
            res = res[0]
        return res
    
    def get_cov_T0_integrand(self, mu12, l1, l2, k1, k2, name, indices=['a','b','c','d']):
        bias_dict = self._get_bias_dict(indices)
        res = self.cov_integrand[name](mu12, l1, l2, k1, k2, self.fgrowth, bias_dict)
        return res

    def get_cov_T0_term_direct(self, l1, l2, k1, k2, name, indices=['a','b','c','d'], epsrel=1e-8, limit=10000):
        bias_dict = self._get_bias_dict(indices)
        integrand = lambda mu: self.get_cov_T0_integrand(mu, l1, l2, k1, k2, name, indices) * self.get_pk_lin(np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu))
        res = quad(integrand, -1, 1, epsrel=epsrel, limit=limit)[0] / 2
        return res

    def get_cov_T2211(self, l1, l2, k1, k2):
        # T0 contribution from T2211 ("snake") term, whose integration is done using FFTLog-based method.

        term1 = 8 * self.get_pk_lin(k1)**2 * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_1', indices=['a','b','c','d'])
        term2 = 8 * self.get_pk_lin(k2)**2 * self.get_cov_T0_term(l1, l2, k2, k1, name='T2211_1', indices=['c','d','a','b'])
        
        factor = 4 * self.get_pk_lin(k1) * self.get_pk_lin(k2)
        term3 = factor * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_2', indices=['a','d','c','b'])
        term4 = factor * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_2', indices=['b','c','d','a'])
        term5 = factor * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_2', indices=['a','c','d','b'])
        term6 = factor * self.get_cov_T0_term(l1, l2, k1, k2, name='T2211_2', indices=['b','d','c','a'])

        term = term1 + term2 + term3 + term4 + term5 + term6
        return term

    def get_cov_T3111(self, l1, l2, k1, k2):
        # T0 contribution from T3111 ("star") term, whose integration is done analytically.

        term1 = 6 * self.get_pk_lin(k1)**2 * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T3111', indices=['a','b','c','d'])
        term2 = 6 * self.get_pk_lin(k1)**2 * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T3111', indices=['a','b','d','c'])
        term3 = 6 * self.get_pk_lin(k2)**2 * self.get_pk_lin(k1) * self.get_cov_T0_term(l1, l2, k2, k1, name='T3111', indices=['c','d','a','b'])
        term4 = 6 * self.get_pk_lin(k2)**2 * self.get_pk_lin(k1) * self.get_cov_T0_term(l1, l2, k2, k1, name='T3111', indices=['c','d','b','a'])
        
        term = term1 + term2 + term3 + term4
        return term

    def get_cov_T_SN_B(self, l1, l2, k1, k2):

        factor = dkron(self.tracer_names['a'], self.tracer_names['c']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['c']) / self.ndens['b']
        term1 = 2 * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_1', indices=['a','b','d','c'])
        term2 = 2 * self.get_pk_lin(k1) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','d','c'])
        term3 = 2 * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k2, k1, name='T_SN_B_2', indices=['a','b','d','c'])
        term_Babd = factor * (term1 + term2 + term3)

        factor = dkron(self.tracer_names['a'], self.tracer_names['d']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['d']) / self.ndens['b']
        term1 = 2 * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_1', indices=['a','b','c','d'])
        term2 = 2 * self.get_pk_lin(k1) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','c','d'])
        term3 = 2 * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k2, k1, name='T_SN_B_2', indices=['a','b','c','d'])
        term_Babc = factor * (term1 + term2 + term3)

        term = term_Babd + term_Babc
        return term

    def get_cov_T_SN_P(self, l1, l2, k1, k2):
        factor = dkron(self.tracer_names['a'], self.tracer_names['c']) * dkron(self.tracer_names['b'], self.tracer_names['d'])
        factor += dkron(self.tracer_names['a'], self.tracer_names['d']) * dkron(self.tracer_names['b'], self.tracer_names['c'])
        factor /= (self.ndens['a'] * self.ndens['b'])
        term = factor * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_P', indices=['a','b','c','d'])
        return term

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
        term_2211 = quad(self.get_T2211_integrand, -1, 1, args=(l1, l2, k1, k2), epsrel=epsrel, limit=limit)[0] / 2
        term_3111 = self.get_cov_T3111(l1, l2, k1, k2)
        cov = (term_2211 + term_3111) / self.vol
        return cov

    def get_cov_T0_SN_direct(self, l1, l2, k1, k2, epsrel=1e-8, limit=10000):
        factor = dkron(self.tracer_names['a'], self.tracer_names['c']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['c']) / self.ndens['b']
        term_Babd = factor * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_1', indices=['a','b','d','c'])
        
        factor = dkron(self.tracer_names['a'], self.tracer_names['d']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['d']) / self.ndens['b']
        term_Babc = factor * self.get_pk_lin(k1) * self.get_pk_lin(k2) * self.get_cov_T0_term(l1, l2, k1, k2, name='T_SN_B_1', indices=['a','b','c','d'])
        
        term_SN_B1 = term_Babd + term_Babc

        term_SN_B2_P = quad(self.get_T0_SN_integrand, -1, 1, args=(l1, l2, k1, k2), epsrel=epsrel, limit=limit)[0] / 2
        cov = (term_SN_B1 + term_SN_B2_P) / self.vol
        return cov

    def get_T2211_integrand(self, mu12, l1, l2, k1, k2):
        # T0 integrand from T2211 ("snake") terms

        term1 = 8 * self.get_pk_lin(k1)**2 * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T2211_1', indices=['a','b','c','d'])
        term2 = 8 * self.get_pk_lin(k2)**2 * self.get_cov_T0_integrand(mu12, l1, l2, k2, k1, name='T2211_1', indices=['c','d','a','b'])

        factor = 4 * self.get_pk_lin(k1) * self.get_pk_lin(k2)
        term3 = factor * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T2211_2', indices=['a','b','c','d'])
        term4 = factor * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T2211_2', indices=['b','c','d','a'])
        term5 = factor * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T2211_2', indices=['a','c','d','b'])
        term6 = factor * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T2211_2', indices=['b','d','c','a'])

        term = term1 + term2 + term3 + term4 + term5 + term6

        k12 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
        integrand = self.get_pk_lin(k12) * term
        return integrand

    def get_T0_SN_integrand(self, mu12, l1, l2, k1, k2):
        # T0 integrand from shot-noise terms

        # T_SN_B term
        factor = dkron(self.tracer_names['a'], self.tracer_names['c']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['c']) / self.ndens['b']
        term1 = self.get_pk_lin(k1) * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','d','c'])
        term2 = self.get_pk_lin(k2) * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','d','c'])
        term_Babd = factor * (term1 + term2)

        factor = dkron(self.tracer_names['a'], self.tracer_names['d']) / self.ndens['a'] + dkron(self.tracer_names['b'], self.tracer_names['d']) / self.ndens['b']
        term1 = self.get_pk_lin(k1) * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','c','d'])
        term2 = self.get_pk_lin(k2) * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T_SN_B_2', indices=['a','b','c','d'])
        term_Babc = factor * (term1 + term2)
        
        term_B2 = term_Babd + term_Babc

        # T_SN_P term
        factor = dkron(self.tracer_names['a'], self.tracer_names['c']) * dkron(self.tracer_names['b'], self.tracer_names['d'])
        factor += dkron(self.tracer_names['a'], self.tracer_names['d']) * dkron(self.tracer_names['b'], self.tracer_names['c'])
        factor /= (self.ndens['a'] * self.ndens['b'])
        term_P = factor * self.get_cov_T0_integrand(mu12, l1, l2, k1, k2, name='T_SN_P', indices=['a','b','c','d'])

        k12 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
        integrand = self.get_pk_lin(k12) * (term_B2 + term_P)
        return integrand
