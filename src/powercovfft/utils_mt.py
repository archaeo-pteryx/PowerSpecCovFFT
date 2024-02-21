import os
import glob, re
import numpy as np
from sympy.parsing.mathematica import parse_mathematica
from sympy import var, lambdify


class CovCoeff:

    def __init__(self, name):
        self.name = name
        
        self.expr = {}
        self.set_ab = set()
        fnames = glob.glob(os.path.dirname(__file__)+'/coeff_func/%s_l*.txt' % (name))
        for fname in fnames:
            l1, l2, a, b = self.get_args_comb(fname)
            with open(fname,'r') as file:
                expr = file.read()
            self.expr[(l1,l2,a,b)] = parse_mathematica(expr)
            self.set_ab.add((a,b))

        k1 = var('k1')
        k2 = var('k2')
        f = var('f')

        bias_args = []
        for bias_name in ['b1', 'b2', 'bG2']:
            bias_args += [var('%s%s' % (bias_name, i)) for i in ['a','b','c','d']]

        self.func = {key: lambdify([k1, k2, f] + bias_args, self.expr[key], modules='numpy') for key in self.expr.keys()}

    def __call__(self, a, b, l1, l2, k1, k2, f, bias):
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        b1 = bias['b1']
        b2 = bias['b2']
        bG2 = bias['bG2']

        return self.func[(l1, l2, a, b)](k1, k2, f, 
                                         b1[0], b1[1], b1[2], b1[3],
                                         b2[0], b2[1], b2[2], b2[3],
                                         bG2[0], bG2[1], bG2[2], bG2[3]
                                        )

    def get_args_comb(self, fname):
        m = re.search(self.name, fname)
        info_text = fname[m.end()+1:]
        info = re.findall(r"\d+", info_text)
        l1 = int(info[0][0])
        l2 = int(info[0][1])
        a = int(info[1])
        b = int(info[2])
        return l1, l2, a, b


class CovIntegral:

    def __init__(self, name):
        self.name = name
        self.expr = {}
        self.expr_diag = {}
        
        fnames = glob.glob(os.path.dirname(__file__)+'/integrand/%s_l*.txt' % (name))
        for fname in fnames:
            l1, l2 = self.get_args_comb(fname)
            with open(fname,'r') as file:
                expr = file.read()
            self.expr[(l1,l2)] = parse_mathematica(expr)

        fnames = glob.glob(os.path.dirname(__file__)+'/integrand/%s_diag_l*.txt' % (name))
        for fname in fnames:
            l1, l2 = self.get_args_comb(fname)
            with open(fname,'r') as file:
                expr = file.read()
            self.expr_diag[(l1,l2)] = parse_mathematica(expr)

        r = var('r')
        logkpm = var('logkpm')
        f = var('f')

        bias_args = []
        for bias_name in ['b1', 'b2', 'bG2', 'b3', 'bG3', 'bdG2', 'bGamma3']:
            bias_args += [var('%s%s' % (bias_name, i)) for i in ['a','b','c','d']]

        # for k1 != k2
        self.func = {key: lambdify([r, logkpm, f] + bias_args, self.expr[key], modules='numpy') for key in self.expr.keys()}
        # for k1 == k2
        self.func_diag = {key: lambdify([f] + bias_args, self.expr_diag[key], modules='numpy') for key in self.expr.keys()}

    def __call__(self, l1, l2, k1, k2, f, bias):
        k1 = np.float128(np.atleast_1d(k1))
        k2 = np.float128(np.atleast_1d(k2))

        r = k2 / k1
        k2dummy = np.where(k2!=k1, k2, k1+0.1)
        logkpm = np.log((k1 + k2) / np.abs(k1 - k2dummy))

        b1 = np.float128(bias['b1'])
        b2 = np.float128(bias['b2'])
        bG2 = np.float128(bias['bG2'])
        b3 = np.float128(bias['b3'])
        bG3 = np.float128(bias['bG3'])
        bdG2 = np.float128(bias['bdG2'])
        bGamma3 = np.float128(bias['bGamma3'])

        res1 = self.func[(l1, l2)](r, logkpm, f, 
                                   b1[0], b1[1], b1[2], b1[3],
                                   b2[0], b2[1], b2[2], b2[3],
                                   bG2[0], bG2[1], bG2[2], bG2[3],
                                   b3[0], b3[1], b3[2], b3[3],
                                   bG3[0], bG3[1], bG3[2], bG3[3],
                                   bdG2[0], bdG2[1], bdG2[2], bdG2[3],
                                   bGamma3[0], bGamma3[1], bGamma3[2], bGamma3[3]
                                   )
        
        res2 = self.func_diag[(l1, l2)](f, 
                                        b1[0], b1[1], b1[2], b1[3],
                                        b2[0], b2[1], b2[2], b2[3],
                                        bG2[0], bG2[1], bG2[2], bG2[3],
                                        b3[0], b3[1], b3[2], b3[3],
                                        bG3[0], bG3[1], bG3[2], bG3[3],
                                        bdG2[0], bdG2[1], bdG2[2], bdG2[3],
                                        bGamma3[0], bGamma3[1], bGamma3[2], bGamma3[3]
                                        )
        res2 = res2 * np.ones(k1.shape)

        res = res1 * np.where(k2!=k1, 1, 0) + res2 * np.where(k2==k1, 1, 0)

        return res

    def get_args_comb(self, fname):
        m = re.search(self.name, fname)
        info_text = fname[m.end()+1:]
        info = re.findall(r"\d+", info_text)
        l1 = int(info[0][0])
        l2 = int(info[0][1])
        return l1, l2


class CovIntegrand:

    def __init__(self, name):
        self.name = name
        self.expr = {}
        self.expr_diag = {}
        
        fnames = glob.glob(os.path.dirname(__file__)+'/integrand/%s_l*.txt' % (name))
        for fname in fnames:
            l1, l2 = self.get_args_comb(fname)
            with open(fname,'r') as file:
                expr = file.read()
            self.expr[(l1,l2)] = parse_mathematica(expr)

        mu12 = var('mu')
        k1 = var('k1')
        k2 = var('k2')
        f = var('f')

        bias_args = []
        for bias_name in ['b1', 'b2', 'bG2']:
            bias_args += [var('%s%s' % (bias_name, i)) for i in ['a','b','c','d']]

        self.func = {key: lambdify([mu12, k1, k2, f] + bias_args, self.expr[key], modules='numpy') for key in self.expr.keys()}

    def __call__(self, mu12, l1, l2, k1, k2, f, bias):
        mu12 = np.atleast_1d(mu12)
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        b1 = bias['b1']
        b2 = bias['b2']
        bG2 = bias['bG2']

        return self.func[(l1, l2)](mu12, k1, k2, f, 
                                   b1[0], b1[1], b1[2], b1[3],
                                   b2[0], b2[1], b2[2], b2[3],
                                   bG2[0], bG2[1], bG2[2], bG2[3]
                                   )

    def get_args_comb(self, fname):
        m = re.search(self.name, fname)
        info_text = fname[m.end()+1:]
        info = re.findall(r"\d+", info_text)
        l1 = int(info[0][0])
        l2 = int(info[0][1])
        return l1, l2
