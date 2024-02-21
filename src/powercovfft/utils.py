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

        b1 = var('b1')
        b2 = var('b2')
        bG2 = var('bG2')

        self.func = {key: lambdify([k1, k2, f, b1, b2, bG2], self.expr[key], modules='numpy') for key in self.expr.keys()}

    def __call__(self, a, b, l1, l2, k1, k2, f, bias):
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        b1 = bias['b1']
        b2 = bias['b2']
        bG2 = bias['bG2']

        return self.func[(l1, l2, a, b)](k1, k2, f, b1, b2, bG2)

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

        b1 = var('b1')
        b2 = var('b2')
        bG2 = var('bG2')
        b3 = var('b3')
        bG3 = var('bG3')
        bdG2 = var('bdG2')
        bGamma3 = var('bGamma3')

        # for k1 != k2
        self.func = {key: lambdify([r, logkpm, f, b1, b2, bG2, b3, bG3, bdG2, bGamma3], self.expr[key], modules='numpy') for key in self.expr.keys()}
        # for k1 == k2
        self.func_diag = {key: lambdify([f, b1, b2, bG2, b3, bG3, bdG2, bGamma3], self.expr_diag[key], modules='numpy') for key in self.expr.keys()}

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

        res1 = self.func[(l1, l2)](r, logkpm, f, b1, b2, bG2, b3, bG3, bdG2, bGamma3)
        res2 = self.func_diag[(l1, l2)](f, b1, b2, bG2, b3, bG3, bdG2, bGamma3) * np.ones(k1.shape)
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

        b1 = var('b1')
        b2 = var('b2')
        bG2 = var('bG2')

        self.func = {key: lambdify([mu12, k1, k2, f, b1, b2, bG2], self.expr[key], modules='numpy') for key in self.expr.keys()}

    def __call__(self, mu12, l1, l2, k1, k2, f, bias):
        mu12 = np.atleast_1d(mu12)
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        b1 = bias['b1']
        b2 = bias['b2']
        bG2 = bias['bG2']

        return self.func[(l1, l2)](mu12, k1, k2, f, b1, b2, bG2)

    def get_args_comb(self, fname):
        m = re.search(self.name, fname)
        info_text = fname[m.end()+1:]
        info = re.findall(r"\d+", info_text)
        l1 = int(info[0][0])
        l2 = int(info[0][1])
        return l1, l2
