import os
import numpy as np
from scipy.special import loggamma
from sympy.parsing.mathematica import parse_mathematica
from sympy import var, lambdify


class MasterIntegral:

    def __init__(self):
        dirname = os.path.dirname(__file__)
        
        fnames = [dirname+'/master_int/poly_b%s.txt' % (b) for b in range(13)]
        self.expr_poly = []
        for fname in fnames:
            with open(fname,'r') as file:
                expr = file.read()
            expr = parse_mathematica(expr)
            self.expr_poly.append(expr)
            
        fnames = [dirname+'/master_int/z0lim20_b%s.txt' % (b) for b in range(13)]
        self.expr_z0lim = []
        for fname in fnames:
            with open(fname,'r') as file:
                expr = file.read()
            expr = parse_mathematica(expr)
            self.expr_z0lim.append(expr)
            
        fnames = [dirname+'/master_int/z1_b%s.txt' % (b) for b in range(13)]
        self.expr_z1 = []
        for fname in fnames:
            with open(fname,'r') as file:
                expr = file.read()
            expr = parse_mathematica(expr)
            self.expr_z1.append(expr)
        
        z = var('z')
        a = var('a')
        
        self.poly = [lambdify([z,a], expr, modules='numpy') for expr in self.expr_poly]
        self.z0lim = [lambdify([z,a], expr, modules='numpy') for expr in self.expr_z0lim]
        self.z1 = [lambdify([a], expr, modules='numpy') for expr in self.expr_z1]

    def __call__(self, a, b, k1, k2, z_switch=0.1):
        a = np.atleast_1d(a)
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)

        z = 2 * k1 * k2 / (k1**2 + k2**2)

        fac = np.exp(loggamma(b+2) - (b+1) * np.log(z) - np.log(np.prod([a-n for n in range(1,b+2)], axis=0)))
        res = fac * (self.poly[b](-z, a) / (1+z)**a - self.poly[b](z, a) / (1 - np.where(z!=1., z, 0))**a)
        
        res_z1 = self.z1[b](a * np.ones_like(z))
        res = res * np.where(z != 1., 1, 0) + res_z1 * np.where(z == 1., 1, 0)
        
        res_z0 = self.z0lim[b](z, a)
        res = res * np.where(np.abs(z) > z_switch, 1, 0) + res_z0 * np.where(np.abs(z) <= z_switch, 1, 0)
        
        res = (k1**2 + k2**2)**(-a) * res / (2 * (b+1))
        return res
