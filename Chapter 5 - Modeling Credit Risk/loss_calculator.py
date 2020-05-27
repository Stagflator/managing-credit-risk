# First import the necessary libraries for calculation
from scipy import integrate
import scipy.optimize as so
from math import pi, inf, sqrt, exp
import numpy as np
import pandas as pn
from statistics import median
import matplotlib.pyplot as plt
import scipy.stats
from openpyxl import load_workbook
import csv
import matplotlib.pyplot as plt

# Returns expected loss and unexpected loss under default and migration modes as a tuple
# Tuple order: Expected loss default, Expected loss migration, Unex. default, Unex migration
def exp_unex_loss_calculator(bond_rating, d, c, mrr, vrr, ne, dirty_price, ld, pd):
    default1 = pd * (dirty_price - mrr)
    default2 = pd * (dirty_price - mrr)**2
    deltay = []
    deltap = []
    delfactor = []
    delfactor2 = []
    for i in range(17):
        deltay.append( -(yspread[int(bond_rating)] - yspread[i]) / yspread[int(bond_rating)]/100)
    for i in range(17):
        deltap.append( dirty_price * d * deltay[i] - 0.5 * dirty_price * c * deltay[i]**2 )
    for i in range(17):
        delfactor.append(helper1[i] * deltap[i])
    for i in range(17):
        delfactor2.append(helper1[i] * deltap[i]**2)

    out1 = sum(delfactor) + default1
    out2 = sum(delfactor2) + default2

    expected_loss = ne * pd * (dirty_price - mrr)
    unexpected_loss = ne * sqrt(pd * vrr**2 + ld**2*pd*(1-pd))
    expected_loss_m = ne * out1
    unexpected_loss_m = ne * sqrt(pd * vrr**2 + out2 - out1**2)
    return int(expected_loss), int(expected_loss_m), int(unexpected_loss), int(unexpected_loss_m)




def upper_limit_finder(bond1_pd, bond2_pd):
    qu = bond1_pd
    qu2 = bond2_pd
    a = qu * sqrt(2 * pi)
    a2 = qu2 * sqrt(2 * pi)

    def integrand(u):
        return exp(-0.5*u**2)
    def func(x):
        return integrate.quad(integrand, -inf, x)[0] - a
    def func1(y):
        return integrate.quad(integrand, -inf, y)[0] - a2

    sol = so.fsolve(func, 0)
    sol2 = so.fsolve(func1, 0)
    return float(sol), float(sol2)



# Default correlation between two obligors
def default_correlation_calc(joint_prob_of_default, pd1, pd2):
    return (joint_prob_of_default - pd1 * pd2) / (sqrt(pd1 * (1-pd1) * pd2 * (1-pd2)))

# Loss correlation between two obligors
def loss_correlation_calc(ne1, ne2, pd1, pd2, ld1, ld2, rok, ul1, ul2):
    return (ne1 * ne2 * sqrt(pd1 * (1-pd1)) * sqrt(pd2 * (1-pd2)) * rok) / (ul1 * ul2)



# Finding the integral limits from z treshold tables
def limits_calculator(bond_rating1, bond_rating_2):
    low_limits = []
    up_limits = []
    x = list(range(0,18))
    for i in x:
        low_limits.append(ztresh[int(bond_rating1)][i])
        up_limits.append(ztresh[int(bond_rating_2)][i])
    return low_limits, up_limits

# Joint probability of default calculator for two obligor pairs
def joint_probability_of_def_calc(low_limit1, up_limit1, low_limit2, up_limit2, asset_return_corr):
    def integrand(x, y):
        return np.exp(-(x**2 - 2*asset_return_corr*x*y + y**2)/(2*abs(1 - asset_return_corr**2)))
    coefficient = 1 / (2*pi*sqrt(1 - asset_return_corr**2))
    re = integrate.dblquad(integrand, low_limit1, up_limit1, lambda x: low_limit2, lambda x: up_limit2)
    return re[0] * coefficient


