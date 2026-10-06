"""Mean-variance portfolio optimization, following Markowitz (1952).

Throughout, `w` is a vector of portfolio weights that sums to one, `mu` is the
vector of expected returns, and `Sigma` is the covariance matrix of returns.
Weights may be negative (short positions are allowed).

The notebook `01_markowitz.ipynb.py` derives everything you need. Your job in
HW 0 is to move that logic out of the notebook and into the two functions
marked TODO below, so that the tests in `test_port_opt.py` pass.
"""

import numpy as np


def portfolio_mean(w, mu):
    """Expected return of a portfolio. Linear in the weights."""
    return w @ mu


def portfolio_variance(w, Sigma):
    """Variance of a portfolio's return. Quadratic in the weights."""
    return w @ Sigma @ w


def find_global_minimum_variance_portfolio(Sigma):
    """Weights of the portfolio with the smallest possible variance.

    Solves: minimize w' Sigma w subject to sum(w) = 1.
    """
    n = Sigma.shape[0]
    ## TODO: YOUR CODE HERE. Replace the equal-weighted placeholder below.
    #wstar = np.ones(n) / n #placeholder
    ones = np.ones(n)
    wstar = np.linalg.solve(Sigma, ones)
    wstar = wstar / wstar.sum()

    return wstar


def find_tangency_portfolio(mu, Sigma, rf):
    """Weights of the portfolio of risky assets with the highest Sharpe ratio.

    Solves: maximize (w' mu - rf) / sqrt(w' Sigma w) subject to sum(w) = 1.
    """
    n = Sigma.shape[0]
    ## TODO: YOUR CODE HERE. Replace the equal-weighted placeholder below.
    #wstar = np.ones(n) / n # placeholder
    wstar = np.linalg.solve(Sigma, mu - rf)
    wstar = wstar / wstar.sum()
    return wstar
