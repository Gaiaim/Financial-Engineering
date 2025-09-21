import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
from scipy.optimize import root_scalar
from scipy.stats import gamma, norm, t
from scipy.special import comb

from utilities.ex1_utilities import get_discount_factor_by_zero_rates_linear_interp
from typing import List, Union 


def compute_KL_equity_alternative(I_values, Ku_e, Kd_e, nu_opt, recovery, rho_model, p, discounts, dates, notional, discount):

    # Compute upper tranche prices for each portfolio size I in I_values
    price_up_KL = np.array([
        Price_KL_tstud_fast(
            np.array([Ku_e]),       # Kd_vec for upper tranche
            np.array([1.0]),        # Ku_vec for upper tranche
            nu_opt,
            recovery,
            I,
            rho_model,
            p,
            discounts,
            dates
        )[0]  # Price_KL_tstud returns an array, take first element
        for I in I_values
    ])

    # Monetary value of upper tranche price
    price_up = price_up_KL * notional * (1 - Ku_e)

    # Expected loss of full portfolio
    expected_loss_p = (1 - recovery) * p

    # Present value of full portfolio
    price_ptf = discount * notional * (1 - expected_loss_p)

    # Equity tranche price (absolute)
    price_KL_equity = price_ptf - price_up

    # Convert to percentage of notional (equity tranche width)
    price_KL_equity_pct = price_KL_equity / (notional * (Ku_e - Kd_e))

    return price_KL_equity_pct, price_up_KL, price_ptf

def Price_HP_tstud(Kd_vec, Ku_vec, nu, recovery, I, rho_vec, p, discounts, dates):
    from Calibrations import calibration_K

    if isinstance(dates, (dt.date, dt.datetime)):
        dates = [dt.datetime.combine(dates, dt.datetime.min.time())]
    elif isinstance(dates, pd.DatetimeIndex):
        dates = [d.to_pydatetime() for d in dates]
    elif isinstance(dates, list) and isinstance(dates[0], dt.date):
        dates = [dt.datetime.combine(d, dt.datetime.min.time()) if isinstance(d, dt.date) else d for d in dates]

    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    n_tranches = len(Kd_vec)
    exact_sol = np.zeros(n_tranches)

    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts)

    # Calibrazione soglia k per t-copula dato p e primo rho
    def eq_k(k):
        return calibration_K(k, rho_vec[0], nu) - p

    sol = root_scalar(eq_k, bracket=[-5, 5], method='brentq')
    k_calibr = sol.root

    # Griglia y per integrazione
    y_grid = np.linspace(-6, 6, 300)
    phi_y = t.pdf(y_grid, nu)

    # m da 0 a I
    m_vec = np.arange(I + 1)
    binom_coeffs = comb(I, m_vec)

    for i in range(n_tranches):
        Kd, Ku, r = Kd_vec[i], Ku_vec[i], rho_vec[i]

        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # Probabilità condizionata P(y), shape (len(y_grid),)
        P_y = t.cdf((k_calibr - np.sqrt(r) * y_grid) / np.sqrt(1 - r), nu)
        P_y = np.clip(P_y, 1e-12, 1 - 1e-12)

        # Calcolo probabilità binomiale condizionata: shape (len(y_grid), I+1)
        prob_m_y = binom_coeffs * (P_y[:, None] ** m_vec) * ((1 - P_y[:, None]) ** (I - m_vec))

        # Funzione perdita tranche per ogni m
        L_m = np.minimum(np.maximum(m_vec / I - d, 0), u - d) / (u - d)

        # Calcolo integrand (shape (len(y_grid), I+1))
        integrand = phi_y[:, None] * prob_m_y

        # Integro su y
        integral_y = np.trapz(integrand, y_grid, axis=0)

        # Somma su m
        tot_sum = np.sum(L_m * integral_y)

        exact_sol[i] = discount * (1 - tot_sum)

    return exact_sol

def Price_HP_vasicek(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates):
    """
    Prices MBS tranches under Gaussian copula (Vasicek model) using binomial expansion
    """
    if isinstance(dates, (dt.date, dt.datetime)):
        dates = [dt.datetime.combine(dates, dt.datetime.min.time())]
    elif isinstance(dates, pd.DatetimeIndex):
        dates = [d.to_pydatetime() for d in dates]
    elif isinstance(dates, list) and isinstance(dates[0], dt.date):
        dates = [dt.datetime.combine(d, dt.datetime.min.time()) if isinstance(d, dt.date) else d for d in dates]


    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    n_tranches = len(Kd_vec)
    exact_sol = np.zeros(n_tranches)

    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts)

    # Griglia per integrazione numerica su y
    y_grid = np.linspace(-6, 6, 300)
    phi_y = norm.pdf(y_grid)

    # Precalcolo coeff binomiali (m = 0,...,I)
    m_vec = np.arange(I + 1)
    binom_coeffs = comb(I, m_vec)

    # Quantile di default
    norm_ppf_p = norm.ppf(p)

    for i in range(n_tranches):
        Kd, Ku, rho = Kd_vec[i], Ku_vec[i], rho_vec[i]
        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # Probabilità condizionata P(y) shape: (len(y_grid),)
        P_y = norm.cdf((norm_ppf_p - np.sqrt(rho) * y_grid) / np.sqrt(1 - rho))
        P_y = np.clip(P_y, 1e-12, 1 - 1e-12)

        # Calcola la probabilità binomiale condizionata su y
        # Shape: (len(y_grid), I+1)
        prob_m_y = binom_coeffs * (P_y[:, None] ** m_vec) * ((1 - P_y[:, None]) ** (I - m_vec))

        # Loss tranche L(m/I)
        L_m = np.minimum(np.maximum(m_vec / I - d, 0), u - d) / (u - d)

        # Integrando su y per ogni m
        integrand = phi_y[:, None] * prob_m_y  # shape (len(y_grid), I+1)
        integral_y = np.trapz(integrand, y_grid, axis=0)  # shape (I+1,)

        # Somma su m
        tot_sum = np.sum(L_m * integral_y)

        exact_sol[i] = discount * (1 - tot_sum)

    return exact_sol

def Price_KL_vasicek_fast(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates):
    # Preprocessing delle date
    if isinstance(dates[0], datetime) is False:
        dates = [datetime.combine(d, datetime.min.time()) for d in dates]

    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    n_tranches = len(Kd_vec)
    approx_sol_all = np.zeros(n_tranches)

    # Discount factor
    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts)

    # Griglie
    y_grid = np.linspace(-6, 6, 200)
    z_grid = np.linspace(1e-4, 1 - 1e-4, 300)
    phi_y = norm.pdf(y_grid)
    norm_ppf_p = norm.ppf(p)

    for idx in range(n_tranches):
        Kd, Ku, rho = Kd_vec[idx], Ku_vec[idx], rho_vec[idx]
        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # L(z)
        Lz = np.minimum(np.maximum(z_grid - d, 0), u - d) / (u - d)

        # P(y)
        P_y = norm.cdf((norm_ppf_p - np.sqrt(rho) * y_grid[:, None]) / np.sqrt(1 - rho))
        P_y = np.clip(P_y, 1e-10, 1 - 1e-10)

        # K(z, P(y))
        Z, P = np.meshgrid(z_grid, P_y)
        Z = np.clip(Z, 1e-10, 1 - 1e-10)
        KzPy = Z * np.log(Z / P) + (1 - Z) * np.log((1 - Z) / (1 - P))

        # C1(z)
        C1_z = np.sqrt(I / (2 * np.pi * z_grid * (1 - z_grid)))

        # Denominatore D(y)
        D_y = np.trapz(C1_z[:, None] * np.exp(-I * KzPy.T), z_grid, axis=0)
        D_y = np.maximum(D_y, 1e-12)  # per evitare overflow

        # Numeratore
        numerator = Lz[:, None] * (C1_z[:, None] / D_y) * np.exp(-I * KzPy.T)

        # Integra su z
        inner_integral = np.trapz(numerator, z_grid, axis=0)

        # Integra su y
        expected_loss = np.trapz(phi_y * inner_integral, y_grid)

        # Prezzo finale
        approx_sol_all[idx] = discount * (1 - expected_loss)

    return approx_sol_all

def Price_KL_tstud_fast(Kd_vec, Ku_vec, nu, recovery, I, rho_vec, p, discounts, dates):
    from Calibrations import calibration_K

    if isinstance(dates[0], datetime) is False:
        dates = [datetime.combine(d, datetime.min.time()) for d in dates]

    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    n_tranches = len(Kd_vec)
    approx_sol_all = np.zeros(n_tranches)

    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts)

    # Griglie per integrazione
    y_grid = np.linspace(-10, 10, 200)   # supporto ampio per t-student
    z_grid = np.linspace(1e-4, 1 - 1e-4, 300)
    phi_y = t.pdf(y_grid, df=nu)

    # calibrazione k per il primo rho (assumendo stesso p e nu)
    def eq_k(k):
        return calibration_K(k, rho_vec[0], nu) - p
    sol = root_scalar(eq_k, bracket=[-5, 5], method='brentq')
    k_calibr = sol.root

    for idx in range(n_tranches):
        Kd, Ku, rho = Kd_vec[idx], Ku_vec[idx], rho_vec[idx]

        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # funzione loss tranche (vectorizzata)
        Lz = np.minimum(np.maximum(z_grid - d, 0), u - d) / (u - d)

        # calcolo P(y)
        # P(y) = tCDF((k_calibr - sqrt(rho)*y) / sqrt(1-rho), nu)
        P_y = t.cdf((k_calibr - np.sqrt(rho) * y_grid[:, None]) / np.sqrt(1 - rho), df=nu)
        P_y = np.clip(P_y, 1e-10, 1 - 1e-10)  # evitare log(0)

        # meshgrid per z e P(y)
        Z, P = np.meshgrid(z_grid, P_y)

        # KL divergence K(z, P)
        KzPy = Z * np.log(Z / P) + (1 - Z) * np.log((1 - Z) / (1 - P))

        # C1(z)
        C1_z = np.sqrt(I / (2 * np.pi * z_grid * (1 - z_grid)))

        # Denominatore D(y)
        # dimensioni: z_grid x y_grid, espandiamo per broadcasting
        # attenzione all'orientamento: dobbiamo moltiplicare C1(z) * exp(-I*K(z,P)) lungo z e poi integrare
        exp_term = np.exp(-I * KzPy.T)  # trasposto per allineare dimensioni
        D_y = np.trapz(C1_z[:, None] * exp_term, z_grid, axis=0)
        D_y = np.maximum(D_y, 1e-12)

        # Numeratore
        numerator = Lz[:, None] * (C1_z[:, None] / D_y) * exp_term

        # Integrazione interna su z
        inner_integral = np.trapz(numerator, z_grid, axis=0)

        # Integrazione esterna su y
        expected_loss = np.trapz(phi_y * inner_integral, y_grid)

        approx_sol_all[idx] = discount * (1 - expected_loss)

    return approx_sol_all

def Price_LHP_tstud(nu, Kd_vec, Ku_vec, recovery, rho_vec, p, discounts_values, dates):
    from Calibrations import calibration_K

    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)

    if isinstance(dates, (dt.date, dt.datetime)):
        dates = [dt.datetime.combine(dates, dt.datetime.min.time())]
    elif isinstance(dates, pd.DatetimeIndex):
        dates = [d.to_pydatetime() for d in dates]
    elif isinstance(dates, list) and isinstance(dates[0], dt.date):
        dates = [dt.datetime.combine(d, dt.datetime.min.time()) if isinstance(d, dt.date) else d for d in dates]

    n_tranches = len(Kd_vec)
    ETL = np.zeros(n_tranches)
    Tstud_sol = np.zeros(n_tranches)

    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts_values)

    def eq_k(k):
        return calibration_K(k, rho_vec[0], nu) - p

    sol = root_scalar(eq_k, bracket=[-5, 5], method='brentq')
    k_calibr = sol.root

    # Griglia z da 0 a 1 per integrazione numerica
    z_grid = np.linspace(0, 1, 1000)[1:-1]  # esclusi 0 e 1 per evitare inf/NaN

    for i in range(n_tranches):
        Kd = Kd_vec[i]
        Ku = Ku_vec[i]
        r = rho_vec[i]

        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # Funzione perdita tranche L(z), vettorializzata
        def L(z):
            return np.minimum(np.maximum(z - d, 0), u - d) / (u - d)

        # inv_P(z) vettorializzato
        t_ppf = t.ppf(z_grid, nu)
        inv_P_z = (k_calibr - np.sqrt(1 - r) * t_ppf) / np.sqrt(r)

        # der_P(z) vettorializzato
        pdf_t_ppf = t.pdf(t_ppf, nu)
        der_P_z = np.abs(-1 / pdf_t_ppf * np.sqrt((1 - r) / r))

        # integrand vettoriale
        integrand = t.pdf(inv_P_z, nu) * der_P_z
        integrand1 = L(z_grid) * integrand

        # integrazione numerica su z_grid
        ETL[i] = np.trapz(integrand1, z_grid)
        Tstud_sol[i] = discount * (1 - ETL[i])

    return Tstud_sol

                

def Price_LHP_vasicek(
    Kd_vec: np.ndarray,
    Ku_vec: np.ndarray,
    recovery: float,
    rho_vec: np.ndarray,
    p: float,
    discounts_values: List[float],
    dates: Union[List[dt.datetime], pd.DatetimeIndex, dt.date]
) -> np.ndarray:
    # Normalizza dates in lista di datetime.datetime
    if isinstance(dates, (dt.date, dt.datetime)):
        dates = [dt.datetime.combine(dates, dt.datetime.min.time())]
    elif isinstance(dates, pd.DatetimeIndex):
        dates = [d.to_pydatetime() for d in dates]
    elif isinstance(dates, list) and isinstance(dates[0], dt.date):
        dates = [dt.datetime.combine(d, dt.datetime.min.time()) if isinstance(d, dt.date) else d for d in dates]

    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    n_tranches = len(Kd_vec)

    ETL_market = np.zeros(n_tranches)
    LHP_sol = np.zeros(n_tranches)

    maturity = datetime(2027, 2, 2)
    today = datetime(2023, 2, 2)
    discount = get_discount_factor_by_zero_rates_linear_interp(today, maturity, dates, discounts_values)

    # Griglia z per integrazione numerica esclusi 0 e 1 per evitare problemi numerici
    z_grid = np.linspace(0.001, 0.999, 1000)

    for i in range(n_tranches):
        Kd = Kd_vec[i]
        Ku = Ku_vec[i]
        rho = rho_vec[i]

        d = Kd / (1 - recovery)
        u = Ku / (1 - recovery)

        # Funzione perdita tranche L(z) vettorializzata
        def L(z):
            return np.minimum(np.maximum(z - d, 0), u - d) / (u - d)

        eps = 1e-10
        z_clipped = np.clip(z_grid, eps, 1 - eps)

        # inv_P(z) vettoriale
        inv_P_z = (norm.ppf(p) - np.sqrt(1 - rho) * norm.ppf(z_clipped)) / np.sqrt(rho)

        # der_P(z) vettoriale
        der_P_z = (np.sqrt(1 - rho) / np.sqrt(rho)) / norm.pdf(norm.ppf(z_clipped))

        # integrand vettoriale
        integrand1 = L(z_grid) * norm.pdf(inv_P_z) * der_P_z


        # integrazione numerica su z_grid
        ETL_market[i] = np.trapz(integrand1, z_grid)


        LHP_sol[i] = discount * (1 - ETL_market[i])

    return LHP_sol


def tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd, Ku, I, recovery, theta):
    
    np.random.seed(0)
    q = 1 - p

    # Simulate systemic factor Y ~ Gamma(1/theta, scale=1)
    y = gamma.rvs(1/theta, scale=1, size=Nsim)

    # Simulate idiosyncratic uniforms
    epsilon = np.random.rand(Nsim, I)

    # Inverse generator for Clayton copula
    s = -np.log(epsilon) / y[:, None]
    u = (1 + s) ** (-1/theta)

    # Count defaults per simulation
    n_defaults = np.sum(u > q, axis=1)

    Loss_ptf = (1 - recovery) * (n_defaults / I)

    portf_loss = np.clip(Loss_ptf - Kd, 0, Ku - Kd)
    Tranche_loss = portf_loss / (Ku - Kd)

    price = discount * (1 - np.mean(Tranche_loss))

    # 95% confidence interval

    stderr = np.std(Tranche_loss, ddof=1) / np.sqrt(Nsim)

    z = norm.ppf(0.975)
    IC = discount * (1 - np.mean(Tranche_loss) + np.array([-1, 1]) * z * stderr)

    return price, IC




def tranchePriceMC_GaussianCopula(Nsim, discount, rho, p, Kd, Ku, I, recovery):
    np.random.seed(0)

    if rho <= 0.0 or rho >= 1.0 or np.isnan(rho):
        return np.nan, (np.nan, np.nan)

    rho = np.clip(rho, 1e-6, 1 - 1e-6)
    k = norm.ppf(p)

    M_vec = np.random.randn(Nsim)             # fattore sistemico
    zi_mat = np.random.randn(I, Nsim)         # fattori idiosincratici
    xi_mat = np.sqrt(rho) * M_vec[np.newaxis, :] + np.sqrt(1 - rho) * zi_mat

    n_def = np.sum(xi_mat <= k, axis=0)

    Loss_ptf = (1 - recovery) * (n_def / I)

    portf_loss = np.clip(Loss_ptf - Kd, 0, Ku - Kd)
    Tranche_loss = portf_loss / (Ku - Kd)

    price = discount * (1 - np.mean(Tranche_loss))

    std_err = np.std(Tranche_loss) / np.sqrt(Nsim)
    z = norm.ppf(0.975)

    IC = discount * (np.array([1, 1]) - np.mean(Tranche_loss) + np.array([-1, 1]) * z * std_err)


    return price, IC




def tranchePriceMC_tCopula(Nsim, discount, rho, p, Kd, Ku, I, recovery, nu):

    np.random.seed(0)  # reproducibility


    if rho <= 0.0 or rho >= 1.0 or np.isnan(rho):
        return np.nan, (np.nan, np.nan)

    rho = np.clip(rho, 1e-6, 1 - 1e-6)

    # Calibrate threshold k such that P(T <= k) = p for Student-t
    sol = root_scalar(lambda kval: t.cdf(kval, nu) - p, bracket=[-10, 10])
    k = sol.root

    # Generate systemic and idiosyncratic t-distributed risk factors
    M = t.rvs(nu, size=Nsim)        # systemic factor
    Z = t.rvs(nu, size=(I, Nsim))   # idiosyncratic factors

    # Latent variables via t-copula
    X = np.sqrt(rho) * M + np.sqrt(1 - rho) * Z

    n_def = np.sum(X <= k, axis=0)

    # Adjust tranche boundaries for recovery
    u = Ku / (1 - recovery)
    d = Kd / (1 - recovery)

    # Calculate normalized tranche loss
    loss_frac = n_def / I
    tranche_loss = np.clip(loss_frac - d, 0, u - d) / (u - d)

    # Discounted expected tranche price
    price = discount * (1 - np.mean(tranche_loss))

    # 95% confidence interval
    std_err = np.std(tranche_loss) / np.sqrt(Nsim)
    z = norm.ppf(0.975)
    IC = discount * (1 - np.mean(tranche_loss) + np.array([-1, 1]) * z * std_err)

    return price, IC

