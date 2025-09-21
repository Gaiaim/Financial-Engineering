import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import fsolve, root_scalar, fminbound, brentq
from scipy.stats import t

from scipy.integrate import quad 

from typing import List, Tuple, Optional, Dict, Any, Union 

from Pricing import (Price_LHP_vasicek, Price_LHP_tstud, tranchePriceMC_GaussianCopula, tranchePriceMC_tCopula, tranchePriceMC_ArchimedeanCopula, Price_HP_vasicek, Price_KL_vasicek_fast, Price_KL_tstud_fast)


def calibration_K(k, rho, nu):
   
    k = np.atleast_1d(k)
    rho = np.atleast_1d(rho)
    integral = np.zeros((len(k), len(rho)))

    for i, k_i in enumerate(k):
        for j, rho_j in enumerate(rho):
            denom = np.sqrt(1 - rho_j)

            def integrand(y):
                arg = (k_i - np.sqrt(rho_j) * y) / denom
                return t.cdf(arg, nu) * t.pdf(y, nu)

            integral[i, j], _ = quad(integrand, -6, 6)

    return integral




def calibration_model_parameters(
    model_name: str,
    params: Dict[str, Any],
    Ku: float,
    recovery: float,
    rho: float,
    p: float,
    dates: Union[List[dt.datetime], pd.DatetimeIndex, dt.date],
    discounts: List[float]
) -> Tuple[
    Any,          # param_opt: float o array di float
    Optional[Any],  # output_extra1: float o array di float (es. nu_opt)
    Optional[Any],  # output_extra2: float o array di float (es. MSE o nu2)
    Optional[Any]   # output_extra3: float (es. mse_opt) o None
]:

    # 1) prezzo mercato equity tranche (Vasicek / Gaussian Copula)
    Kd = np.atleast_1d(0)
    Ku = np.atleast_1d(Ku)
    rho = np.atleast_1d(rho)
    price_equity_mkt = Price_LHP_vasicek(Kd, Ku, recovery, rho, p, discounts, dates)

    # per root-finding iniziale
    bracket = (-0.5, 0.5)  # o [0,1] se rho must be in [0,1]

    # helper per ottimizzazioni: obj_nu, obj_nu_diff, obj_nu_KL, ...
    # devono restituire (mse, rho_vec) o mse, a seconda del modello

    if model_name == 'double_t':
        if params['flag_nu']:
            # ------ calibrazione su nu ------
            Ku_vec   = params['Ku_vec']
            rho_vec  = params['rho_vec']
            Kd_vec  = np.zeros_like(Ku_vec) 

            def mse_vs_nu(nu: float) -> float:
                mse, _ = obj_nu(
                    nu, Kd_vec,
                    Ku_vec=Ku_vec, p=p, recovery=recovery,
                    rho_vec=rho_vec, dates=dates, discounts=discounts
                )
                return mse

            nu_opt = fminbound(
                mse_vs_nu,
                2.0,
                100.0
            )
            # ---- Calcolo finale ----
            mse_opt, rho_opt = obj_nu(
                nu_opt,
                Kd_vec=Kd_vec,
                Ku_vec=Ku_vec,
                p=p,
                recovery=recovery,
                rho_vec=rho_vec,
                dates=dates,
                discounts=discounts
            )

            # ---- Plot: MSE vs ν ----
            nu_vals = np.linspace(2, 100, 50)
            mse_vals = [mse_vs_nu(nu) for nu in nu_vals]

            plt.figure(figsize=(8, 5))
            plt.plot(nu_vals, mse_vals, '-o', color='blue', linewidth=1.5)
            plt.xlabel('Degrees of freedom ν')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.title('Calibration Error vs ν',  fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # ---- Output stile MATLAB ----
            print("\n>> Calibration:")
            print(f" ν optimal (degrees of freedom) = {nu_opt:.4f}")
            print(f" Mean Squared Error = {mse_opt:.6f}")

            return rho_opt, nu_opt, mse_opt, None

        else:
            # nu fisso, calibro rho con fzero
            nu = params['nu']
            def f_rho(r):
                return (
                    Price_LHP_tstud(nu, 0, Ku, recovery, r, p, dates, discounts) - price_equity_mkt)
            sol = root_scalar(f_rho, bracket=bracket, method='bisect')
            return sol.root, None, None, None


    elif model_name == 'KL':
        # ------ calibrazione su nu con approssimazione KL ------
        Ku_vec  = params['Ku_vec']
        rho_vec = params['rho_vec']
        I       = params['I']
        Kd_allzeros = np.zeros_like(Ku_vec)

        # Objective function for MSE vs nu
        def mse_vs_nu(nu):
            mse, _ = obj_nu_KL(
                nu,
                Kd_vec=Kd_allzeros,
                Ku_vec=Ku_vec,
                p=p,
                recovery=recovery,
                rho_vec=rho_vec,
                dates_list=dates,
                discounts=discounts,
                I=I
            )
            return mse

        # Minimize MSE with respect to nu
        nu_opt = fminbound(mse_vs_nu, 2.0, 100.0, disp=0)
        mse_opt, rho_model = obj_nu_KL(
            nu_opt,
            Kd_vec=Kd_allzeros,
            Ku_vec=Ku_vec,
            p=p,
            recovery=recovery,
            rho_vec=rho_vec,
            dates_list=dates,
            discounts=discounts,
            I=I
        )

        # ---- Plot: MSE vs nu ----
        nu_vals_KL = np.linspace(2.0, 100.0, 50)
        mse_vals_KL = [mse_vs_nu(nu_val) for nu_val in nu_vals_KL]

        plt.figure(figsize=(8, 5))
        plt.plot(nu_vals_KL, mse_vals_KL, '-o', linewidth=1.5, color='b')
        plt.xlabel('Degrees of Freedom ν')
        plt.ylabel('MSE')
        plt.title('MSE vs ν (KL Approximation)', fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ---- Output summary ----
        print("\n>> Calibration:")
        print(f" ν optimal (degrees of freedom) = {nu_opt:.4f}")
        print(f" Mean Squared Error = {mse_opt:.6f}")

        return rho_model, nu_opt, mse_opt, None



    elif model_name == 'gaussian_copula':
        Nsim     = params['Nsim']
        I        = params['I']
        discount = params['discount']

        def f_rho(r):
          
            price, _ = tranchePriceMC_GaussianCopula(Nsim, discount, r, p, 0.0, Ku, I, recovery)
            return price - price_equity_mkt

        def f_rho_plot(r):
  
            return tranchePriceMC_GaussianCopula(Nsim, discount, r, p, 0.0, Ku, I, recovery)


        
        bracket2 = (0.0001, 0.9999)

        sol = root_scalar(f_rho, bracket=bracket2, method='brentq')
        rho_opt = sol.root

        # Plot model price vs rho
        rho_range = np.linspace(0.0001, 0.9999, 50)

        values = [f_rho_plot(r)[0] for r in rho_range]  # prendi solo il prezzo

        plt.figure(figsize=(8, 5))
        plt.plot(rho_range, values, 'm-', linewidth=2.4, label='Model Price')
        plt.axhline(price_equity_mkt, color='b', linestyle='-', linewidth=2.4, label='Market Price')
        plt.plot(rho_opt, f_rho_plot(rho_opt)[0], 'ks', markersize=12, markerfacecolor='k', label='Optimal ρ')

        plt.xlabel('ρ')
        plt.ylabel('Equity Tranche Price')
        plt.title('Gaussian Copula Calibration',  fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return rho_opt, None, None, None


    elif model_name == 't_student_copula':
        Nsim     = params['Nsim']
        I        = params['I']
        discount = params['discount']
        nu       = params['nu']

        def f_rho(r):
            price, _ = tranchePriceMC_tCopula(Nsim, discount, r, p, 0.0, Ku, I, recovery,nu)
            return price - price_equity_mkt

        def f_rho_plot(r):
            return tranchePriceMC_tCopula(Nsim, discount, r, p, 0.0, Ku, I, recovery, nu)
        
        bracket2 = (0.0001, 0.9999)
        sol = root_scalar(f_rho, bracket=bracket2, method='brentq')
        rho_opt = sol.root

        # Plot model price vs rho
        rho_range = np.linspace(0.0001, 0.9999, 50)
        values = [f_rho_plot(r)[0] for r in rho_range]  # prendi solo il prezzo

        plt.figure(figsize=(8, 5))
        plt.plot(rho_range, values, 'm-', linewidth=2.4, label='Model Price')
        plt.axhline(price_equity_mkt, color='b', linestyle='-', linewidth=2.4, label='Market Price')
        plt.plot(rho_opt, f_rho_plot(rho_opt)[0], 'ks', markersize=12, markerfacecolor='k', label='Optimal ρ')
        plt.xlabel('ρ')
        plt.ylabel('Equity Tranche Price')
        plt.title('t-Student Copula Calibration', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return rho_opt, None, None, None


    elif model_name == 'archimedean_copula_clayton':
        Nsim     = params['Nsim']
        I        = params['I']
        discount = params['discount']

        def f_theta(theta):
            price,_ = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, 0.0, Ku, I, recovery, theta)
            return price - price_equity_mkt
        
        def f_theta_plot(theta):
            return tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, 0.0, Ku, I, recovery, theta)
        
        bracket2 = (0.0001, 1.9999)

        sol = root_scalar(f_theta, bracket = bracket2, method='brentq')
        theta_opt = sol.root

        # Plot model price vs theta
        theta_range = np.linspace(0.001, 2.0, 50)
        values = [f_theta_plot(r)[0] for r in theta_range] 

        plt.figure(figsize=(8, 5))
        plt.plot(theta_range, values, 'm-', linewidth=2.4, label='Model Price')
        plt.axhline(price_equity_mkt, color='b', linestyle='-', linewidth=2.4, label='Market Price')
        plt.plot(theta_opt, f_theta_plot(theta_opt)[0], 'ks', markersize=12, markerfacecolor='k', label='Optimal θ')
        plt.xlabel('θ')
        plt.ylabel('Equity Tranche Price')
        plt.title('Clayton Copula Calibration',  fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return theta_opt, None, None, None


    else:
        raise ValueError(f"Unknown model: {model_name}")







def find_rho_implied(Kd_vec, Ku_vec, recovery, I, rho_vec, p, dates, discounts, price_model, flag):
   
    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)

    
    price_model = np.atleast_1d(price_model)
    discounts = np.atleast_1d(discounts)
    dates = np.array(dates)

    if isinstance(dates, (dt.date, dt.datetime)):
        dates = [dt.datetime.combine(dates, dt.datetime.min.time())]
    elif isinstance(dates, pd.DatetimeIndex):
        dates = [d.to_pydatetime() for d in dates]
    elif isinstance(dates, list) and isinstance(dates[0], dt.date):
        dates = [dt.datetime.combine(d, dt.datetime.min.time()) if isinstance(d, dt.date) else d for d in dates]


    n_tranches = len(Kd_vec)
    
    rho_implied = np.zeros(n_tranches)
  
    # Choose pricing function based on flag
    for i in range(n_tranches):

        if flag == "LHP":
            def fun(r):
                return Price_LHP_vasicek(Kd_vec[i], Ku_vec[i], recovery, r, p, discounts, dates) - price_model[i]
        elif flag == "HP":
            def fun(r):
                return Price_HP_vasicek(Kd_vec[i], Ku_vec[i], recovery, I, r, p, discounts, dates) - price_model[i]

        elif flag == "KL":
            def fun(r):
                return Price_KL_vasicek_fast(Kd_vec[i], Ku_vec[i], recovery, I, r, p, discounts, dates) - price_model[i]

        else:
            raise ValueError('Invalid flag. Use "LHP", "HP", or "KL".')

        # Solve for implied rho, starting from rho_vec[i]
        rho_implied[i] = brentq(fun, 0.01, 0.99)

            
                
    # Plot comparison of market vs implied correlations
    x_labels = ['0-3', '0-6', '0-9', '0-12', '0-22']
    x = np.arange(1, len(x_labels) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(x, rho_vec, 'o-', color='blue', linewidth=2.4, markersize=5, markerfacecolor='blue', label='Market ρ')
    plt.plot(x, rho_implied, 'o-', color='magenta', linewidth=2.4, markersize=5, markerfacecolor='magenta', label='Implied ρ')
    plt.xticks(x, x_labels)
    plt.xlabel('Tranches')
    plt.ylabel('Correlation ρ')
    plt.title('Comparison of Market and Implied Correlations', fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True) 
    plt.grid(True)
    plt.show()
    
    return rho_implied


    

def obj_nu(nu, Kd_vec, Ku_vec, p, recovery, rho_vec, dates, discounts):
    
    Kd_vec = np.atleast_1d(Kd_vec)
    Ku_vec = np.atleast_1d(Ku_vec)
    rho_vec = np.atleast_1d(rho_vec)
    
    # Market prices using Vasicek Gaussian copula
    Price_market = Price_LHP_vasicek(Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates)

    # Calibrate rho_model so model price of equity tranche matches market price
    def objective_rho(r):
        return Price_LHP_tstud(nu, Kd_vec[0], Ku_vec[0], recovery, r, p, discounts, dates) - Price_market[0]

    rho_model = fsolve(objective_rho, x0=rho_vec[0])[0]

    # Compute model prices with calibrated rho_model
    errors = []
    for i in range(len(Ku_vec)):
        model_price = Price_LHP_tstud(nu, Kd_vec[i], Ku_vec[i], recovery, rho_model, p, discounts, dates)
        errors.append(model_price - Price_market[i])

    errors = np.array(errors)

    # Mean squared error
    MSE = np.mean(errors ** 2)

    return MSE, rho_model

def obj_nu_KL(nu, Kd_vec, Ku_vec, p, recovery, rho_vec, dates_list, discounts, I):
    # Prezzi di mercato con modello Vasicek + KL approximation
    Price_market = Price_KL_vasicek_fast(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates_list)

    # Funzione obiettivo calibrazione rho sul tranche equity
    def objective_rho(r):
        return Price_KL_tstud_fast(Kd_vec[0], Ku_vec[0], nu, recovery, I, r, p, discounts, dates_list) - Price_market[0]

    # Calibrazione rho con fsolve
    rho_model_KL = fsolve(objective_rho, rho_vec[0])[0]

    # Calcolo MSE su tutti i tranches con rho_model_KL
    errors = []
    for i in range(len(Ku_vec)):
        price_model = Price_KL_tstud_fast(Kd_vec[i], Ku_vec[i], nu, recovery, I, rho_model_KL, p, discounts, dates_list)
        errors.append((price_model - Price_market[i]) ** 2)

    MSE = np.mean(errors)

    return MSE, rho_model_KL

def obj_theta(price_vasicek, Nsim, discount, p, Kd_vec, Ku_vec, I, recovery, theta):
    
    errors = []
    for i in range(len(Ku_vec)):
        # Compute tranche price using Archimedean copula Monte Carlo simulation
        model_price,_ = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd_vec[i], Ku_vec[i], I, recovery, theta)
        # Calculate difference with Vasicek price
        errors.append(model_price - price_vasicek[i])

    # Mean squared error over all tranches
    MSE = np.mean(np.square(errors))
    return MSE
