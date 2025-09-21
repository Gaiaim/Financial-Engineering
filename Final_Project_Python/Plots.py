import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from Pricing import (Price_HP_vasicek, Price_LHP_vasicek, Price_KL_vasicek_fast,
                      Price_HP_tstud, Price_LHP_tstud, Price_KL_tstud_fast, tranchePriceMC_GaussianCopula, tranchePriceMC_tCopula)



def compareTCopulaVsExact(tranche_labels, Kd_allzeros, Ku, p, rho, rho_model, recovery, I, nu_opt, discounts, dates, Nsim, discount):
   
    rho = np.atleast_1d(rho)
    rho_model = np.atleast_1d(rho_model)
    Kd_allzeros = np.atleast_1d(Kd_allzeros)
    Ku = np.atleast_1d(Ku)
    tranche_labels = list(tranche_labels)

    n_tranches = len(Ku)

    # Se rho è scalare, estendilo per tutti i tranches
    if rho.size == 1:
        rho = np.full(n_tranches, rho[0])
    if rho_model.size == 1:
        rho_model = np.full(n_tranches, rho_model[0])

    price_tcopula_model = np.zeros(n_tranches)
    price_tcopula_correct = np.zeros(n_tranches)
    price_exact = np.zeros(n_tranches)
    IC_tcopulamodel = np.zeros((n_tranches, 2))
    IC_tcopulamarket = np.zeros((n_tranches, 2))



    for i in range(n_tranches):
        print(f'\nSimulation {tranche_labels[i]} ({i+1}/{n_tranches})...')


        # t-Copula using model rho
        price_tcopula_model[i], IC_tcopulamodel[i, :] = tranchePriceMC_tCopula(
            Nsim, discount, rho_model[i], p, Kd_allzeros[i], Ku[i], I, recovery, nu_opt)


        # t-Copula using market rho
        price_tcopula_correct[i], IC_tcopulamarket[i, :] = tranchePriceMC_tCopula(
            Nsim, discount, rho[i], p, Kd_allzeros[i], Ku[i], I, recovery, nu_opt)

        # Exact pricing via HP Vasicek
        price_exact[i] = Price_HP_vasicek(Kd_allzeros[i], Ku[i], recovery, I, rho[i], p, discounts, dates)
    



    mse_model = np.mean((price_exact - price_tcopula_model)**2)
    mse_market = np.mean((price_exact - price_tcopula_correct)**2)

    # Print summary
    print(f'{"--- Comparison: t-Copula vs Exact (HP) ---":^80}')
    print(f'{"Tranche":<15}   {"Copula (rho_model) [Lower, Price, Upper]":<35}   {"Copula (rho_market) [Lower, Price, Upper]":<40}   {"Exact (HP)"}')
    print('-'*120)

    for i in range(n_tranches):
        print(f'{tranche_labels[i]:<10}   '
              f'[{IC_tcopulamodel[i,0]:10.6f}, {price_tcopula_model[i]:10.6f}, {IC_tcopulamodel[i,1]:10.6f}]   '
              f'[{IC_tcopulamarket[i,0]:10.6f}, {price_tcopula_correct[i]:10.6f}, {IC_tcopulamarket[i,1]:10.6f}]   '
              f'{price_exact[i]:20.8f}')

    print(f'\n{"MSE (t-Copula with rho_model vs Exact):":<40} {mse_model:20.8e}')
    print(f'{"MSE (t-Copula with rho_market vs Exact):":<40} {mse_market:20.8e}')

    return mse_model, mse_market, price_tcopula_model, price_tcopula_correct, IC_tcopulamodel, IC_tcopulamarket

def compareCopulaVsExact(Nsim, discount, rho_model, rho, p, Kd_allzeros, Ku, I, recovery, discounts, dates, tranche_labels):
    
    # Assicura array numpy 1D coerenti
    rho = np.atleast_1d(rho)
    rho_model = np.atleast_1d(rho_model)
    Kd_allzeros = np.atleast_1d(Kd_allzeros)
    Ku = np.atleast_1d(Ku)
    tranche_labels = list(tranche_labels)

    n_tranches = len(Ku)

    # Se rho è scalare, estendilo per tutti i tranches
    if rho.size == 1:
        rho = np.full(n_tranches, rho[0])
    if rho_model.size == 1:
        rho_model = np.full(n_tranches, rho_model[0])

    price_copula_model = np.zeros(n_tranches)
    price_copula_correct = np.zeros(n_tranches)
    price_exact = np.zeros(n_tranches)
    IC_gaussianmodel = np.zeros((n_tranches, 2))
    IC_gaussianmarket = np.zeros((n_tranches, 2))



    for i in range(n_tranches):
        print(f'Simulation {tranche_labels[i]} ({i+1}/{n_tranches})...')

        # Gaussian Copula with model correlation
   
        price_copula_model[i], IC_gaussianmodel[i, :] = tranchePriceMC_GaussianCopula(
            Nsim, discount, rho_model[i], p, Kd_allzeros[i], Ku[i], I, recovery)

        # Gaussian Copula with market correlation

        price_copula_correct[i], IC_gaussianmarket[i, :] = tranchePriceMC_GaussianCopula(
            Nsim, discount, rho[i], p, Kd_allzeros[i], Ku[i], I, recovery)

        # Exact HP Vasicek price
        price_exact[i] = Price_HP_vasicek(Kd_allzeros[i], Ku[i], recovery, I, rho[i], p, discounts, dates).item()



    # Mean squared errors
    mse_gaussianmodel_exact = np.mean((price_exact - price_copula_model) ** 2)
    mse_gaussianmarket_exact = np.mean((price_exact - price_copula_correct) ** 2)

    # Print formatted results with confidence intervals
    print(f'{"--- Comparison: Gaussian Copula vs Exact (HP) ---":^80}')
    print(f'{"Tranche":<15}   {"Copula (rho_model) [Lower, Price, Upper]":<35}   '
          f'{"Copula (rho_market) [Lower, Price, Upper]":<40}   {"Exact (HP)"}')
    print('-' * 120)

    for i in range(n_tranches):
        print(f'{tranche_labels[i]:<10}   '
              f'[{IC_gaussianmodel[i, 0]:10.6f}, {price_copula_model[i]:10.6f}, {IC_gaussianmodel[i, 1]:10.6f}]   '
              f'[{IC_gaussianmarket[i, 0]:10.6f}, {price_copula_correct[i]:10.6f}, {IC_gaussianmarket[i, 1]:10.6f}]   '
              f'{price_exact[i]:20.8f}')

    print(f'\n{"MSE (Gaussian Copula with rho_model vs Exact):":<40} {mse_gaussianmodel_exact:20.8e}')
    print(f'{"MSE (Gaussian Copula with rho_market vs Exact):":<40} {mse_gaussianmarket_exact:20.8e}')

    return mse_gaussianmodel_exact, mse_gaussianmarket_exact, price_copula_model, price_copula_correct, IC_gaussianmodel, IC_gaussianmarket




def plotTranchePricesModel(model_type, use_shifted, Kd_vec, Ku_vec, recovery, rho_vec, nu_opt, rho_model_vec, p, discounts, dates):

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

    # Number of obligors (I) to test, logarithmically spaced from 10 to 1000
    I_values = np.floor(np.logspace(1, np.log10(1000), 12)).astype(int)

    # Create default tranche labels in percentage format (e.g., '0%–3%')
    tranche_labels = [f"{kd*100:.0f}%–{ku*100:.0f}%" for kd, ku in zip(Kd_vec, Ku_vec)]

    # If shifted tranche prices, adjust labels to show consecutive ranges (e.g., '0%–3%', '3%–7%')
    if use_shifted:
        tranche_bounds = Ku_vec * 100
        tranche_labels = [f"{0 if i == 0 else tranche_bounds[i-1]:.0f}%–{tranche_bounds[i]:.0f}%" for i in range(len(tranche_bounds))]

    # Initialize arrays to store prices for all I and tranches
    price_exact_all = np.zeros((len(I_values), n_tranches))
    price_LHP_all = np.zeros((len(I_values), n_tranches))
    price_KL_all = np.zeros((len(I_values), n_tranches))

    for idx, I_curr in enumerate(I_values):
        # Calculate prices depending on chosen model
        if model_type.lower() == 'vasicek':
            rho_vec_subset = rho_vec[:n_tranches]
            price_exact = Price_HP_vasicek(Kd_vec, Ku_vec, recovery, I_curr, rho_vec_subset, p, discounts, dates)
            price_LHP = Price_LHP_vasicek(Kd_vec, Ku_vec, recovery, rho_vec_subset, p, discounts, dates)
            price_KL = Price_KL_vasicek_fast(Kd_vec, Ku_vec, recovery, I_curr, rho_vec_subset, p, discounts, dates)

        elif model_type.lower() == 'tstudent':
            rho_vec_subset = rho_model_vec[:n_tranches]
            price_exact = Price_HP_tstud(Kd_vec, Ku_vec, nu_opt, recovery, I_curr, rho_vec_subset, p, discounts, dates)
            price_LHP = Price_LHP_tstud(nu_opt, Kd_vec, Ku_vec, recovery, rho_vec_subset, p, discounts, dates)
            price_KL = Price_KL_tstud_fast(Kd_vec, Ku_vec, nu_opt, recovery, I_curr, rho_vec_subset, p, discounts, dates)

        else:
            raise ValueError('Model type must be "vasicek" or "tstudent"')


        price_exact_all[idx, :] = price_exact[:n_tranches]
        price_LHP_all[idx, :] = price_LHP[:n_tranches]
        price_KL_all[idx, :] = price_KL[:n_tranches]

    # Plot results
    plt.figure(figsize=(15, 5))

    title_prefix = f"Tranche Prices under {model_type.upper()} Model"
    if use_shifted:
        title_prefix += " with shifted tranches"

    plt.suptitle(title_prefix, fontsize=20, fontweight='bold')

    for tranche_idx in range(n_tranches):
        plt.subplot(1, n_tranches, tranche_idx + 1)
        plt.xscale('log')
        plt.plot(I_values, price_exact_all[:, tranche_idx], 'g-o', linewidth=2.4, markersize=5, label='Exact')
        plt.plot(I_values, price_KL_all[:, tranche_idx], 'r-o', linewidth=2.4, markersize=5, label='KL')
        plt.plot(I_values, price_LHP_all[:, tranche_idx], 'b-o', linewidth=2.4, markersize=5, label='LHP')
        plt.title(f"Tranche: {tranche_labels[tranche_idx]}", fontsize=14, fontweight='bold')
        plt.xlabel('Number of obligors I (Log Scale)')
        plt.ylabel('Normalized price')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()





def plot_KL_equity_comparison(I_values, price_eq_exact, price_eq_KL, price_KL_equity_new, price_eq_LHP):
  
    price_eq_LHP = float(np.squeeze(price_eq_LHP))  


    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xscale('log')
    plt.plot(I_values, price_eq_exact, 'g-o', linewidth=2.4, markersize=5, label='Exact price')
    plt.plot(I_values, price_KL_equity_new, 'r-o', linewidth=2.4, markersize=5, label='KL price (Alternative)')
    if np.isscalar(price_eq_LHP):
        plt.plot(I_values, price_eq_LHP * np.ones_like(I_values), 'b-o', linewidth=2.4, markersize=5, label='LHP price')
    else:
        plt.plot(I_values, price_eq_LHP, 'b-o', linewidth=2.4, markersize=5, label='LHP price')
    plt.xlabel('Number of mortgages')
    plt.ylabel('Prices')
    plt.title('Percentage Prices of Equity Tranche with Alternative Method', fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()

    # Calculate percentage errors
    error_eq = np.abs(price_eq_KL[:len(I_values)] - price_eq_exact) / price_eq_exact
    error_eq_alternative = np.abs(price_KL_equity_new[:len(I_values)] - price_eq_exact) / price_eq_exact

    # Plot price comparison and errors in subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for ax in axs:
        ax.grid(True)
        ax.set_xscale('log')

    # Subplot 1: Prices
    axs[0].plot(I_values, price_eq_KL, '-o', color='blue', linewidth=2.4, markersize=5, label='Price KL equity Standard Method')
    axs[0].plot(I_values, price_KL_equity_new, '-o', color='magenta', linewidth=2.4, markersize=5, label='Price KL equity Alternative Method')
    axs[0].set_xlabel('Number of mortgages')
    axs[0].set_ylabel('Prices')
    axs[0].set_title('Price KL for Equity Tranche: Standard vs Alternative Method', fontsize=14, fontweight='bold')
    axs[0].legend()

    # Subplot 2: Percentage errors
    axs[1].plot(I_values, error_eq, '--', color='blue', linewidth=2.4, markersize=5, label='Error Standard method')
    axs[1].plot(I_values, error_eq_alternative, '--', color='magenta', linewidth=2.4, markersize=5, label='Error Alternative method')
    axs[1].set_xlabel('Number of mortgages')
    axs[1].set_ylabel('Errors')
    axs[1].set_title('Percentage Errors of KL solution for Equity Tranche: Standard vs Alternative Method', fontsize=14, fontweight='bold')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
