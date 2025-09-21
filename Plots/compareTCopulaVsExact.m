%% FUNCTION: compareTCopulaVsExact
% Compare t-Copula simulated tranche prices (using both model and market rho)
% against analytical HP Vasicek prices, with confidence intervals and MSE evaluation.
%
% [mse_model, mse_market, price_tcopula_model, price_tcopula_correct, IC_tcopulamodel, IC_tcopulamarket] = ...
%     compareTCopulaVsExact(tranche_labels, Kd, Ku, p, rho, rho_model, recovery, I, nu_opt, discounts, dates, Nsim, discount)
%
% INPUTS:
%   tranche_labels        -    cell array of tranche names (e.g. {'0-3', '3-6', ...})
%   Kd                    -    vector of lower attachment points
%   Ku                    -    vector of upper detachment points
%   p                     -    default probability
%   rho                   -    market correlation vector
%   rho_model             -    calibrated correlation vector
%   recovery              -    recovery rate
%   I                     -    number of obligors
%   nu_opt                -    degrees of freedom for the t-Copula
%   discounts             -    discount factor curve
%   dates                 -    corresponding dates for discount curve
%   Nsim                  -    number of Monte Carlo simulations
%   discount              -    discount factor for tranche payoff
%
% OUTPUTS:
%   mse_model             -    MSE between t-Copula (model rho) and HP prices
%   mse_market            -    MSE between t-Copula (market rho) and HP prices
%   price_tcopula_model   -    simulated prices using calibrated rho
%   price_tcopula_correct -    simulated prices using market rho
%   IC_tcopulamodel       -    confidence intervals for model prices [low, high]
%   IC_tcopulamarket      -    confidence intervals for market prices [low, high]

function [mse_model, mse_market, price_tcopula_model, price_tcopula_correct, IC_tcopulamodel, IC_tcopulamarket] = ...
    compareTCopulaVsExact(tranche_labels, Kd, Ku, p, rho, rho_model, recovery, I, nu_opt, discounts, dates, Nsim, discount)

    n_tranches = length(Ku);

    % Preallocate arrays
    price_tcopula_model   = zeros(1, n_tranches); 
    price_tcopula_correct = zeros(1, n_tranches); 
    price_exact           = zeros(1, n_tranches); 
    IC_tcopulamodel       = zeros(n_tranches, 2);
    IC_tcopulamarket      = zeros(n_tranches, 2);

    tic_global = tic;

    for i = 1:n_tranches
        fprintf('\nSimulation %s (%d/%d)...\n', tranche_labels{i}, i, n_tranches); 

        % --- t-Copula price with model-calibrated rho ---
        t0 = tic;
        [price_tcopula_model(i), IC_tcopulamodel(i,:)] = tranchePriceMC_tCopula( ...
            Nsim, discount, rho_model, p, Kd(i), Ku(i), I, recovery, nu_opt);
        fprintf('  -> t-Copula (rho_model) completed in %.2f s\n', toc(t0)); 

        % --- t-Copula price with market rho ---
        t0 = tic;
        [price_tcopula_correct(i), IC_tcopulamarket(i,:)] = tranchePriceMC_tCopula( ...
            Nsim, discount, rho(i), p, Kd(i), Ku(i), I, recovery, nu_opt);
        fprintf('  -> t-Copula (rho_market) completed in %.2f s\n', toc(t0)); 

        % --- Analytical HP Vasicek price ---
        t0 = tic;
        price_exact(i) = Price_HP_Vasicek(Kd(i), Ku(i), recovery, I, rho(i), p, discounts, dates);
        fprintf('  -> Exact (HP) completed in %.2f s\n', toc(t0));
    end

    total_time = toc(tic_global);  
    fprintf('Total simulation completed in %.2f seconds.\n', total_time);

    % Compute mean squared errors
    mse_model  = mean((price_exact - price_tcopula_model).^2);
    mse_market = mean((price_exact - price_tcopula_correct).^2);

    % Display results with confidence intervals
    fprintf('\n');
    title = '--- Comparison: t-Copula vs Exact (HP) ---';
    fprintf('%s%s\n', repmat(' ', 1, floor((80 - strlength(title)) / 2)), title);
    fprintf('%-15s   %-35s   %-40s   %-20s\n', 'Tranche', 'Copula (rho_model) [Lower, Price, Upper]', 'Copula (rho_market) [Lower, Price, Upper]', 'Exact (HP)');
    fprintf('%s\n', repmat('-', 1, 120));

    for i = 1:n_tranches
        fprintf('%-10s   [%10.6f, %10.6f, %10.6f]   [%10.6f, %10.6f, %10.6f]   %20.8f\n', ...
            tranche_labels{i}, ...
            IC_tcopulamodel(i,1), price_tcopula_model(i), IC_tcopulamodel(i,2), ...
            IC_tcopulamarket(i,1), price_tcopula_correct(i), IC_tcopulamarket(i,2), ...
            price_exact(i));
    end

    % Summary of MSE values
    fprintf('\n%-40s %20.8e\n', 'MSE (t-Copula with rho_model vs Exact):', mse_model);
    fprintf('%-40s %20.8e\n', 'MSE (t-Copula with rho_market vs Exact):', mse_market);
end
