%% FUNCTION: compareCopulaVsExact
% Compare tranche prices from Gaussian Copula simulations (with model and market correlations)
% against exact analytical prices from the Vasicek model (Hull-White approach).
%
% [mse_gaussianmodel_exact, mse_gaussianmarket_exact, price_copula_model, price_copula_correct, IC_gaussianmodel, IC_gaussianmarket] = ...
%     compareCopulaVsExact(Nsim, discount, rho_model, rho, p, Kd_allzeros, Ku, I, recovery, discounts, dates, tranche_labels)
%
% INPUTS:
%   Nsim                 -    number of Monte Carlo simulations
%   discount             -    discount factor
%   rho_model            -    vector of correlations from model calibration
%   rho                  -    market correlation vector
%   p                    -    default probability
%   Kd_allzeros          -    lower attachment points (typically zeros)
%   Ku                   -    upper detachment points
%   I                    -    number of obligors
%   recovery             -    recovery rate
%   discounts            -    discount curve values
%   dates                -    corresponding dates for the curve
%   tranche_labels       -    cell array of tranche names (e.g. {'0-3', '3-6', ...})
%
% OUTPUTS:
%   mse_gaussianmodel_exact   -    MSE between prices from model rho and exact prices
%   mse_gaussianmarket_exact  -    MSE between prices from market rho and exact prices
%   price_copula_model        -    simulated prices using rho_model
%   price_copula_correct      -    simulated prices using rho (market)
%   IC_gaussianmodel          -    confidence intervals [low, high] for rho_model prices
%   IC_gaussianmarket         -    confidence intervals [low, high] for rho (market) prices

function [mse_gaussianmodel_exact, mse_gaussianmarket_exact, price_copula_model, price_copula_correct, IC_gaussianmodel, IC_gaussianmarket] = ...
    compareCopulaVsExact(Nsim, discount, rho_model, rho, p, Kd_allzeros, Ku, I, recovery, discounts, dates, tranche_labels)

    n_tranches = length(Ku);

    % Preallocate result containers
    price_copula_model   = zeros(1, n_tranches);     % using calibrated rho
    price_copula_correct = zeros(1, n_tranches);     % using market rho
    price_exact          = zeros(1, n_tranches);     % using analytical HP model
    IC_gaussianmodel     = zeros(n_tranches, 2);     % CI for calibrated rho
    IC_gaussianmarket    = zeros(n_tranches, 2);     % CI for market rho

    tic_global = tic;

    for i = 1:n_tranches
        fprintf('Simulation %s (%d/%d)...\n', tranche_labels{i}, i, n_tranches); 
        t0 = tic;

        % Gaussian copula price with calibrated rho_model
        [price_copula_model(i), IC_gaussianmodel(i,:)] = tranchePriceMC_GaussianCopula( ...
            Nsim, discount, rho_model, p, Kd_allzeros(i), Ku(i), I, recovery);
        fprintf('  -> Gaussian Copula (rho_model) completed in %.2f s\n', toc(t0)); 

        % Gaussian copula price with market rho
        t0 = tic;
        [price_copula_correct(i), IC_gaussianmarket(i,:)] = tranchePriceMC_GaussianCopula( ...
            Nsim, discount, rho(i), p, Kd_allzeros(i), Ku(i), I, recovery);
        fprintf('  -> Gaussian Copula (rho_market) completed in %.2f s\n', toc(t0)); 

        % Exact price using HP Vasicek formulation
        t0 = tic;
        price_exact(i) = Price_HP_Vasicek(Kd_allzeros(i), Ku(i), recovery, I, rho(i), p, discounts, dates);
        fprintf('  -> Exact (HP) completed in %.2f s\n\n', toc(t0));
    end

    total_time = toc(tic_global);  
    fprintf('Total simulation completed in %.2f seconds.\n', total_time);

    % Compute mean squared errors
    mse_gaussianmodel_exact  = mean((price_exact - price_copula_model).^2);
    mse_gaussianmarket_exact = mean((price_exact - price_copula_correct).^2);

    % Print results in formatted table
    fprintf('\n');
    title = '--- Comparison: Gaussian Copula vs Exact (HP) ---';
    fprintf('%s%s\n', repmat(' ', 1, floor((80 - strlength(title)) / 2)), title);
    fprintf('%-15s   %-35s   %-40s   %-20s\n', 'Tranche', 'Copula (rho_model) [Lower, Price, Upper]', 'Copula (rho_market) [Lower, Price, Upper]', 'Exact (HP)');
    fprintf('%s\n', repmat('-', 1, 120));

    for i = 1:n_tranches
        fprintf('%-10s   [%10.6f, %10.6f, %10.6f]   [%10.6f, %10.6f, %10.6f]   %20.8f\n', ...
            tranche_labels{i}, ...
            IC_gaussianmodel(i,1), price_copula_model(i), IC_gaussianmodel(i,2), ...
            IC_gaussianmarket(i,1), price_copula_correct(i), IC_gaussianmarket(i,2), ...
            price_exact(i));
    end

    % Print summary MSEs
    fprintf('\n%-40s %20.8e\n', 'MSE (Gaussian Copula with rho_model vs Exact):', mse_gaussianmodel_exact);
    fprintf('%-40s %20.8e\n', 'MSE (Gaussian Copula with rho_market vs Exact):', mse_gaussianmarket_exact);
end
