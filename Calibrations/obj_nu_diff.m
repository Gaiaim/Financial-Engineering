%% FUNCTION obj_nu_diff
% Computes mean squared error (MSE) between model-implied and market-implied tranche prices
% using a double t-Student copula with degrees of freedom nu_M (market) and nu_Zi (idiosyncratic).
%
% INPUTS:
% - nu_M, nu_Zi: degrees of freedom of t-copula for market and idiosyncratic risks
% - Kd_vec, Ku_vec: vectors of lower and upper detachment points for each tranche
% - p: default probability
% - recovery: recovery rate (decimal)
% - rho_vec: vector of market-implied correlations
% - dates, discounts: used for present value computation
%
% OUTPUTS:
% - MSE: mean squared error between model and market prices
% - rho_model: calibrated model correlation

function [MSE, rho_model] = obj_nu_diff(nu_M, nu_Zi, Kd_vec, Ku_vec, p, recovery, rho_vec, dates, discounts)
    % Compute target tranche prices from Vasicek model
    Price_market = Price_LHP_Vasicek(Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates);

    % Calibrate model-implied rho using only the equity tranche
    f = @(r) Price_LHP_tstud_diff(nu_M, nu_Zi, Kd_vec(1), Ku_vec(1), recovery, r, p, discounts, dates) - Price_market(1);
    options = optimoptions('fsolve', 'Display', 'off');
    rho_model = fsolve(f, rho_vec(1), options);

    % Compute mean squared error across all tranches
    errors = zeros(length(Ku_vec), 1);
    for i = 1:length(Ku_vec)
        model_price = Price_LHP_tstud_diff(nu_M, nu_Zi, Kd_vec(i), Ku_vec(i), recovery, rho_model, p, discounts, dates);
        errors(i) = (model_price - Price_market(i))^2;
    end

    MSE = mean(errors);
end
