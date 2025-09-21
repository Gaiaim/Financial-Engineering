%% FUNCTION: obj_nu
% Compute MSE between market and model prices using t-Student copula
%
% Inputs:
% - nu: degrees of freedom of the t-Student copula
% - Kd_vec: vector of lower detachment points
% - Ku_vec: vector of upper detachment points
% - p: default probability
% - recovery: recovery rate
% - rho_vec: vector of market-implied correlations
% - dates, discounts: bootstrapped discount curve info
%
% Outputs:
% - MSE: mean squared error between model and market prices
% - rho_model: calibrated correlation for equity tranche

function [MSE, rho_model] = obj_nu(nu, Kd_vec, Ku_vec, p, recovery, rho_vec, dates, discounts)

    % Market prices using Vasicek
    Price_market = Price_LHP_Vasicek(Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates);

    % Calibrate rho_model using only the equity tranche
    f = @(r) Price_LHP_tstud(nu, Kd_vec(1), Ku_vec(1), recovery, r, p, discounts, dates) - Price_market(1);
    options = optimoptions('fsolve', 'Display', 'off');
    rho_model = fsolve(f, rho_vec(1), options);

    % Compute model-implied prices for all tranches with calibrated rho_model
    errors = zeros(length(Ku_vec), 1);
    for i = 1:length(Ku_vec)
        errors(i) = Price_LHP_tstud(nu, Kd_vec(i), Ku_vec(i), recovery, rho_model, p, discounts, dates) - Price_market(i);
    end

    % Mean squared error
    MSE = mean(errors.^2);
end

