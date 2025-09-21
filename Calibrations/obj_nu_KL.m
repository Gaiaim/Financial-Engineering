%% FUNCTION: obj_nu_KL
% Computes the mean squared error between market-implied and model-implied tranche prices
% under a t-Student copula model, using a single correlation (rho_model_KL) calibrated
% to match the first (equity) tranche.
% Inputs:
% - nu:       Degrees of freedom for the t-Student copula
% - Kd_vec:   Vector of lower detachment points
% - Ku_vec:   Vector of upper detachment points
% - p:        Default probability
% - recovery: Recovery rate (decimal)
% - rho_vec:  Vector of market-implied correlations (per tranche)
% - dates:    Payment dates
% - discounts: Discount factors
% - I:        Notional profile or integration weights
%
% Outputs:
% - MSE:            Mean squared error across all tranches
% - rho_model_KL:   Calibrated correlation (used for all tranches)


function [MSE, rho_model_KL] = obj_nu_KL(nu, Kd_vec, Ku_vec, p, recovery, rho_vec, dates, discounts, I)

    % Market prices using Vasicek model with KL approximation
    Price_market = Price_KL_Vasicek(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates);

    % Calibrate rho using the equity tranche
    f = @(r) Price_KL_tstud(Kd_vec(1), Ku_vec(1), nu, recovery, I, r, p, discounts, dates) - Price_market(1);
    options = optimoptions('fsolve', 'Display', 'off');
    rho_model_KL = fsolve(f, rho_vec(1), options);

    % Compute MSE between model prices (with rho_model_KL) and market prices
    errors = zeros(length(Ku_vec), 1);
    for i = 1:length(Ku_vec)
        price_model = Price_KL_tstud(Kd_vec(i), Ku_vec(i), nu, recovery, I, rho_model_KL, p, discounts, dates);
        errors(i) = (price_model - Price_market(i))^2;
    end
    
    MSE = mean(errors);

end
