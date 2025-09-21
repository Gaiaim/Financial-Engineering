%% FUNCTION: compute_KL_equity_alternative
% Compute equity tranche price and portfolio valuation using KL method under a t-Student model
%
% [price_KL_equity_pct, price_up_KL, price_ptf] = compute_KL_equity_alternative(I_values, Ku_e, Kd_e, nu_opt, recovery, rho_model, p, discounts, dates, notional, discount)
%
% INPUTS:
%   I_values    -    vector of systemic factor values for KL approximation
%   Ku_e        -    upper attachment point of the equity tranche
%   Kd_e        -    lower attachment point of the equity tranche
%   nu_opt      -    degrees of freedom of the t-Student copula
%   recovery    -    recovery rate
%   rho_model   -    vector of asset correlations
%   p           -    default probability
%   discounts   -    discount factor curve
%   dates       -    corresponding dates for discount curve
%   notional    -    portfolio notional amount
%   discount    -    discount factor at maturity
%
% OUTPUTS:
%   price_KL_equity_pct -    percentage price of the equity tranche
%   price_up_KL         -    KL-based price of the upper tranche
%   price_ptf           -    present value of the full portfolio

function [price_KL_equity_pct, price_up_KL, price_ptf] = compute_KL_equity_alternative(I_values, Ku_e, Kd_e, nu_opt, recovery, rho_model, p, discounts, dates, notional, discount)
    
    % Compute KL-based price for the upper tranche (Ku_e to 1)
    price_up_KL = arrayfun(@(i) Price_KL_tstud(Ku_e, 1, nu_opt, recovery, i, rho_model, p, discounts, dates), I_values);
    
    % Convert upper tranche price from unit price to monetary value
    price_up = price_up_KL * notional * (1 - Ku_e);
    
    % Compute expected loss of the full portfolio
    expected_loss_p = (1 - recovery) * p;
    
    % Compute present value of the full portfolio
    price_ptf = discount * notional * (1 - expected_loss_p);
    
    % Residual value gives the absolute equity tranche price
    price_KL_equity = price_ptf - price_up;
    
    % Convert equity tranche price to percentage of notional
    price_KL_equity_pct = price_KL_equity ./ (notional * (Ku_e - Kd_e));

end
