%% FUNCTION: error_between_KL_methods

function error_between_KL_methods(Kd_e, Ku_e, nu_opt, recovery, I, rho_model, p, discounts, dates, notional, discount)
%
% Compares the results of three methods for pricing the equity tranche:
%     1. Standard KL method using t-Student copula
%     2. Alternative KL method 
%     3. HP method
% It computes and prints the relative errors between both KL methods and HP.
%
% INPUTS:
%   Kd_e        : Lower attachment point for equity tranche (scalar, usually 0)
%   Ku_e        : Upper attachment point for equity tranche (scalar, e.g., 0.03)
%   nu_opt      : Degrees of freedom for the t-Student copula
%   recovery    : Recovery rate (scalar between 0 and 1)
%   I           : Number of Monte Carlo simulations
%   rho_model   : Correlation parameter for the model
%   p           : Vector of cumulative default probabilities (per time step)
%   discounts   : Vector of discount factors for each time step
%   dates       : Vector of corresponding dates or time points
%   notional    : Notional amount of the tranche
%   discount    : Final discount factor applied (e.g., risk-free discounting)


    % Compute equity tranche price using standard KL method with t-copula 
    price_KL_standard_500 = Price_KL_tstud(Kd_e, Ku_e, nu_opt, recovery, I, rho_model, p, discounts, dates);

    % Compute equity tranche price using alternative KL method
    [price_KL_alternative_500, ~, ~] = compute_KL_equity_alternative(I, Ku_e, Kd_e, nu_opt, recovery, rho_model, p, discounts, dates, notional, discount);

    % Compute equity tranche price using HP method
    price_HP_500 = Price_HP_tstud(Kd_e, Ku_e, nu_opt, recovery, I, rho_model, p, discounts, dates);

    % Compute relative errors with respect to HP price
    relative_error_KL_standard = abs(price_KL_standard_500 - price_HP_500) / price_HP_500 * 100;
    relative_error_KL_alternative = abs(price_KL_alternative_500 - price_HP_500) / price_HP_500 * 100;

    % Print results
    fprintf('\nRelative Error between standard KL and HP:      %.4f%%\n', relative_error_KL_standard);
    fprintf('Relative Error between alternative KL and HP:   %.4f%%\n\n', relative_error_KL_alternative);

end
