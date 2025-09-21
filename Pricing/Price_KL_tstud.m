%% FUNCTION: Price_KL_tstud
% VECTORIAL VERSION: computes approximate MBS tranche prices via KL with t-Student copula
%
% approx_sol = Price_KL_tstud(Kd_vec, Ku_vec, nu, recovery, I, rho_vec, p, discounts, dates)
%
% INPUTS:
%   Kd_vec     -    vector of lower attachment points
%   Ku_vec     -    vector of upper detachment points
%   nu         -    degrees of freedom of the t-Student copula
%   recovery   -    recovery rate (e.g., 0.4)
%   I          -    number of obligors in the portfolio
%   rho_vec    -    vector of asset correlations (one per tranche)
%   p          -    marginal default probability
%   discounts  -    discount factor curve
%   dates      -    corresponding dates for discount curve
%
% OUTPUT:
%   approx_sol -    vector of discounted tranche prices

function approx_sol = Price_KL_tstud(Kd_vec, Ku_vec, nu, recovery, I, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    approx_sol = zeros(n_tranches, 1);

    % Compute discount factor for common maturity
    date = datetime('02-Feb-2027');  
    today = datetime('02-Feb-2023'); 
    discount = interpolation_vector(dates, discounts, date, today);
    
    % Calibrate threshold k from marginal probability (based on first rho)
    k_calibr = fzero(@(k) calibration_K(k, rho_vec(1), nu) - p, -4);

    % t-Student density function
    phi = @(y) tpdf(y, nu);

    % KL divergence function
    K_func = @(z, x) z .* log(z ./ x) + (1 - z) .* log((1 - z) ./ (1 - x));

    % KL normalization constant
    C1 = @(z) sqrt(I ./ (2 * pi * (1 - z) .* z));

    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        rho = rho_vec(i);

        % Rescale attachment points
        d = Kd / (1 - recovery);
        u = Ku / (1 - recovery);

        % Define tranche loss profile
        L = @(z) min(max(z - d, 0), u - d) / (u - d);

        % Conditional default probability
        P = @(y) tcdf((k_calibr - sqrt(rho) * y) ./ sqrt(1 - rho), nu);

        % Denominator of KL correction term
        D = @(y) arrayfun(@(yy) quadgk(@(z) C1(z) .* exp(-I .* K_func(z, P(yy))), eps, 1 - eps), y);

        % Full KL correction constant
        C = @(z, y) C1(z) ./ D(y);

        % Loss-weighted integrand
        integrand2 = @(y, z) L(z) .* C(z, y) .* exp(-I .* K_func(z, P(y)));

        % Integrate over systemic factor y
        integrand3 = @(y) phi(y) .* arrayfun(@(yy) quadgk(@(z) integrand2(yy, z), eps, 1 - eps), y);

        % Expected tranche loss
        expected_loss = quadgk(integrand3, -6, 6);

        % Discounted tranche value
        approx_sol(i) = discount * (1 - expected_loss);
    end
end
