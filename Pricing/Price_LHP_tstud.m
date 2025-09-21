%% FUNCTION: Price_LHP_tstud
% Compute tranche prices under a t-Student copula model (Double t)
%
% Inputs:
% - nu: degrees of freedom of the t-Student copula
% - Kd_vec: lower detachment points
% - Ku_vec: upper detachment points
% - recovery: recovery rate
% - rho_vec: asset correlation(s)
% - p: default probability
% - dates, discounts: market discount curve
%
% Output:
% - Tstud_sol: vector of discounted tranche prices

function Tstud_sol = Price_LHP_tstud(nu, Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    ETL = zeros(n_tranches, 1);  
    Tstud_sol = zeros(n_tranches, 1);

    % Compute discount factor (same for all tranches)
    date = datetime('02-Feb-2027');
    today = datetime('02-Feb-2023');
    discount = interpolation_vector(dates, discounts, date, today);

    % Calibrate threshold k so that t-CDF matches marginal default probability p
    k_calibr = fzero(@(k) calibration_K(k, rho_vec(1), nu) - p, -4);

    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        r = rho_vec(i);

        % Rescale tranche attachment points by (1 - recovery)
        d = Kd / (1 - recovery);
        u = Ku / (1 - recovery);

        % Tranche loss function normalized between d and u
        L = @(z) min(max(z - d, 0), u - d) / (u - d);

        % Inverse conditional default probability function
        inv_P = @(z) (k_calibr - sqrt(1 - r) .* tinv(z, nu)) ./ sqrt(r);

        % Derivative of inverse function (absolute value)
        der_P = @(z) abs(-1 ./ tpdf(tinv(z, nu), nu) .* sqrt((1 - r) / r));

        % Integrand: t-density of inv_P times derivative for change of variables
        integrand = @(z) tpdf(inv_P(z), nu) .* der_P(z);

        % Integrand weighted by tranche loss function
        integrand1 = @(z) L(z) .* integrand(z);

        % Compute expected tranche loss via numerical integration
        ETL(i) = quadgk(integrand1, 0, 1);

        % Discount expected tranche payoff
        Tstud_sol(i) = discount * (1 - ETL(i));
    end
end
