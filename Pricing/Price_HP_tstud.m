%% FUNCTION: Price_HP_tstud
% Prices MBS tranches under a single t-Student copula using binomial expansion
%
% INPUTS:
% - Kd_vec, Ku_vec: detachment points for tranches
% - nu: degrees of freedom of t-distribution
% - recovery: recovery rate
% - I: number of obligors
% - rho_vec: vector of correlations (one per tranche)
% - p: default probability
% - discounts, dates: used to compute discount factor
%
% OUTPUT:
% - exact_sol: present value of each tranche

function exact_sol = Price_HP_tstud(Kd_vec, Ku_vec, nu, recovery, I, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    exact_sol = zeros(n_tranches, 1);

    % Compute discount factor to maturity
    date = datetime('02-Feb-2027');
    today = datetime('02-Feb-2023');
    discount = interpolation_vector(dates, discounts, date, today);

    % Calibrate t-threshold k such that t-copula default prob = p
    k_calibr = fzero(@(k) calibration_K(k, rho_vec(1), nu) - p, 0);

    % Precompute binomial coefficients 
    binom_coeffs = arrayfun(@(m) nchoosek(I, m), 0:I);

    % Define phi 
    phi = @(y) tpdf(y, nu);

    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        r = rho_vec(i);

        % Conditional default probability P(y)
        P = @(y) tcdf((k_calibr - sqrt(r) * y) ./ sqrt(1 - r), nu);

        % Loss bounds scaled by (1 - recovery)
        d = Kd / (1 - recovery);
        u = Ku / (1 - recovery);

        % Tranche loss function
        L = @(z) min(max(z - d, 0), u - d) / (u - d);

        tot_sum = 0;
        for m = 0:I
            coeff = binom_coeffs(m+1); % +1 for MATLAB indexing
            integrand = @(y) phi(y) .* coeff .* (P(y).^m) .* (1 - P(y)).^(I - m);
            pd = quadgk(integrand, -6, 6);
            tot_sum = tot_sum + L(m / I) * pd;
        end

        exact_sol(i) = discount * (1 - tot_sum);
    end
end
