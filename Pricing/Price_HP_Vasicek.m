%% FUNCTION: Price_HP_Vasicek
% Prices MBS tranches using Gaussian copula (Vasicek model) with binomial expansion
%
% INPUTS:
% - Kd_vec, Ku_vec: detachment points
% - recovery: recovery rate
% - I: number of obligors
% - rho_vec: correlation vector
% - p: default probability
% - discounts, dates: discount curve inputs
%
% OUTPUT:
% - exact_sol: present value of each tranche

function exact_sol = Price_HP_Vasicek(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    exact_sol = zeros(n_tranches, 1);

    % Get discount factor to maturity
    date = datetime('02-Feb-2027');
    today = datetime('02-Feb-2023');
    discount = interpolation_vector(dates, discounts, date, today);

    % Precompute binomial coefficients once
    binom_coeffs = arrayfun(@(m) nchoosek(I, m), 0:I);

    % Define standard normal pdf once
    phi = @(y) normpdf(y);

    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        rho = rho_vec(i);

        % Conditional default probability P(y)
        P = @(y) normcdf((norminv(p) - sqrt(rho) * y) / sqrt(1 - rho));

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
