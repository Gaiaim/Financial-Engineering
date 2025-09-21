%% FUNCTION: Price_LHP_tstud_diff
% Prices a single tranche using the double t-Student copula under the large homogeneous portfolio (LHP) approximation
%
% INPUTS:
% - nu_M: degrees of freedom for the market (systemic) t-distribution
% - nu_Zi: degrees of freedom for the idiosyncratic t-distribution
% - Kd, Ku: lower and upper detachment points
% - recovery: recovery rate (as decimal)
% - rho: asset correlation
% - p: single-name default probability
% - dates, discounts: vectors for computing the discount factor
%
% OUTPUT:
% - price: present value of the tranche [Kd, Ku]

function prices = Price_LHP_tstud_diff(nu_M, nu_Zi, Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    prices = zeros(n_tranches, 1);

    % Discount factor (assume same maturity for all tranches)
    date = datetime('02-Feb-2027');
    today = datetime('02-Feb-2023');
    discount = interpolation_vector(dates, discounts, date, today);

    
    % Calibrate k to match marginal default probability p
    k = fzero(@(x) calibration_K_diff(x, rho_vec(1), nu_M, nu_Zi) - p, 0);


    % Loop over each tranche
    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        rho = rho_vec(i);
        

        % Cumulative distribution under double t
        F = @(l) tcdf((-k + sqrt(1 - rho) * tinv(l / (1 - recovery), nu_Zi)) / sqrt(rho), nu_M);

        % Expected Tranche Loss (ETL)
        integrand = @(l) 1 - F(l);
        ETL = integral(integrand, Kd, Ku) / (Ku - Kd);

        % Discounted tranche price
        prices(i) = discount * (1 - ETL);
    end
end
