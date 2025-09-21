%% FUNCTION: Price_LHP_Vasicek
% Compute tranche prices under the Vasicek (Gaussian) copula model
%
% Inputs:
% - Kd_vec, Ku_vec: lower and upper detachment points
% - recovery: recovery rate
% - rho_vec: asset correlations
% - p: default probability
% - dates, discounts: discount curve data
%
% Output:
% - LHP_sol: vector of discounted tranche prices (market-implied)

function LHP_sol = Price_LHP_Vasicek(Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates)

    n_tranches = length(Kd_vec);
    ETL_market = zeros(n_tranches, 1);
    LHP_sol = zeros(n_tranches, 1);

    % Compute discount factor to maturity date
    date = datetime('02-Feb-2027');
    today = datetime('02-Feb-2023');
    discount = interpolation_vector(dates, discounts, date, today);

    % Loop over each tranche to compute expected tranche loss and discounted price
    for i = 1:n_tranches
        Kd = Kd_vec(i);
        Ku = Ku_vec(i);
        rho = rho_vec(i);

        % Rescale tranche attachment points by (1 - recovery)
        d = Kd / (1 - recovery);
        u = Ku / (1 - recovery);

        % Tranche loss function normalized between d and u
        L = @(z) min(max(z - d, 0), u - d) / (u - d);

        % Change of variable for conditional default probability
        inv_P = @(z) (norminv(p) - sqrt(1 - rho) .* norminv(z)) ./ sqrt(rho);

        % Derivative of inverse conditional default CDF
        der_P = @(z) 1 ./ normpdf(norminv(z)) .* sqrt((1 - rho) / rho);

        % Integrand for expected tranche loss calculation
        integrand = @(z) normpdf(inv_P(z)) .* der_P(z);
        integrand1 = @(z) L(z) .* integrand(z);

        % Calculate expected tranche loss via numerical integration
        ETL_market(i) = quadgk(integrand1, 0, 1);

        % Calculate discounted tranche price as notional minus ETL
        LHP_sol(i) = discount * (1 - ETL_market(i));
    end
end