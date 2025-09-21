%% FUNCTION: Price_KL_Vasicek
% Computes the approximate KL price vectorially for vectors of Kd, Ku, and rho
%
% Inputs:
% - Kd_vec, Ku_vec, rho_vec: parameter vectors (same length)
% - recovery, I, p, discounts, dates: scalar or fixed vectors
%
% Output:
% - approx_sol_all: vector of prices corresponding to input parameters

function approx_sol_all = Price_KL_Vasicek(Kd_vec, Ku_vec, recovery, I, rho_vec, p, discounts, dates)

    n = length(Kd_vec);              
    approx_sol_all = zeros(n,1);     

    % Discount factor (comune a tutte le tranche)
    date = datetime('02-Feb-2027');  
    today = datetime('02-Feb-2023'); 
    discount = interpolation_vector(dates, discounts, date, today);

    % Standard normal PDF, definita una volta
    phi = @(y) normpdf(y);

    % KL divergence, definita una volta, vettoriale su z e x
    Kfun = @(z, x) z .* log(z ./ x) + (1 - z) .* log((1 - z) ./ (1 - x));

    % Normalization factor C1(z)
    C1 = @(z) sqrt(I ./ (2 * pi * (1 - z) .* z));

    for idx = 1:n
        Kd = Kd_vec(idx);
        Ku = Ku_vec(idx);
        rho = rho_vec(idx);

        % Rescaling tranche boundaries
        d = Kd / (1 - recovery);
        u = Ku / (1 - recovery);

        % Loss function
        L = @(z) min(max(z - d, 0), u - d) / (u - d);

        % Conditional default probability data-dependent
        P = @(y) normcdf((norminv(p) - sqrt(rho) * y) ./ sqrt(1 - rho));

        % Denominator integrand over z per y
        integrand1 = @(z, y) C1(z) .* exp(-I * Kfun(z, P(y)));

        % Denominator D(y) integral vettoriale
        D = @(y) arrayfun(@(yy) quadgk(@(z) integrand1(z, yy), eps, 1 - eps), y);

        % Normalization C(z,y)
        C = @(z, y) C1(z) ./ D(y);

        % Numerator integrand con perdita ponderata
        integrand2 = @(y, z) L(z) .* C(z, y) .* exp(-I .* Kfun(z, P(y)));

        % Integrale totale su y e z
        integrand3 = @(y) phi(y) .* arrayfun(@(yy) quadgk(@(z) integrand2(yy, z), eps, 1 - eps), y);

        % Calcolo integrale finale expected loss
        expected_loss = quadgk(integrand3, -6, 6);

        % Prezzo finale tranche scontato
        approx_sol_all(idx) = discount * (1 - expected_loss);
    end

    
end
