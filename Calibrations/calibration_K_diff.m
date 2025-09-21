%% FUNCTION: calibration_K_diff
% Computes an integral used in the double t-Student copula calibration
% when two different degrees of freedom are used for the systemic and idiosyncratic components.
%
% INPUTS:
% - k:      threshold parameter (scalar)
% - rho:    vector of correlation values
% - nu_M:   degrees of freedom for the market factor
% - nu_Zi:  degrees of freedom for the idiosyncratic components
%
% OUTPUT:
% - integral: vector of integral results for each rho

function integral = calibration_K_diff(k, rho, nu_M, nu_Zi)

    integral = zeros(size(rho));  % Preallocate 
    for i = 1:length(rho)
        rho_i = rho(i);
        denom = sqrt(1 - rho_i);
        sqrt_rho = sqrt(rho_i);

        % Define integrand using mixed t-distributions
        integrand = @(y) tcdf((k - sqrt_rho * y) ./ denom, nu_Zi) .* tpdf(y, nu_M);

        % Numerical integration over y in [-6, 6]
        integral(i) = quadgk(integrand, -6, 6);
    end
end
