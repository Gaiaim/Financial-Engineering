%% FUNCTION: calibration_K
% Computes an integral used in the calibration of the t-Student copula model.
%
% INPUTS:
% - k:    Threshold parameter (scalar)
% - rho:  Correlation values (vector)
% - nu:   Degrees of freedom for the t-distribution
%
% OUTPUT:
% - integral: Vector of integral values corresponding to each rho

function integral = calibration_K(k, rho, nu)
    % Preallocate output
    integral = zeros(size(rho));  

    for i = 1:length(rho)
        rho_i = rho(i); 
        denom = sqrt(1 - rho_i);

        % Define the integrand
        integrand = @(y) tcdf((k - sqrt(rho_i) * y) ./ denom, nu) .* tpdf(y, nu);
        
        % Compute integral over y ∈ [-6, 6]
        integral(i) = quadgk(integrand, -6, 6);
    end
end

