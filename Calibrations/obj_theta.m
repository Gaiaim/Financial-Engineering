%% FUNCTION: obj_theta
% Objective function for calibrating the Clayton copula parameter (θ)
% Computes mean squared error (MSE) between Archimedean MC prices and Vasicek reference prices
%
% MSE = obj_theta(price_vasicek, Nsim, discount, p, Kd_vec, Ku_vec, I, recovery, theta)
%
% INPUTS:
%   price_vasicek -    reference tranche prices (e.g., from Vasicek model)
%   Nsim          -    number of Monte Carlo simulations
%   discount      -    discount factor
%   p             -    default probability
%   Kd_vec        -    vector of lower attachment points
%   Ku_vec        -    vector of upper detachment points
%   I             -    number of obligors
%   recovery      -    recovery rate
%   theta         -    Archimedean (Clayton) copula parameter
%
% OUTPUT:
%   MSE           -    mean squared error between model and reference prices

function MSE = obj_theta(price_vasicek, Nsim, discount, p, Kd_vec, Ku_vec, I, recovery, theta)

    errors = zeros(length(Ku_vec), 1);

    for i = 1:length(Ku_vec)
        % Compute model price using MC under Archimedean copula
        model_price = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd_vec(i), Ku_vec(i), I, recovery, theta);

        % Error against Vasicek benchmark
        errors(i) = model_price - price_vasicek(i);
    end

    % Mean squared error as calibration loss function
    MSE = mean(errors.^2);
end
