%% FUNCTION: tranchePriceMC_tCopula
% Fast pricing of a tranche using vectorized t-copula Monte Carlo simulation
%
% INPUTS:
%   Nsim     - number of simulations
%   discount - discount factor
%   rho      - correlation coefficient
%   p        - cumulative default probability
%   kd       - attachment point
%   ku       - detachment point
%   I        - number of obligors
%   recovery - recovery rate
%   nu       - degrees of freedom of the t distribution
%
% OUTPUTS:
%   price    - expected discounted tranche value
%   IC       - 95% confidence interval for the price

function [price, IC] = tranchePriceMC_tCopula(Nsim, discount, rho, p, Kd, Ku, I, recovery, nu)

    rng('default')
    % Calibrate threshold k such that P(T <= k) = p for t-distribution
    tic;
    k = fzero(@(kval) tcdf(kval, nu) - p, -4);

    % Generate systemic and idiosyncratic t-distributed risk factors
    M = trnd(nu, 1, Nsim);         % systemic factor
    Z = trnd(nu, I, Nsim);         % idiosyncratic factors

    % Simulate latent variables for obligors using the t-copula model
    X = sqrt(rho) .* M + sqrt(1 - rho) .* Z;

    % Count defaults in each simulation (latent variable below threshold k)
    n_defaults = sum(X <= k, 1);

    % Adjust tranche boundaries for recovery rate
    u = Ku / (1 - recovery);
    d = Kd / (1 - recovery);

    % Calculate tranche loss fraction (normalized by tranche width)
    loss_frac = n_defaults / I;
    tranche_loss = min(max(loss_frac - d, 0), u - d) / (u - d);

    % Compute discounted expected tranche price
    price = discount * (1 - mean(tranche_loss));

    % Compute confidence interval only if requested
    if nargout > 1
        std_err = std(tranche_loss) / sqrt(Nsim);
        z = norminv(0.975);  % z-score for 95% confidence
        IC = discount * (1 - mean(tranche_loss) + [-1, 1] * z * std_err);
    end
end
