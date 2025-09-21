%% FUNCTION: tranchePriceMC_tCopula_parallel
% Parallel pricing of a tranche using t-copula Monte Carlo simulation
%
% price = tranchePriceMC_tCopula_parallel(Nsim, discount, rho, p, kd, ku, I, recovery, nu)
%
% INPUTS:
%   Nsim      -    total number of simulations
%   discount  -    discount factor applied to expected payoff
%   rho       -    asset correlation coefficient
%   p         -    marginal default probability
%   kd        -    lower attachment point of the tranche
%   ku        -    upper detachment point of the tranche
%   I         -    number of obligors in the portfolio
%   recovery  -    recovery rate
%   nu        -    degrees of freedom for the t-distribution
%
% OUTPUT:
%   price     -    expected discounted value of the tranche

function price = tranchePriceMC_tCopula_parallel(Nsim, discount, rho, p, kd, ku, I, recovery, nu)

    % Calibrate threshold k such that CDF_t(k) = p
    k = fzero(@(kval) tcdf(kval, nu) - p, -4);

    % Convert attachment and detachment points to loss space
    u = ku / (1 - recovery);
    d = kd / (1 - recovery);

    % Set up parallel simulation parameters
    block_size = 1e5;
    n_blocks = ceil(Nsim / block_size);
    loss_sum = zeros(1, n_blocks);

    % Parallel loop over blocks
    parfor b = 1:n_blocks
        this_N = min(block_size, Nsim - (b - 1) * block_size);

        % Generate systemic and idiosyncratic factors from t-distribution
        M = trnd(nu, 1, this_N);
        Z = trnd(nu, I, this_N);

        % Compute latent variables using one-factor t-copula model
        X = sqrt(rho) .* M + sqrt(1 - rho) .* Z;

        % Count defaults per scenario
        n_defaults = sum(X <= k, 1);

        % Compute normalized tranche loss
        loss_frac = n_defaults / I;
        tranche_loss = min(max(loss_frac - d, 0), u - d) / (u - d);

        % Accumulate tranche losses
        loss_sum(b) = sum(tranche_loss);
    end

    % Compute expected tranche price
    avg_loss = sum(loss_sum) / Nsim;
    price = discount * (1 - avg_loss);
end
