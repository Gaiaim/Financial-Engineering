%% FUNCTION: tranchePriceMC_GaussianCopula
% Prices a tranche using vectorized one-factor Gaussian copula Monte Carlo simulation
%
% INPUTS:
%   Nsim     - number of simulations
%   discount - discount factor
%   rho      - correlation parameter for the one-factor model
%   p        - cumulative default probability
%   kd       - attachment point (fraction of portfolio)
%   ku       - detachment point (fraction of portfolio)
%   I        - number of obligors
%   recovery - common recovery rate
%
% OUTPUTS:
%   price    - expected discounted tranche value
%   IC       - 95% confidence interval for the price

function [price, IC] = tranchePriceMC_GaussianCopula(Nsim, discount, rho, p, Kd, Ku, I, recovery)

    rng('default')
    % Compute default threshold
    k = norminv(p);

    % Generate random factors
    M_vec = randn(1, Nsim);        % systemic
    zi_mat = randn(I, Nsim);       % idiosyncratic

    % Latent variables (Gaussian Copula)
    xi_mat = sqrt(rho) * M_vec + sqrt(1 - rho) * zi_mat;

    % Default count
    n_def = sum(xi_mat <= k, 1);

    % Portfolio loss
    Loss_ptf = (1 - recovery) * (n_def / I);

    % Tranche loss
    portf_loss = min(max(Loss_ptf - Kd, 0), Ku - Kd);
    Tranche_loss = portf_loss / (Ku - Kd);

    % Discounted expected price
    price = discount * (1 - mean(Tranche_loss));

    % Return confidence interval only if requested
    if nargout > 1
        std_err = std(Tranche_loss) / sqrt(Nsim);
        z = norminv(0.975);  % 95% CI
        IC = discount * ([1 1] - mean(Tranche_loss) + [-1 1] * z * std_err);
    end
end
