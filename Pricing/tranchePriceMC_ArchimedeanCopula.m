%% FUNCTION: tranchePriceMC_ArchimedeanCopula
% Price a tranche using Monte Carlo simulation under an Archimedean copula model (Clayton)
%
% [price, ci] = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd, Ku, I, recovery, theta)
%
% INPUTS:
%   Nsim      -    number of Monte Carlo simulations
%   discount  -    discount factor
%   p         -    marginal default probability
%   Kd        -    lower attachment point of the tranche
%   Ku        -    upper detachment point of the tranche
%   I         -    number of obligors in the portfolio
%   recovery  -    recovery rate
%   theta     -    dependence parameter of the Archimedean (Clayton) copula
%
% OUTPUTS:
%   price     -    expected discounted tranche price
%   IC        -    confidence interval for the tranche price (95%)

function [price, IC] = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd, Ku, I, recovery, theta)

    rng('default');  % Set random seed for reproducibility
    q = 1 - p;

    % Simulate systemic factor from Gamma distribution
    y = gamrnd(1/theta, 1, Nsim, 1);

    % Generate idiosyncratic uniform variables
    epsilon = rand(Nsim, I);

    % Apply inverse generator transformation
    s = (-1 ./ y) .* log(epsilon);
    u = (1 + s) .^ (-1/theta);

    % Count defaults (u > q means default)
    n_defaults = sum(u > q, 2);

    % Compute portfolio-level loss
    Loss_ptf = (1 - recovery) * (n_defaults / I);

    % Clip and normalize to compute tranche loss
    portf_loss = min(max(Loss_ptf - Kd, 0), Ku - Kd);
    Tranche_loss = portf_loss / (Ku - Kd);

    % Compute discounted expected tranche payoff
    price = discount * (1 - mean(Tranche_loss));

    % Compute 95% confidence interval if requested
    if nargout > 1
        alpha = 0.05;
        stderr = std(Tranche_loss) / sqrt(Nsim);
        z = norminv(1 - alpha/2);
        IC = discount * [1 - mean(Tranche_loss) - z * stderr, ...
                         1 - mean(Tranche_loss) + z * stderr];
    end
end
