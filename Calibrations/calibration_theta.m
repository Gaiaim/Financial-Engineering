%% FUNCTION: calibration_theta
% Price a tranche under an Archimedean (Clayton) copula using Monte Carlo simulation
%
% price = calibration_theta(Nsim, discount, p, Kd, Ku, I, recovery, theta)
%
% INPUTS:
%   Nsim      -    number of Monte Carlo simulations
%   discount  -    discount factor
%   p         -    default probability
%   Kd        -    lower attachment point of the tranche
%   Ku        -    upper detachment point of the tranche
%   I         -    number of obligors
%   recovery  -    recovery rate
%   theta     -    copula dependence parameter (Clayton generator)
%
% OUTPUT:
%   price     -    expected discounted value of the tranche

function price = calibration_theta(Nsim, discount, p, Kd, Ku, I, recovery, theta)

    rng('default');  % For reproducibility
    q = 1 - p;

    % Preallocate loss vectors
    Loss_ptf = zeros(Nsim, 1);
    portf_loss = zeros(Nsim, 1);
    Tranche_loss = zeros(Nsim, 1);

    for n = 1:Nsim
        % Simulate systemic factor from Gamma distribution
        y = gamrnd(1/theta, 1, 1, 1);

        % Simulate uniform idiosyncratic variables
        epsilon = rand(I, 1);

        % Define inverse of generator function
        inv_phi = @(s) (1 + s).^(-1/theta);

        % Compute copula values u(i) for each obligor
        u = zeros(I, 1);
        for i = 1:I
            u(i) = inv_phi((-1/y) * log(epsilon(i)));
        end

        % Survival test: u(i) <= q ⇒ survival
        n_survivals = sum(u <= q);
        n_defaults = I - n_survivals;

        % Compute portfolio loss
        Loss_ptf(n) = (1 - recovery) * (n_defaults / I);

        % Clip loss to tranche boundaries
        portf_loss(n) = min(max(Loss_ptf(n) - Kd, 0), Ku - Kd);

        % Normalize tranche loss by tranche width
        Tranche_loss(n) = portf_loss(n) / (Ku - Kd);
    end

    % Final discounted price of the tranche
    price = discount * (1 - mean(Tranche_loss));
end
