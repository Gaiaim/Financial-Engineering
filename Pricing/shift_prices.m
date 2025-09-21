%% FUNCTION: shift_prices
% Computes prices of shifted tranches (e.g., 3-6, 6-9) starting from cumulative tranche prices
% (e.g., 0-3, 0-6, 0-9)
%
% INPUTS:
% - cumulative_prices: vector of cumulative prices P(0-x), e.g. P(0-3), P(0-6), P(0-9)
% - tranche_bounds: vector of upper tranche bounds, e.g. [3, 6, 9]
%
% OUTPUT:
% - shifted_prices: vector of prices for shifted tranches, e.g. P(3-6), P(6-9)

function shifted_prices = shift_prices(cumulative_prices, tranche_bounds)

    n = length(cumulative_prices);          % Number of cumulative tranches
    shifted_prices = zeros(1, n);            % Initialize output vector

    % The first tranche price equals the first cumulative price [0 - K1]
    shifted_prices(1) = cumulative_prices(1);

    % Calculate individual tranche prices using weighted differences
    for i = 2:n
        width_total = tranche_bounds(i);     % Current tranche upper bound (K_i)
        width_prev = tranche_bounds(i-1);    % Previous tranche upper bound (K_{i-1})

        % Compute tranche price for interval [K_{i-1}, K_i]
        shifted_prices(i) = (width_total * cumulative_prices(i) - width_prev * cumulative_prices(i-1)) / (width_total - width_prev);
    end

    shifted_prices = shifted_prices'; % Return as column vector
end