%% FUNCTION: bootstrap
% Computes the discount curve from market instruments using bootstrapping.
%
% INPUTS:
%   - datesSet: struct containing market dates (settlement, depos, futures, swaps)
%   - ratesSet: struct containing corresponding market rates (depos, futures, swaps)
%
% OUTPUTS:
%   - dates: vector of maturity dates
%   - discounts: vector of corresponding discount factors

function [dates, discounts] = bootstrap(datesSet, ratesSet)

    % Deposit instruments
    L_depos = mean(ratesSet.depos, 2);  % Average deposit rates
    today = datesSet.settlement;
    y_frac_depos = yearfrac(today, datesSet.depos, 2);  % Act/360 convention

    discounts_depos = 1 ./ (1 + y_frac_depos .* L_depos);  % Deposit discount factors

    % Futures instruments
    y_frac_futurs = yearfrac(datesSet.futures(:,1), datesSet.futures(:,2), 2);  % Time between start and end
    mid_rates_futures = mean(ratesSet.futures, 2);  % Mid rates

    fwd_disc = 1 ./ (1 + y_frac_futurs .* mid_rates_futures);  % Forward discount factors

    discounts_futures = zeros(7, 1);  % Preallocate vector for futures discounts

    % Interpolate sequentially using previous known discount values
    discounts_futures(1) = interpolation(datesSet.depos(3), datesSet.depos(4), discounts_depos(3), discounts_depos(4), datesSet.futures(1,1), fwd_disc(1), today);
    discounts_futures(2) = interpolation(datesSet.depos(4), datesSet.futures(1,2), discounts_depos(4), discounts_futures(1), datesSet.futures(2,1), fwd_disc(2), today);
    discounts_futures(3) = interpolation(datesSet.futures(1,2), datesSet.futures(2,2), discounts_futures(1), discounts_futures(2), datesSet.futures(3,1), fwd_disc(3), today);
    discounts_futures(4) = discounts_futures(3) * fwd_disc(4);
    discounts_futures(5) = discounts_futures(4) * fwd_disc(5);
    discounts_futures(6) = interpolation(datesSet.futures(4,2), datesSet.futures(5,2), discounts_futures(4), discounts_futures(5), datesSet.futures(6,1), fwd_disc(6), today);
    discounts_futures(7) = interpolation(datesSet.futures(5,2), datesSet.futures(6,2), discounts_futures(5), discounts_futures(6), datesSet.futures(7,1), fwd_disc(7), today);

    % Swap instruments
    dt = datetime(2023, 02, 02);  % Starting date
    datesSet_add = Add_dates(dt, 50);  % Generate swap dates up to 50 years
    swap_dates = datenum(datesSet_add);  % Convert to numeric format

    y_frac_swaps = yearfrac(swap_dates(1:end-1), swap_dates(2:end), 6);  % 30/360 convention

    mid_rates_swaps = mean(ratesSet.swaps, 2);
    interpolated_rates = spline(datesSet.swaps, mid_rates_swaps, swap_dates);  % Spline interpolation

    discounts_swaps = zeros(length(swap_dates) - 1, 1);  % Preallocate vector

    % Bootstrap first swap discount with interpolation
    discounts_swaps(1) = interpolation(datesSet.futures(3,2), datesSet.futures(4,2), ...
                                       discounts_futures(3), discounts_futures(4), ...
                                       swap_dates(2), 1, today);

    % Recursive bootstrapping for subsequent swap discounts
    for i = 2:(length(swap_dates) - 1)
        x = sum(y_frac_swaps(1:i-1) .* discounts_swaps(1:i-1));  % Weighted previous discounts
        discounts_swaps(i) = (1 - interpolated_rates(i+1) * x) / ...
                             (1 + y_frac_swaps(i) * interpolated_rates(i+1));
    end

    % Aggregate results
    % Combine all discount dates and factors
    dates = [today; datesSet.depos(1:4); datesSet.futures(1:7,2); swap_dates(3:end)];
    discounts = [1; discounts_depos(1:4); discounts_futures; discounts_swaps(2:end)];

    % Sort chronologically
    [dates_sorted, idx] = sort(dates);
    discounts_sorted = discounts(idx);

    % Output sorted results
    dates = dates_sorted;
    discounts = discounts_sorted;

end
