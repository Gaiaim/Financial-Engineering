%% FUNCTION: interpolation
% Computes the interpolated discount factor for a given target date using zero rate linear interpolation.
%
% INPUTS:
%   - start_date: Start date of the interpolation interval
%   - end_date: End date of the interpolation interval
%   - start_B: Discount factor at the start date
%   - end_B: Discount factor at the end date
%   - date: Target date for which to compute the discount factor
%   - fwd: Forward adjustment factor (usually 1 unless specified)
%   - today: Valuation date (anchor date for year fractions)
%
% OUTPUT:
%   - discount: Interpolated discount factor for the target date

function discount = interpolation(start_date, end_date, start_B, end_B, date, fwd, today)

    % Compute year fractions between today and each relevant date (using Act/365 convention)
    y_frac_start = yearfrac(today, start_date, 3); 
    y_frac_end = yearfrac(today, end_date, 3);
    y_frac_date = yearfrac(today, date, 3);

    % Compute implied zero rates (epsilons) for start and end discount factors
    eps_start = -log(start_B) / y_frac_start;
    eps_end = -log(end_B) / y_frac_end;

    % Interpolate the zero rate for the target date using linear interpolation
    if date > start_date && date < end_date
        y = interp1([start_date, end_date], [eps_start, eps_end], date, 'linear');
    else
        y = eps_end;  % Use flat extrapolation beyond known range
    end

    % Compute the discount factor for the target date and apply forward adjustment
    discount = exp(-y * y_frac_date) * fwd;

end
