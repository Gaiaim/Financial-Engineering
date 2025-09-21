%% FUNCTION: interpolation_vector
% INTERPOLATION_VECTOR Interpolates discount factors for a set of input dates using known discount data.
%
% INPUTS:
%   - dates: Vector of known dates corresponding to existing discount factors
%   - discounts: Vector of discount factors associated with the known dates
%   - dates_interp: Vector of target dates to interpolate discount factors for
%   - today: Valuation date used for year fraction calculations
%
% OUTPUT:
%   - disc_interp: Vector of interpolated discount factors for dates_interp

function disc_interp = interpolation_vector(dates, discounts, dates_interp, today)

    % Ensure all date inputs are in serial date number format
    if isdatetime(dates)
        dates = datenum(dates);
    end
    if isdatetime(dates_interp)
        dates_interp = datenum(dates_interp);
    end
    if isdatetime(today)
        today = datenum(today);
    end

    % Initialize output vector
    num_interp = length(dates_interp);
    disc_interp = zeros(num_interp, 1);

    % Loop over each interpolation date
    for i = 1:num_interp
        interp_date = dates_interp(i);

        % Locate the closest bounding dates in the known dataset
        idx_prev = find(dates <= interp_date, 1, 'last');    % Most recent known date before interp_date
        idx_next = find(dates >= interp_date, 1, 'first');   % Closest known date after interp_date

        % Case: exact match
        if ~isempty(idx_prev) && dates(idx_prev) == interp_date
            disc_interp(i) = discounts(idx_prev);
        else
            % Case: proper bounding dates found
            if ~isempty(idx_prev) && ~isempty(idx_next)
                start_date = dates(idx_prev);
                end_date = dates(idx_next);
                start_B = discounts(idx_prev);
                end_B = discounts(idx_next);

                % Perform interpolation using helper function
                disc_interp(i) = interpolation(start_date, end_date, start_B, end_B, interp_date, 1, today);
            else
                % Out-of-bounds case (no extrapolation)
                disc_interp(i) = NaN;
            end
        end
    end

end
