%% FUNCTION: Add_dates
function datesSet_add = Add_dates(dt, n)
% ADD_DATES Generates a series of future dates, adjusted for weekends.
%
%   datesSet_add = Add_dates(dt, n) creates an array of n+1 future dates, 
%   starting from the given date (dt) and spaced by one year each. If a 
%   generated date falls on a weekend, it is adjusted to the next business 
%   day (Monday).
%
%   Inputs:
%   - dt: Initial date (datetime format).
%   - n: Number of years to generate.
%
%   Output:
%   - datesSet_add: An (n+1)x1 array of adjusted future dates.

    % Initialize an array of NaT (Not-a-Time) values for storing dates
    datesSet_add = NaT(n+1, 1);

    % Generate the sequence of dates
    for i = 0:n
        datesSet_add(i+1) = dt + calyears(i); % Add i years to the initial date
        
        % Check if the generated date falls on a weekend and adjust to the next business day
        if weekday(datesSet_add(i+1)) == 7
            datesSet_add(i+1) = datesSet_add(i+1) + days(2);  % Saturday: move to Monday
        elseif weekday(datesSet_add(i+1)) == 1
            datesSet_add(i+1) = datesSet_add(i+1) + days(1);  % Sunday: move to Monday
        end
    end
end
