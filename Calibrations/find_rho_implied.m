%% FUNCTION: find_rho_implied
% Calibrate implied correlations for tranches using different copula pricing methods (LHP, HP, KL)
%
% rho_implied = find_rho_implied(Kd_vec, Ku_vec, recovery, I, rho_vec, p, dates, discounts, price_model, flag)
%
% INPUTS:
%   Kd_vec      -    vector of lower attachment points
%   Ku_vec      -    vector of upper detachment points
%   recovery    -    recovery rate
%   I           -    number of obligors (used in HP and KL methods)
%   rho_vec     -    initial guess or market correlation vector
%   p           -    default probability
%   dates       -    dates for the discount curve
%   discounts   -    discount factors
%   price_model -    observed or target tranche prices
%   flag        -    pricing method: 'LHP', 'HP', or 'KL'
%
% OUTPUT:
%   rho_implied -    vector of implied correlations (one per tranche)

function rho_implied = find_rho_implied(Kd_vec, Ku_vec, recovery, I, rho_vec, p, dates, discounts, price_model, flag)

    rho_implied = zeros(size(rho_vec));
    n_tranches = length(Kd_vec);

    % Loop over tranches and invert the pricing function to find implied rho
    if flag == "LHP"
        for i = 1:n_tranches
            fun = @(r) Price_LHP_Vasicek(Kd_vec(i), Ku_vec(i), recovery, r, p, discounts, dates) - price_model(i);
            options = optimoptions('fsolve', 'Display', 'off');
            rho_implied(i) = fsolve(fun, rho_vec(i), options);
        end

    elseif flag == "HP"
        for i = 1:n_tranches
            fun = @(r) Price_HP_Vasicek(Kd_vec(i), Ku_vec(i), recovery, I, r, p, discounts, dates) - price_model(i);
            options = optimoptions('fsolve', 'Display', 'off');
            rho_implied(i) = fsolve(fun, rho_vec(i), options);
        end

    elseif flag == "KL"
        for i = 1:n_tranches
            fun = @(r) Price_KL_Vasicek(Kd_vec(i), Ku_vec(i), recovery, I, r, p, discounts, dates) - price_model(i);
            options = optimoptions('fsolve', 'Display', 'off');
            rho_implied(i) = fsolve(fun, rho_vec(i), options);
        end

    else
        error('Invalid flag. Use "LHP", "HP", or "KL".');
    end

    % === Plot implied vs market correlations ===
    x_labels = {'0-3', '0-6', '0-9', '0-12', '0-22'};  % Custom tranche labels
    x = 1:numel(Kd_vec);

    figure;
    set(gcf, 'Color', 'w'); hold on; set(gca, 'Color', 'w'); 

    % Market rho
    plot(x, rho_vec, 'o-', 'Color', 'b', 'LineWidth', 2.4, ...
        'MarkerSize', 5, 'MarkerFaceColor', 'b', 'DisplayName', 'Market \rho');

    % Implied rho
    plot(x, rho_implied, 'o-', 'Color', 'm', 'LineWidth', 2.4, ...
        'MarkerSize', 5, 'MarkerFaceColor', 'm', 'DisplayName', 'Implied \rho');

    % Axes and legend
    xticks(x); xticklabels(x_labels); 
    xlabel('Tranches'); ylabel('Correlation \rho');
    title('Comparison of Market and Implied Correlations', 'FontSize', 14, 'FontWeight', 'bold'); 
    legend('Location', 'best', 'Box', 'on');
    grid on; hold off;

end
