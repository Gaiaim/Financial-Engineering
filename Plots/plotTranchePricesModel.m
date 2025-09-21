%% FUNCTION: plotTranchePricesModel
% Plot tranche prices computed with Exact, KL and LHP methods across multiple obligor sizes (I)
% under Vasicek or t-Student copula models, optionally with shifted (cumulative) tranches.
%
% plotTranchePricesModel(model_type, use_shifted, Kd_vec, Ku_vec, recovery, rho_vec, nu_opt, rho_model_vec, p, discounts, dates)
%
% INPUTS:
%   model_type      -    'vasicek' or 'tstudent' (specifies the copula model)
%   use_shifted     -    logical flag for plotting shifted tranches
%   Kd_vec          -    vector of lower attachment points
%   Ku_vec          -    vector of upper detachment points
%   recovery        -    recovery rate
%   rho_vec         -    vector of market correlations (for Vasicek model)
%   nu_opt          -    degrees of freedom (used for t-Student model)
%   rho_model_vec   -    vector of model-implied correlations (for t-Student)
%   p               -    default probability
%   discounts       -    discount factor curve
%   dates           -    dates corresponding to discount factors

function plotTranchePricesModel(model_type, use_shifted, Kd_vec, Ku_vec, recovery, rho_vec, nu_opt, rho_model_vec, p, discounts, dates)

    I_values = floor(logspace(1, log10(1000), 12));  % Portfolio sizes
    n_tranches = length(Kd_vec);

    % Generate tranche labels as percentages
    tranche_labels = strcat(string(Kd_vec * 100), '%–', string(Ku_vec * 100), '%');

    % Use shifted (cumulative) representation if flag is active
    if use_shifted
        tranche_bounds = Ku_vec * 100;
        tranche_labels = strcat(string([0, tranche_bounds(1:end-1)]), '%–', string(tranche_bounds), '%');
    end

    % Initialize price matrices [#I x #tranches]
    price_exact_all = zeros(length(I_values), n_tranches);
    price_LHP_all   = zeros(length(I_values), n_tranches);
    price_KL_all    = zeros(length(I_values), n_tranches);

    % Loop over all obligor sizes
    for idx = 1:length(I_values)
        I_curr = I_values(idx);

        switch lower(model_type)
            case 'vasicek'
                % Use market rho for all tranches
                rho_vec = rho_vec(1:n_tranches);
                price_exact = Price_HP_Vasicek(Kd_vec, Ku_vec, recovery, I_curr, rho_vec, p, discounts, dates);
                price_LHP   = Price_LHP_Vasicek(Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates);
                price_KL    = Price_KL_Vasicek(Kd_vec, Ku_vec, recovery, I_curr, rho_vec, p, discounts, dates);

            case 'tstudent'
                % Use calibrated rho for all tranches
                rho_vec = rho_model_vec(1:n_tranches);
                price_exact = Price_HP_tstud(Kd_vec, Ku_vec, nu_opt, recovery, I_curr, rho_vec, p, discounts, dates);
                price_LHP   = Price_LHP_tstud(nu_opt, Kd_vec, Ku_vec, recovery, rho_vec, p, discounts, dates);
                price_KL    = Price_KL_tstud(Kd_vec, Ku_vec, nu_opt, recovery, I_curr, rho_vec, p, discounts, dates);

            otherwise
                error('Model type must be "vasicek" or "tstudent"');
        end

        % Store or shift prices
        if use_shifted
            price_exact_all(idx, :) = shift_prices(price_exact(1:n_tranches), tranche_bounds);
            price_LHP_all(idx, :)   = shift_prices(price_LHP(1:n_tranches), tranche_bounds);
            price_KL_all(idx, :)    = shift_prices(price_KL(1:n_tranches), tranche_bounds);
        else
            price_exact_all(idx, :) = price_exact(1:n_tranches);
            price_LHP_all(idx, :)   = price_LHP(1:n_tranches);
            price_KL_all(idx, :)    = price_KL(1:n_tranches);
        end
    end

    % ---------------------- Plotting ----------------------
    figure; set(gcf, 'Position', [300, 100, 1200, 400]);

    if use_shifted
        sgtitle(['Tranche Prices under ', upper(model_type), ' Model with shifted tranches'], ...
                'FontSize', 20, 'FontWeight', 'bold');
    else
        sgtitle(['Tranche Prices under ', upper(model_type), ' Model'], ...
                'FontSize', 20, 'FontWeight', 'bold');
    end

    % One subplot per tranche
    for tranche_idx = 1:n_tranches
        subplot(1, n_tranches, tranche_idx);
        hold on; set(gca, 'XScale', 'log'); set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');

        % Plot each method
        semilogx(I_values, price_exact_all(:, tranche_idx), '-o', 'Color', 'g', ...
                 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'Exact');
        semilogx(I_values, price_KL_all(:, tranche_idx), '-o', 'Color', 'r', ...
                 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'KL');
        semilogx(I_values, price_LHP_all(:, tranche_idx), '-o', 'Color', 'b', ...
                 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'LHP');

        % Titles and labels
        title(['Tranche: ' tranche_labels{tranche_idx}], 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Number of obligors I (Log Scale)');
        ylabel('Normalized price');
        legend('Location', 'best', 'Box', 'on');
        grid on; set(gca, 'GridAlpha', 0.4); hold off;
    end
end
