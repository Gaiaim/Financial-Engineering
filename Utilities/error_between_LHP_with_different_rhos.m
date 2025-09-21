%% FUNCTION: error_between_LHP_with_different_rhos

% Computes and visualizes the pricing error introduced by using a single equity-implied correlation (rho_e) instead of tranche-specific market-implied 
% correlations (rho) when pricing tranches under the Vasicek LHP model.
% It also plots how tranche prices vary as a function of correlation (rho).
%
% INPUTS:
%   Ku         : Vector of upper detachment points for each tranche
%   recovery   : Recovery rate
%   rho        : Vector of tranche-specific market-implied correlations
%   rho_e      : Scalar equity-implied correlation
%   p          : Vector of default probabilities for each time period
%   discounts  : Discount factors for each time period
%   dates      : Corresponding dates for the default probabilities


function error_between_LHP_with_different_rhos(Ku, recovery, rho, rho_e, p, discounts, dates)

    n_tranches = length(Ku);  % Number of tranches
    Kd_allzeros = zeros(size(Ku));  % Lower attachment points
    rho_vec_eq = ones(size(Ku)) * rho_e;  % Vector of rho_e 

    % Prices using equity-implied correlation for all tranches
    price_LHP_wrong = Price_LHP_Vasicek(Kd_allzeros, Ku, recovery, rho_vec_eq, p, discounts, dates);

    % Prices using market-implied correlation for each tranche
    price_LHP_correct = Price_LHP_Vasicek(Kd_allzeros, Ku, recovery, rho, p, discounts, dates);

    % Compute relative percentage error for each tranche
    relative_error = abs(price_LHP_wrong - price_LHP_correct) ./ price_LHP_correct * 100;

    % Print errors
    rho_char = char(961);  % Unicode for Greek letter rho
    fprintf('\nRelative errors between LHP prices with %s_{mkt} and %s_e for all tranches:\n', rho_char, rho_char);
    fprintf('--------------------------------------------------------------------------\n');
    fprintf('%-18s %-18s %-18s %-15s\n', ...
            'Tranche', sprintf('Price (%s_e)', rho_char), sprintf('Price (%s_{mkt})', rho_char), 'Rel. Error (%)');
    fprintf('--------------------------------------------------------------------------\n');
    
    for i = 1:n_tranches
        fprintf('[%2.0f%% - %2.0f%%]         %-.6f           %-.6f           %-.4f\n', ...
            Kd_allzeros(i)*100, Ku(i)*100, ...
            price_LHP_wrong(i), price_LHP_correct(i), relative_error(i));
    end
    
    fprintf('--------------------------------------------------------------------------\n\n');

    % plot
    rho_grid = linspace(0, 1, 100);
    prices_all = zeros(n_tranches, length(rho_grid));

    % For each rho value in the grid, compute tranche prices assuming same rho for all
    for j = 1:length(rho_grid)
        rho_temp = ones(size(Ku)) * rho_grid(j); 
        price_temp = Price_LHP_Vasicek(Kd_allzeros, Ku, recovery, rho_temp, p, discounts, dates);
        prices_all(:, j) = price_temp;
    end
    color_order = lines(n_tranches); figure; hold on;
    legend_handles = gobjects(2 * n_tranches, 1);
    legend_labels = cell(2 * n_tranches, 1);
    for i = 1:n_tranches
        h_curve = plot(rho_grid, prices_all(i, :), 'LineWidth', 2, 'Color', color_order(i, :));
        h_rho = xline(rho(i), '--', 'LineWidth', 1.5, 'Color', color_order(i, :));
       
        legend_handles(2*i - 1) = h_curve;
        legend_handles(2*i)     = h_rho;
        tranche_label = sprintf('Tranche price [0%% - %d%%]', round(Ku(i)*100));
        rho_label = ['rho\_mkt for [0% - ' num2str(round(Ku(i)*100)) '%]'];
        legend_labels{2*i - 1} = tranche_label;
        legend_labels{2*i}     = rho_label;
    end

    xlabel('$\rho$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Tranche Price', 'FontSize', 12);
    title('LHP tranche prices vs correlation $\rho$', 'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0 1]);
    legend(legend_handles, legend_labels, 'Interpreter', 'latex', 'Location', 'best');
    hold off;

end
    