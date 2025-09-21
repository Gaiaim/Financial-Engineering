%% FUNCTION: plot_KL_equity_comparison
% Visual comparison between equity tranche prices using KL approximation:
% exact, alternative, and LHP methods; includes error analysis.
%
% plot_KL_equity_comparison(I_values, price_eq_exact, price_eq_KL, price_KL_equity_new, price_eq_LHP)
%
% INPUTS:
%   I_values             -    vector of portfolio sizes (number of mortgages)
%   price_eq_exact       -    vector of exact equity tranche prices (HP or benchmark)
%   price_eq_KL          -    vector of equity prices from standard KL method
%   price_KL_equity_new  -    vector of equity prices from alternative KL method
%   price_eq_LHP         -    scalar or vector of LHP equity price for comparison
%
% OUTPUTS:
%   (none)               -    generates two comparative figures with price and error plots

function plot_KL_equity_comparison(I_values, price_eq_exact, price_eq_KL, price_KL_equity_new, price_eq_LHP)

    % --------- Figure 1: Equity prices comparison (Exact vs KL vs LHP) ----------
    figure(); set(gcf, 'Color', 'w'); hold on; set(gca, 'Color', 'w'); set(gca, 'XScale', 'log');
    semilogx(I_values, price_eq_exact, 'g-o', 'LineWidth', 2.4, 'MarkerSize', 5); 
    semilogx(I_values, price_KL_equity_new, 'r-o', 'LineWidth', 2.4, 'MarkerSize', 5);
    semilogx(I_values, price_eq_LHP * ones(1, length(I_values)), 'b-o', 'LineWidth', 2.4, 'MarkerSize', 5);
    legend('Exact price', 'KL price (Alternative)', 'LHP price');
    ylabel('Prices'); xlabel('Number of mortgages');
    title('Percentage Prices of Equity Tranche with Alternative Method', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; hold off;

    % --------- Error computation: relative to exact ---------
    error_eq = abs(price_eq_KL(1:length(I_values)) - price_eq_exact) ./ price_eq_exact;
    error_eq_alternative = abs(price_KL_equity_new(1:length(I_values)) - price_eq_exact) ./ price_eq_exact;

    % --------- Figure 2: Prices & percentage errors (Standard vs Alternative KL) ----------
    figure; set(gcf, 'Color', 'w'); set(gca, 'Color', 'w'); 

    % --- Subplot 1: Equity prices ---
    subplot(2,1,1);
    semilogx(I_values, price_eq_KL, '-o', 'Color', 'b', 'LineWidth', 2.4, 'MarkerSize', 5); hold on;
    semilogx(I_values, price_KL_equity_new, '-o', 'Color', 'm', 'LineWidth', 2.4, 'MarkerSize', 5);
    legend('Price KL equity Standard Method', 'Price KL equity Alternative Method');
    xlabel('Number of mortgages'); ylabel('Prices');
    title('Price KL for Equity Tranche: Standard vs Alternative Method', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;

    % --- Subplot 2: Percentage errors ---
    subplot(2,1,2);
    semilogx(I_values, error_eq, '--', 'Color', 'b', 'LineWidth', 2.4, 'MarkerSize', 5); hold on;
    semilogx(I_values, error_eq_alternative, '--', 'Color', 'm', 'LineWidth', 2.4, 'MarkerSize', 5);
    legend('Error Standard method', 'Error Alternative method');
    xlabel('Number of mortgages'); ylabel('Errors');
    title('Percentage Errors of KL solution for Equity Tranche: Standard vs Alternative Method', ...
        'FontSize', 14, 'FontWeight', 'bold');
    grid on; hold off;

end
