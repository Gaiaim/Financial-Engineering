%% FUNCTION: checkClaytonVsVasicek
% Function to compare tranche prices between Vasicek (as a function of ρ) and Clayton (fixed)
% INPUTS:
%   Kd_allzeros     - lower detachment points
%   Ku              - upper detachment points
%   recovery        - recovery rate
%   p               - default probabilities
%   discounts       - discount factors
%   dates           - vector of dates
%   price_clayton   - tranche prices from Clayton copula

function checkClaytonVsVasicek(Kd_allzeros, Ku, recovery, p, discounts, dates, price_clayton)

    rho_vals = linspace(0, 0.99, 200); 
    n = length(Kd_allzeros); 
    tranche_labels = {'0-3', '0-6', '0-9', '0-12', '0-22'};

    figure; 
    set(gcf, 'Color', 'w'); 
    set(gcf, 'Position', [100, 100, 900, 600]); 
    hold on; 
    set(gca, 'Color', 'w');

    for i = 1:n
        prices = arrayfun(@(r) Price_LHP_Vasicek(Kd_allzeros(i), Ku(i), recovery, r, p, discounts, dates), rho_vals);
        subplot(2, 3, i); 
        plot(rho_vals, prices, 'Color', 'b', 'LineWidth', 2.4); 
        hold on; 
        yline(price_clayton(i), '--m', 'LineWidth', 2.4);
        legend('Price tranche Vasicek', 'Price tranche Clayton', 'Location', 'best'); 
        xlabel('\rho'); ylabel('Price'); 
        title(sprintf('Tranche %s', tranche_labels{i})); 
        grid on; axis square;
    end

    hold off;
end
