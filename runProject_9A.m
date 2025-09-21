%% Project 9A: Multi-Name Credit Product --- Federico Crivellaro, Gaia Imperatore ---

clear; close all; clc; 
addpath(genpath(fullfile(pwd, 'Datas')));
addpath(genpath(fullfile(pwd, 'Pricing')));
addpath(genpath(fullfile(pwd, 'Calibrations')));
addpath(genpath(fullfile(pwd, 'Plots')));
addpath(genpath(fullfile(pwd, 'Utilities')));

load('cSelect.mat'); load('datesSet.mat'); load('ratesSet.mat'); load('Ku.mat'); load('rho.mat'); load('Kd.mat'); load('Kd_allzeros.mat');
formatData = 'dd/mm/yyyy';

    % Identify all .m files and dependencies required to execute the main script 'runProject.m'
deps = matlab.codetools.requiredFilesAndProducts('runProject_9A.m');
for i = 1:length(deps)
    fprintf('%s\n', deps{i});
end

%% Read market data (we loaded data above)
% [datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap', formatData);
% Start global timer
tStart = datetime('now');

%% Bootstrap the discount curve
% Construct the discount factors from the market zero-coupon rates and dates
[dates, discounts] = bootstrap(datesSet, ratesSet); 

%% INPUTS
I = 500;                          % Total number of mortgages in the portfolio
p = 0.06;                         % Individual default probability for each mortgage
recovery = 0.40;                  % Average recovery rate on defaulted mortgages
LGD = 1 - recovery;               % Loss given default
single_notional = 2e6;            % Notional exposure per mortgage
notional = single_notional * I;   % Total portfolio notional amount
maturity = 4;                     % Maturity horizon in years

%% Useful variables for the analysis
Ku_e = Ku(1);                     % Upper detachment point for equity tranche
rho_e = rho(1);                   % First Correlation (equity tranche)
Kd_e = 0;                         % Lower detachment for equity tranche (usually zero)
flag = 'LHP';                     % Model flag used in the function find_rho_implied
n = length(Ku);                   % Number of tranches considered in total
n_tranches = 3;                   % Only 3 tranches for some calculations as asked

today = datetime(datesSet.settlement, 'ConvertFrom', 'datenum');
date_maturity = Add_dates(today, maturity); 
date_maturity = date_maturity(end);     % Final maturity date
discount = interpolation_vector(dates, discounts, date_maturity, today); % Calculate discount factor interpolated at maturity

%% Question a): Calibrate the Double t-Student model
% Calibrate correlation parameter and degrees of freedom nu
params = struct('Ku_vec', Ku, 'rho_vec', rho, 'flag_nu', "true");
[rho_model, nu_opt, mse_opt] = calibration_model_parameters('double_t', params, Ku_e, recovery, rho_e, p, dates, discounts);
rho_model_vec = rho_model * ones(n, 1); % Model correlation parameter vector 

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated double t-student model 
price_tstudent = Price_LHP_tstud(nu_opt, Kd_allzeros, Ku, recovery, rho_model_vec, p, discounts, dates);
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_doublet = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_tstudent, flag);

%% EXTRA ANALYSIS: Calibration of Double t-Student model with two different degrees of freedom 
% calibrate nu_M and nu_Zi separately to minimize the MSE between model and market prices
params = struct('Ku_vec', Ku, 'rho_vec', rho, 'flag_nu', "true");
[rho_model_2, nu_M_opt, nu_Zi_opt, mse_opt_2] = calibration_model_parameters('double_t_diff', params, Ku_e, recovery, rho_e, p, dates, discounts);
rho_model_2_vec = rho_model_2 * ones(n, 1); % Model correlation parameter vector 

% Compute tranche prices from calibrated double t-student model with 2 different nu 
price_tstudent_diff = Price_LHP_tstud_diff(nu_M_opt, nu_Zi_opt, Kd_allzeros, Ku, recovery, rho_model_2_vec, p, discounts, dates);

% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_2 = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_tstudent_diff, flag);

%% Question b): Pricing comparison
%% 1) Exact pricing under Double t-Student and Vasicek models
warning('off');
% Compute exact prices using the Homogeneous Portfolio assumption (HP)
price_vasicek_exact = Price_HP_Vasicek(Kd_allzeros, Ku, recovery, I, rho, p, discounts, dates);
price_tstudent_exact = Price_HP_tstud(Kd_allzeros, Ku, nu_opt, recovery, I, rho_model_vec, p, discounts, dates);

%% 2) KL approximation for Double t-Student and Vasicek models
% Compute prices using the Kullback-Leibler (KL) approximation for faster evaluation
price_vasicek_KL = Price_KL_Vasicek(Kd_allzeros, Ku, recovery, I, rho, p, discounts, dates);
price_tstudent_KL = Price_KL_tstud(Kd_allzeros, Ku, nu_opt, recovery, I, rho_model_vec, p, discounts, dates);

%% 3) Large Homogeneous Portfolio (LHP) approximation for Double t-Student and Vasicek
% Compute prices using the LHP approximation
price_vasicek = Price_LHP_Vasicek(Kd_allzeros, Ku, recovery, rho, p, discounts, dates);
price_tstudent = Price_LHP_tstud(nu_opt, Kd_allzeros, Ku, recovery, rho_model_vec, p, discounts, dates);

%% Compare prices: Vasicek vs Double t-Student
% Summarize all calculated tranche prices in a table for comparison
tranche_labels = ["0 - 3"; "0 - 6"; "0 - 9"; "0 - 12"; "0 - 22"];
T = table(tranche_labels, price_vasicek_exact, price_vasicek, price_vasicek_KL, price_tstudent_exact, price_tstudent, price_tstudent_KL, 'VariableNames', {'Tranche', 'HP_Vasicek', 'LHP_Vasicek', 'KL_Vasicek', 'HP_tStudent', 'LHP_tStudent', 'KL_tStudent'});
disp('Price Comparison: Vasicek vs t-Student (HP, LHP, KL):');
disp(T);

%% Plots: Tranche prices with respect to the number of obligors I 
% VASICEK: cumulative and not cumulative tranches
plotTranchePricesModel('vasicek', false, Kd_allzeros, Ku, recovery, rho, nu_opt, rho_model_vec, p, discounts, dates);
plotTranchePricesModel('vasicek', true, Kd_allzeros, Ku, recovery, rho, nu_opt, rho_model_vec, p, discounts, dates);

% DOUBLE T-STUDENT: cumulative and not cumulative tranches
plotTranchePricesModel('tstudent', false, Kd_allzeros, Ku, recovery, rho, nu_opt, rho_model_vec, p, discounts, dates);
plotTranchePricesModel('tstudent', true, Kd_allzeros, Ku, recovery, rho, nu_opt, rho_model_vec, p, discounts, dates);

%% EQUITY TRANCHE
% Compute equity tranche prices with HP, (standard) KL, LHP solution across different I  
I_values = floor(logspace(1, log10(10000), 12));
price_eq_exact = arrayfun(@(i)  Price_HP_tstud(Kd_e, Ku_e, nu_opt, recovery, i, rho_model, p, discounts, dates), I_values); 
price_eq_KL = arrayfun(@(i)  Price_KL_tstud(Kd_e, Ku_e, nu_opt, recovery, i, rho_model, p, discounts, dates), I_values);
price_eq_LHP = Price_LHP_tstud(nu_opt, Kd_e, Ku_e, recovery, rho_model, p, discounts, dates);

% Plot equity tranche prices under the double t-Student model 
plotTranchePricesModel('tstudent', false, Kd_e, Ku_e, recovery, rho, nu_opt, rho_model_vec, p, discounts, dates);

%% Alternative method for KL
% Find the equity price subtracting the total portfolio value by all tranches except the equity (from 3% to 100%)
[price_KL_equity_new, price_up_KL, price_ptf] = compute_KL_equity_alternative(I_values, Ku_e, Kd_e, nu_opt, recovery, rho_model, p, discounts, dates, notional, discount);
% Plot a comparison of different equity tranche pricing methods
plot_KL_equity_comparison(I_values, price_eq_exact, price_eq_KL, price_KL_equity_new, price_eq_LHP);
error_between_KL_methods(Kd_e, Ku_e, nu_opt, recovery, I, rho_model, p, discounts, dates, notional, discount);

%% Question c): KL calibration
% Calibrate correlation parameter and degrees of freedom nu
params = struct('Ku_vec', Ku, 'rho_vec', rho, 'I', I, 'flag_nu', "true");
[rho_model_KL, nu_KL_opt, mse_KL_opt] = calibration_model_parameters('KL', params, Ku_e, recovery, rho_e, p, dates, discounts);
rho_model_KL_vec = rho_model_KL * ones(n, 1);  % Model correlation parameter vector 
disp(rho_model_KL)

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated double t-student model with KL approximation
price_tstudent_KL_calibrated = Price_KL_tstud(Kd_allzeros, Ku, nu_opt, recovery, I, rho_model_KL_vec, p, discounts, dates);
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_KL = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_tstudent_KL_calibrated, 'KL');

%% Question d): Price tranches (0-3, 0-6, 0-9) with Vasicek model using correlation rho equal to the implied correlation of the equity tranche.
% Use the implied rho from the equity tranche to price the other tranches
rho_vec_eq = rho_e * ones(n_tranches, 1);
error_between_LHP_with_different_rhos(Ku(1:n_tranches), recovery, rho(1:n_tranches), rho_e, p, discounts, dates);
%% Question e): Gaussian Copula
Nsim = 1e6; % Number of Monte Carlo simulations
rng('default');  % Set random seed for reproducibility
% Compare Gaussian copula pricing model against exact pricing
[nMSE_gaussianmodel, nMSE_gaussianmarket, price_copula_model, price_copula_correct, IC_gaussianmodel, IC_gaussianmarket] = compareCopulaVsExact(Nsim, discount, rho_model, rho, p, Kd_allzeros(1:n_tranches), Ku(1:n_tranches), I, recovery, discounts, dates, tranche_labels);

%% Calibration of Rho for Gaussian copula
params = struct('Nsim', Nsim, 'discount', discount, 'I', I);
% Calibrate correlation parameter for the Gaussian copula
rho_gaussian = calibration_model_parameters('gaussian_copula', params, Ku_e, recovery, rho_e, p, dates, discounts);

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated Gaussian copula model 
price_gaussian_copula = zeros(n, 1);
for i = 1:n
    price_gaussian_copula(i) = tranchePriceMC_GaussianCopula(Nsim, discount, rho_gaussian, p, Kd_allzeros(i), Ku(i), I, recovery);
end
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_gaussian = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_gaussian_copula, flag);

%% EXTRA ANALYSIS: t-Copula
Nsim = 1e4; % Number of Monte Carlo simulations for double t copula and Clayton copula
% Compare t-Copula pricing model against exact pricing
[mse_model, mse_market, price_tcopula_model, price_tcopula_correct, IC_tcopulamodel, IC_tcopulamarket] = compareTCopulaVsExact(tranche_labels, Kd_allzeros(1:n_tranches), Ku(1:n_tranches), p, rho, rho_model, recovery, I, nu_opt, discounts, dates, Nsim, discount);

%% Calibration of Rho for t-Student copula

params = struct('Nsim', Nsim, 'discount', discount, 'I', I, 'nu', nu_opt);
% Calibrate correlation for the t-Student copula, taking into account degrees of freedom nu_opt
rho_tstudent = calibration_model_parameters('t_student_copula', params, Ku_e, recovery, rho_e, p, dates, discounts);

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated Gaussian copula model 
price_tstudent_copula = zeros(n, 1);
for i = 1:n
    price_tstudent_copula(i) = tranchePriceMC_tCopula(Nsim, discount, rho_tstudent, p, Kd_allzeros(i), Ku(i), I, recovery, nu_opt);
end
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_tstudent = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_tstudent_copula, flag);

%% EXTRA ANALYSIS: Archimedean Copula - Clayton

%% Calibration for theta for Clayton Copula
params = struct('Nsim', Nsim, 'discount', discount, 'I', I);
theta_opt = calibration_model_parameters('archimedean_copula_clayton', params, Ku_e, recovery, rho_e, p, dates, discounts);

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated Clayton copula model 
price_clayton = zeros(size(Kd_allzeros));
for i = 1:n
    price_clayton(i) = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd_allzeros(i), Ku(i), I, recovery, theta_opt);
end
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_clayton = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_clayton, flag);

%% Alternative approach: find optimal theta by minimizing MSE across all tranches
obj = @(theta) obj_theta(price_vasicek, Nsim, discount, p, Kd_allzeros, Ku, I, recovery, theta);
theta_opt_mse = fminbnd(obj, 0, 10); 

%% Find implied correlation (rho) for each cumulative tranche from model prices
% Compute tranche prices from calibrated Clayton copula model
price_clayton_mse = zeros(size(Kd_allzeros));
for i = 1:n
    price_clayton_mse(i) = tranchePriceMC_ArchimedeanCopula(Nsim, discount, p, Kd_allzeros(i), Ku(i), I, recovery, theta_opt_mse);
end
% Find implied rho that equate market price to the calibrated model price (invert Vasicek)
rho_impl_clayton_mse = find_rho_implied(Kd_allzeros, Ku, recovery, I, rho, p, dates, discounts, price_clayton_mse, flag);

%% Plot - Clayton Copula with the 2 different approaches 
x_labels = {'0-3', '0-6', '0-9', '0-12', '0-22'}; x = 1:5; figure; set(gcf, 'Color', 'w'); hold on; set(gca, 'Color', 'w'); 
plot(x, rho, 'o-', 'Color', 'k', 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'Market \rho');
plot(x, rho_impl_clayton, 'o-', 'Color', 'b', 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'Implied \rho Clayton Copula'); % Clayton - standard approach
plot(x, rho_impl_clayton_mse, 'o-', 'Color', 'm', 'LineWidth', 2.4, 'MarkerSize', 5,'DisplayName', 'Implied \rho Clayton Copula Alternative Approach'); % Clayton - alternative approach
xticks(x); xticklabels(x_labels); xlim([0.5, 5.5]); ylim padded; xlabel('Tranches', 'FontSize', 12); ylabel('Correlation (\rho)', 'FontSize', 12); legend('Location', 'best', 'Box', 'on'); 
title('Clayton Copula - Implied Correlations', 'FontSize', 14, 'FontWeight', 'bold'); grid on; hold off;

%% Global comparison plot
figure; set(gcf, 'Color', 'w'); hold on; set(gca, 'Color', 'w'); 
plot(x, rho, 'o-', 'Color', 'k', 'LineWidth', 2.4, 'MarkerSize', 5, 'DisplayName', 'Market \rho');
plot(x, rho_impl_clayton, 'o-', 'Color', 'g', 'LineWidth', 3, 'MarkerSize', 5, 'DisplayName', 'Implied \rho Clayton Copula');
plot(x, rho_impl_gaussian, 'o-', 'Color', 'm', 'LineWidth', 3, 'MarkerSize', 5, 'DisplayName', 'Implied \rho Gaussian Copula');
plot(x, rho_impl_tstudent, 'o-', 'Color', 'c', 'LineWidth', 3, 'MarkerSize', 5, 'DisplayName', 'Implied \rho t-Student Copula');
plot(x, rho_impl_doublet, 'o-', 'Color', 'b', 'LineWidth', 3, 'MarkerSize', 5, 'DisplayName', 'Implied \rho Double t-Student');
xticks(x); xticklabels(x_labels); xlim([0.5, 5.5]); ylim padded; xlabel('Tranches', 'FontSize', 12); ylabel('Correlation (\rho)', 'FontSize', 12); legend('Location', 'best', 'Box', 'on');
title('Comparison of Market and Implied Correlations', 'FontSize', 14, 'FontWeight', 'bold'); grid on; hold off;

%% Check for uniqueness and stability of the solution
% For each tranche, plot price as function of rho under Vasicek and compare with Clayton price 
checkClaytonVsVasicek(Kd_allzeros, Ku, recovery, p, discounts, dates, price_clayton);

%% Stop timer and compute execution time 
tEnd = datetime('now');
disp(tEnd - tStart) 