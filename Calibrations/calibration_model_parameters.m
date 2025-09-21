%% FUNCTION: calibration_model_parameters
% Calibrate copula-based model parameters to match the market price of an equity tranche.
%
% [param_opt, output_extra1, output_extra2, output_extra3] = calibration_model_parameters(modelName, params, Ku, recovery, rho, p, dates, discounts)
%
% INPUTS:
%   modelName     -    name of the model to calibrate ('double_t', 'double_t_diff', 'KL', etc.)
%   params        -    structure with model-specific parameters (e.g., nu, Nsim, I, Ku_vec, etc.)
%   Ku            -    detachment point of the equity tranche
%   recovery      -    recovery rate
%   rho           -    (initial) correlation level for Vasicek pricing
%   p             -    default probability
%   dates         -    dates for the discount curve
%   discounts     -    discount factors
%
% OUTPUTS:
%   param_opt     -    calibrated parameter(s): rho, nu, theta or vector of correlations
%   output_extra1 -    optional output: calibrated nu or theta
%   output_extra2 -    optional output: secondary parameter or MSE
%   output_extra3 -    optional output: final MSE (when relevant)

function [param_opt, output_extra1, output_extra2, output_extra3] = calibration_model_parameters(modelName, params, Ku, recovery, rho, p, dates, discounts)

    % Compute market price of equity tranche using Vasicek (Gaussian Copula)
    price_equity_mkt = Price_LHP_Vasicek(0, Ku, recovery, rho, p, discounts, dates);
    x0 = [0, 1];  % Initial interval for fzero

    switch modelName

        % ----------- DOUBLE T-COPULA (common ν) -----------
        case 'double_t'
            if params.flag_nu == "true"
                % Setup attachment points (zero lower bounds)
                Kd_allzeros = zeros(size(params.Ku_vec));

                % Objective: minimize MSE over ν
                obj_fun = @(nu) obj_nu(nu, Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts);

                % Minimize using bounded scalar optimizer
                options = optimset('Display', 'off');
                nu_opt = fminbnd(obj_fun, 2, 100, options);

                % Retrieve optimal correlation structure and final error
                [mse_opt, param_opt] = obj_nu(nu_opt, Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts);
                output_extra1 = nu_opt;
                output_extra2 = mse_opt;

                % Plot calibration error vs ν
                nu_vals = linspace(2, 100, 50);
                mse_vals = arrayfun(obj_fun, nu_vals);
                figure; set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
                plot(nu_vals, mse_vals, '-o', 'LineWidth', 1.5, 'Color', 'b');
                xlabel('Degrees of freedom \nu'); ylabel('MSE'); title('Calibration error vs \nu'); grid on;

                % Print summary
                fprintf('\n>> Calibration:\n');
                fprintf(' ν optimal (degrees of freedom) = %.4f\n Mean Squared Error = %.6f\n', nu_opt, mse_opt);
                return
            else
                % If ν is fixed, calibrate rho to match market price
                nu = params.nu;
                fun = @(r) Price_LHP_tstud(nu, 0, Ku, recovery, r, p, discounts, dates) - price_equity_mkt;
            end

        % ----------- DOUBLE T-COPULA (different ν_M and ν_Zi) -----------
        case 'double_t_diff'
            if params.flag_nu == "true"
                Kd_allzeros = zeros(size(params.Ku_vec));

                % Objective: MSE based on (ν_M, ν_Zi)
                obj_fun = @(nu_vec) obj_nu_diff(nu_vec(1), nu_vec(2), Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts);

                % Initial guess and bounds
                x0 = [10, 10]; lb = [2, 2]; ub = [100, 100];
                options = optimset('Display', 'off');

                % Run constrained optimization
                [nu_opt_vec, mse_opt_2] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);
                [mse_opt, rho_model] = obj_nu_diff(nu_opt_vec(1), nu_opt_vec(2), Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts);

                param_opt = rho_model;
                output_extra1 = nu_opt_vec(1);  % ν_M
                output_extra2 = nu_opt_vec(2);  % ν_Zi
                output_extra3 = mse_opt;

                fprintf('\n>> Calibration:\n');
                fprintf(' ν_M optimal = %.4f\n ν_Zi optimal = %.4f\n Mean Squared Error = %.6f\n', nu_opt_vec(1), nu_opt_vec(2), mse_opt_2);
                return
            else
                nu_M = params.nu_M;
                nu_Zi = params.nu_Zi;
                fun = @(r) obj_nu_diff(nu_M, nu_Zi, 0, Ku, p, recovery, r, dates, discounts) - price_equity_mkt;
            end

        % ----------- KL APPROXIMATION MODEL -----------
        case 'KL'
            Kd_allzeros = zeros(size(params.Ku_vec));
            obj_fun_KL = @(nu) obj_nu_KL(nu, Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts, params.I);

            options = optimset('Display', 'off');
            [nu_opt, mse_opt] = fminbnd(obj_fun_KL, 2, 100, options);

            % Plot calibration MSE vs ν
            nu_vals_KL = linspace(2, 100, 50);
            mse_vals_KL = arrayfun(obj_fun_KL, nu_vals_KL);
            figure; set(gcf, 'Color', 'w');
            plot(nu_vals_KL, mse_vals_KL, '-o', 'LineWidth', 1.5, 'Color', 'b');
            xlabel('Degrees of Freedom \nu'); ylabel('MSE');
            title('MSE vs \nu (KL Approximation)');
            grid on;

            [~, rho_model] = obj_nu_KL(nu_opt, Kd_allzeros, params.Ku_vec, p, recovery, params.rho_vec, dates, discounts, params.I);
            param_opt = rho_model;
            output_extra1 = nu_opt;
            output_extra2 = mse_opt;

            fprintf('\n>> Calibration:\n');
            fprintf(' ν optimal (degrees of freedom) = %.4f\n Mean Squared Error = %.6f\n', nu_opt, mse_opt);
            return

        % ----------- GAUSSIAN COPULA MODEL -----------
        case 'gaussian_copula'
            fun = @(r) tranchePriceMC_GaussianCopula(params.Nsim, params.discount, r, p, 0, Ku, params.I, recovery) - price_equity_mkt;
            opts = optimset('Display', 'off');
            param_opt = fzero(fun, x0, opts);

            % Plot fitted vs market price
            f_plot = @(r) tranchePriceMC_GaussianCopula(params.Nsim, params.discount, r, p, 0, Ku, params.I, recovery);
            rho_range = linspace(0, 1, 50); values = arrayfun(f_plot, rho_range);
            figure; hold on;
            plot(rho_range, values, 'm', 'LineWidth', 2.4);
            yline(price_equity_mkt, '-b', 'LineWidth', 2.4);
            plot(param_opt, f_plot(param_opt), 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
            xlabel('\rho'); ylabel('Price Equity tranche');
            title('Gaussian Copula Calibration');
            legend('Model Price', 'Market Price', 'Location', 'southeast'); grid on; hold off;

        % ----------- T-STUDENT COPULA MODEL -----------
        case 't_student_copula'
            nu = params.nu;
            fun = @(r) tranchePriceMC_tCopula(params.Nsim, params.discount, r, p, 0, Ku, params.I, recovery, nu) - price_equity_mkt;
            opts = optimset('Display', 'off');
            param_opt = fzero(fun, x0, opts);

            f_plot = @(r) tranchePriceMC_tCopula(params.Nsim, params.discount, r, p, 0, Ku, params.I, recovery, nu);
            rho_range = linspace(0, 1, 50); values = arrayfun(f_plot, rho_range);
            figure; hold on;
            plot(rho_range, values, 'm', 'LineWidth', 2.4);
            yline(price_equity_mkt, '-b', 'LineWidth', 2.4);
            plot(param_opt, f_plot(param_opt), 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
            xlabel('\rho'); ylabel('Price Equity tranche');
            title('t-Student Copula Calibration');
            legend('Model Price', 'Market Price', 'Location', 'southeast'); grid on; hold off;

        % ----------- ARCHIMEDEAN (CLAYTON) COPULA MODEL -----------
        case 'archimedean_copula_clayton'
            fun = @(theta) tranchePriceMC_ArchimedeanCopula(params.Nsim, params.discount, p, 0, Ku, params.I, recovery, theta) - price_equity_mkt;
            opts = optimset('Display', 'off');
            theta_opt = fzero(fun, [0.01, 10], opts);
            param_opt = theta_opt;

            f_plot = @(theta) tranchePriceMC_ArchimedeanCopula(params.Nsim, params.discount, p, 0, Ku, params.I, recovery, theta);
            theta_range = linspace(0.001, 2, 50); values = arrayfun(f_plot, theta_range);
            figure; hold on;
            plot(theta_range, values, 'm', 'LineWidth', 2.4);
            yline(price_equity_mkt, '-b', 'LineWidth', 2.4);
            plot(theta_opt, f_plot(theta_opt), 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
            xlabel('\theta'); ylabel('Price Equity tranche');
            title('Clayton Copula Calibration');
            legend('Model Price', 'Market Price', 'Location', 'southeast'); grid on; hold off;
            return

        otherwise
            error('Unknown model name');
    end

    % If fallthrough: perform last optimization
    opts = optimset('Display', 'off');
    param_opt = fzero(fun, x0, opts);
end
