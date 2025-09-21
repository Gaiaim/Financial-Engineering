# Final Project 9A – Financial Engineering

## Structured Credit: Tranche Pricing and Correlation Modeling

This repository contains the full implementation and documentation of the final assignment for the Financial Engineering course at Politecnico di Milano. The project focuses on pricing Collateralized Debt Obligation (CDO) tranches on a large homogeneous portfolio of mortgages, using various credit risk models and correlation structures.

The objective is to calibrate, analyze, and compare the following models:
- Double t-Student copula with real-valued degrees of freedom
- Vasicek (Gaussian one-factor) copula
- Li's Gaussian copula model with constant intensity (optional)

Model calibration is based on market-implied correlations provided via tranche quotes, and pricing accuracy is evaluated under the Large Homogeneous Portfolio (LHP) approximation. The project also includes an analysis of pricing sensitivity with respect to portfolio size and tranche bounds.

---

## 📁 Project Structure

```
Final_Project_9A/
├── code/           # MATLAB scripts for model calibration and pricing
├── data/           # Input data (e.g., TrancheCorrelations.xlsx)
├── report/         # LaTeX report and figures
├── runProject.m    # Main script to execute and reproduce results
└── README.md       # Project overview and instructions
```

---

## ▶️ How to Run the Project

1. Open `runProject.m` in MATLAB.
2. Ensure that `TrancheCorrelations.xlsx` is located in the `data/` folder.
3. Execute the script to:
   - Calibrate the double t-Student model to market tranche data.
   - Plot and compare pricing results under different approximations (Exact, KL, LHP).
   - Evaluate the effect of portfolio size on tranche pricing (log-scale analysis).

Optional Python implementations may be included for extended analysis and validation.

---

## 🔧 Requirements

- MATLAB R2022b or later
- Toolboxes:
  - Financial Toolbox
  - Statistics and Machine Learning Toolbox
- (Optional) Python 3.10+ with NumPy, SciPy, and matplotlib

---

## 📊 Key Features

- Calibration of correlation parameters and degrees of freedom to market data
- Visual and quantitative comparison between exact and approximate pricing
- Implementation of multiple models under a unified framework
- Plot generation for report inclusion

---

## 👥 Authors

- Federico Crivellaro  
- Gaia Imperatore  

---

## 🎓 Academic Information

Final assignment – *Financial Engineering*  
Politecnico di Milano, Academic Year 2024/2025  
Instructor: [Insert professor’s name]  
Submission email: `financial.engineering.polimi@gmail.com`

---

## 📄 License

This repository is intended solely for educational purposes related to the coursework of Financial Engineering at PoliMi. All rights reserved to the authors.
