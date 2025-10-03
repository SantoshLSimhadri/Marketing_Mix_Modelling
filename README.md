# Marketing_Mix_Modelling

Overview

This project demonstrates an end-to-end Marketing Mix Modeling (MMM) pipeline built in Python. It uses synthetic weekly data to simulate marketing investments across multiple channels (TV, Search, Social, Display, Email) and models their impact on sales using statistical and machine learning approaches.

The pipeline includes feature engineering (adstock & saturation), model training, performance evaluation, channel contribution analysis, and budget optimization.

Objectives

Measure and optimize marketing performance.

Estimate channel contributions (incremental sales impact).

Provide ROI insights for each marketing channel.

Recommend optimal budget allocations across channels.

Features

Data Simulation

Synthetic dataset of weekly sales, marketing spends, promotions, seasonality, and competitor effects.

Feature Engineering

Adstock transformation (carryover effect).

Hill function (saturation effect).

Modeling Approaches

Ridge Regression

Lasso Regression

Random Forest Regressor

Evaluation Metrics

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

Channel Contribution Analysis

Estimate incremental contribution of each marketing channel.

ROI proxy = contribution ÷ spend.

Budget Optimizer

Simulates incremental budget allocation.

Optimizes spend across channels to maximize predicted sales.


How to Run

Install dependencies:

pip install numpy pandas scikit-learn scipy

Run the project:

python mmm_from_scratch.py

Results will be saved to the outputs/ folder.

Key Outputs

Model Performance: Compare Ridge, Lasso, and RandomForest.

Channel ROI: Identify which channels drive the most efficient growth.

Budget Recommendations: Data-driven strategy for incremental spend allocation.Model Performance: Compare Ridge, Lasso, and RandomForest.

Channel ROI: Identify which channels drive the most efficient growth.

Budget Recommendations: Data-driven strategy for incremental spend allocation.
