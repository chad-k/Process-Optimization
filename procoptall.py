# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:29:42 2025
@author: ckaln
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from scipy.optimize import minimize

# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv("C:\\Users\\chad\\Documents\\AI\\synthetic_process_data.csv")
specs = pd.read_csv("C:\\Users\\chad\\Documents\\AI\\spec_limits.csv")

# Merge specs into main data
data = data.merge(specs, on="part_number", how="left")

# ------------------------------
# Optimization Function
# ------------------------------
def optimize_parameters(model, X, y, target, bounds):
    model.fit(X, y)

    def objective(params):
        prediction = model.predict([params])[0]
        return (prediction - target) ** 2

    initial_guess = [np.mean([b[0], b[1]]) for b in bounds]
    result = minimize(objective, x0=initial_guess, bounds=bounds)

    if result.success:
        optimized = result.x
        predicted = model.predict([optimized])[0]
        return optimized, predicted
    else:
        return None, None

# ------------------------------
# Main Optimization Loop
# ------------------------------
results = []
bounds = [(170, 200), (45, 60), (1100, 1300)]  # temperature, pressure, speed
soft_tol_percent = 0.10  # ±10% soft band

model_choices = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

for (part, machine), group in data.groupby(["part_number", "machine_number"]):
    X = group[["temperature", "pressure", "speed"]].values
    y = group["measurement"].values
    target = group["target"].iloc[0]
    lsl = group["lower_spec_limit"].iloc[0]
    usl = group["upper_spec_limit"].iloc[0]

    best_result = None
    lowest_error = float("inf")

    for model_name, model in model_choices.items():
        optimized, predicted = optimize_parameters(model, X, y, target, bounds)
        if optimized is not None:
            error = abs(predicted - target)
            if error < lowest_error:
                lowest_error = error
                soft_lower = target * (1 - soft_tol_percent)
                soft_upper = target * (1 + soft_tol_percent)
                soft_pass = soft_lower <= predicted <= soft_upper

                best_result = {
                    "part_number": part,
                    "machine_number": machine,
                    "model_type": model_name,
                    "optimized_temperature": round(optimized[0], 2),
                    "optimized_pressure": round(optimized[1], 2),
                    "optimized_speed": round(optimized[2], 2),
                    "predicted_measurement": round(predicted, 3),
                    "target": target,
                    "abs_error": round(error, 3),
                    "percent_error": round(100 * error / target, 2),
                    "in_spec": lsl <= predicted <= usl,
                    "soft_pass": soft_pass
                }

    if best_result:
        results.append(best_result)
    else:
        print(f"⚠️ Optimization failed for {part} - {machine}")

# ------------------------------
# Save Results
# ------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["part_number", "machine_number"])
results_df.to_csv("optimized_machine_settings.csv", index=False)

# ------------------------------
# Summary
# ------------------------------
total = len(results_df)
in_spec_count = results_df["in_spec"].sum()
soft_pass_count = results_df["soft_pass"].sum()
avg_error = results_df["abs_error"].mean()

print(f"✅ Optimization complete. {in_spec_count}/{total} results are within spec ({100 * in_spec_count / total:.2f}%)")
print(f"✨ Soft pass (within ±10% of target): {soft_pass_count}/{total} ({100 * soft_pass_count / total:.2f}%)")
print(f"📦 Saved to 'optimized_machine_settings.csv'")
print(f"📉 Average absolute error: {avg_error:.3f}")
