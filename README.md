# Week 4 Task: Predictive Modeling and Optimization in Logistics Systems

## Project Overview
This project demonstrates a Python workflow for forecasting shipment delivery time and proposing data-driven ways to improve logistics operations. The dataset is simulated for educational purposes; the results should not be treated as evidence from a real logistics company.

## Objectives
- Simulate hypothetical shipment records.
- Explore logistics features and their relationships with delivery time.
- Compare a baseline, Linear Regression, and Random Forest Regressor.
- Evaluate predictions using MAE, RMSE, and R-squared.
- Use validation and hyperparameter tuning.
- Recommend operational actions and measurable KPIs.

## Dataset
The synthetic dataset includes distance, traffic level, weather severity, dispatch delay, package weight, number of stops, warehouse processing time, and delivery time (target, in hours).

## Tools
Python, pandas, NumPy, scikit-learn, and Matplotlib.

## Methodology
1. Generate reproducible synthetic shipment data.
2. Check data quality and explore the feature relationships.
3. Split data into training and testing sets.
4. Train a median baseline, Linear Regression, and Random Forest.
5. Compare MAE, RMSE, and R-squared on held-out data.
6. Tune Random Forest parameters using cross-validation.
7. Review residuals and propose logistics optimization actions.

## Evaluation Metrics
- **MAE:** Average absolute prediction error, in hours.
- **RMSE:** Error metric that penalizes larger misses more strongly.
- **R-squared:** Relative measure of how much target variation the model explains.

The code prints measured metrics when executed. No model scores are claimed in this README because they depend on running the code.

## Optimization Recommendations
- Flag shipments whose predicted delivery time exceeds the available promised window.
- Use forecast durations to support route planning and vehicle/driver allocation, while respecting capacity, safety, and working-hour constraints.
- Monitor warehouse processing and dispatch delays by facility and shift.
- Track on-time delivery rate, MAE in minutes, route duration, fuel cost per shipment, and alert precision.
- Run a shadow test and a controlled pilot before operational rollout.

## Repository Structure
```text
week-4-logistics-predictive-modeling/
├── README.md
├── requirements.txt
├── report/
│   └── Week_4_Predictive_Modeling_Optimization_Logistics.docx
├── notebooks/
│   └── delivery_time_model.ipynb
├── src/
│   └── train_model.py
└── outputs/
    └── .gitkeep
```

## How to Run
```bash
python -m pip install -r requirements.txt
python src/train_model.py
```

## Important Limitations
The dataset is simulated. For real deployment, replace it with appropriately governed historical shipment data, use a time-based test split, check for data leakage, and evaluate model errors by route and operating condition.

## Conclusion
Predictive modeling can help logistics teams identify delivery risks earlier and make better-informed planning decisions. Operational value must be confirmed through a measured pilot against a baseline.
