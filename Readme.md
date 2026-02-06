Approach & Methodology
1. Data Source

YouTube Trending Videos dataset

Contains metadata of videos that appeared in YouTube’s trending section

2. Feature Engineering (Pre-upload only)

The following features are derived before video upload, avoiding data leakage:

Feature	Description
title_length	Length of the video title
has_num	Whether the title contains numbers
tags_count	Number of tags
publish_hour	Hour of upload (0–23)
publish_day	Day of the week (0–6)
category_id	Content category (mapped internally)

Post-upload signals such as likes, comments, and engagement were intentionally excluded to prevent data leakage.

3. Target Transformation

View counts follow a power-law distribution

Applied log1p(views) to stabilize variance

Predictions are inverse-transformed using expm1 for interpretability

4. Model Selection

Random Forest Regressor

Chosen for:

Handling non-linear relationships

Robustness to noise

Strong performance on tabular data

No need for feature scaling

5. Evaluation

Metrics evaluated on test data only

Achieved:

Realistic MAE on log-scale

Strong R² on transformed target

Model intentionally produces conservative predictions to avoid speculative virality






How to Run the App Locally
1. Install Dependencies
pip install streamlit scikit-learn pandas numpy joblib

2. Run the App
streamlit run app.py