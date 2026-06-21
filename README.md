# Predicting Digital Burnout from Smartphone Usage Patterns

Smartphones have become a near-constant presence in daily life, and while they bring real benefits, heavy or unhealthy usage patterns are increasingly linked to digital burnout — mental and emotional exhaustion driven by continuous online engagement. This project uses real smartphone behavioral data (rather than self-reported surveys) to predict who's at higher risk of digital burnout, and applies explainable AI so the predictions aren't just accurate but also understandable.

Two supervised models were trained — Logistic Regression and Random Forest — and the Random Forest model came out ahead with 94% accuracy. SHAP and feature importance analysis were then used to figure out exactly which behaviors drive that prediction.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Hypotheses](#hypotheses)
- [Research Methodology](#research-methodology)
- [Datasets](#datasets)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Descriptive Statistics](#descriptive-statistics)
- [Risk vs. Non-Risk Comparison](#risk-vs-non-risk-comparison)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Confusion Matrix](#confusion-matrix)
- [Feature Importance and SHAP](#feature-importance-and-shap)
- [Implementation](#implementation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction

Smartphones today are used for nearly everything — communication, work, entertainment, education, social interaction. That convenience comes with a cost for some users: excessive screen time, social media overuse, and constant notifications have all been tied to stress, fatigue, poor concentration, sleep problems, and reduced productivity. This broader pattern is what's referred to as digital burnout.

Most existing research on this topic leans heavily on self-reported surveys, which can be biased or inaccurate — people aren't always great at estimating their own screen time. This project takes a different approach by working directly with behavioral usage data and applying machine learning to surface patterns that might not be obvious from a questionnaire.

Beyond just building a predictive model, this project also focuses on explainability. A model that says "this user is at risk" without saying *why* isn't especially useful — SHAP and feature importance analysis were used here specifically to open up that black box.

## Problem Statement

Spotting who's at risk of digital burnout is harder than it sounds. Most public datasets don't come with a direct "burnout" label, and a lot of the research that does exist depends on self-reported data that may not match actual behavior. There's a real need for models built on real usage data — screen time, app activity, social media hours, gaming behavior — and for those models to be interpretable enough that the reasoning behind a prediction is actually clear. This project sets out to build exactly that.

## Objectives

1. Analyze smartphone usage behavior and identify patterns connected to digital burnout.
2. Build machine learning models that predict heavy smartphone users.
3. Apply explainability techniques (SHAP, feature importance) to make those predictions interpretable.

## Hypotheses

- **H1** — Machine learning models built on smartphone behavioral data can achieve acceptable predictive performance for identifying heavy users.
- **H2** — Behavioral features like screen time, social media usage, and app activity meaningfully influence smartphone addiction and burnout.
- **H3** — Explainable AI techniques can improve model interpretability without hurting predictive performance.

## Research Methodology

This is a quantitative, experimental study built on supervised machine learning. It relies on measurable numerical variables — daily screen time, social media hours, notifications per day, app opens, sleep hours, weekend screen time — rather than subjective self-reports.

It qualifies as supervised learning because the primary dataset includes a clear target variable, `addicted_label`:

| Target Variable | Meaning |
|---|---|
| `addicted_label = 0` | Non-addicted / lower burnout risk |
| `addicted_label = 1` | Addicted / higher burnout risk |

Two models were implemented — **Logistic Regression** and **Random Forest** — along with **SHAP** explainability analysis to interpret what's driving the predictions.

## Datasets

Three CSV files were initially collected, but two turned out to contain identical records and features — so the duplicate was dropped before any analysis, since training or evaluating on repeated rows would have skewed the results.

| Dataset | Rows | Columns | Target Variable | Role in Project |
|---|---|---|---|---|
| Smartphone Usage & Addiction Prediction Dataset | 7,500 | 16 | `addicted_label` | Main classification dataset |
| Smartphone Behavioral Dataset | 1,000 | 10 | Created risk level | Behavioral analysis & feature engineering |

The first dataset became the core classification dataset since it already had a labeled target. The second didn't come with a direct addiction label, but its usage features were still useful for a supporting behavioral analysis — a custom risk score was engineered for it (more on that below).

## Data Cleaning

Before any modeling, the data went through a standard cleanup pass:

1. Examined all uploaded CSV files
2. Identified and removed the duplicated 7,500-row dataset
3. Checked for and handled missing values
4. Selected `addicted_label` as the target variable
5. Dropped non-predictive identifiers like `User_ID`
6. Encoded categorical variables numerically
7. Scaled/normalized numerical variables where needed

### Correlation Heatmap

A correlation heatmap was generated early on to get a feel for how the behavioral variables relate to each other and to the target.

![Correlation heatmap between behavioral features](readme_images/correlation_heatmap.png)

*Figure 1. Correlation heatmap between behavioral features.*

The strongest relationships showed up between `daily_screen_time_hours`, `social_media_hours`, `weekend_screen_time`, and `addicted_label` — in other words, the more time someone spends on their phone (especially on weekends and social media), the more strongly that lines up with higher burnout risk.

## Exploratory Data Analysis

The target variable distribution skewed heavily toward the higher-risk class:

| Class | Count | Percent |
|---|---|---|
| 1 (Addicted) | 5,308 | 70.77% |
| 0 (Non-addicted) | 2,192 | 29.23% |

Breaking addiction down by severity level:

| Addiction Level | Count | Percent |
|---|---|---|
| Moderate | 2,874 | 38.32% |
| Severe | 2,434 | 32.45% |
| Mild | 1,373 | 18.31% |
| Not addicted / Not assigned | 819 | 10.92% |

Moderate and severe levels together make up the bulk of the dataset, which says something about how common heavier smartphone usage patterns are among the sampled users.

## Descriptive Statistics

A statistical summary of the main behavioral variables:

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| daily_screen_time_hours | 7.50 | 2.61 | 3.00 | 12.00 |
| social_media_hours | 3.27 | 1.59 | 0.50 | 6.00 |
| gaming_hours | 2.01 | 1.15 | 0.00 | 4.00 |
| work_study_hours | 3.24 | 1.60 | 0.50 | 6.00 |
| sleep_hours | 6.74 | 1.28 | 4.50 | 9.00 |
| notifications_per_day | 134.26 | 66.59 | 20.00 | 250.00 |
| app_opens_per_day | 97.83 | 48.42 | 15.00 | 180.00 |
| weekend_screen_time | 9.24 | 2.72 | 3.58 | 14.88 |

Average daily screen time sits at 7.5 hours, and that climbs to 9.24 hours on weekends — a pretty clear sign of prolonged digital engagement across the dataset as a whole.

## Risk vs. Non-Risk Comparison

Splitting users by `addicted_label` makes the behavioral gap between groups much more obvious:

| Feature | Lower Risk (0) | Higher Risk (1) |
|---|---|---|
| daily_screen_time_hours | 5.16 | 8.47 |
| social_media_hours | 2.25 | 3.70 |
| sleep_hours | 6.67 | 6.77 |
| notifications_per_day | 134.33 | 134.23 |
| app_opens_per_day | 97.00 | 98.18 |
| weekend_screen_time | 6.89 | 10.21 |

The higher-risk group logs noticeably more daily screen time (8.47 vs. 5.16 hours) and weekend screen time (10.21 vs. 6.89 hours). Interestingly, notification counts and app opens are nearly identical across both groups — it's not *how often* people check their phones that separates the groups, it's *how long* they stay on it once they do.

## Feature Engineering

The second dataset didn't come with a built-in burnout label, so a custom **digital burnout risk score** was constructed from:

- daily screen time
- total app usage hours
- social media usage hours
- gaming usage hours
- number of apps used

That score was then bucketed into three risk levels:

| Risk Level | Count | Percent |
|---|---|---|
| Low | 330 | 33% |
| Medium | 330 | 33% |
| High | 340 | 34% |

This gave the behavioral dataset a usable target for analysis, even without an original addiction label.

A quick statistical profile of this dataset (1,000 records, 10 columns):

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Age | 38.74 | 12.19 | 18.00 | 59.00 |
| Total_App_Usage_Hours | 6.41 | 3.13 | 1.00 | 11.97 |
| Daily_Screen_Time_Hours | 7.70 | 3.71 | 1.01 | 14.00 |
| Number_of_Apps_Used | 16.65 | 7.62 | 3.00 | 29.00 |
| Social_Media_Usage_Hours | 2.46 | 1.44 | 0.00 | 4.99 |
| Productivity_App_Usage_Hours | 2.50 | 1.44 | 0.00 | 5.00 |
| Gaming_App_Usage_Hours | 2.48 | 1.45 | 0.01 | 5.00 |

The high-risk bucket consistently scored highest across screen time, app usage, social media, and gaming activity — reinforcing the same pattern seen in the main dataset.

## Model Development

Two classification models were trained:

| Model | Reason for Selection |
|---|---|
| Logistic Regression | Baseline classifier |
| Random Forest | Captures complex, non-linear relationships between behavioral features |

**Inputs:** `age`, `gender`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`

**Target:** `addicted_label`

**Split:** 80% training / 20% testing

## Model Evaluation

Four metrics were used to judge each model — accuracy, precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 89% | 89% | 89% | 89% |
| **Random Forest** | **94%** | **94%** | **94%** | **94%** |

Random Forest came out ahead on every metric, which tracks with expectations — it's better suited to picking up on the non-linear interactions between screen time, social media usage, and weekend behavior that a linear model like Logistic Regression can't fully capture.

The detailed classification report for the model:

![Classification report showing precision, recall, and F1-score by class](readme_images/logistic_regression_classification_report.png)

*Figure 2. Detailed classification metrics by class.*

## Confusion Matrix

| Actual \ Predicted | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 618 | 65 |
| **Actual 1** | 71 | 1,496 |

![Confusion matrix for the Random Forest model](readme_images/confusion_matrix_random_forest.png)

*Figure 3. Confusion matrix for the Random Forest model.*

Out of the full test set, the model correctly classified 618 non-addicted users and 1,496 addicted users, with relatively few misclassifications on either side — a solid sign that the model isn't just accurate on paper but genuinely separating the two groups well.

## Feature Importance and SHAP

Random Forest's built-in feature importance scores point to a clear ranking of what matters most:

| Feature | Importance |
|---|---|
| social_media_hours | 0.42 |
| daily_screen_time_hours | 0.35 |
| weekend_screen_time | 0.25 |

![Feature importance ranking from the Random Forest model](readme_images/feature_importance_random_forest.png)

*Figure 4. Feature importance for the Random Forest model.*

SHAP analysis backed this up, confirming that `social_media_hours` and `daily_screen_time_hours` carry the strongest influence on the model's predictions.

![SHAP value distribution showing feature impact on predictions](readme_images/shap_explainability.png)

*Figure 5. SHAP explainability visualization.*

In plain terms: social media usage and overall screen exposure aren't just correlated with burnout risk, they're the features the model leans on most heavily when making its predictions — and that lines up well with intuition.

## Implementation

### Tools Used

| Tool | Purpose |
|---|---|
| Python | Core implementation language |
| Pandas | Reading and preprocessing CSV data |
| Scikit-learn | Model training and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SHAP | Explainable AI |

### Key Code

**Logistic Regression**

```python
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))
```

**Random Forest**

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
```

**SHAP Explainability**

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Pipeline Steps

1. Import the uploaded CSV datasets
2. Examine row/column counts and feature names
3. Identify and remove duplicate datasets
4. Select `addicted_label` as the target
5. Drop non-predictive identifiers (`User_ID`)
6. Encode categorical variables
7. Engineer additional behavioral features
8. Split into training/testing sets
9. Train Logistic Regression and Random Forest
10. Evaluate with accuracy, precision, recall, F1-score
11. Generate confusion matrix and feature importance
12. Apply SHAP explainability
13. Interpret the relationship between usage behavior and burnout risk

## Conclusion

This project set out to predict digital burnout using real smartphone behavioral data rather than self-reported surveys, and the results support that approach. Logistic Regression reached about 89% accuracy as a baseline, while Random Forest pushed that to roughly 94% by picking up on more complex relationships in the data.

Across both the modeling and the explainability analysis, the same three features kept surfacing as the strongest predictors: `social_media_hours`, `daily_screen_time_hours`, and `weekend_screen_time`. SHAP confirmed that these aren't just statistically correlated with burnout — they're what the model actually relies on most when making predictions.

Taken together, the findings support all three hypotheses: machine learning models built on behavioral data can predict burnout risk reasonably well (H1), specific usage behaviors meaningfully drive that risk (H2), and explainability techniques like SHAP can make those predictions transparent without sacrificing performance (H3).

## Future Work

- Use larger, more diverse datasets spanning different age groups, countries, occupations, and educational backgrounds to improve generalizability
- Add richer behavioral features — late-night usage, notification response time, emotional state, physical activity, mental health indicators
- Test deep learning approaches (ANNs, LSTMs, transformer-based models) against the current classical ML baselines
- Build a real-time monitoring tool or mobile app that can flag early signs of digital burnout and offer personalized recommendations
- Validate model predictions against real participants using established psychological burnout assessment tools

## References

1. World Health Organization. (2019). *Burn-out an occupational phenomenon.*
2. Montag, C., & Walla, P. (2016). Carpe diem instead of losing your social mind: Beyond digital addiction and why we all suffer from digital overuse. *Addictive Behaviors Reports, 4*, 1–3.
3. Lin, Y.-H., Chiang, C.-L., Lin, P.-H., Chang, L.-R., Ko, C.-H., & Lee, Y.-H. (2016). Proposed diagnostic criteria for smartphone addiction. *PLOS ONE, 11*(11), e0163010.
4. Zahranusratt. *Smartphone Usage and Addiction Analysis Dataset.* Kaggle. [Link](https://www.kaggle.com/datasets/zahranusratt/smartphone-usage-and-addiction-analysis-dataset/data)
5. Mohit Bhadra. *Smartphone Usage and Behavioral Dataset.* Kaggle. [Link](https://www.kaggle.com/datasets/bhadramohit/smartphone-usage-and-behavioral-dataset)
6. *Smartphone Usage and Addiction Prediction Dataset.* Kaggle. [Link](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)
7. Alam, S., & Alam, M. A. U. (2026). Explainable multitask burnout prediction using adaptive deep learning (EMBRACE) for resident physicians: Algorithm development and validation study. *JMIR AI, 5*, e57025.
8. An, R., Qian, G., Mumtaz, A., Alotaibi, K. A., & Wang, X. (2025). Digital fatigue and academic resilience among university students with grit and flexibility as mediators. *Scientific Reports, 15*.
9. Ibrahim, R. K., Khaled, M., Almansoori, M., Almazrouei, M., Ashraf, A., Alahmedi, S. H., & Hendy, A. (2025). Screen time and stress: Understanding how digital burnout influences health among nursing students. *BMC Nursing*.
10. Mazurets, O., Vit, R., Molchanova, M., Sobko, O., Wierzbicki, A., & Chumachenko, D. (2025). Neural network detection of digital fatigue and burnout with interpretable thematic segmentation. *CEUR Workshop Proceedings*.
11. Song, T., Zhu, H., Yang, K., Chang, W., & Ni, J. (2025). How mobile phone addiction leads to college students' learning burnout: The role of depression as a mediator and fear of missing out as a moderator. *Frontiers in Psychiatry, 16*.
12. Zhao, L., Zhao, J., Cao, E., Li, K., Pan, L., Zou, Y., & Sun, X. (2025). Development and validation of a digital burnout scale in artificial intelligence era. *Frontiers in Psychology, 16*.
