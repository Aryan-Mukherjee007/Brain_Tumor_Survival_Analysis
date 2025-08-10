#Imports
!pip install lifelines
!pip install scikit-survival openpyxl --quiet

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter
from scipy.stats import expon, weibull_min
from lifelines.statistics import proportional_hazard_test, logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from IPython.display import display
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Non-informative Censoring Test

#Loading and preparing the data
filename = '/content/pone.0148733.s001.xlsx'
df = pd.read_excel(filename, sheet_name='data')

columns_to_use = ['OS', 'status', 'Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']
df_clean = df[columns_to_use].dropna().copy()
df_clean.rename(columns={'OS': 'duration', 'status': 'event'}, inplace=True)

#Fitting Weibull model to data
aft = WeibullAFTFitter()
aft.fit(df_clean, duration_col='duration', event_col='event')

#Function imputes additional or less survival time for censored subjects
def impute_and_fit(df, gamma, n_imputations=10):
    stats = []
    rho = np.exp(aft.params_['rho_'])
    for _ in range(n_imputations):
        df_imp = df.copy()
        pred_median = aft.predict_median(df_imp)
        for idx, row in df_imp[df_imp['event'] == 0].iterrows():
            lam = 1 / pred_median.loc[idx]
            lam_adj = lam * gamma
            t0 = row['duration']
            add_time = weibull_min(c=rho, scale=1/lam_adj).rvs()
            df_imp.at[idx, 'duration'] = t0 + add_time
            df_imp.at[idx, 'event'] = 1
        cph = CoxPHFitter()
        cph.fit(df_imp, duration_col='duration', event_col='event')
        #Note: below line can be modified for other covariates in the data
        stats.append(cph.hazard_ratios_['KI'])
    return np.mean(stats), np.std(stats)

#Sets sensitivity parameters
gammas = np.linspace(0.5, 2.0, 7)
means, stds = [], []

#Runs above function imputing survival times and records mean and standard deviation of the hazard ratio for Karnofsky Index (for each sensitivity value)
for gamma in gammas:
    mean_hr, std_hr = impute_and_fit(df_clean, gamma, n_imputations=20)
    means.append(mean_hr)
    stds.append(std_hr)
    print(f"gamma={gamma:.2f}: mean HR(KI)={mean_hr:.3f}, std={std_hr:.3f}")

#Computes and records the range and standard deviation of the mean hazard ratios from each imputed trial
hr_range = max(means) - min(means)
hr_std = np.std(means)
print(f"\nRange of mean HR(KI) across gamma: {hr_range:.3f}")
print(f"Std of mean HR(KI) across gamma: {hr_std:.3f}")

#Plot results (mean hazard ratio versus gamma)
plt.errorbar(gammas, means, yerr=stds, fmt='o-', capsize=5)
plt.xlabel('Sensitivity Parameter')
plt.ylabel('Mean Hazard Ratio for KI')
plt.title('Sensitivity of HR')
plt.show()

#Shoenfeld Test for Proportional Hazards

#Data loading and standardizing for lifelines module
filename = '/content/pone.0148733.s001.xlsx'
df = pd.read_excel(filename, sheet_name='data')

covariates = ['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']
df_clean = df[['OS', 'status'] + covariates].dropna().copy()
df_clean.rename(columns={'OS': 'duration', 'status': 'event'}, inplace=True)

#Fitting Cox model on data
cph = CoxPHFitter()
cph.fit(df_clean, duration_col='duration', event_col='event')

#Schoenfeld residual test with log(t_i)
schoenfeld_test = proportional_hazard_test(cph, df_clean, time_transform='log')

#Performing test for each covariate
results = schoenfeld_test.summary.loc[covariates].copy()
results = results.reset_index()

#Formatting and table preparation
def format_float(x):
    s = f"{x:.4f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

results['Covariate'] = results['index']
results['Test_statistic'] = results['test_statistic'].apply(format_float)
results['P-value'] = results['p'].apply(format_float)
results['-log₂(P)'] = (-np.log2(results['p'])).apply(format_float)

table = results[['Covariate', 'Test_statistic', 'P-value', '-log₂(P)']]

styled_table = (
    table.style
    .set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    )
    .set_properties(**{'text-align': 'center'})
    .hide(axis='index')
)
display(styled_table)

#Markdown format (test format, not relevant to paper)
print('\n**Schoenfeld Test for Proportional Hazards (log(time) transformation)**\n')
print('| Covariate            | Test_statistic | P-value   | -log₂(P) |')
print('|:--------------------:|:--------------:|:---------:|:--------:|')
for _, row in table.iterrows():
    print(f"| {row['Covariate']} | {row['Test_statistic']} | {row['P-value']} | {row['-log₂(P)']} |")

#Kaplan Meir Curve (unstratified)

#Loading and preparing data
filename = '/content/pone.0148733.s001.xlsx'
df = pd.read_excel(filename, sheet_name='data')

df_kaplan = df[['OS', 'status']].dropna().copy()
df_kaplan.rename(columns={'OS': 'duration', 'status': 'event'}, inplace=True)

#Fitting the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=df_kaplan['duration'], event_observed=df_kaplan['event'])

#Plotting survival curve from fitted estimator
plt.figure(figsize=(8, 6))
kmf.plot(ci_show=True, linewidth=2)
plt.title('Kaplan-Meier Survival Curve for All Patients', fontsize=16)
plt.xlabel('Time (months)', fontsize=14)
plt.ylabel('Survival Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#Comparison of Cox model and Exponential AFT

#Loading and preparing data
filename = '/content/pone.0148733.s001.xlsx'
df = pd.read_excel(filename, sheet_name='data')

df_comparison = df[['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods', 'OS', 'status']].dropna().copy()
df_comparison.rename(columns={'OS': 'duration', 'status': 'event'}, inplace=True)

#Fitting exponential model
exp_aft = WeibullAFTFitter(penalizer=0.01)
exp_aft.fit(df_comparison, duration_col='duration', event_col='event')

#Computing c-index
cox_hazard = cph.predict_partial_hazard(df_comparison)
exp_predictions = exp_aft.predict_median(df_comparison)
cindex_cox = concordance_index(df_comparison['duration'], -cox_hazard, df_comparison['event'])
cindex_exp = concordance_index(df_comparison['duration'], -exp_predictions, df_comparison['event'])

#Creating and displaying table of information
summary_df = pd.DataFrame({
    'Model': ['Cox Proportional Hazards', 'Exponential AFT'],
    'C-Index': [cindex_cox, cindex_exp],
    'AIC': [cph.AIC_partial_, exp_aft.AIC_]
}).round(4)

styled_summary = (
    summary_df
    .style
    .set_caption("Table 2: Model Performance Comparison")
    .format({'C-Index': '{:.3f}', 'AIC': '{:.1f}'})
    .hide(axis='index')
    .set_properties(subset=summary_df.columns, **{
        'text-align': 'center'
    })
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'caption',
         'props': [('font-size', '14pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'table', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]}
    ])
)
display(styled_summary)

#Random Survival Forest for survival function

#Loading, preparing, and splitting the data
data = pd.read_excel('pone.0148733.s001.xlsx', sheet_name='data')
data = data.dropna(subset=['OS', 'status'])

X = data[['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']]
y = Surv.from_arrays(event=data['status'].astype(bool), time=data['OS'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Fitting RSF model
rsf = RandomSurvivalForest(max_features='sqrt', random_state=42)
rsf.fit(X_train, y_train)

#Calculating and projecting c-index
cindex_test = rsf.score(X_test, y_test)

summary_df = pd.DataFrame({
    'Model': ['Random Survival Forest'],
    'C-Index': [cindex_test]
}).round(4)

styled_summary = (
    summary_df
    .style
    .set_caption("Table: Model Performance (Test Data)")
    .format({'C-Index': '{:.3f}'})
    .hide(axis='index')
    .set_properties(subset=summary_df.columns, **{'text-align': 'center'})
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'caption',
         'props': [('font-size', '14pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'table', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]}
    ])
)

display(styled_summary)

#Permutation feature importance from RSF

#Function permutes all feature vectors and computes c-index drop in the RSF model for each permutation
def rsf_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_pred = model.predict(X)
    baseline_cindex = concordance_index_censored(
        y['event'], y['time'], baseline_pred
    )[0]
    importances = []
    stds = []
    for col in X.columns:
        scores = []
        X_permuted = X.copy()
        for _ in range(n_repeats):
            X_permuted[col] = rng.permutation(X_permuted[col].values)
            pred = model.predict(X_permuted)
            cindex = concordance_index_censored(
                y['event'], y['time'], pred
            )[0]
            scores.append(baseline_cindex - cindex)
        importances.append(np.mean(scores))
        stds.append(np.std(scores))
    return np.array(importances), np.array(stds)

#Stores mean and standard deviation of c-index drop
importances, stds = rsf_permutation_importance(rsf, X_test, y_test, n_repeats=20)

#(For paper only) ordering and renaming treatment feature for easy display before plotting
indices = np.argsort(importances)[::-1]
feature_names = X_test.columns[indices]

display_names = [name if name != "Stereotactic methods" else "Treatment" for name in feature_names]

plt.figure(figsize=(8, 5))
plt.title("RSF Feature Importance")
plt.bar(range(len(importances)), importances[indices], yerr=stds[indices], align="center", capsize=5)
plt.xticks(range(len(importances)), display_names, rotation=45, ha='right')
plt.ylabel("Mean C-index Decrease")
plt.xlabel("Covariate")
plt.tight_layout()
plt.show()

#Running cox regression

#Loading and preparing the data
data_path = '/content/pone.0148733.s001.xlsx'
df = pd.read_excel(data_path, sheet_name='data')

covariates = ['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']
df_clean = df[['OS', 'status'] + covariates].dropna().copy()
df_clean.rename(columns={'OS': 'duration', 'status': 'event'}, inplace=True)

#Fitting the Cox PH model
cph = CoxPHFitter()
cph.fit(df_clean, duration_col='duration', event_col='event')

#Labeling table
label_map = {
    'Sex': 'Sex (male/female)',
    'Diagnosis': 'Diagnosis (per category increase: meningioma→LG glioma→HG glioma→other)',
    'Location': 'Location (supratentorial/infratentorial)',
    'KI': 'Karnofsky Index (per point increase)',
    'GTV': 'Gross Tumor Volume (per cm³ increase)',
    'Stereotactic methods': 'Stereotactic method (SRT/SRS)'
}

#Creating table of resultant hazard ratios and confidence intervals from analysis
summary = cph.summary.copy()
summary['Covariate'] = summary.index.map(label_map)
summary['Hazard Ratio'] = np.exp(summary['coef'])
summary['95% CI Lower'] = np.exp(summary['coef lower 95%'])
summary['95% CI Upper'] = np.exp(summary['coef upper 95%'])

def format_float(x):
    return f"{x:.4f}".rstrip('0').rstrip('.') if '.' in f"{x:.4f}" else f"{x:.4f}"

summary['Hazard Ratio'] = summary['Hazard Ratio'].apply(format_float)
summary['95% CI'] = summary.apply(lambda row: f"[{format_float(row['95% CI Lower'])}, {format_float(row['95% CI Upper'])}]", axis=1)

final_table = summary[['Covariate', 'Hazard Ratio', '95% CI']].reset_index(drop=True)

print("\n**Cox Proportional Hazards Regression Results**\n")
print("| Covariate                                         | Hazard Ratio | 95% CI             |")
print("|:--------------------------------------------------|:------------:|:------------------:|")
for _, row in final_table.iterrows():
    print(f"| {row['Covariate']:<50} | {row['Hazard Ratio']:^12} | {row['95% CI']:^18} |")

#Predicted Survival Function for Model Patient from Cox PH

#Loading and preparing data and selecting patient
file_path = "pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

patient_id = 86
covariates = ['Sex', 'Diagnosis', 'Location', 'KI', 'GTV', 'Stereotactic methods']

#When predicting a patient's survival time, it would be silly to include their post-mortem result in the training
df_omit = df[df['ID'] != patient_id].copy()
df_omit_clean = df_omit[['OS', 'status'] + covariates].dropna()

#Fitting the Cox PH model
cph = CoxPHFitter()
cph.fit(df_omit_clean, duration_col='OS', event_col='status')

#Finding maximum possible survival time from patient's covariates
patient_row = df[df['ID'] == patient_id]
patient_covariates = patient_row[covariates]

max_time = df['OS'].max()
print(f"Maximum observed survival time in the full dataset: {max_time:.2f} months")

#Using Cox model to predict probability of patient's survival, given their covariate row vector, at each point in a time during 200 evenly spaced increments between inception and max_time
time_grid = np.linspace(0, max_time, 200)
surv_func = cph.predict_survival_function(patient_covariates, times=time_grid)

#Plotting these discrete probabilities onto a curve
plt.figure(figsize=(8,6))
plt.plot(surv_func.index, surv_func.values, label='Predicted survival for patient 86', color='blue', linewidth=2)
plt.xlim(0, max_time + 2)
plt.ylim(0, 1.05)
plt.title('Cox Model Predicted Survival Function for Patient 86')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Kaplan-Meier curves, stratified by sex

#Loading and preparing the data
file_path = "pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

sex_map = {0: "Female", 1: "Male"}
df = df[df['Sex'].isin([0,1])]

#Masking to select only one sex, fitting Kaplan-Meier estimator to it, repeating for other sex
plt.figure(figsize=(8,6))

for sex_code, sex_label in sex_map.items():
    mask = df['Sex'] == sex_code
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[mask, 'OS'], event_observed=df.loc[mask, 'status'], label=sex_label)
    kmf.plot(ci_show=True, linewidth=2)

#Plotting resultant Kaplan-Meier survival curves
plt.title('Kaplan-Meier Survival Curves by Sex')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Kaplan-meier curves, stratified by tumor location

file_path = "pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

location_map = {0: "Infratentorial", 1: "Supratentorial"}
df = df[df['Location'].isin([0, 1])]

plt.figure(figsize=(8,6))

for loc_code, loc_label in location_map.items():
    mask = df['Location'] == loc_code
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[mask, 'OS'], event_observed=df.loc[mask, 'status'], label=loc_label)
    kmf.plot(ci_show=True, linewidth=2)

plt.title('Kaplan-Meier Survival Curves by Tumor Location')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Kaplan-meier curves, stratified by treatment

file_path = "pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

stereo_map = {0: "SRS", 1: "SRT"}
df = df[df['Stereotactic methods'].isin([0, 1])]

plt.figure(figsize=(8,6))

for method_code, method_label in stereo_map.items():
    mask = df['Stereotactic methods'] == method_code
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[mask, 'OS'], event_observed=df.loc[mask, 'status'], label=method_label)
    kmf.plot(ci_show=True, linewidth=2)

plt.title('Kaplan-Meier Survival Curves by Stereotactic Method')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Log-rank test results (two groups)

#Loading and preparing the data
file_path = "pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

group_defs = {
    "Sex": ("Sex", {0: "Female", 1: "Male"}),
    "Location": ("Location", {0: "Infratentorial", 1: "Supratentorial"}),
    "Stereotactic methods": ("Stereotactic methods", {0: "SRS", 1: "SRT"}),
}

results = []

#Loops through each feature and compares survival time between groups post-split
for var, (col, label_map) in group_defs.items():
    mask = df[col].isin(label_map.keys())
    group0 = df[mask & (df[col] == list(label_map.keys())[0])]
    group1 = df[mask & (df[col] == list(label_map.keys())[1])]
    lr = logrank_test(
        group0['OS'], group1['OS'],
        event_observed_A=group0['status'],
        event_observed_B=group1['status']
    )
    n0 = len(group0)
    n1 = len(group1)
    results.append({
        "Variable": var,
        "Group 1": f"{label_map[list(label_map.keys())[0]]} (n={n0})",
        "Group 2": f"{label_map[list(label_map.keys())[1]]} (n={n1})",
        "p-value": lr.p_value
    })

#Storing p-value results in a dictionary
results_df = pd.DataFrame(results)

#Producing tablee (no longer relevant for paper)
styled_table = (
    results_df
    .style
    .set_caption("Table 1: Log-rank Test Results for Categorical Variables")
    .format({'p-value': '{:.4f}'})
    .hide(axis='index')
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid black',
        'padding': '8px'
    })
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '1px solid black'),
            ('background-color', '#f2f2f2'),
            ('padding', '8px')
        ]},
        {'selector': 'caption', 'props': [
            ('font-size', '14pt'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('caption-side', 'top'),
            ('margin-bottom', '10px')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('border', '1px solid black')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin-left', 'auto'),
            ('margin-right', 'auto'),
            ('width', 'auto')
        ]}
    ])
)

display(styled_table)

#Kaplan-meier curves, stratified by diagnosis (same as other stratifications but with four curves)

df = pd.read_excel("pone.0148733.s001.xlsx", sheet_name='data')
diagnosis_map = {0: "Meningioma", 1: "LG glioma", 2: "HG glioma", 3: "other"}

df = df.dropna(subset=['Diagnosis'])
df['Diagnosis'] = df['Diagnosis'].astype(int)

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple']

for diagnosis_code, color in zip(sorted(df['Diagnosis'].unique()), colors):
    mask = df['Diagnosis'] == diagnosis_code
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[mask, 'OS'],
            event_observed=df.loc[mask, 'status'],
            label=diagnosis_map[diagnosis_code])
    kmf.plot(ci_show=True, color=color, linewidth=2)

plt.title('Kaplan-Meier Survival Curves by Tumor Diagnosis')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Diagnosis')
plt.tight_layout()
plt.show()

#Multivariate Log-rank test results

#Loading and preparing the data
file_path = "/content/pone.0148733.s001.xlsx"
df = pd.read_excel(file_path, sheet_name='data')

df_clean = df.dropna(subset=['Diagnosis']).copy()
df_clean['Diagnosis'] = df_clean['Diagnosis'].astype(int)

#Calling multivariate_logrank_test function that's built into the lifelines package (and storing results)
results = multivariate_logrank_test(
    event_durations=df_clean['OS'],
    groups=df_clean['Diagnosis'],
    event_observed=df_clean['status']
)

#Preparing and displaying p-value from test
results_table = pd.DataFrame({
    'Covariate': ['Diagnosis'],
    'Log-Rank Chi2': [results.test_statistic],
    'p-value': [results.p_value]
})

results_table['p-value'] = results_table['p-value'].apply(lambda x: f"{x:.2e}")
results_table['Log-Rank Chi2'] = results_table['Log-Rank Chi2'].apply(lambda x: f"{x:.4f}")

styled_table = (
    results_table
    .style
    .set_caption("Table: Log-rank Test for Diagnosis (4 Groups)")
    .hide(axis='index')
    .set_properties(**{
        'text-align': 'center',
        'border': '1px solid black',
        'padding': '8px'
    })
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '1px solid black'),
            ('background-color', '#f2f2f2'),
            ('padding', '8px')
        ]},
        {'selector': 'caption', 'props': [
            ('font-size', '13pt'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('caption-side', 'top'),
            ('margin-bottom', '10px')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('border', '1px solid black')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin-left', 'auto'),
            ('margin-right', 'auto'),
            ('width', 'auto')
        ]}
    ])
)

display(styled_table)
