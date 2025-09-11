#Imports and package installations
!pip install lifelines
!pip install scikit-survival openpyxl --quiet

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter, LogLogisticFitter
from scipy import optimize
from scipy.optimize import minimize
from scipy.stats import expon, weibull_min, norm
from scipy.special import gamma, gammainc
from lifelines.statistics import proportional_hazard_test, logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from IPython.display import display
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Data processing

#Initializing the data
df = pd.read_excel("pone.0148733.s001.xlsx", sheet_name="data")

#Cleaning the data
df = df.dropna()

#Constants and defining covariate array
event_col = "status"
duration_col = "OS"
covariates = ["Sex", "Diagnosis", "Location", "KI", "GTV", "Stereotactic methods"]

#Fitting cox model

#Fitting and outputting cox model
cph = CoxPHFitter()
cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)

print("\nCox Model Summary (beta coefficients):")
print(cph.summary)

#Baseline survival
baseline_surv = cph.baseline_survival_
print("\nBaseline Survival Function:")

#First and last 6 points (for paper)
print(baseline_surv.head(6))
print(cph.baseline_survival_.tail(6))

#AIC for cox model
cox_aic = cph.AIC_partial_
print(f"AIC (Cox PH model): {cox_aic:.3f}")

#C-index for cox model
cox_cindex = concordance_index(df["OS"], -cph.predict_partial_hazard(df), df["status"])
print(f"C-index (Cox PH model): {cox_cindex:.3f}")

#Performing log-likelihood

#Splitting covariates by discrete v.s. continous
continuous_covs = ['KI', 'GTV']  # continuous vars
categorical_covs = ['Sex', 'Diagnosis', 'Location', 'Stereotactic methods']  # ID removed

#Computing mean and standard deviations for the continuous covariates
scalers = {}
for c in continuous_covs:
    mu, sigma = df[c].mean(), df[c].std()
    scalers[c] = (mu, sigma)
    df[c] = (df[c] - mu) / sigma
covariates = categorical_covs + continuous_covs

#Creating the design matrix
T = df[duration_col].astype(float).to_numpy()
E = df[event_col].astype(int).to_numpy()
X = df[covariates].astype(float).to_numpy()
X_design = np.column_stack([np.ones(len(T)), X])
colnames = ['Intercept'] + covariates

#Function defining negative log-likelihood computation
def nll(theta):
    beta = theta[:-1]
    log_gamma = theta[-1]
    gamma = np.exp(log_gamma)
    eta = X_design @ beta
    log_t = np.log(T + 1e-300)
    z = np.exp(gamma * (log_t + eta))
    logf = np.log(gamma) + (gamma - 1.0) * log_t + gamma * eta - 2.0 * np.log1p(z)
    logS = -np.log1p(z)
    ll = (E * logf + (1 - E) * logS).sum()
    return -ll

#Fitting under the optimizer
p = X_design.shape[1]
theta0 = np.zeros(p + 1)

opt = minimize(nll, theta0, method='trust-constr')
if not opt.success:
    print("trust-constr failed, retrying with BFGS")
    opt = minimize(nll, theta0, method='BFGS')

theta_hat = opt.x
beta_hat  = theta_hat[:-1]
log_g_hat = theta_hat[-1]
gamma_hat = np.exp(log_g_hat)

#Computing the variance-covariance matrix
H_inv = opt.hess_inv if hasattr(opt, 'hess_inv') else None
vcov = np.array(H_inv) if H_inv is not None else np.full((len(theta_hat), len(theta_hat)), np.nan)
se = np.sqrt(np.diag(vcov)) if np.all(np.isfinite(vcov)) else np.full(len(theta_hat), np.nan)

#Performing Wald test
z_vals = beta_hat / se[:-1]
p_vals = 2 * (1 - norm.cdf(np.abs(z_vals)))
z_logg = log_g_hat / se[-1] if np.isfinite(se[-1]) else np.nan
p_logg = 2 * (1 - norm.cdf(np.abs(z_logg))) if np.isfinite(z_logg) else np.nan
se_gamma = gamma_hat * se[-1] if np.isfinite(se[-1]) else np.nan

# -----------------------------
# 6) Summaries
# -----------------------------
coef_table = pd.DataFrame({
    'coef (beta)': beta_hat,
    'std err':     se[:-1],
    'z':           z_vals,
    'p>|z|':       p_vals
}, index=colnames)

gamma_table = pd.DataFrame({
    'estimate (gamma)': [gamma_hat],
    'std err (gamma)':  [se_gamma],
    'log(gamma)':       [log_g_hat],
    'std err log(gamma)':[se[-1]],
    'z for log(gamma)': [z_logg],
    'p>|z| (log gamma)':[p_logg]
}, index=['shape'])

print("\n=== Log-Logistic AFT (MLE, with standardized covariates, ID excluded) ===")
print(coef_table.round(6))
print("\n=== Shape parameter ===")
print(gamma_table.round(6))

#Defining estimated survival function
def S_hat(t, x_row):
    t = np.asarray(t, dtype=float)
    if isinstance(x_row, dict):
        x_vec = []
        for c in covariates:
            if c in continuous_covs:
                mu, sigma = scalers[c]
                x_vec.append((x_row.get(c, mu) - mu) / sigma)
            else:
                x_vec.append(x_row.get(c, 0.0))
        x_vec = np.array(x_vec, dtype=float)
    else:
        x_vec = np.asarray(x_row, dtype=float)
        assert x_vec.shape[0] == len(covariates)

    eta = beta_hat[0] + np.dot(beta_hat[1:], x_vec)
    z = (t * np.exp(eta)) ** gamma_hat
    return 1.0 / (1.0 + z)

#Log-likelihood accuracy tests

#Storing value of maximized log-likelihood
loglik = -nll(theta_hat)

#Extracting and storing the of estimated parameters (betas + log gamma)
k = len(theta_hat)
n = len(T)

#Getting both AIC and AICc (will use AIC in paper, AICc used for personal comparison)
AIC = 2 * k - 2 * loglik
AICc = AIC + (2 * k * (k + 1)) / (n - k - 1) if n > (k + 1) else np.nan

#Computing c-index from Harrell
eta = X_design @ beta_hat
risk_scores = -eta  # convention: larger = riskier

c_index = concordance_index(T, risk_scores, E)

print(f"AIC: {AIC:.4f}")
print(f"AICc: {AICc:.4f}")
print(f"C-index: {c_index:.4f}")

#Exponential

from autograd import numpy as anp

#No exponential available- must define Weibull with rho = 1
class ExponentialAFTFitter(WeibullAFTFitter):
    def _log_likelihood(self, params, T, E, Xs):
        params_ = params.copy()
        params_["rho_"] = anp.array([1.0])
        return super()._log_likelihood(params_, T, E, Xs)

#Fitting and outputting the exponential model
exp_aft = ExponentialAFTFitter()
exp_aft.fit(df, duration_col=duration_col, event_col=event_col)

print("=== Exponential AFT Model Coefficients ===")
print(exp_aft.summary)

#AIC and c-index calculations
print("\nAIC:", exp_aft.AIC_)
predicted_median = exp_aft.predict_median(df)
c_index = concordance_index(
    df[duration_col],
    -predicted_median,
    df[event_col]
)
print("C-index:", c_index)

#Preparing survival data (for the scikit package)

y_struct = np.array([(bool(e), t) for e, t in zip(df['status'], df['OS'])],
                    dtype=[('event', 'bool'), ('time', 'float')])

#Extracting the feature matrix
X = df[covariates]

#Fitting the RSF model (based on recommended hyperparameters)
rsf = RandomSurvivalForest(
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
rsf.fit(X, y_struct)

#C-index (on full data due to sample size constrictions)
cindex = concordance_index_censored(
    y_struct["event"], y_struct["time"], rsf.predict(X)
)[0]
print("Random Survival Forest C-index:", cindex)

#Log-rank test results (two groups)

#Explicit group differentiation
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

#Multivariate Log-rank test results

#Ensuring diagnosis specifically is cleaned (again)
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

#Stratifying Kaplan-Meier curves by treatment

#Further preparing the data
stereo_map = {0: "SRS", 1: "SRT"}
df = df[df['Stereotactic methods'].isin([0, 1])]
plt.figure(figsize=(8, 6))

#Colors and color matching (for paper)
colors = ['red', 'blue']
stereo_methods = sorted(stereo_map.keys())

for i, method_code in enumerate(stereo_methods):
    method_label = stereo_map[method_code]
    mask = df['Stereotactic methods'] == method_code
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[mask, 'OS'], event_observed=df.loc[mask, 'status'], label=method_label)
    kmf.plot(ci_show=True, linewidth=2, color=colors[i])


#Plotting everything
plt.title('Kaplan-Meier Survival Curves by Stereotactic Method')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Kaplan-meier curves, stratified by diagnosis (same as other stratifications but with four curves)
diagnosis_map = {0: "Meningioma", 1: "LG glioma", 2: "HG glioma", 3: "other"}
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

#Appendix information

#Splitting covariates
categorical_covariates = ['Sex', 'Diagnosis', 'Location', 'Stereotactic methods', 'status']
quantitative_covariates = ['KI', 'GTV', 'OS']

#Picking colors for paper
color = 'skyblue'
edgecolor = 'black'

#8 plots (6 in paper, 2 for personal investigation)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

#Bar charts for categorical
for i, cov in enumerate(categorical_covariates):
    counts = df[cov].value_counts().sort_index()
    axes[i].bar(counts.index.astype(str), counts.values, color=color, edgecolor=edgecolor)
    axes[i].set_title(f'Frequency of {cov}')
    axes[i].set_xlabel(cov)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

#Histograms for quantitative
for j, cov in enumerate(quantitative_covariates):
    ax_idx = len(categorical_covariates) + j  # maps to axes[5], [6], [7]
    axes[ax_idx].hist(df[cov].dropna(), bins=15, color=color, edgecolor=edgecolor)
    axes[ax_idx].set_title(f'Distribution of {cov}')
    axes[ax_idx].set_xlabel(cov)
    axes[ax_idx].set_ylabel('Frequency')


plt.tight_layout()
plt.show()

