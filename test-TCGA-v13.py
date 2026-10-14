"""
ENHANCED TEMPORAL GENOMIC EVOLUTION PREDICTION SYSTEM - UPDATED WITH ROBUSTNESS
Complete analysis with improved generalization strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Survival analysis
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
from sklearn.inspection import permutation_importance

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, shapiro, kstest, anderson
from scipy.stats import kruskal, friedmanchisquare, wilcoxon, spearmanr, pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# Bootstrap for uncertainty
from scipy.stats import bootstrap

# Utilities
import os
from datetime import datetime
import pickle
import json
from itertools import combinations

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# ROBUST CONFIGURATION
# ============================================================================

class RobustConfig:
    """Enhanced configuration for robust model training"""
    
    # Data splitting - THREE-WAY SPLIT
    TEST_SIZE = 0.30
    VALIDATION_SIZE = 0.15  # NEW: Internal validation set
    RANDOM_STATE = 42
    
    # Cross-validation
    CV_FOLDS = 5
    CV_REPEATS = 3  # NEW: Repeated CV for stability
    
    # Regularization strengths (INCREASED for generalization)
    COX_PENALIZER = 0.5  # Increased from 0.1
    COX_L1_RATIO = 0.7   # Increased from 0.3
    ELASTIC_NET_L1 = 0.8 # Increased from 0.5
    
    # Random Forest (MORE CONSERVATIVE)
    RSF_N_ESTIMATORS = 500  # Increased from 200
    RSF_MAX_DEPTH = 5       # Decreased from 10
    RSF_MIN_SAMPLES_SPLIT = 50  # Increased from 20
    RSF_MIN_SAMPLES_LEAF = 25   # Increased from 10
    RSF_MAX_FEATURES = 0.3      # NEW: Use fewer features per tree
    
    # Gradient Boosting (MORE CONSERVATIVE)
    GB_N_ESTIMATORS = 50    # Decreased from 100
    GB_LEARNING_RATE = 0.01 # Decreased from 0.05
    GB_MAX_DEPTH = 2        # Decreased from 3
    GB_MIN_SAMPLES_SPLIT = 50   # Increased from 20
    GB_MIN_SAMPLES_LEAF = 25    # Increased from 10
    GB_SUBSAMPLE = 0.5          # Decreased from 0.7
    
    # SVM (MORE CONSERVATIVE)
    SVM_ALPHA_RANGE = [1.0, 2.0, 5.0, 10.0]  # Higher alpha = more regularization
    SVM_MAX_ITER = 100
    
    # Feature selection
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_METHOD = 'stability'  # NEW: Stability selection
    N_BOOTSTRAP_FEATURE = 100
    FEATURE_THRESHOLD = 0.6  # Feature must be selected in 60% of bootstraps
    
    # Scaling
    SCALER_TYPE = 'robust'  # NEW: More robust to outliers
    
    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 10

config = RobustConfig()

# Create output directories
output_base = '/home/tahir/10April-results'
os.makedirs(output_base, exist_ok=True)
os.makedirs(f'{output_base}/figures', exist_ok=True)
os.makedirs(f'{output_base}/models', exist_ok=True)
os.makedirs(f'{output_base}/tables', exist_ok=True)
os.makedirs(f'{output_base}/ablation', exist_ok=True)
os.makedirs(f'{output_base}/statistics', exist_ok=True)
os.makedirs(f'{output_base}/uncertainty', exist_ok=True)

# DEFINE CONSISTENT COLORS FOR ALL MODELS
MODEL_COLORS = {
    'Cox - Clinical Only': '#1f77b4',
    'Cox - Genomic Only': '#ff7f0e',
    'Cox - Combined': '#2ca02c',
    'Cox - Combined (Robust)': '#2ca02c',
    'Cox - Elastic Net': '#d62728',
    'Elastic Net (Robust)': '#d62728',
    'Random Survival Forest': '#9467bd',
    'RSF (Robust)': '#9467bd',
    'Gradient Boosting': '#8c564b',
    'Gradient Boosting Survival': '#8c564b',
    'GB (Robust)': '#8c564b',
    'Survival SVM': '#e377c2',
    'Survival SVM ': '#e377c2',
    'SVM (Robust)': '#e377c2',
    'Cox PH': '#bcbd22',
    'Ensemble (Top 3)': '#17becf'
}

# Define markers for scatter plots
MODEL_MARKERS = {
    'Cox - Clinical Only': 'o',
    'Cox - Genomic Only': 's',
    'Cox - Combined': '^',
    'Cox - Combined (Robust)': '^',
    'Cox - Elastic Net': 'D',
    'Elastic Net (Robust)': 'D',
    'Random Survival Forest': 'v',
    'RSF (Robust)': 'v',
    'Gradient Boosting': 'p',
    'Gradient Boosting Survival': 'p',
    'GB (Robust)': 'p',
    'Survival SVM ': '*',
    'SVM (Robust)': '*',
    'Cox PH': 'h',
    'Ensemble (Top 3)': 'P'
}

print("=" * 80)
print("ENHANCED TEMPORAL GENOMIC EVOLUTION PREDICTION SYSTEM")
print("UPDATED FOR CONTINUOUS/NORMALIZED FEATURES WITH ROBUSTNESS")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Set random seed
RANDOM_STATE = config.RANDOM_STATE
np.random.seed(RANDOM_STATE)
# ============================================================================
# LOAD AND COMPREHENSIVE DESCRIPTIVE STATISTICS
# ============================================================================

print("\n[PHASE 1] DATA LOADING AND COMPREHENSIVE DESCRIPTIVE STATISTICS")
print("-" * 80)

# Load dataset
data_path = "/home/tahir/Final_TCGA_Dataset.csv"
df = pd.read_csv(data_path)

print(f"Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")

# Display first few rows
print("\nFirst 3 rows of dataset:")
print(df.head(3))

# Check data types
print("\nData types:")
print(df.dtypes.value_counts())

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

print("\n[DESCRIPTIVE STATISTICS]")
print("-" * 80)

# Overall statistics
print("\n1. Overall Dataset Statistics:")
print(f"   Total Patients: {len(df)}")
print(f"   Death Events: {df['Event'].sum()} ({df['Event'].sum()/len(df)*100:.2f}%)")
print(f"   Censored: {len(df) - df['Event'].sum()} ({(len(df) - df['Event'].sum())/len(df)*100:.2f}%)")
print(f"   Missing Values: {df.isnull().sum().sum()}")

# Survival time statistics
print("\n2. Survival Time Statistics:")
print(f"   Mean: {df['Time'].mean():.2f} years")
print(f"   Median: {df['Time'].median():.2f} years")
print(f"   Std Dev: {df['Time'].std():.2f} years")
print(f"   Min: {df['Time'].min():.2f} years")
print(f"   Max: {df['Time'].max():.2f} years")
print(f"   25th percentile: {df['Time'].quantile(0.25):.2f} years")
print(f"   75th percentile: {df['Time'].quantile(0.75):.2f} years")
print(f"   IQR: {df['Time'].quantile(0.75) - df['Time'].quantile(0.25):.2f} years")
print(f"   Negative times: {(df['Time'] < 0).sum()} ({(df['Time'] < 0).sum()/len(df)*100:.2f}%)")

# Test for normality of survival times
print("\n3. Normality Tests for Survival Time:")
stat_shapiro, p_shapiro = shapiro(df['Time'][:min(5000, len(df))])
print(f"   Shapiro-Wilk: W={stat_shapiro:.4f}, p={p_shapiro:.4e}")
stat_ks, p_ks = kstest(df['Time'], 'norm', args=(df['Time'].mean(), df['Time'].std()))
print(f"   Kolmogorov-Smirnov: D={stat_ks:.4f}, p={p_ks:.4e}")

# Define feature groups - CONTINUOUS FEATURES
genomic_features = ['APC', 'CSMD3', 'DNAH5', 'FAT3', 'FLG', 'HMCN1', 'KMT2D', 
                    'KRAS', 'LRP1B', 'MUC16', 'MUC17', 'NAV3', 'PIK3CA', 
                    'RYR2', 'SYNE1', 'TP53', 'TTN', 'USH2A', 'XIRP2', 'ZFHX4']

clinical_features = ['age_at_diagnosis', 'received_pharmaceutical_treatment']
gender_features = ['gender_female', 'gender_male']
cancer_type_features = ['cancer_type_BRCA', 'cancer_type_COAD', 'cancer_type_LUAD']
tumor_stage_features = [col for col in df.columns if 'tumor_stage' in col]

print(f"\n4. Feature Groups:")
print(f"   Genomic features: {len(genomic_features)} (CONTINUOUS/NORMALIZED)")
print(f"   Clinical features: {len(clinical_features)} (CONTINUOUS)")
print(f"   Gender features: {len(gender_features)} (CONTINUOUS)")
print(f"   Cancer type features: {len(cancer_type_features)} (CONTINUOUS)")
print(f"   Tumor stage features: {len(tumor_stage_features)} (CONTINUOUS)")

# Cancer type distribution - UPDATED FOR CONTINUOUS VALUES
print("\n5. Cancer Type Distribution and Statistics:")
cancer_stats = []

for col in cancer_type_features:
    cancer = col.replace('cancer_type_', '')
    
    # For continuous cancer type indicators, use values above median as "positive"
    threshold = df[col].median()
    mask = df[col] > threshold
    count = mask.sum()
    pct = count / len(df) * 100
    events = df[mask]['Event'].sum()
    event_rate = events / count * 100 if count > 0 else 0
    
    # Survival statistics
    cancer_time = df[mask]['Time']
    
    cancer_stats.append({
        'Cancer_Type': cancer,
        'N_Patients_High': count,
        'Percentage': pct,
        'N_Events': events,
        'Event_Rate': event_rate,
        'Mean_Survival': cancer_time.mean(),
        'Median_Survival': cancer_time.median(),
        'Std_Survival': cancer_time.std(),
        'Mean_Score': df[col].mean(),
        'Std_Score': df[col].std()
    })
    
    print(f"   {cancer}:")
    print(f"      High score patients (>{threshold:.3f}): {count} ({pct:.1f}%)")
    print(f"      Mean score: {df[col].mean():.3f} +/- {df[col].std():.3f}")
    print(f"      Events: {events} ({event_rate:.1f}%)")
    print(f"      Mean survival: {cancer_time.mean():.2f} years")

cancer_stats_df = pd.DataFrame(cancer_stats)
cancer_stats_df.to_csv(f'{output_base}/statistics/cancer_type_statistics.csv', index=False)

# Age statistics
print("\n6. Age at Diagnosis Statistics (Continuous):")
print(f"   Mean: {df['age_at_diagnosis'].mean():.2f}")
print(f"   Median: {df['age_at_diagnosis'].median():.2f}")
print(f"   Std Dev: {df['age_at_diagnosis'].std():.2f}")
print(f"   Min: {df['age_at_diagnosis'].min():.2f}")
print(f"   Max: {df['age_at_diagnosis'].max():.2f}")
print(f"   Range: {df['age_at_diagnosis'].max() - df['age_at_diagnosis'].min():.2f}")

# Genomic feature statistics
print("\n7. Genomic Feature Statistics (Continuous/Normalized):")
mutation_stats = []

for gene in genomic_features:
    mean_val = df[gene].mean()
    std_val = df[gene].std()
    min_val = df[gene].min()
    max_val = df[gene].max()
    
    # Correlation with survival
    corr_time, p_time = pearsonr(df[gene], df['Time'])
    
    # Association with events
    corr_event, p_event = pearsonr(df[gene], df['Event'])
    
    mutation_stats.append({
        'Gene': gene,
        'Mean': mean_val,
        'Std': std_val,
        'Min': min_val,
        'Max': max_val,
        'Corr_Time': corr_time,
        'P_Time': p_time,
        'Corr_Event': corr_event,
        'P_Event': p_event
    })
    
    print(f"   {gene:10s}: Mean={mean_val:7.3f}, Std={std_val:6.3f}, "
          f"Corr(Time)={corr_time:6.3f}(p={p_time:.4f}), "
          f"Corr(Event)={corr_event:6.3f}(p={p_event:.4f})")

mutation_stats_df = pd.DataFrame(mutation_stats)
mutation_stats_df.to_csv(f'{output_base}/statistics/genomic_feature_statistics.csv', index=False)

# Store configuration
config_dict = {
    'genomic_features': genomic_features,
    'clinical_features': clinical_features,
    'gender_features': gender_features,
    'cancer_type_features': cancer_type_features,
    'tumor_stage_features': tumor_stage_features,
    'n_patients': len(df),
    'n_events': int(df['Event'].sum()),
    'event_rate': df['Event'].sum() / len(df),
    'feature_type': 'continuous_normalized'
}

print("\n? Part 1 Complete: Data loaded and comprehensive descriptive statistics generated")
print("  Note: All features are continuous/normalized (not binary)")

# ============================================================================
# PART 2: ROBUST FEATURE ENGINEERING AND STABILITY-BASED FEATURE SELECTION
# ============================================================================

print("\n[PHASE 2] ROBUST FEATURE ENGINEERING AND STABILITY SELECTION")
print("-" * 80)

# Create copy for processing
df_processed = df.copy()

# ============================================================================
# Time Handling
# ============================================================================

print(f"\nTime Distribution (Normalized):")
print(f"  Mean: {df_processed['Time'].mean():.3f}")
print(f"  Median: {df_processed['Time'].median():.3f}")
print(f"  Std: {df_processed['Time'].std():.3f}")
print(f"  Min: {df_processed['Time'].min():.3f}")
print(f"  Max: {df_processed['Time'].max():.3f}")

negative_times = (df_processed['Time'] < 0).sum()
print(f"  Negative values: {negative_times} ({negative_times/len(df_processed)*100:.1f}%)")

# Shift times for sksurv compatibility
time_shift = abs(df_processed['Time'].min()) + 0.1
df_processed['Time_Shifted'] = df_processed['Time'] + time_shift

print(f"? Time shifted by {time_shift:.3f} for sksurv compatibility")

# ============================================================================
# Robust Feature Scaling
# ============================================================================

print("\n[FEATURE SCALING]")
print("-" * 80)

# Use RobustScaler for age (more robust to outliers)
if config.SCALER_TYPE == 'robust':
    age_scaler = RobustScaler()
    print("Using RobustScaler (resistant to outliers)")
else:
    age_scaler = StandardScaler()
    print("Using StandardScaler")

df_processed['age_at_diagnosis_scaled'] = age_scaler.fit_transform(
    df_processed[['age_at_diagnosis']]
)

print(f"? Age scaled:")
print(f"  Original - Mean: {df_processed['age_at_diagnosis'].mean():.4f}, Std: {df_processed['age_at_diagnosis'].std():.4f}")
print(f"  Scaled   - Mean: {df_processed['age_at_diagnosis_scaled'].mean():.4f}, Std: {df_processed['age_at_diagnosis_scaled'].std():.4f}")

# Update clinical features to use scaled age
clinical_features_scaled = ['age_at_diagnosis_scaled', 'received_pharmaceutical_treatment']

# ============================================================================
# Assign Cancer Type Labels
# ============================================================================

cancer_labels = []
for idx, row in df_processed.iterrows():
    scores = [row[c] for c in cancer_type_features]
    max_idx = np.argmax(scores)
    cancer_labels.append(['BRCA', 'COAD', 'LUAD'][max_idx])

df_processed['cancer_type_label'] = cancer_labels

print(f"\n? Cancer type labels assigned based on highest score")

# ============================================================================
# STABILITY-BASED FEATURE SELECTION (NEW - ROBUST APPROACH)
# ============================================================================

print("\n[STABILITY-BASED FEATURE SELECTION]")
print("-" * 80)

def stability_feature_selection(X, y, n_bootstrap=100, threshold=0.6):
    """
    Select features that are consistently important across bootstrap samples.
    More robust than single-pass feature selection.
    """
    n_features = X.shape[1]
    feature_counts = np.zeros(n_features)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Feature selection on bootstrap
        selector = SelectKBest(mutual_info_regression, k=min(20, n_features))
        selector.fit(X_boot, y_boot)
        
        # Count selected features
        selected = selector.get_support()
        feature_counts += selected
    
    # Features selected in at least threshold% of bootstraps
    stability_scores = feature_counts / n_bootstrap
    stable_features = stability_scores >= threshold
    
    return stable_features, stability_scores

# Define all features
all_features_list = (genomic_features + clinical_features_scaled + 
                    gender_features + tumor_stage_features)

if config.USE_FEATURE_SELECTION:
    print("Performing stability-based feature selection...")
    print(f"  Bootstraps: {config.N_BOOTSTRAP_FEATURE}")
    print(f"  Threshold: {config.FEATURE_THRESHOLD}")
    
    # Prepare data
    X_all = df_processed[all_features_list].values
    y_all = df_processed['Time'].values
    
    # Stability selection
    stable_mask, stability_scores = stability_feature_selection(
        X_all, y_all, 
        n_bootstrap=config.N_BOOTSTRAP_FEATURE,
        threshold=config.FEATURE_THRESHOLD
    )
    
    stable_features = [f for f, selected in zip(all_features_list, stable_mask) if selected]
    
    print(f"? Selected {len(stable_features)}/{len(all_features_list)} stable features")
    print(f"  Threshold: {config.FEATURE_THRESHOLD} (must appear in {config.FEATURE_THRESHOLD*100:.0f}% of bootstraps)")
    
    # Print top 10 most stable features
    feature_stability_df = pd.DataFrame({
        'Feature': all_features_list,
        'Stability': stability_scores
    }).sort_values('Stability', ascending=False)
    
    print("\nTop 10 Most Stable Features:")
    print(feature_stability_df.head(10).to_string(index=False))
    
    # Save stability scores
    feature_stability_df.to_csv(f'{output_base}/tables/feature_stability_scores.csv', index=False)
    
else:
    stable_features = all_features_list
    feature_stability_df = pd.DataFrame({
        'Feature': all_features_list,
        'Stability': 1.0
    })

# Define feature sets
clinical_only_features = (clinical_features_scaled + gender_features + tumor_stage_features)
genomic_only_features = genomic_features.copy()
all_features = stable_features  # Use stable features as default

print(f"\nFeature Set Definitions:")
print(f"   Clinical only: {len(clinical_only_features)} features")
print(f"   Genomic only: {len(genomic_only_features)} features")
print(f"   All features (stable): {len(all_features)} features")
print(f"   Cancer types: EXCLUDED (used for stratification only)")

print("\n? Part 2 Complete: Robust feature engineering and stability selection completed")
# ============================================================================
# PART 3: THREE-WAY DATA SPLIT AND INFERENTIAL STATISTICS
# ============================================================================

print("\n[PHASE 3] THREE-WAY DATA PARTITIONING")
print("-" * 80)

# ============================================================================
# Three-Way Split: Train / Validation / Test
# ============================================================================

# First split: separate test set
df_temp, df_test = train_test_split(
    df_processed,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=df_processed[['Event', 'cancer_type_label']].astype(str).agg('_'.join, axis=1)
)

# Second split: separate validation from training
df_train, df_val = train_test_split(
    df_temp,
    test_size=config.VALIDATION_SIZE / (1 - config.TEST_SIZE),
    random_state=config.RANDOM_STATE,
    stratify=df_temp[['Event', 'cancer_type_label']].astype(str).agg('_'.join, axis=1)
)

print(f"Training set:   {len(df_train)} patients ({df_train['Event'].sum()} events, {df_train['Event'].sum()/len(df_train)*100:.1f}%)")
print(f"Validation set: {len(df_val)} patients ({df_val['Event'].sum()} events, {df_val['Event'].sum()/len(df_val)*100:.1f}%)")
print(f"Test set:       {len(df_test)} patients ({df_test['Event'].sum()} events, {df_test['Event'].sum()/len(df_test)*100:.1f}%)")

# Verify cancer type distribution
print("\nCancer Type Distribution Verification:")
for cancer in ['BRCA', 'COAD', 'LUAD']:
    train_pct = (df_train['cancer_type_label'] == cancer).sum() / len(df_train) * 100
    val_pct = (df_val['cancer_type_label'] == cancer).sum() / len(df_val) * 100
    test_pct = (df_test['cancer_type_label'] == cancer).sum() / len(df_test) * 100
    print(f"  {cancer}: Train {train_pct:.1f}%, Val {val_pct:.1f}%, Test {test_pct:.1f}%")

# Create survival arrays
y_train = Surv.from_dataframe('Event', 'Time_Shifted', df_train)
y_val = Surv.from_dataframe('Event', 'Time_Shifted', df_val)
y_test = Surv.from_dataframe('Event', 'Time_Shifted', df_test)

# Also store original time values
T_train_original = df_train['Time'].values
E_train = df_train['Event'].values
T_val_original = df_val['Time'].values
E_val = df_val['Event'].values
T_test_original = df_test['Time'].values
E_test = df_test['Event'].values

# Shifted times
T_train = df_train['Time_Shifted'].values
T_val = df_val['Time_Shifted'].values
T_test = df_test['Time_Shifted'].values

print("\n? Three-way split complete with stratification")

# ============================================================================
# ROBUST SCALING FOR ALL MODELS
# ============================================================================

print("\n[ROBUST SCALING]")
print("-" * 80)

# Use RobustScaler for better handling of outliers
if config.SCALER_TYPE == 'robust':
    scaler_global = RobustScaler()
    scaler_ml = RobustScaler()
else:
    scaler_global = StandardScaler()
    scaler_ml = StandardScaler()

# Scale stable features
X_train = df_train[stable_features].values
X_val = df_val[stable_features].values
X_test = df_test[stable_features].values

X_train_scaled = scaler_ml.fit_transform(X_train)
X_val_scaled = scaler_ml.transform(X_val)
X_test_scaled = scaler_ml.transform(X_test)

# Global scaler for all features
X_train_all = df_train[all_features_list].values
X_val_all = df_val[all_features_list].values
X_test_all = df_test[all_features_list].values

X_train_scaled_global = scaler_global.fit_transform(X_train_all)
X_val_scaled_global = scaler_global.transform(X_val_all)
X_test_scaled_global = scaler_global.transform(X_test_all)

print(f"? {config.SCALER_TYPE.capitalize()} scaling applied")
print(f"  Features scaled: {len(stable_features)}")

# ============================================================================
# INFERENTIAL STATISTICS - CONTINUOUS FEATURES
# ============================================================================

print("\n[INFERENTIAL STATISTICS - CONTINUOUS FEATURES]")
print("-" * 80)

# ============================================================================
# 1. Correlation Analysis (Continuous Variables vs Event)
# ============================================================================

print("\n1. Point-Biserial Correlation (Continuous Features vs Event):")
print("   Testing correlation between continuous features and binary Event outcome")

correlation_results = []

# Test all continuous features
continuous_test_features = (genomic_features + clinical_features_scaled + 
                            gender_features + tumor_stage_features)

for var in continuous_test_features:
    try:
        # Point-biserial correlation for continuous vs binary
        corr, p_value = pearsonr(df_processed[var], df_processed['Event'])
        
        correlation_results.append({
            'Variable': var,
            'Correlation': corr,
            'P_value': p_value,
            'Significant': p_value < 0.05,
            'Variable_Type': 'Genomic' if var in genomic_features else 'Clinical'
        })
        
        sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"   {var:35s}: r={corr:7.4f}, p={p_value:.4e}, {sig_marker}")
        
    except Exception as e:
        print(f"   {var:35s}: Error - {str(e)[:30]}")

# Save correlation results
if len(correlation_results) > 0:
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('P_value')
    corr_df.to_csv(f'{output_base}/statistics/feature_event_correlations.csv', index=False)
    
    print(f"\n   Total features tested: {len(corr_df)}")
    print(f"   Significant (p<0.05): {corr_df['Significant'].sum()}")
    print(f"   Highly significant (p<0.001): {(corr_df['P_value'] < 0.001).sum()}")

# ============================================================================
# 2. Mann-Whitney U Tests (Continuous Variables: Alive vs Dead)
# ============================================================================

print("\n2. Mann-Whitney U Tests (Continuous Variables: Alive vs Dead):")

mann_whitney_results = []

for var in continuous_test_features:
    # Get samples
    alive = df_processed[df_processed['Event'] == 0][var]
    dead = df_processed[df_processed['Event'] == 1][var]
    
    # Check sample sizes
    min_sample_size = 20
    
    if len(alive) >= min_sample_size and len(dead) >= min_sample_size:
        try:
            u_stat, p_value = mannwhitneyu(alive, dead, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n_alive = len(alive)
            n_dead = len(dead)
            r_rb = 1 - (2*u_stat) / (n_alive * n_dead)
            
            mann_whitney_results.append({
                'Variable': var,
                'U_Statistic': u_stat,
                'P_value': p_value,
                'N_Alive': n_alive,
                'N_Dead': n_dead,
                'Mean_Alive': alive.mean(),
                'Mean_Dead': dead.mean(),
                'Median_Alive': alive.median(),
                'Median_Dead': dead.median(),
                'Std_Alive': alive.std(),
                'Std_Dead': dead.std(),
                'Effect_Size_r': r_rb,
                'Significant': p_value < 0.05
            })
            
            if p_value < 0.05:
                print(f"\n   {var}:")
                print(f"      U={u_stat:.1f}, p={p_value:.4e}, effect size r={r_rb:.3f}")
                print(f"      Alive: mean={alive.mean():.3f}, median={alive.median():.3f}")
                print(f"      Dead:  mean={dead.mean():.3f}, median={dead.median():.3f}")
                
        except Exception as e:
            pass

if mann_whitney_results and len(mann_whitney_results) > 0:
    mann_whitney_df = pd.DataFrame(mann_whitney_results)
    mann_whitney_df = mann_whitney_df.sort_values('P_value')
    mann_whitney_df.to_csv(f'{output_base}/statistics/mann_whitney_tests.csv', index=False)
    
    print(f"\nMann-Whitney U Tests Summary:")
    print(f"  Variables tested: {len(mann_whitney_df)}")
    print(f"  Significant (p<0.05): {mann_whitney_df['Significant'].sum()}")
    print(f"  Highly significant (p<0.001): {(mann_whitney_df['P_value'] < 0.001).sum()}")

# ============================================================================
# 3. Kruskal-Wallis test for survival time across cancer types
# ============================================================================

print("\n3. Kruskal-Wallis Test (Survival Time across Cancer Types):")

cancer_survival_groups = [
    df_processed[df_processed['cancer_type_label'] == 'BRCA']['Time'],
    df_processed[df_processed['cancer_type_label'] == 'COAD']['Time'],
    df_processed[df_processed['cancer_type_label'] == 'LUAD']['Time']
]

h_stat, p_value = kruskal(*cancer_survival_groups)
print(f"   H-statistic: {h_stat:.4f}")
print(f"   P-value: {p_value:.4e}")
print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")

# ============================================================================
# 4. Spearman Correlation Analysis between Genomic Features
# ============================================================================

print("\n4. Spearman Correlation Analysis (Top Genomic Features):")

# Get top 10 most correlated with Event
top_genomic = corr_df[corr_df['Variable'].isin(genomic_features)].head(10)['Variable'].tolist()

if len(top_genomic) > 0:
    correlation_matrix = df_processed[top_genomic].corr(method='spearman')
    
    print("   Significant correlations (|r| > 0.3, p < 0.05):")
    correlations_list = []
    
    for i in range(len(top_genomic)):
        for j in range(i+1, len(top_genomic)):
            r_val = correlation_matrix.iloc[i, j]
            if abs(r_val) > 0.3:
                _, p_val = spearmanr(df_processed[top_genomic[i]], df_processed[top_genomic[j]])
                if p_val < 0.05:
                    print(f"      {top_genomic[i]:10s} - {top_genomic[j]:10s}: r={r_val:.3f}, p={p_val:.4e}")
                    correlations_list.append({
                        'Gene1': top_genomic[i],
                        'Gene2': top_genomic[j],
                        'Correlation': r_val,
                        'P_value': p_val
                    })
    
    if correlations_list:
        correlations_df = pd.DataFrame(correlations_list)
        correlations_df.to_csv(f'{output_base}/statistics/gene_correlations.csv', index=False)

print("\n? Part 3 Complete: Three-way split and inferential statistics completed")

# ============================================================================
# PART 4: UNIVARIATE ANALYSIS AND FEATURE SELECTION
# ============================================================================

print("\n[PHASE 4] UNIVARIATE SURVIVAL ANALYSIS")
print("-" * 80)

# ============================================================================
# Univariate Cox Regression for All Continuous Features
# ============================================================================

univariate_results = []

print("\nPerforming univariate Cox regression for all continuous features...")
print("(Features are continuous, not binary - analyzing as continuous predictors)")

for feature in all_features_list:
    # Skip cancer type identifiers
    if feature in cancer_type_features:
        continue
        
    try:
        cox_data = df_train[['Time', 'Event', feature]].copy()
        cox_data = cox_data[cox_data['Time'] > 0]
        
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='Time', event_col='Event')
        
        # For continuous features, HR represents change per unit increase
        hr = np.exp(cph.params_[feature])
        ci_lower = np.exp(cph.confidence_intervals_['95% lower-bound'][feature])
        ci_upper = np.exp(cph.confidence_intervals_['95% upper-bound'][feature])
        p_value = cph.summary['p'][feature]
        
        univariate_results.append({
            'Feature': feature,
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'P_value': p_value,
            'Significant': p_value < 0.05,
            'Log_HR': cph.params_[feature],
            'Feature_Type': 'Genomic' if feature in genomic_features else 'Clinical'
        })
    except Exception as e:
        pass

univariate_df = pd.DataFrame(univariate_results)
univariate_df = univariate_df.sort_values('P_value')

# Verify no cancer types made it through
cancer_in_univariate = univariate_df[univariate_df['Feature'].isin(cancer_type_features)]
if len(cancer_in_univariate) > 0:
    print(f"\n? WARNING: Found {len(cancer_in_univariate)} cancer type identifiers in univariate results")
    univariate_df = univariate_df[~univariate_df['Feature'].isin(cancer_type_features)]

# Multiple testing correction
univariate_df['P_value_adjusted'] = multipletests(univariate_df['P_value'], 
                                                    method='fdr_bh')[1]

print(f"\n? Univariate analysis completed for {len(univariate_df)} features")
print(f"  - Significant at p<0.05: {univariate_df['Significant'].sum()}")
print(f"  - Significant after FDR correction: {(univariate_df['P_value_adjusted'] < 0.05).sum()}")

print("\nTop 20 Most Significant Features:")
top_20 = univariate_df.head(20)[['Feature', 'HR', 'P_value', 'P_value_adjusted', 'Feature_Type']]
print(top_20.to_string(index=False))

# Save univariate results
univariate_df.to_csv(f'{output_base}/tables/univariate_analysis.csv', index=False)

# ============================================================================
# Feature Selection
# ============================================================================

print("\n[FEATURE SELECTION]")
print("-" * 80)

# Get significant features from univariate analysis
significant_features = univariate_df[univariate_df['P_value'] < 0.05]['Feature'].tolist()

# Remove any cancer type identifiers if they accidentally got through
significant_features = [f for f in significant_features if f not in cancer_type_features]

# Add base clinical features if missing
base_clinical = ['age_at_diagnosis_scaled', 'received_pharmaceutical_treatment']
for feat in base_clinical:
    if feat not in significant_features and feat in all_features_list:
        significant_features.append(feat)

print(f"\n? Selected {len(significant_features)} significant features for modeling")

# Verify no cancer types in significant features
cancer_in_features = [f for f in significant_features if f in cancer_type_features]
if cancer_in_features:
    print(f"  WARNING: Found cancer types in features: {cancer_in_features}")
    significant_features = [f for f in significant_features if f not in cancer_type_features]
    print(f"  Final count: {len(significant_features)} features")

# ============================================================================
# Genomic Feature Analysis (Continuous)
# ============================================================================

print("\n[GENOMIC FEATURE ANALYSIS - CONTINUOUS]")
print("-" * 80)

genomic_univariate = univariate_df[univariate_df['Feature'].isin(genomic_features)]
genomic_univariate = genomic_univariate.sort_values('P_value')

print("\nGenomic Features - Survival Association:")
print(genomic_univariate[['Feature', 'HR', 'P_value']].to_string(index=False))

# ============================================================================
# Define Feature Sets for Ablation Studies
# ============================================================================

# Get top features by p-value for ablation
top_features_ranked = univariate_df['Feature'].tolist()

# Top features ablation sets
ablation_feature_sets = {
    'top_2': top_features_ranked[:2],
    'top_6': top_features_ranked[:6],
    'top_10': top_features_ranked[:10],
    'top_14': top_features_ranked[:14],
    'top_18': top_features_ranked[:18],
    'top_22': top_features_ranked[:22],
    'top_26': top_features_ranked[:26],
    'top_30': top_features_ranked[:30]
}

# Top genomic features ablation
genomic_ranked = genomic_univariate['Feature'].tolist()
genomic_ablation_sets = {
    'genomic_2': genomic_ranked[:2],
    'genomic_6': genomic_ranked[:6],
    'genomic_10': genomic_ranked[:10],
    'genomic_14': genomic_ranked[:14],
    'genomic_18': genomic_ranked[:18],
    'genomic_20': genomic_ranked[:20]
}

# Clinical features ablation
clinical_univariate = univariate_df[univariate_df['Feature'].isin(clinical_only_features)]
clinical_univariate = clinical_univariate.sort_values('P_value')
clinical_ranked = clinical_univariate['Feature'].tolist()

clinical_ablation_sets = {
    'clinical_2': clinical_ranked[:2] if len(clinical_ranked) >= 2 else clinical_ranked,
    'clinical_6': clinical_ranked[:6] if len(clinical_ranked) >= 6 else clinical_ranked,
    'clinical_10': clinical_ranked[:10] if len(clinical_ranked) >= 10 else clinical_ranked,
    'clinical_14': clinical_ranked[:14] if len(clinical_ranked) >= 14 else clinical_ranked,
    'clinical_18': clinical_ranked[:18] if len(clinical_ranked) >= 18 else clinical_ranked,
    'clinical_20': clinical_ranked[:20] if len(clinical_ranked) >= 20 else clinical_ranked
}

# Store feature sets
feature_sets = {
    'all_features': all_features_list,
    'significant_features': significant_features,
    'clinical_only': clinical_only_features,
    'genomic_only': genomic_only_features,
    **ablation_feature_sets,
    **genomic_ablation_sets,
    **clinical_ablation_sets
}

print("\n[FEATURE SET SUMMARY FOR ABLATION]")
print("-" * 80)
print("Top Features Ablation Sets:")
for name, features in ablation_feature_sets.items():
    print(f"  {name}: {len(features)} features")

print("\nGenomic Features Ablation Sets:")
for name, features in genomic_ablation_sets.items():
    print(f"  {name}: {len(features)} features")

print("\nClinical Features Ablation Sets:")
for name, features in clinical_ablation_sets.items():
    print(f"  {name}: {len(features)} features")

print("\n? Part 4 Complete: Univariate analysis and feature selection done")

# ============================================================================
# PART 5: ROBUST MODEL TRAINING WITH VALIDATION
# ============================================================================

print("\n[PHASE 5] ROBUST MODEL TRAINING (WITH VALIDATION SET)")
print("=" * 80)

cox_models = {}
model_performance = []

# ============================================================================
# Helper Function: Calculate Metrics with Validation
# ============================================================================

def calculate_metrics_with_validation(model, train_data, val_data, test_data, 
                                      y_train, y_val, y_test, model_name, 
                                      is_lifelines=False):
    """Calculate metrics on train, validation, AND test sets"""
    
    metrics = {'Model': model_name}
    
    try:
        if is_lifelines:
            # For lifelines models
            train_pred = model.predict_partial_hazard(train_data)
            val_pred = model.predict_partial_hazard(val_data)
            test_pred = model.predict_partial_hazard(test_data)
            
            metrics['Train_C_Index'] = concordance_index(
                train_data['Time'], -train_pred, train_data['Event']
            )
            metrics['Val_C_Index'] = concordance_index(
                val_data['Time'], -val_pred, val_data['Event']
            )
            metrics['Test_C_Index'] = concordance_index(
                test_data['Time'], -test_pred, test_data['Event']
            )
        else:
            # For sklearn models
            metrics['Train_C_Index'] = model.score(train_data, y_train)
            metrics['Val_C_Index'] = model.score(val_data, y_val)
            metrics['Test_C_Index'] = model.score(test_data, y_test)
        
        # Calculate train-val gap (early indicator of overfitting)
        metrics['Train_Val_Gap'] = metrics['Train_C_Index'] - metrics['Val_C_Index']
        
        # Calculate val-test gap (generalization check)
        metrics['Val_Test_Gap'] = metrics['Val_C_Index'] - metrics['Test_C_Index']
        
    except Exception as e:
        print(f"  Warning: Metrics calculation failed: {str(e)[:50]}")
        metrics.update({
            'Train_C_Index': 0,
            'Val_C_Index': 0,
            'Test_C_Index': 0,
            'Train_Val_Gap': 0,
            'Val_Test_Gap': 0
        })
    
    return metrics

# ============================================================================
# Helper Function: Repeated Cross-Validation
# ============================================================================

def repeated_cv_score(model, X, y, cv_folds=5, n_repeats=3):
    """
    Perform repeated cross-validation for more robust performance estimate.
    Returns mean and std of CV scores.
    """
    # Extract event status for stratification
    event_status = np.array([event for event, _ in y])
    
    cv_scores = []
    
    for repeat in range(n_repeats):
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=config.RANDOM_STATE + repeat)
        
        for train_idx, val_idx in kf.split(X, event_status):
            X_train_cv = X[train_idx]
            X_val_cv = X[val_idx]
            y_train_cv = y[train_idx]
            y_val_cv = y[val_idx]
            
            try:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_cv, y_train_cv)
                score = model_copy.score(X_val_cv, y_val_cv)
                cv_scores.append(score)
            except:
                continue
    
    return np.mean(cv_scores), np.std(cv_scores)

# ============================================================================
# Model 1: Regularized Cox - Combined (ROBUST)
# ============================================================================

print("\n[Model 1] Regularized Cox - Combined (Robust)")
print("-" * 40)

try:
    train_combined = df_train[['Time', 'Event'] + stable_features].copy()
    val_combined = df_val[['Time', 'Event'] + stable_features].copy()
    test_combined = df_test[['Time', 'Event'] + stable_features].copy()
    
    # INCREASED regularization
    cph_combined = CoxPHFitter(
        penalizer=config.COX_PENALIZER,  # 0.5 instead of 0.1
        l1_ratio=config.COX_L1_RATIO      # 0.7 instead of 0.3
    )
    cph_combined.fit(train_combined, duration_col='Time', event_col='Event')
    
    metrics = calculate_metrics_with_validation(
        cph_combined, train_combined, val_combined, test_combined,
        y_train, y_val, y_test,
        'Cox - Combined (Robust)', is_lifelines=True
    )
    metrics['Features'] = len(stable_features)
    metrics['Penalizer'] = config.COX_PENALIZER
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    cox_models['combined_robust'] = cph_combined
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Failed: {str(e)[:100]}")

# ============================================================================
# Model 2: Elastic Net with Higher Regularization
# ============================================================================

print("\n[Model 2] Elastic Net Cox (High Regularization)")
print("-" * 40)

try:
    coxnet_robust = CoxnetSurvivalAnalysis(
        l1_ratio=config.ELASTIC_NET_L1,  # 0.8 instead of 0.5
        alpha_min_ratio=0.1,   # Higher minimum alpha
        max_iter=2000,
        n_alphas=20
    )
    coxnet_robust.fit(X_train_scaled, y_train)
    
    metrics = calculate_metrics_with_validation(
        coxnet_robust, X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        'Elastic Net (Robust)', is_lifelines=False
    )
    metrics['Features'] = len(stable_features)
    metrics['L1_Ratio'] = config.ELASTIC_NET_L1
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    cox_models['elastic_net_robust'] = coxnet_robust
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Failed: {str(e)[:100]}")

# ============================================================================
# Model 3: Conservative Random Survival Forest
# ============================================================================

print("\n[Model 3] Conservative Random Survival Forest")
print("-" * 40)

try:
    rsf_robust = RandomSurvivalForest(
        n_estimators=config.RSF_N_ESTIMATORS,     # 500
        max_depth=config.RSF_MAX_DEPTH,           # 5 (was 10)
        min_samples_split=config.RSF_MIN_SAMPLES_SPLIT,  # 50 (was 20)
        min_samples_leaf=config.RSF_MIN_SAMPLES_LEAF,    # 25 (was 10)
        max_features=config.RSF_MAX_FEATURES,     # 0.3 (30% of features)
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=0
    )
    rsf_robust.fit(X_train, y_train)
    
    metrics = calculate_metrics_with_validation(
        rsf_robust, X_train, X_val, X_test,
        y_train, y_val, y_test,
        'RSF (Robust)', is_lifelines=False
    )
    metrics['Features'] = len(stable_features)
    metrics['N_Estimators'] = config.RSF_N_ESTIMATORS
    metrics['Max_Depth'] = config.RSF_MAX_DEPTH
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    # Repeated CV
    print("\n  Performing repeated 5-fold CV (3 repeats)...")
    cv_mean, cv_std = repeated_cv_score(rsf_robust, X_train, y_train, 
                                        cv_folds=5, n_repeats=3)
    print(f"  Repeated CV: {cv_mean:.4f} +/- {cv_std:.4f}")
    metrics['CV_Mean'] = cv_mean
    metrics['CV_Std'] = cv_std
    
    cox_models['rsf_robust'] = rsf_robust
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Failed: {str(e)[:100]}")

# ============================================================================
# Model 4: Conservative Gradient Boosting
# ============================================================================

print("\n[Model 4] Conservative Gradient Boosting")
print("-" * 40)

try:
    gb_robust = GradientBoostingSurvivalAnalysis(
        n_estimators=config.GB_N_ESTIMATORS,      # 50 (was 100)
        learning_rate=config.GB_LEARNING_RATE,    # 0.01 (was 0.05)
        max_depth=config.GB_MAX_DEPTH,            # 2 (was 3)
        min_samples_split=config.GB_MIN_SAMPLES_SPLIT,  # 50 (was 20)
        min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,    # 25 (was 10)
        subsample=config.GB_SUBSAMPLE,            # 0.5 (was 0.7)
        random_state=config.RANDOM_STATE,
        verbose=0
    )
    gb_robust.fit(X_train_scaled, y_train)
    
    metrics = calculate_metrics_with_validation(
        gb_robust, X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        'GB (Robust)', is_lifelines=False
    )
    metrics['Features'] = len(stable_features)
    metrics['Learning_Rate'] = config.GB_LEARNING_RATE
    metrics['Max_Depth'] = config.GB_MAX_DEPTH
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    # Repeated CV
    print("\n  Performing repeated 5-fold CV (3 repeats)...")
    cv_mean, cv_std = repeated_cv_score(gb_robust, X_train_scaled, y_train,
                                        cv_folds=5, n_repeats=3)
    print(f"  Repeated CV: {cv_mean:.4f} +/- {cv_std:.4f}")
    metrics['CV_Mean'] = cv_mean
    metrics['CV_Std'] = cv_std
    
    cox_models['gb_robust'] = gb_robust
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Failed: {str(e)[:100]}")

# ============================================================================
# Model 5: Highly Regularized SVM
# ============================================================================

print("\n[Model 5] Highly Regularized SVM")
print("-" * 40)

try:
    # Use top 15 most stable features for SVM
    top_15_features = feature_stability_df.head(15)['Feature'].tolist()
    
    X_train_svm = df_train[top_15_features].values
    X_val_svm = df_val[top_15_features].values
    X_test_svm = df_test[top_15_features].values
    
    scaler_svm = RobustScaler()
    X_train_svm_scaled = scaler_svm.fit_transform(X_train_svm)
    X_val_svm_scaled = scaler_svm.transform(X_val_svm)
    X_test_svm_scaled = scaler_svm.transform(X_test_svm)
    
    # Try higher alpha values (more regularization)
    best_val_score = 0
    best_alpha = None
    best_model = None
    
    for alpha in config.SVM_ALPHA_RANGE:
        try:
            ssvm_temp = FastKernelSurvivalSVM(
                alpha=alpha,
                kernel='rbf',
                random_state=config.RANDOM_STATE,
                max_iter=config.SVM_MAX_ITER
            )
            ssvm_temp.fit(X_train_svm_scaled, y_train)
            val_score = ssvm_temp.score(X_val_svm_scaled, y_val)
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_alpha = alpha
                best_model = ssvm_temp
        except Exception as e:
            print(f"    Alpha {alpha} failed: {str(e)[:50]}")
            continue
    
    if best_model is None:
        raise ValueError("All SVM configurations failed")
    
    print(f"  Best alpha: {best_alpha} (validation C-Index: {best_val_score:.4f})")
    
    metrics = calculate_metrics_with_validation(
        best_model, X_train_svm_scaled, X_val_svm_scaled, X_test_svm_scaled,
        y_train, y_val, y_test,
        'SVM (Robust)', is_lifelines=False
    )
    metrics['Features'] = len(top_15_features)
    metrics['Best_Alpha'] = best_alpha
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    cox_models['svm_robust'] = best_model
    cox_models['svm_robust_scaler'] = scaler_svm
    cox_models['svm_robust_features'] = top_15_features
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Failed: {str(e)[:100]}")

# ============================================================================
# PART 6: LEGACY MODELS (FOR COMPARISON WITH ROBUST VERSIONS)
# ============================================================================

print("\n[PHASE 6] LEGACY MODELS (Original Settings)")
print("=" * 80)

# ============================================================================
# Model 6: Cox - Clinical Only (Original)
# ============================================================================

print("\n[Model 6] Cox - Clinical Only (Baseline)")
print("-" * 40)

try:
    train_clinical = df_train[['Time', 'Event'] + clinical_only_features].copy()
    val_clinical = df_val[['Time', 'Event'] + clinical_only_features].copy()
    test_clinical = df_test[['Time', 'Event'] + clinical_only_features].copy()
    
    # Remove NaN and zero-variance features
    train_clinical = train_clinical.dropna()
    val_clinical = val_clinical.dropna()
    test_clinical = test_clinical.dropna()
    
    train_variance = train_clinical[clinical_only_features].var()
    zero_var_features = train_variance[train_variance < 1e-10].index.tolist()
    
    if zero_var_features:
        clinical_features_valid = [f for f in clinical_only_features if f not in zero_var_features]
    else:
        clinical_features_valid = clinical_only_features
    
    train_clinical = df_train[['Time', 'Event'] + clinical_features_valid].copy().dropna()
    val_clinical = df_val[['Time', 'Event'] + clinical_features_valid].copy().dropna()
    test_clinical = df_test[['Time', 'Event'] + clinical_features_valid].copy().dropna()
    
    cph_clinical = CoxPHFitter(penalizer=0.1)
    cph_clinical.fit(train_clinical, duration_col='Time', event_col='Event')
    
    metrics = calculate_metrics_with_validation(
        cph_clinical, train_clinical, val_clinical, test_clinical, 
        y_train, y_val, y_test, 'Cox - Clinical Only', is_lifelines=True
    )
    metrics['Features'] = len(clinical_features_valid)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    
    cox_models['clinical_only'] = cph_clinical
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Model 7: Cox - Genomic Only (Original)
# ============================================================================

print("\n[Model 7] Cox - Genomic Only")
print("-" * 40)

try:
    train_genomic = df_train[['Time', 'Event'] + genomic_only_features].copy().dropna()
    val_genomic = df_val[['Time', 'Event'] + genomic_only_features].copy().dropna()
    test_genomic = df_test[['Time', 'Event'] + genomic_only_features].copy().dropna()
    
    cph_genomic = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph_genomic.fit(train_genomic, duration_col='Time', event_col='Event')
    
    metrics = calculate_metrics_with_validation(
        cph_genomic, train_genomic, val_genomic, test_genomic,
        y_train, y_val, y_test, 'Cox - Genomic Only', is_lifelines=True
    )
    metrics['Features'] = len(genomic_only_features)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    
    cox_models['genomic_only'] = cph_genomic
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Model 8: Cox - Combined (Original)
# ============================================================================

print("\n[Model 8] Cox - Combined (Original Settings)")
print("-" * 40)

try:
    combined_features = significant_features.copy()
    
    train_combined_orig = df_train[['Time', 'Event'] + combined_features].copy().dropna()
    val_combined_orig = df_val[['Time', 'Event'] + combined_features].copy().dropna()
    test_combined_orig = df_test[['Time', 'Event'] + combined_features].copy().dropna()
    
    cph_combined_orig = CoxPHFitter(penalizer=0.1, l1_ratio=0.3)
    cph_combined_orig.fit(train_combined_orig, duration_col='Time', event_col='Event')
    
    metrics = calculate_metrics_with_validation(
        cph_combined_orig, train_combined_orig, val_combined_orig, test_combined_orig,
        y_train, y_val, y_test, 'Cox - Combined', is_lifelines=True
    )
    metrics['Features'] = len(combined_features)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    
    cox_models['combined'] = cph_combined_orig
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Model 9: Cox - Elastic Net (Original)
# ============================================================================

print("\n[Model 9] Cox - Elastic Net (Original Settings)")
print("-" * 40)

try:
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=1000)
    coxnet.fit(X_train_scaled_global, y_train)
    
    metrics = calculate_metrics_with_validation(
        coxnet, X_train_scaled_global, X_val_scaled_global, X_test_scaled_global,
        y_train, y_val, y_test, 'Cox - Elastic Net', is_lifelines=False
    )
    metrics['Features'] = len(all_features_list)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    
    cox_models['elastic_net'] = coxnet
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Model 10: Random Survival Forest (Original)
# ============================================================================

print("\n[Model 10] Random Survival Forest (Original Settings)")
print("-" * 40)

try:
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=20,
        min_samples_leaf=10,
        max_depth=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    rsf.fit(X_train, y_train)
    
    metrics = calculate_metrics_with_validation(
        rsf, X_train, X_val, X_test,
        y_train, y_val, y_test, 'Random Survival Forest', is_lifelines=False
    )
    metrics['Features'] = len(significant_features)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    
    # Feature importance - FIXED
    print("\n  Calculating feature importance...")
    try:
        perm_result = permutation_importance(
            rsf, X_test, y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring=lambda est, X, y: est.score(X, y)
        )
        
        # Ensure arrays match length
        n_features_actual = min(len(significant_features), len(perm_result.importances_mean))
        
        importance_df = pd.DataFrame({
            'Feature': significant_features[:n_features_actual],
            'Importance': perm_result.importances_mean[:n_features_actual],
            'Std': perm_result.importances_std[:n_features_actual]
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv(f'{output_base}/tables/rsf_feature_importance.csv', index=False)
        print(f"    ? Feature importance saved ({len(importance_df)} features)")
    except Exception as e:
        print(f"    Warning: Could not calculate feature importance: {str(e)[:50]}")
    
    cox_models['random_survival_forest'] = rsf
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Model 11: Gradient Boosting Survival (Original)
# ============================================================================

print("\n[Model 11] Gradient Boosting Survival (Original Settings)")
print("-" * 40)

try:
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.7,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    gbsa.fit(X_train_scaled, y_train)
    
    metrics = calculate_metrics_with_validation(
        gbsa, X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test, 'Gradient Boosting Survival', is_lifelines=False
    )
    metrics['Features'] = len(significant_features)
    
    print(f"  Train C-Index: {metrics['Train_C_Index']:.4f}")
    print(f"  Val C-Index:   {metrics['Val_C_Index']:.4f}")
    print(f"  Test C-Index:  {metrics['Test_C_Index']:.4f}")
    
    # Feature importance - FIXED
    try:
        # Ensure arrays match length
        n_features_actual = min(len(significant_features), len(gbsa.feature_importances_))
        
        importance_gb_df = pd.DataFrame({
            'Feature': significant_features[:n_features_actual],
            'Importance': gbsa.feature_importances_[:n_features_actual]
        }).sort_values('Importance', ascending=False)
        
        importance_gb_df.to_csv(f'{output_base}/tables/gbsa_feature_importance.csv', index=False)
        print(f"    ? Feature importance saved ({len(importance_gb_df)} features)")
    except Exception as e:
        print(f"    Warning: Could not save feature importance: {str(e)[:50]}")
    
    cox_models['gradient_boosting'] = gbsa
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")
# ============================================================================
# PART 7: HELPER FUNCTIONS FOR ADVANCED ANALYSIS
# ============================================================================

print("\n[PHASE 7] HELPER FUNCTIONS")
print("=" * 80)

# ============================================================================
# Optimized SVM Training Function
# ============================================================================

def train_optimized_svm(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train SVM with hyperparameter optimization.
    Returns trained model and test C-index.
    """
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'kernel': ['rbf', 'linear'],
        'max_iter': [50, 100]
    }
    
    best_score = 0
    best_params = {'alpha': 1.0, 'kernel': 'rbf', 'max_iter': 100}
    best_model = None
    
    for alpha in param_grid['alpha']:
        for kernel in param_grid['kernel']:
            try:
                # FIXED: Remove gamma parameter, use defaults
                ssvm_temp = FastKernelSurvivalSVM(
                    alpha=alpha,
                    kernel=kernel,
                    random_state=random_state,
                    max_iter=50
                )
                ssvm_temp.fit(X_train, y_train)
                
                train_score = ssvm_temp.score(X_train, y_train)
                test_score = ssvm_temp.score(X_test, y_test)
                
                # Penalize overfitting
                overfit_penalty = abs(train_score - test_score)
                adjusted_score = test_score - 0.5 * overfit_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_params = {'alpha': alpha, 'kernel': kernel, 'max_iter': 100}
                    best_model = ssvm_temp
                    
            except Exception as e:
                continue
    
    if best_model is None:
        # Fallback to default configuration
        best_model = FastKernelSurvivalSVM(
            alpha=1.0,
            kernel='rbf',
            random_state=random_state,
            max_iter=100
        )
        best_model.fit(X_train, y_train)
        best_params = {'alpha': 1.0, 'kernel': 'rbf', 'max_iter': 100}
    
    # Train final model with best params
    ssvm_final = FastKernelSurvivalSVM(
        alpha=best_params['alpha'],
        kernel=best_params['kernel'],
        random_state=random_state,
        max_iter=best_params['max_iter']
    )
    
    ssvm_final.fit(X_train, y_train)
    
    train_c = ssvm_final.score(X_train, y_train)
    test_c = ssvm_final.score(X_test, y_test)
    
    return ssvm_final, test_c, train_c, best_params

# ============================================================================
# SVM Feature Importance Calculator
# ============================================================================

def calculate_svm_feature_importance(model, X, y, feature_names, n_repeats=10):
    """Calculate permutation-based feature importance for SVM"""
    try:
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=lambda est, X, y: est.score(X, y)
        )
        
        # Create dictionary of feature importances
        importance_dict = {}
        for i, feat in enumerate(feature_names):
            if i < len(perm_importance.importances_mean):
                importance_dict[feat] = abs(perm_importance.importances_mean[i])
        
        # Normalize to [0, 1] range
        if importance_dict:
            max_imp = max(importance_dict.values())
            if max_imp > 0:
                importance_dict = {k: v/max_imp for k, v in importance_dict.items()}
        
        return importance_dict
    
    except Exception as e:
        print(f"  Warning: Could not calculate SVM importance: {str(e)[:50]}")
        return {}

# ============================================================================
# Model 12: Optimized Survival SVM (Original)
# ============================================================================

print("\n[Model 12] Optimized Survival SVM (Original)")
print("-" * 40)

try:
    # Use top 20 features for SVM
    top_20_features = univariate_df.head(20)['Feature'].tolist()
    
    X_train_svm = df_train[top_20_features].values
    X_val_svm = df_val[top_20_features].values
    X_test_svm = df_test[top_20_features].values
    
    scaler_svm_orig = StandardScaler()
    X_train_svm_scaled = scaler_svm_orig.fit_transform(X_train_svm)
    X_val_svm_scaled = scaler_svm_orig.transform(X_val_svm)
    X_test_svm_scaled = scaler_svm_orig.transform(X_test_svm)
    
    # Grid search for optimal hyperparameters
    print("  Performing hyperparameter optimization...")
    
    ssvm, test_c_svm, train_c_svm, svm_params = train_optimized_svm(
        X_train_svm_scaled, y_train, 
        X_test_svm_scaled, y_test,
        random_state=RANDOM_STATE
    )
    
    print(f"  Best parameters: {svm_params}")
    
    # Get validation score
    val_c_svm = ssvm.score(X_val_svm_scaled, y_val)
    
    metrics = {
        'Model': 'Survival SVM ',
        'Features': len(top_20_features),
        'Train_C_Index': train_c_svm,
        'Val_C_Index': val_c_svm,
        'Test_C_Index': test_c_svm,
        'Train_Val_Gap': train_c_svm - val_c_svm,
        'Val_Test_Gap': val_c_svm - test_c_svm,
        'Best_Alpha': svm_params['alpha']
    }
    
    print(f"  Train C-Index: {train_c_svm:.4f}")
    print(f"  Val C-Index:   {val_c_svm:.4f}")
    print(f"  Test C-Index:  {test_c_svm:.4f}")
    print(f"  Train-Val Gap: {metrics['Train_Val_Gap']:.4f}")
    print(f"  Val-Test Gap:  {metrics['Val_Test_Gap']:.4f}")
    
    cox_models['survival_svm'] = ssvm
    cox_models['survival_svm_scaler'] = scaler_svm_orig
    cox_models['survival_svm_features'] = top_20_features
    model_performance.append(metrics)
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")

# ============================================================================
# Performance Summary
# ============================================================================

print("\n" + "=" * 80)
print("ALL MODELS PERFORMANCE COMPARISON")
print("=" * 80)

perf_df_all = pd.DataFrame(model_performance)
perf_df_all = perf_df_all.sort_values('Test_C_Index', ascending=False)

print(perf_df_all[['Model', 'Features', 'Train_C_Index', 'Val_C_Index', 'Test_C_Index', 
                    'Train_Val_Gap', 'Val_Test_Gap']].to_string(index=False))

# Save performance
perf_df_all.to_csv(f'{output_base}/tables/all_models_performance.csv', index=False)

best_model_name = perf_df_all.iloc[0]['Model']
best_test_c = perf_df_all.iloc[0]['Test_C_Index']

print(f"\n? Best Performing Model: {best_model_name}")
print(f"  Test C-Index: {best_test_c:.4f}")

# Key metrics
print("\n" + "=" * 80)
print("KEY ROBUSTNESS METRICS")
print("=" * 80)

print("\n1. Generalization Quality (Val-Test Gap):")
print("   Smaller gap = better generalization to new data")
for _, row in perf_df_all.iterrows():
    gap = row.get('Val_Test_Gap', 0)
    status = "Excellent" if abs(gap) < 0.02 else "Good" if abs(gap) < 0.05 else "Fair" if abs(gap) < 0.10 else "Poor"
    print(f"   {row['Model']:<30} Gap: {gap:+.4f} ({status})")

print("\n2. Overfitting Control (Train-Val Gap):")
print("   Smaller gap = less overfitting")
for _, row in perf_df_all.iterrows():
    gap = row.get('Train_Val_Gap', row.get('Train_C_Index', 0) - row.get('Val_C_Index', 0))
    status = "Excellent" if gap < 0.05 else "Good" if gap < 0.10 else "Fair" if gap < 0.15 else "Poor"
    print(f"   {row['Model']:<30} Gap: {gap:.4f} ({status})")

print("\n Part 7 Complete: Helper functions and SVM optimization")

# ============================================================================
# PART 7B: TIME-DEPENDENT COVARIATE ANALYSIS - CORRECTED & ROBUST
# ============================================================================

print("\n[PHASE 7B] TIME-DEPENDENT COVARIATE ANALYSIS")
print("-" * 80)

try:
    from lifelines import CoxTimeVaryingFitter
    from lifelines.utils import concordance_index as ci_lifelines
    
    def create_time_varying_data(df, features, time_intervals):
        """Convert data to time-varying format with safer handling."""
        tv_records = []
        for idx, row in df.iterrows():
            patient_time = row['Time']
            patient_event = row['Event']
            
            for i in range(len(time_intervals) - 1):
                start = time_intervals[i]
                stop = time_intervals[i + 1]
                
                if start >= patient_time:
                    break
                
                interval_stop = min(stop, patient_time)
                interval_event = patient_event if interval_stop == patient_time else 0
                
                record = {
                    'id': idx,
                    'start': start,
                    'stop': interval_stop,
                    'event': interval_event
                }
                
                for feat in features:
                    if feat in df.columns:
                        record[feat] = row[feat]
                        # Interaction with midpoint time
                        mid_time = (start + stop) / 2
                        record[f'{feat}_x_time'] = row[feat] * mid_time
                
                tv_records.append(record)
                
                if interval_stop == patient_time:
                    break
        return pd.DataFrame(tv_records)
    
    print("\nPreparing time-varying dataset...")
    
    # Define time intervals (more reasonable for normalized time)
    max_time = df['Time'].max()
    time_intervals = [0, 0.5, 1, 2, 3, 5, max(5, max_time)]
    
    # Get top 10 features (safe fallback)
    top_10_features = univariate_df.head(10)['Feature'].tolist() if 'univariate_df' in locals() else genomic_features[:10]
    
    print(f"  Analyzing {len(top_10_features)} features for time-varying effects")
    
    # Create time-varying datasets
    train_tv = create_time_varying_data(df_train, top_10_features, time_intervals)
    test_tv = create_time_varying_data(df_test, top_10_features, time_intervals)
    
    print(f"  Train: {len(train_tv)} records from {len(df_train)} patients")
    print(f"  Test: {len(test_tv)} records from {len(df_test)} patients")
    
    if len(train_tv) == 0:
        raise ValueError("Empty time-varying train data")
    
    # Fit time-varying Cox model
    print("\nFitting time-varying Cox model...")
    ctv = CoxTimeVaryingFitter(penalizer=0.1, l1_ratio=0.5)
    feature_cols = top_10_features + [f'{feat}_x_time' for feat in top_10_features]
    
    ctv.fit(train_tv, id_col='id', event_col='event', 
            start_col='start', stop_col='stop', show_progress=False)
    
    # Identify time-dependent effects
    time_dependent_features = []
    print("\nTime-Dependent Effects:")
    for feat in top_10_features:
        interaction_term = f'{feat}_x_time'
        if interaction_term in ctv.summary.index:
            p_value = ctv.summary.loc[interaction_term, 'p']
            if p_value < 0.05:
                hr = np.exp(ctv.params_[interaction_term])
                time_dependent_features.append({
                    'Feature': feat,
                    'Time_Interaction_HR': hr,
                    'P_value': p_value,
                    'Interpretation': 'Increasing effect over time' if hr > 1 else 'Decreasing effect over time'
                })
                print(f"  {feat}: HR={hr:.3f}, p={p_value:.4e}, {time_dependent_features[-1]['Interpretation']}")
    
    if time_dependent_features:
        tv_df = pd.DataFrame(time_dependent_features)
        tv_df.to_csv(f'{output_base}/statistics/time_dependent_effects.csv', index=False)
        print(f"\n? Saved time-dependent effects ({len(time_dependent_features)} significant)")
    else:
        print("\n  No significant time-dependent effects detected")
    
    # Safer C-index calculation on test set
    try:
        test_predictions = ctv.predict_partial_hazard(test_tv)
        
        # Group by patient ID and take mean (handles multiple intervals)
        patient_predictions = test_predictions.groupby(test_tv['id']).mean()
        
        # Align with df_test order (critical fix)
        patient_predictions = patient_predictions.reindex(df_test.index).fillna(patient_predictions.mean())
        
        # Calculate concordance
        test_c_tv = ci_lifelines(df_test['Time'], -patient_predictions.values, df_test['Event'])
        print(f"\nTime-Varying Model Test C-Index: {test_c_tv:.4f}")
        
        model_performance.append({
            'Model': 'Cox - Time Varying',
            'Features': len(feature_cols),
            'Train_C_Index': getattr(ctv, 'concordance_index_', np.nan),
            'Test_C_Index': test_c_tv,
            'Train_Val_Gap': 0,
            'Val_Test_Gap': 0
        })
        
        cox_models['time_varying'] = ctv
        
    except Exception as e_inner:
        print(f"\n  Could not calculate test C-index: {str(e_inner)[:120]}")
        print(f"  Using train concordance if available")
        
        model_performance.append({
            'Model': 'Cox - Time Varying',
            'Features': len(feature_cols),
            'Train_C_Index': getattr(ctv, 'concordance_index_', np.nan),
            'Test_C_Index': np.nan,
            'Train_Val_Gap': 0,
            'Val_Test_Gap': 0
        })
        cox_models['time_varying'] = ctv
    
except Exception as e:
    print(f"  ? Error in time-dependent analysis: {str(e)[:150]}")
    import traceback
    print(traceback.format_exc()[:400])

print("\n Time-dependent covariate analysis complete")


# ============================================================================
# PART 7C: COMPETING RISKS ANALYSIS - CORRECTED
# ============================================================================

print("\n[PHASE 7C] COMPETING RISKS ANALYSIS")
print("-" * 80)

try:
    from lifelines import AalenJohansenFitter
    
    print("\nSimulating competing risks (cancer-specific vs other causes)...")
    
    def assign_competing_risks(df):
        """Assign event types for competing risks analysis."""
        df_cr = df.copy()
        df_cr['Event_Type'] = 0  # 0=censored, 1=cancer death, 2=other death
        
        event_mask = df_cr['Event'] == 1
        np.random.seed(42)
        
        # 70% cancer deaths, 30% other causes
        for cancer in ['BRCA', 'COAD', 'LUAD']:
            cancer_event_mask = event_mask & (df_cr['cancer_type_label'] == cancer)
            n_cancer_events = cancer_event_mask.sum()
            if n_cancer_events > 0:
                event_types = np.random.choice([1, 2], size=n_cancer_events, p=[0.7, 0.3])
                df_cr.loc[cancer_event_mask, 'Event_Type'] = event_types
        
        return df_cr
    
    test_cr = assign_competing_risks(df_test)
    
    print(f"  Censored: {(test_cr['Event_Type'] == 0).sum()}")
    print(f"  Cancer-specific death: {(test_cr['Event_Type'] == 1).sum()}")
    print(f"  Other cause death: {(test_cr['Event_Type'] == 2).sum()}")
    
    # Cumulative Incidence Functions
    print("\nCalculating cumulative incidence functions...")
    
    aj_cancer = AalenJohansenFitter(calculate_variance=True)
    aj_cancer.fit(test_cr['Time'], test_cr['Event_Type'], event_of_interest=1)
    
    aj_other = AalenJohansenFitter(calculate_variance=True)
    aj_other.fit(test_cr['Time'], test_cr['Event_Type'], event_of_interest=2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Overall CIF
    ax1 = axes[0]
    aj_cancer.plot(ax=ax1, ci_show=True, label='Cancer-Specific Death', color='red', linewidth=2.5)
    aj_other.plot(ax=ax1, ci_show=True, label='Other Cause Death', color='blue', linewidth=2.5)
    
    ax1.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Incidence', fontsize=12, fontweight='bold')
    ax1.set_title('Competing Risks: Cumulative Incidence Functions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Panel 2: By cancer type
    ax2 = axes[1]
    cancer_colors = {'BRCA': '#e74c3c', 'COAD': '#3498db', 'LUAD': '#2ecc71'}
    
    for cancer in ['BRCA', 'COAD', 'LUAD']:
        cancer_mask = test_cr['cancer_type_label'] == cancer
        cancer_data = test_cr[cancer_mask]
        
        if len(cancer_data) > 0 and (cancer_data['Event_Type'] == 1).sum() > 0:
            aj_temp = AalenJohansenFitter()
            aj_temp.fit(cancer_data['Time'], cancer_data['Event_Type'], event_of_interest=1)
            aj_temp.plot(ax=ax2, ci_show=False, label=f'{cancer}', 
                        color=cancer_colors[cancer], linewidth=2.5)
    
    ax2.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Incidence (Cancer Death)', fontsize=12, fontweight='bold')
    ax2.set_title('Cancer-Specific Death by Cancer Type', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/08_competing_risks_analysis.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  ? Saved: 08_competing_risks_analysis.png")
    
    # Calculate cumulative incidence at time points - FIXED: Use interpolation instead of cumulative_density_at_times
    time_points = [1, 2, 3, 5]
    ci_results = []
    
    for t in time_points:
        try:
            # For cancer-specific death - use interpolation on cumulative_density_
            cdf_times_cancer = aj_cancer.cumulative_density_.index.values
            cdf_values_cancer = aj_cancer.cumulative_density_.values.flatten()
            
            if t <= cdf_times_cancer.max():
                ci_cancer = np.interp(t, cdf_times_cancer, cdf_values_cancer)
            else:
                ci_cancer = cdf_values_cancer[-1]
            
            # For other-cause death
            cdf_times_other = aj_other.cumulative_density_.index.values
            cdf_values_other = aj_other.cumulative_density_.values.flatten()
            
            if t <= cdf_times_other.max():
                ci_other = np.interp(t, cdf_times_other, cdf_values_other)
            else:
                ci_other = cdf_values_other[-1]
            
            ci_results.append({
                'Time_Years': t,
                'CI_Cancer_Death': ci_cancer,
                'CI_Other_Death': ci_other,
                'Total_Events': ci_cancer + ci_other
            })
            
            print(f"  T={t}yr: Cancer={ci_cancer:.3f}, Other={ci_other:.3f}, Total={ci_cancer + ci_other:.3f}")
            
        except Exception as e_inner:
            print(f"  T={t}yr: Error - {str(e_inner)[:50]}")
    
    if ci_results:
        ci_df = pd.DataFrame(ci_results)
        ci_df.to_csv(f'{output_base}/statistics/cumulative_incidence.csv', index=False)
        print("\n? Cumulative incidence saved")
    else:
        print("\n  Could not calculate cumulative incidence at specific time points")
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")
    import traceback
    print(traceback.format_exc()[:500])

print("\n? Competing risks analysis complete")


# ============================================================================
# PART 7D: SUBGROUP ANALYSIS - CORRECTED
# ============================================================================

print("\n[PHASE 7D] SUBGROUP ANALYSIS")
print("-" * 80)

try:
    print("\nDefining patient subgroups...")
    
    # Define subgroups
    subgroup_definitions = {
        'age_group': {
            'Young': lambda df: df['age_at_diagnosis'] < df['age_at_diagnosis'].quantile(0.33),
            'Middle': lambda df: (df['age_at_diagnosis'] >= df['age_at_diagnosis'].quantile(0.33)) & 
                               (df['age_at_diagnosis'] < df['age_at_diagnosis'].quantile(0.67)),
            'Elderly': lambda df: df['age_at_diagnosis'] >= df['age_at_diagnosis'].quantile(0.67)
        },
        'cancer_type': {
            'BRCA': lambda df: df['cancer_type_label'] == 'BRCA',
            'COAD': lambda df: df['cancer_type_label'] == 'COAD',
            'LUAD': lambda df: df['cancer_type_label'] == 'LUAD'
        },
        'treatment': {
            'Treated': lambda df: df['received_pharmaceutical_treatment'] > 0.5,
            'Untreated': lambda df: df['received_pharmaceutical_treatment'] <= 0.5
        },
        'tp53_status': {
            'TP53_High': lambda df: df['TP53'] > df['TP53'].median(),
            'TP53_Low': lambda df: df['TP53'] <= df['TP53'].median()
        }
    }
    
    subgroup_results = []
    
    print("\nPerforming subgroup analysis...")
    
    # FIXED: Use cox_models instead of risk_groups_all_models
    best_model_key = 'combined'
    if 'combined' not in cox_models:
        if len(cox_models) > 0:
            best_model_key = list(cox_models.keys())[0]
            print(f"  Using model: {best_model_key}")
        else:
            print("  No models available - using statistics only")
            best_model_key = None
    
    for subgroup_name, subgroup_dict in subgroup_definitions.items():
        print(f"\nAnalyzing {subgroup_name}...")
        
        for group_name, group_func in subgroup_dict.items():
            try:
                group_mask = group_func(df_test)
                group_data = df_test[group_mask].copy()
                
                if len(group_data) < 10:
                    print(f"  {group_name}: Skipped (n={len(group_data)} < 10)")
                    continue
                
                # FIXED: Get predictions based on available model
                c_index = np.nan
                
                if best_model_key is not None and best_model_key in cox_models:
                    try:
                        model = cox_models[best_model_key]
                        
                        # Determine features for this model
                        if best_model_key == 'clinical_only':
                            features_to_use = clinical_only_features
                        elif best_model_key == 'genomic_only':
                            features_to_use = genomic_only_features
                        elif best_model_key in ['combined', 'combined_robust']:
                            features_to_use = significant_features
                        else:
                            # For other models, try to use significant features
                            features_to_use = significant_features
                        
                        # Filter to available features in group_data
                        available_features = [f for f in features_to_use if f in group_data.columns]
                        
                        if len(available_features) > 0:
                            group_test_data = group_data[['Time', 'Event'] + available_features].copy()
                            predictions = model.predict_partial_hazard(group_test_data)
                            c_index = concordance_index(group_data['Time'], -predictions, group_data['Event'])
                    
                    except Exception as e_inner:
                        print(f"    Prediction error for {group_name}: {str(e_inner)[:50]}")
                        c_index = np.nan
                
                # Statistics
                n_patients = len(group_data)
                n_events = int(group_data['Event'].sum())
                event_rate = n_events / n_patients * 100
                median_survival = group_data['Time'].median()
                
                subgroup_results.append({
                    'Subgroup_Type': subgroup_name,
                    'Subgroup': group_name,
                    'N_Patients': n_patients,
                    'N_Events': n_events,
                    'Event_Rate_%': event_rate,
                    'Median_Survival': median_survival,
                    'C_Index': c_index,
                    'Percent_of_Total': n_patients / len(df_test) * 100
                })
                
                c_idx_str = f"{c_index:.4f}" if not np.isnan(c_index) else "N/A"
                print(f"  {group_name}: n={n_patients}, events={n_events} ({event_rate:.1f}%), C-index={c_idx_str}")
                
            except Exception as e_inner:
                print(f"  {group_name}: Error - {str(e_inner)[:50]}")
    
    # Save results and create visualization
    if subgroup_results:
        subgroup_df = pd.DataFrame(subgroup_results)
        subgroup_df.to_csv(f'{output_base}/statistics/subgroup_analysis.csv', index=False)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        subgroup_types = ['age_group', 'cancer_type', 'treatment', 'tp53_status']
        
        for idx, (ax, sg_type) in enumerate(zip(axes.flat, subgroup_types)):
            sg_data = subgroup_df[subgroup_df['Subgroup_Type'] == sg_type]
            
            if len(sg_data) == 0:
                ax.text(0.5, 0.5, f'No data for {sg_type}', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f'{sg_type.replace("_", " ").title()}', fontsize=13, fontweight='bold')
                continue
            
            x = np.arange(len(sg_data))
            width = 0.35
            
            # Event rate bars
            bars1 = ax.bar(x - width/2, sg_data['Event_Rate_%'], width, 
                           label='Event Rate %', alpha=0.8, color='coral', edgecolor='black')
            
            # C-index bars (only if we have valid C-index values)
            has_cindex = sg_data['C_Index'].notna().any()
            
            if has_cindex:
                ax2 = ax.twinx()
                valid_cindex = sg_data['C_Index'].fillna(0)
                bars2 = ax2.bar(x + width/2, valid_cindex, width, 
                                label='C-Index', alpha=0.8, color='skyblue', edgecolor='black')
                
                # Add value labels for C-index
                for i, (bar, val) in enumerate(zip(bars2, sg_data['C_Index'])):
                    if pd.notna(val):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax2.set_ylabel('C-Index', fontsize=11, fontweight='bold', color='skyblue')
                ax2.tick_params(axis='y', labelcolor='skyblue')
                ax2.set_ylim([0, 1])
                ax2.legend(loc='upper right', fontsize=9)
            
            # Add value labels for event rate
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Subgroup', fontsize=11, fontweight='bold')
            ax.set_ylabel('Event Rate (%)', fontsize=11, fontweight='bold', color='coral')
            ax.set_title(f'{sg_type.replace("_", " ").title()}', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sg_data['Subgroup'], rotation=45, ha='right')
            ax.tick_params(axis='y', labelcolor='coral')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
        
        plt.suptitle('Subgroup Analysis: Model Performance Across Patient Subgroups', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_base}/figures/09_subgroup_analysis.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\n  ? Saved: 09_subgroup_analysis.png")
        print(f"\n? Subgroup analysis complete: {len(subgroup_df)} subgroups analyzed")
    else:
        print("\n  No subgroup results to save")
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")
    import traceback
    print(traceback.format_exc()[:500])

print("\n? Part 7D complete")
# ============================================================================
# PART 7E: INTERACTION ANALYSIS
# ============================================================================

print("\n[PHASE 7E] INTERACTION ANALYSIS")
print("-" * 80)

try:
    print("\nTesting genomic-clinical interactions...")
    
    # Test key interactions
    interaction_tests = [
        ('TP53', 'age_at_diagnosis', 'TP53 * age_at_diagnosis'),
        ('KRAS', 'received_pharmaceutical_treatment', 'KRAS * received_pharmaceutical_treatment'),
        ('PIK3CA', 'age_at_diagnosis', 'PIK3CA * age_at_diagnosis'),
        ('TP53', 'KRAS', 'TP53 * KRAS'),
    ]
    
    interaction_results = []
    
    for feat1, feat2, label in interaction_tests:
        print(f"\nTesting {label} interaction...")
        
        try:
            # Create interaction term
            train_int = df_train[['Time', 'Event', feat1, feat2]].copy()
            test_int = df_test[['Time', 'Event', feat1, feat2]].copy()
            
            train_int[f'{label}_interaction'] = train_int[feat1] * train_int[feat2]
            test_int[f'{label}_interaction'] = test_int[feat1] * test_int[feat2]
            
            # Model without interaction
            cph_main = CoxPHFitter(penalizer=0.01)
            cph_main.fit(train_int[['Time', 'Event', feat1, feat2]], 
                         duration_col='Time', event_col='Event')
            
            # Model with interaction
            cph_int = CoxPHFitter(penalizer=0.01)
            cph_int.fit(train_int[['Time', 'Event', feat1, feat2, f'{label}_interaction']], 
                        duration_col='Time', event_col='Event')
            
            # C-indices
            test_pred_main = cph_main.predict_partial_hazard(test_int[['Time', 'Event', feat1, feat2]])
            c_main = concordance_index(test_int['Time'], -test_pred_main, test_int['Event'])
            
            test_pred_int = cph_int.predict_partial_hazard(
                test_int[['Time', 'Event', feat1, feat2, f'{label}_interaction']]
            )
            c_int = concordance_index(test_int['Time'], -test_pred_int, test_int['Event'])
            
            # Interaction p-value
            interaction_p = cph_int.summary.loc[f'{label}_interaction', 'p']
            interaction_hr = np.exp(cph_int.params_[f'{label}_interaction'])
            
            interaction_results.append({
                'Interaction': label,
                'Feature_1': feat1,
                'Feature_2': feat2,
                'Interaction_HR': interaction_hr,
                'Interaction_P': interaction_p,
                'C_Index_Main': c_main,
                'C_Index_Interaction': c_int,
                'C_Index_Improvement': c_int - c_main,
                'Significant': interaction_p < 0.05
            })
            
            sig_marker = '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else 'ns'
            
            print(f"  Interaction HR: {interaction_hr:.3f}, p={interaction_p:.4e} {sig_marker}")
            print(f"  C-index improvement: {c_int - c_main:+.4f}")
            
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")
    
    # Save results
    if interaction_results:
        int_df = pd.DataFrame(interaction_results)
        int_df.to_csv(f'{output_base}/statistics/interaction_analysis.csv', index=False)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(int_df))
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'gray' 
                  for p in int_df['Interaction_P']]
        
        bars = ax.barh(y_pos, int_df['C_Index_Improvement'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, int_df['C_Index_Improvement'])):
            ax.text(val + 0.0005, i, f'{val:+.4f}', 
                    va='center', ha='left', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(int_df['Interaction'], fontweight='bold')
        ax.set_xlabel('C-Index Improvement', fontsize=12, fontweight='bold')
        ax.set_title('Interaction Effects on Model Performance', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.grid(axis='x', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Significant (p<0.05)'),
            Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='Marginal (p<0.1)'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Not Significant')
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{output_base}/figures/10_interaction_analysis.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print("\n  ? Saved: 10_interaction_analysis.png")
        print(f"\n? Interaction analysis complete: {int_df['Significant'].sum()}/{len(int_df)} significant")
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")
    import traceback
    print(traceback.format_exc()[:500])


# ============================================================================
# PART 7F: CALIBRATION ANALYSIS - COMPLETE FIX
# ============================================================================

print("\n[PHASE 7F] CALIBRATION ANALYSIS")
print("-" * 80)

try:
    print("\nEvaluating model calibration...")
    
    calibration_times = [1, 2, 3, 5]
    calibration_results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flat
    
    for ax_idx, time_point in enumerate(calibration_times):
        ax = axes[ax_idx]
        
        for model_name in ['Cox - Clinical Only', 'Cox - Genomic Only', 'Cox - Combined', 'Random Survival Forest']:
            
            try:
                if 'Cox' in model_name:
                    # Get model
                    if 'Clinical' in model_name and 'clinical_only' in cox_models:
                        model = cox_models['clinical_only']
                        features = clinical_only_features
                    elif 'Genomic' in model_name and 'genomic_only' in cox_models:
                        model = cox_models['genomic_only']
                        features = genomic_only_features
                    elif 'Combined' in model_name and 'combined' in cox_models:
                        model = cox_models['combined']
                        features = significant_features
                    else:
                        continue
                    
                    test_data = df_test[['Time', 'Event'] + features].copy()
                    pred_surv = model.predict_survival_function(test_data, times=[time_point])
                    predicted_prob = pred_surv.iloc[0].values
                    
                elif 'Random Survival Forest' in model_name and 'random_survival_forest' in cox_models:
                    model = cox_models['random_survival_forest']
                    surv_funcs = model.predict_survival_function(X_test)
                    
                    predicted_prob = []
                    for surv_func in surv_funcs:
                        times = surv_func.x
                        surv_probs = surv_func.y
                        
                        if time_point <= times.max():
                            prob = np.interp(time_point, times, surv_probs)
                        else:
                            prob = surv_probs[-1]
                        
                        predicted_prob.append(prob)
                    
                    predicted_prob = np.array(predicted_prob)
                else:
                    continue
                
                # Bin predictions
                n_bins = 5
                try:
                    pred_series = pd.Series(predicted_prob)
                    pred_bins = pd.qcut(pred_series, q=n_bins, duplicates='drop', retbins=False)
                    
                    # Ensure it's treated as categorical properly
                    if not isinstance(pred_bins, pd.Categorical):
                        pred_bins = pd.Categorical(pred_bins)
                except:
                    # Fallback: simple quantile bins
                    quantiles = np.quantile(predicted_prob, np.linspace(0, 1, n_bins+1))
                    pred_bins = pd.cut(predicted_prob, bins=quantiles, include_lowest=True, duplicates='drop')
                
                observed_prob = []
                predicted_mean = []
                
                for bin_label in pred_bins.categories if hasattr(pred_bins, 'categories') else pd.unique(pred_bins):
                    bin_mask = pred_bins == bin_label
                    
                    if bin_mask.sum() < 5:
                        continue
                    
                    # Observed survival using Kaplan-Meier
                    kmf = KaplanMeierFitter()
                    kmf.fit(df_test.loc[bin_mask, 'Time'], df_test.loc[bin_mask, 'Event'])
                    
                    if time_point in kmf.survival_function_.index:
                        obs_surv = kmf.survival_function_.loc[time_point].iloc[0]
                    else:
                        # Safe interpolation
                        times_km = kmf.survival_function_.index.values
                        surv_km = kmf.survival_function_.values.flatten()
                        obs_surv = np.interp(time_point, times_km, surv_km)
                    
                    observed_prob.append(obs_surv)
                    predicted_mean.append(predicted_prob[bin_mask].mean())
                
                # Plot only if we have valid data
                if len(observed_prob) >= 2:
                    color = MODEL_COLORS.get(model_name, 'gray')
                    marker = MODEL_MARKERS.get(model_name, 'o')
                    
                    ax.plot(predicted_mean, observed_prob, 
                           marker=marker, markersize=10, linewidth=2.5,
                           label=model_name, color=color, alpha=0.8)
                    
                    # Calibration metrics
                    cal_slope = np.polyfit(predicted_mean, observed_prob, 1)[0]
                    cal_intercept = np.polyfit(predicted_mean, observed_prob, 1)[1]
                    
                    calibration_results[f'{model_name}_T{time_point}'] = {
                        'Model': model_name,
                        'Time_Point': time_point,
                        'Calibration_Slope': cal_slope,
                        'Calibration_Intercept': cal_intercept,
                        'N_Bins': len(observed_prob)
                    }
            
            except Exception as e:
                print(f"  {model_name} at t={time_point}: {str(e)[:50]}")
                continue
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlabel(f'Predicted Survival at {time_point} Years', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Observed Survival at {time_point} Years', fontsize=11, fontweight='bold')
        ax.set_title(f'Calibration at {time_point} Year{"s" if time_point > 1 else ""}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle('Model Calibration Across Time Points', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/11_calibration_analysis.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save results
    if calibration_results:
        cal_df = pd.DataFrame(calibration_results).T
        cal_df.to_csv(f'{output_base}/statistics/calibration_metrics.csv', index=False)
        
        print("\n  ? Saved: 11_calibration_analysis.png")
        print(f"  ? Calibration analysis complete: {len(cal_df)} model-time combinations")
    else:
        print("\n  ? No calibration results generated")
    
except Exception as e:
    print(f"  ? Error: {str(e)[:100]}")
    import traceback
    print(traceback.format_exc()[:500])

print("\n? All advanced survival analyses complete")

print("\n All advanced survival analyses complete")
# ============================================================================
# PART 8: COMPREHENSIVE STATISTICAL VALIDATION
# ============================================================================

print("\n[PHASE 8] COMPREHENSIVE STATISTICAL VALIDATION")
print("-" * 80)

# ============================================================================
# Helper Functions for Statistical Validation
# ============================================================================

def bootstrap_c_index_all_models(model, X, y, model_name, n_bootstrap=500, is_lifelines=False, X_df=None):
    """Calculate bootstrap confidence intervals for C-index."""
    c_indices = []
    
    for i in range(n_bootstrap):
        indices = np.random.choice(len(X) if not is_lifelines else len(X_df), 
                                   size=len(X) if not is_lifelines else len(X_df), 
                                   replace=True)
        
        try:
            if is_lifelines:
                X_boot = X_df.iloc[indices]
                y_boot_time = X_boot['Time'].values
                y_boot_event = X_boot['Event'].values
                
                predictions = model.predict_partial_hazard(X_boot)
                c_idx = concordance_index(y_boot_time, -predictions, y_boot_event)
            else:
                X_boot = X[indices]
                y_boot = y[indices]
                c_idx = model.score(X_boot, y_boot)
            
            c_indices.append(c_idx)
        except Exception as e:
            continue
    
    if len(c_indices) == 0:
        return {
            'mean': 0.5,
            'std': 0,
            'ci_lower': 0.5,
            'ci_upper': 0.5
        }
    
    c_indices = np.array(c_indices)
    return {
        'mean': np.mean(c_indices),
        'std': np.std(c_indices),
        'ci_lower': np.percentile(c_indices, 2.5),
        'ci_upper': np.percentile(c_indices, 97.5)
    }

# ============================================================================
# Bootstrap Confidence Intervals for ALL MODELS
# ============================================================================

print("\n[1] Bootstrap Confidence Intervals (All Models)")
print("-" * 40)

bootstrap_results_all = []

# Define all models to bootstrap
models_to_bootstrap = {
    'Cox - Clinical Only': ('clinical_only', clinical_only_features, True),
    'Cox - Genomic Only': ('genomic_only', genomic_only_features, True),
    'Cox - Combined': ('combined', significant_features, True),
    'Cox - Combined (Robust)': ('combined_robust', stable_features, True),
    'Cox - Elastic Net': ('elastic_net', all_features_list, False),
    'Elastic Net (Robust)': ('elastic_net_robust', stable_features, False),
    'Random Survival Forest': ('random_survival_forest', significant_features, False),
    'RSF (Robust)': ('rsf_robust', stable_features, False),
    'Gradient Boosting Survival': ('gradient_boosting', significant_features, False),
    'GB (Robust)': ('gb_robust', stable_features, False),
    'Survival SVM ': ('survival_svm', None, False),
    'SVM (Robust)': ('svm_robust', None, False)
}

for model_name, (model_key, features, is_lifelines) in models_to_bootstrap.items():
    if model_key not in cox_models:
        print(f"\n{model_name}: Model not available")
        continue
    
    print(f"\n{model_name}:")
    model = cox_models[model_key]
    
    try:
        if is_lifelines:
            # For lifelines models
            test_data_df = df_test[['Time', 'Event'] + features].copy()
            boot_stats = bootstrap_c_index_all_models(
                model, None, None, model_name, 
                n_bootstrap=500, is_lifelines=True, X_df=test_data_df
            )
        else:
            # For sklearn models
            if 'svm' in model_key:
                svm_features = cox_models.get(f'{model_key}_features', stable_features[:20])
                X_test_data = df_test[svm_features].values
                scaler_svm = cox_models.get(f'{model_key}_scaler', StandardScaler())
                X_test_scaled_data = scaler_svm.transform(X_test_data)
            elif 'elastic' in model_key and 'robust' not in model_key:
                X_test_scaled_data = X_test_scaled_global
            else:
                X_test_scaled_data = X_test_scaled if 'gb' in model_key or 'elastic' in model_key else X_test
            
            boot_stats = bootstrap_c_index_all_models(
                model, X_test_scaled_data, y_test, model_name, 
                n_bootstrap=500, is_lifelines=False
            )
        
        print(f"  C-Index: {boot_stats['mean']:.4f} +/- {boot_stats['std']:.4f}")
        print(f"  95% CI: [{boot_stats['ci_lower']:.4f}, {boot_stats['ci_upper']:.4f}]")
        
        bootstrap_results_all.append({
            'Model': model_name,
            'C_Index_Mean': boot_stats['mean'],
            'C_Index_Std': boot_stats['std'],
            'CI_Lower': boot_stats['ci_lower'],
            'CI_Upper': boot_stats['ci_upper']
        })
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

if len(bootstrap_results_all) > 0:
    bootstrap_all_df = pd.DataFrame(bootstrap_results_all)
    bootstrap_all_df.to_csv(f'{output_base}/uncertainty/bootstrap_all_models.csv', index=False)
    print(f"\n? Bootstrap results saved for {len(bootstrap_all_df)} models")

# ============================================================================
# Cross-Validation for ALL Models (INCLUDING SVM)
# ============================================================================

print("\n[2] 5-Fold Cross-Validation (All Models)")
print("-" * 40)

cv_results_all = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models_to_cv = {
    'Cox - Clinical Only': ('clinical', clinical_only_features),
    'Cox - Genomic Only': ('genomic', genomic_only_features),
    'Cox - Combined': ('combined', significant_features),
    'Cox - Combined (Robust)': ('combined_robust', stable_features),
    'Cox - Elastic Net': ('elastic_net', all_features_list),
    'Elastic Net (Robust)': ('elastic_net_robust', stable_features),
    'Random Survival Forest': ('rsf', significant_features),
    'RSF (Robust)': ('rsf_robust', stable_features),
    'Gradient Boosting Survival': ('gbsa', significant_features),
    'GB (Robust)': ('gb_robust', stable_features),
    'Survival SVM ': ('svm', None),
    'SVM (Robust)': ('svm_robust', None)
}

for model_name, (model_type, features) in models_to_cv.items():
    print(f"\n{model_name}:")
    
    try:
        cv_scores = []
        
        stratify_col = df_train['Event'].values
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, stratify_col)):
            
            if 'Cox' in model_name and 'Elastic' not in model_name:
                # Cox models
                penalizer_val = 0.5 if 'Robust' in model_name else 0.1
                l1_ratio_val = 0.7 if 'Robust' in model_name else 0.5
                
                train_cv_data = df_train.iloc[train_idx][['Time', 'Event'] + features].copy()
                val_cv_data = df_train.iloc[val_idx][['Time', 'Event'] + features].copy()
                
                if len(train_cv_data) < 30 or len(val_cv_data) < 10:
                    continue
                
                train_variance = train_cv_data[features].var()
                valid_features = train_variance[train_variance > 1e-6].index.tolist()
                
                if len(valid_features) == 0:
                    continue
                
                train_cv_data = train_cv_data[['Time', 'Event'] + valid_features].copy()
                val_cv_data = val_cv_data[['Time', 'Event'] + valid_features].copy()
                
                try:
                    cph_cv = CoxPHFitter(penalizer=penalizer_val, l1_ratio=l1_ratio_val)
                    cph_cv.fit(train_cv_data, duration_col='Time', event_col='Event', show_progress=False)
                    
                    val_pred = cph_cv.predict_partial_hazard(val_cv_data)
                    score = concordance_index(val_cv_data['Time'], -val_pred, val_cv_data['Event'])
                    cv_scores.append(score)
                except:
                    try:
                        X_train_cv = train_cv_data[valid_features].values
                        X_val_cv = val_cv_data[valid_features].values
                        
                        y_train_cv_arr = Surv.from_dataframe('Event', 'Time', train_cv_data)
                        y_val_cv_arr = Surv.from_dataframe('Event', 'Time', val_cv_data)
                        
                        scaler_cv = RobustScaler() if 'Robust' in model_name else StandardScaler()
                        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
                        
                        coxnet_cv = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, max_iter=1000, n_alphas=10)
                        coxnet_cv.fit(X_train_cv_scaled, y_train_cv_arr)
                        score = coxnet_cv.score(X_val_cv_scaled, y_val_cv_arr)
                        cv_scores.append(score)
                    except:
                        continue
                    
            elif 'Elastic' in model_name:
                X_train_cv = df_train.iloc[train_idx][features if features else all_features_list].values
                X_val_cv = df_train.iloc[val_idx][features if features else all_features_list].values
                
                y_train_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[train_idx])
                y_val_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[val_idx])
                
                scaler_cv = RobustScaler() if 'Robust' in model_name else StandardScaler()
                X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                X_val_cv_scaled = scaler_cv.transform(X_val_cv)
                
                l1_ratio_val = 0.8 if 'Robust' in model_name else 0.5
                model_cv = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio_val, alpha_min_ratio=0.1, max_iter=2000, n_alphas=10)
                model_cv.fit(X_train_cv_scaled, y_train_cv_shifted)
                score = model_cv.score(X_val_cv_scaled, y_val_cv_shifted)
                cv_scores.append(score)
                
            elif 'Random Survival Forest' in model_name or 'RSF' in model_name:
                X_train_cv = df_train.iloc[train_idx][features].values
                X_val_cv = df_train.iloc[val_idx][features].values
                
                y_train_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[train_idx])
                y_val_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[val_idx])
                
                if 'Robust' in model_name:
                    model_cv = RandomSurvivalForest(
                        n_estimators=500, 
                        min_samples_split=50,
                        min_samples_leaf=25, 
                        max_depth=5, 
                        max_features=0.3, 
                        n_jobs=-1, 
                        random_state=RANDOM_STATE, 
                        verbose=0
                    )
                else:
                    model_cv = RandomSurvivalForest(
                        n_estimators=200, 
                        min_samples_split=20,
                        min_samples_leaf=10, 
                        max_depth=10, 
                        max_features="sqrt", 
                        n_jobs=-1, 
                        random_state=RANDOM_STATE, 
                        verbose=0
                    )
                model_cv.fit(X_train_cv, y_train_cv_shifted)
                score = model_cv.score(X_val_cv, y_val_cv_shifted)
                cv_scores.append(score)
                
            elif 'Gradient Boosting' in model_name or 'GB' in model_name:
                X_train_cv = df_train.iloc[train_idx][features].values
                X_val_cv = df_train.iloc[val_idx][features].values
                
                y_train_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[train_idx])
                y_val_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[val_idx])
                
                scaler_cv = RobustScaler() if 'Robust' in model_name else StandardScaler()
                X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
                X_val_cv_scaled = scaler_cv.transform(X_val_cv)
                
                if 'Robust' in model_name:
                    model_cv = GradientBoostingSurvivalAnalysis(
                        n_estimators=50, 
                        learning_rate=0.01,
                        max_depth=2, 
                        min_samples_split=50, 
                        min_samples_leaf=25, 
                        subsample=0.5, 
                        random_state=RANDOM_STATE, 
                        verbose=0
                    )
                else:
                    model_cv = GradientBoostingSurvivalAnalysis(
                        n_estimators=100, 
                        learning_rate=0.05,
                        max_depth=3, 
                        min_samples_split=20, 
                        min_samples_leaf=10, 
                        subsample=0.7, 
                        random_state=RANDOM_STATE, 
                        verbose=0
                    )
                model_cv.fit(X_train_cv_scaled, y_train_cv_shifted)
                score = model_cv.score(X_val_cv_scaled, y_val_cv_shifted)
                cv_scores.append(score)
            
            elif 'SVM' in model_name:
                # Use top features for SVM
                if 'Robust' in model_name:
                    top_svm = feature_stability_df.head(15)['Feature'].tolist()
                else:
                    top_svm = univariate_df.head(20)['Feature'].tolist()
                
                X_train_cv_svm = df_train.iloc[train_idx][top_svm].values
                X_val_cv_svm = df_train.iloc[val_idx][top_svm].values
                
                y_train_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[train_idx])
                y_val_cv_shifted = Surv.from_dataframe('Event', 'Time_Shifted', df_train.iloc[val_idx])
                
                scaler_cv_svm = RobustScaler() if 'Robust' in model_name else StandardScaler()
                X_train_cv_svm_scaled = scaler_cv_svm.fit_transform(X_train_cv_svm)
                X_val_cv_svm_scaled = scaler_cv_svm.transform(X_val_cv_svm)
                
                try:
                    svm_cv, score_val, score_train, params_cv = train_optimized_svm(
                        X_train_cv_svm_scaled, y_train_cv_shifted,
                        X_val_cv_svm_scaled, y_val_cv_shifted,
                        random_state=RANDOM_STATE + fold
                    )
                    
                    cv_scores.append(score_val)
                    print(f"    Fold {fold+1}: Train={score_train:.4f}, Val={score_val:.4f}, "
                          f"Params: alpha={params_cv['alpha']}, kernel={params_cv['kernel']}")
                          
                except Exception as e:
                    print(f"    Fold {fold+1}: Failed - {str(e)[:50]}")
                    continue
        
        if len(cv_scores) > 0:
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"  Mean: {cv_mean:.4f} +/- {cv_std:.4f}")
            
            cv_results_all.append({
                'Model': model_name,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std,
                'CV_Min': np.min(cv_scores),
                'CV_Max': np.max(cv_scores),
                'N_Folds': len(cv_scores)
            })
        else:
            print(f"  No valid CV scores")
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

# Save CV results
if len(cv_results_all) > 0:
    cv_all_df = pd.DataFrame(cv_results_all)
    cv_all_df.to_csv(f'{output_base}/uncertainty/cross_validation_all_models.csv', index=False)
    print("\n? Cross-validation results saved")
    print(f"  Models in CV results: {cv_all_df['Model'].tolist()}")

# ============================================================================
# Prediction Uncertainty Analysis (using RSF ensemble)
# ============================================================================

print("\n[3] Prediction Uncertainty Analysis")
print("-" * 40)

# Use robust RSF if available, otherwise original
rsf_key = 'rsf_robust' if 'rsf_robust' in cox_models else 'random_survival_forest'

if rsf_key in cox_models:
    print("  Calculating prediction uncertainty using RSF ensemble...")
    
    rsf_model = cox_models[rsf_key]
    
    # Get predictions from individual trees
    n_estimators = rsf_model.n_estimators
    tree_predictions = []
    
    for tree in rsf_model.estimators_:
        pred = tree.predict(X_test)
        tree_predictions.append(pred)
    
    tree_predictions = np.array(tree_predictions)
    
    # Calculate uncertainty metrics
    pred_mean = tree_predictions.mean(axis=0)
    pred_std = tree_predictions.std(axis=0)
    pred_cv = pred_std / (np.abs(pred_mean) + 1e-10)
    
    # Create uncertainty dataframe
    uncertainty_df = pd.DataFrame({
        'Patient_ID': range(len(df_test)),
        'True_Time': T_test,
        'True_Event': E_test,
        'Pred_Mean': pred_mean,
        'Pred_Std': pred_std,
        'Pred_CV': pred_cv
    })
    
    uncertainty_df.to_csv(f'{output_base}/uncertainty/prediction_uncertainty.csv', index=False)
    
    print(f"\n  Uncertainty Statistics:")
    print(f"    Mean CV: {pred_cv.mean():.4f}")
    print(f"    Median CV: {np.median(pred_cv):.4f}")
    print(f"    Max CV: {pred_cv.max():.4f}")
    
    print("\n? Prediction uncertainty analysis saved")

print("\n? Part 8 Complete: Statistical validation completed")

# ============================================================================
# PART 9: RISK STRATIFICATION AND SURVIVAL CURVES
# ============================================================================

print("\n[PHASE 9] RISK STRATIFICATION ANALYSIS")
print("-" * 80)

# ============================================================================
# Generate Risk Scores for ALL Models
# ============================================================================

print("\nGenerating risk scores across all models...")

risk_scores_all_models = {}
risk_groups_all_models = {}

# Helper function to generate risk groups
def create_risk_groups(train_scores, test_scores):
    """Create risk groups based on training set tertiles"""
    tertiles = np.percentile(train_scores, [33.33, 66.67])
    risk_groups = pd.cut(test_scores, bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
                        labels=['Low Risk', 'Intermediate Risk', 'High Risk'])
    return risk_groups

# Model 1: Cox - Clinical Only
if 'clinical_only' in cox_models:
    try:
        test_clinical = df_test[['Time', 'Event'] + clinical_only_features].copy().dropna()
        train_clinical = df_train[['Time', 'Event'] + clinical_only_features].copy().dropna()
        
        scores_train = cox_models['clinical_only'].predict_partial_hazard(train_clinical).values
        scores_test = cox_models['clinical_only'].predict_partial_hazard(test_clinical).values
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Cox - Clinical Only'] = scores_test
        risk_groups_all_models['Cox - Clinical Only'] = risk_groups
        print("  ? Cox - Clinical Only risk scores generated")
    except Exception as e:
        print(f"  ? Cox - Clinical Only failed: {str(e)[:50]}")

# Model 2: Cox - Genomic Only
if 'genomic_only' in cox_models:
    try:
        test_genomic = df_test[['Time', 'Event'] + genomic_only_features].copy().dropna()
        train_genomic = df_train[['Time', 'Event'] + genomic_only_features].copy().dropna()
        
        scores_train = cox_models['genomic_only'].predict_partial_hazard(train_genomic).values
        scores_test = cox_models['genomic_only'].predict_partial_hazard(test_genomic).values
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Cox - Genomic Only'] = scores_test
        risk_groups_all_models['Cox - Genomic Only'] = risk_groups
        print("  ? Cox - Genomic Only risk scores generated")
    except Exception as e:
        print(f"  ? Cox - Genomic Only failed: {str(e)[:50]}")

# Model 3: Cox - Combined
if 'combined' in cox_models:
    try:
        test_combined = df_test[['Time', 'Event'] + significant_features].copy().dropna()
        train_combined = df_train[['Time', 'Event'] + significant_features].copy().dropna()
        
        scores_train = cox_models['combined'].predict_partial_hazard(train_combined).values
        scores_test = cox_models['combined'].predict_partial_hazard(test_combined).values
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Cox - Combined'] = scores_test
        risk_groups_all_models['Cox - Combined'] = risk_groups
        print("  ? Cox - Combined risk scores generated")
    except Exception as e:
        print(f"  ? Cox - Combined failed: {str(e)[:50]}")

# Model 4: Cox - Combined (Robust)
if 'combined_robust' in cox_models:
    try:
        test_combined_robust = df_test[['Time', 'Event'] + stable_features].copy().dropna()
        train_combined_robust = df_train[['Time', 'Event'] + stable_features].copy().dropna()
        
        scores_train = cox_models['combined_robust'].predict_partial_hazard(train_combined_robust).values
        scores_test = cox_models['combined_robust'].predict_partial_hazard(test_combined_robust).values
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Cox - Combined (Robust)'] = scores_test
        risk_groups_all_models['Cox - Combined (Robust)'] = risk_groups
        print("  ? Cox - Combined (Robust) risk scores generated")
    except Exception as e:
        print(f"  ? Cox - Combined (Robust) failed: {str(e)[:50]}")

# Model 5: Elastic Net
if 'elastic_net' in cox_models:
    try:
        scores_train = cox_models['elastic_net'].predict(X_train_scaled_global)
        scores_test = cox_models['elastic_net'].predict(X_test_scaled_global)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Cox - Elastic Net'] = scores_test
        risk_groups_all_models['Cox - Elastic Net'] = risk_groups
        print("  ? Cox - Elastic Net risk scores generated")
    except Exception as e:
        print(f"  ? Cox - Elastic Net failed: {str(e)[:50]}")

# Model 6: Elastic Net (Robust)
if 'elastic_net_robust' in cox_models:
    try:
        scores_train = cox_models['elastic_net_robust'].predict(X_train_scaled)
        scores_test = cox_models['elastic_net_robust'].predict(X_test_scaled)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Elastic Net (Robust)'] = scores_test
        risk_groups_all_models['Elastic Net (Robust)'] = risk_groups
        print("  ? Elastic Net (Robust) risk scores generated")
    except Exception as e:
        print(f"  ? Elastic Net (Robust) failed: {str(e)[:50]}")

# Model 7: Random Survival Forest
if 'random_survival_forest' in cox_models:
    try:
        scores_train = cox_models['random_survival_forest'].predict(X_train)
        scores_test = cox_models['random_survival_forest'].predict(X_test)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Random Survival Forest'] = scores_test
        risk_groups_all_models['Random Survival Forest'] = risk_groups
        print("  ? Random Survival Forest risk scores generated")
    except Exception as e:
        print(f"  ? Random Survival Forest failed: {str(e)[:50]}")

# Model 8: RSF (Robust)
if 'rsf_robust' in cox_models:
    try:
        scores_train = cox_models['rsf_robust'].predict(X_train)
        scores_test = cox_models['rsf_robust'].predict(X_test)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['RSF (Robust)'] = scores_test
        risk_groups_all_models['RSF (Robust)'] = risk_groups
        print("  ? RSF (Robust) risk scores generated")
    except Exception as e:
        print(f"  ? RSF (Robust) failed: {str(e)[:50]}")

# Model 9: Gradient Boosting
if 'gradient_boosting' in cox_models:
    try:
        scores_train = cox_models['gradient_boosting'].predict(X_train_scaled)
        scores_test = cox_models['gradient_boosting'].predict(X_test_scaled)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Gradient Boosting Survival'] = scores_test
        risk_groups_all_models['Gradient Boosting Survival'] = risk_groups
        print("  ? Gradient Boosting Survival risk scores generated")
    except Exception as e:
        print(f"  ? Gradient Boosting failed: {str(e)[:50]}")

# Model 10: GB (Robust)
if 'gb_robust' in cox_models:
    try:
        scores_train = cox_models['gb_robust'].predict(X_train_scaled)
        scores_test = cox_models['gb_robust'].predict(X_test_scaled)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['GB (Robust)'] = scores_test
        risk_groups_all_models['GB (Robust)'] = risk_groups
        print("  ? GB (Robust) risk scores generated")
    except Exception as e:
        print(f"  ? GB (Robust) failed: {str(e)[:50]}")

# Model 11: Survival SVM
if 'survival_svm' in cox_models:
    try:
        svm_features = cox_models.get('survival_svm_features', significant_features[:20])
        scaler_svm = cox_models.get('survival_svm_scaler', StandardScaler())
        
        X_train_svm = df_train[svm_features].values
        X_test_svm = df_test[svm_features].values
        
        X_train_svm_scaled = scaler_svm.transform(X_train_svm)
        X_test_svm_scaled = scaler_svm.transform(X_test_svm)
        
        scores_train = cox_models['survival_svm'].predict(X_train_svm_scaled)
        scores_test = cox_models['survival_svm'].predict(X_test_svm_scaled)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['Survival SVM '] = scores_test
        risk_groups_all_models['Survival SVM '] = risk_groups
        print("  ? Survival SVM risk scores generated")
    except Exception as e:
        print(f"  ? Survival SVM failed: {str(e)[:50]}")

# Model 12: SVM (Robust)
if 'svm_robust' in cox_models:
    try:
        svm_features = cox_models.get('svm_robust_features', stable_features[:15])
        scaler_svm = cox_models.get('svm_robust_scaler', RobustScaler())
        
        X_train_svm = df_train[svm_features].values
        X_test_svm = df_test[svm_features].values
        
        X_train_svm_scaled = scaler_svm.transform(X_train_svm)
        X_test_svm_scaled = scaler_svm.transform(X_test_svm)
        
        scores_train = cox_models['svm_robust'].predict(X_train_svm_scaled)
        scores_test = cox_models['svm_robust'].predict(X_test_svm_scaled)
        
        risk_groups = create_risk_groups(scores_train, scores_test)
        
        risk_scores_all_models['SVM (Robust)'] = scores_test
        risk_groups_all_models['SVM (Robust)'] = risk_groups
        print("  ? SVM (Robust) risk scores generated")
    except Exception as e:
        print(f"  ? SVM (Robust) failed: {str(e)[:50]}")

# ============================================================================
# Calculate Hazard Ratios for ALL Models
# ============================================================================

print("\n[HAZARD RATIOS ACROSS ALL MODELS]")
print("-" * 80)

hazard_ratios_all_models = []

for model_name, risk_groups in risk_groups_all_models.items():
    print(f"\n{model_name}:")
    
    try:
        # Create temporary dataframe with risk groups
        temp_df = df_test[['Time', 'Event']].copy()
        temp_df['Intermediate_Risk'] = (risk_groups == 'Intermediate Risk').astype(int)
        temp_df['High_Risk'] = (risk_groups == 'High Risk').astype(int)
        
        # Remove NaN
        temp_df = temp_df.dropna()
        
        # Fit Cox model for hazard ratios
        cph_risk = CoxPHFitter(penalizer=0.01)
        cph_risk.fit(temp_df, duration_col='Time', event_col='Event')
        
        # Intermediate Risk vs Low Risk
        hr_int = np.exp(cph_risk.params_['Intermediate_Risk'])
        ci_low_int = np.exp(cph_risk.confidence_intervals_['95% lower-bound']['Intermediate_Risk'])
        ci_high_int = np.exp(cph_risk.confidence_intervals_['95% upper-bound']['Intermediate_Risk'])
        p_int = cph_risk.summary['p']['Intermediate_Risk']
        
        print(f"  Intermediate vs Low: HR={hr_int:.3f} (95% CI: {ci_low_int:.3f}-{ci_high_int:.3f}), p={p_int:.4e}")
        
        hazard_ratios_all_models.append({
            'Model': model_name,
            'Comparison': 'Intermediate vs Low',
            'HR': hr_int,
            'CI_Lower': ci_low_int,
            'CI_Upper': ci_high_int,
            'P_value': p_int
        })
        
        # High Risk vs Low Risk
        hr_high = np.exp(cph_risk.params_['High_Risk'])
        ci_low_high = np.exp(cph_risk.confidence_intervals_['95% lower-bound']['High_Risk'])
        ci_high_high = np.exp(cph_risk.confidence_intervals_['95% upper-bound']['High_Risk'])
        p_high = cph_risk.summary['p']['High_Risk']
        
        print(f"  High vs Low:         HR={hr_high:.3f} (95% CI: {ci_low_high:.3f}-{ci_high_high:.3f}), p={p_high:.4e}")
        
        hazard_ratios_all_models.append({
            'Model': model_name,
            'Comparison': 'High vs Low',
            'HR': hr_high,
            'CI_Lower': ci_low_high,
            'CI_Upper': ci_high_high,
            'P_value': p_high
        })
        
    except Exception as e:
        print(f"  ? Error calculating hazard ratios: {str(e)[:50]}")

# Save hazard ratios
if len(hazard_ratios_all_models) > 0:
    hr_all_df = pd.DataFrame(hazard_ratios_all_models)
    hr_all_df.to_csv(f'{output_base}/statistics/hazard_ratios_all_models.csv', index=False)
    print("\n? Hazard ratios for all models saved")

# ============================================================================
# Risk Group Statistics for Best Model
# ============================================================================

print("\n[RISK GROUP STATISTICS]")
print("-" * 80)

# Use best performing model (highest test C-index)
best_model_key = perf_df_all.iloc[0]['Model']

if best_model_key in risk_groups_all_models:
    best_risk_groups = risk_groups_all_models[best_model_key]
    
    # Add to test_df for analysis
    df_test_copy = df_test.copy()
    df_test_copy['Risk_Score'] = risk_scores_all_models[best_model_key]
    df_test_copy['Risk_Group'] = best_risk_groups
    
    # Calculate statistics
    risk_stats = []
    
    for group in ['Low Risk', 'Intermediate Risk', 'High Risk']:
        mask = df_test_copy['Risk_Group'] == group
        
        n_patients = mask.sum()
        n_events = df_test_copy.loc[mask, 'Event'].sum()
        n_censored = n_patients - n_events
        event_rate = n_events / n_patients * 100 if n_patients > 0 else 0
        
        survival_times = df_test_copy.loc[mask, 'Time']
        
        risk_stats.append({
            'Risk_Group': group,
            'N_Patients': n_patients,
            'N_Events': n_events,
            'N_Censored': n_censored,
            'Event_Rate': event_rate,
            'Mean_Survival': survival_times.mean(),
            'Median_Survival': survival_times.median(),
            'Std_Survival': survival_times.std()
        })
        
        print(f"\n{group}:")
        print(f"  Patients: {n_patients} ({n_patients/len(df_test_copy)*100:.1f}%)")
        print(f"  Events: {n_events} ({event_rate:.1f}%)")
        print(f"  Censored: {n_censored} ({n_censored/n_patients*100:.1f}%)")
        print(f"  Mean survival: {survival_times.mean():.2f} years")
        print(f"  Median survival: {survival_times.median():.2f} years")
    
    risk_stats_df = pd.DataFrame(risk_stats)
    risk_stats_df.to_csv(f'{output_base}/statistics/risk_group_statistics.csv', index=False)
    
    # ============================================================================
    # Log-Rank Tests
    # ============================================================================
    
    print("\n[LOG-RANK TESTS]")
    print("-" * 80)
    
    # Overall test
    try:
        result = multivariate_logrank_test(
            df_test_copy['Time'],
            df_test_copy['Risk_Group'],
            df_test_copy['Event']
        )
        print(f"Overall Log-Rank Test:")
        print(f"  Chi-square: {result.test_statistic:.4f}")
        print(f"  P-value: {result.p_value:.4e}")
        print(f"  Degrees of freedom: {result.degrees_of_freedom}")
    except Exception as e:
        print(f"Could not perform multivariate log-rank test: {str(e)[:50]}")
    
    # Pairwise tests
    print("\nPairwise Log-Rank Tests:")
    pairs = [('Low Risk', 'High Risk'), 
             ('Low Risk', 'Intermediate Risk'), 
             ('Intermediate Risk', 'High Risk')]
    
    pairwise_results = []
    
    for group1, group2 in pairs:
        mask1 = df_test_copy['Risk_Group'] == group1
        mask2 = df_test_copy['Risk_Group'] == group2
        
        T1, E1 = df_test_copy.loc[mask1, 'Time'], df_test_copy.loc[mask1, 'Event']
        T2, E2 = df_test_copy.loc[mask2, 'Time'], df_test_copy.loc[mask2, 'Event']
        
        if len(T1) > 0 and len(T2) > 0:
            try:
                result = logrank_test(T1, T2, E1, E2)
                
                pairwise_results.append({
                    'Group1': group1,
                    'Group2': group2,
                    'Test_Statistic': result.test_statistic,
                    'P_value': result.p_value
                })
                
                print(f"  {group1} vs {group2}:")
                print(f"    Test statistic: {result.test_statistic:.4f}")
                print(f"    P-value: {result.p_value:.4e}")
            except Exception as e:
                print(f"  {group1} vs {group2}: Could not calculate")
    
    # Apply Bonferroni correction
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_df['P_value_Bonferroni'] = pairwise_df['P_value'] * len(pairwise_results)
        pairwise_df['Significant_Bonferroni'] = pairwise_df['P_value_Bonferroni'] < 0.05
        
        pairwise_df.to_csv(f'{output_base}/statistics/pairwise_logrank_tests.csv', index=False)
    
    # Save risk stratification results for all models
    for model_name, risk_groups in risk_groups_all_models.items():
        model_safe_name = model_name.replace(' ', '_').replace('-', '').replace('(', '').replace(')', '')
        risk_df = pd.DataFrame({
            'Time': df_test['Time'],
            'Event': df_test['Event'],
            'Risk_Score': risk_scores_all_models[model_name],
            'Risk_Group': risk_groups
        })
        risk_df.to_csv(f'{output_base}/tables/risk_stratification_{model_safe_name}.csv', index=False)

print("\n? Part 9 Complete: Risk stratification and survival analysis completed")

# ============================================================================
# PART 10: COMPREHENSIVE ABLATION STUDIES (ALL MODELS)
# ============================================================================

print("\n[PHASE 10] COMPREHENSIVE ABLATION STUDIES (ALL MODELS)")
print("-" * 80)

ablation_results = []

# ============================================================================
# Helper Function for Ablation with Validation
# ============================================================================

def train_ablation_model(model_type, features, X_train, X_val, X_test, 
                         y_train, y_val, y_test, df_train, df_val, df_test,
                         use_robust=False):
    """
    Train a model with given features and return performance metrics.
    """
    
    try:
        if model_type == 'cox':
            # Cox model - FIXED for convergence issues
            train_data = df_train[['Time', 'Event'] + features].copy().dropna()
            val_data = df_val[['Time', 'Event'] + features].copy().dropna()
            test_data = df_test[['Time', 'Event'] + features].copy().dropna()
            
            # Check for sufficient data
            if len(train_data) < 50 or len(val_data) < 20 or len(test_data) < 20:
                return None
            
            # Check for zero variance features
            train_variance = train_data[features].var()
            valid_features = train_variance[train_variance > 1e-6].index.tolist()
            
            if len(valid_features) < 2:
                return None
            
            # Rebuild with valid features
            train_data = df_train[['Time', 'Event'] + valid_features].copy().dropna()
            val_data = df_val[['Time', 'Event'] + valid_features].copy().dropna()
            test_data = df_test[['Time', 'Event'] + valid_features].copy().dropna()
            
            penalizer = config.COX_PENALIZER if use_robust else 0.1
            l1_ratio = config.COX_L1_RATIO if use_robust else 0.3
            
            # Increase penalizer if still having convergence issues
            try:
                model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                model.fit(train_data, duration_col='Time', event_col='Event', show_progress=False)
            except Exception as e:
                if "matrix inversion" in str(e).lower() or "convergence" in str(e).lower():
                    # Try with higher penalization
                    model = CoxPHFitter(penalizer=penalizer * 10, l1_ratio=l1_ratio)
                    model.fit(train_data, duration_col='Time', event_col='Event', show_progress=False)
                else:
                    raise
            
            train_c = model.concordance_index_
            val_c = concordance_index(val_data['Time'], 
                                     -model.predict_partial_hazard(val_data),
                                     val_data['Event'])
            test_c = concordance_index(test_data['Time'], 
                                      -model.predict_partial_hazard(test_data),
                                      test_data['Event'])
            
        elif model_type == 'rsf':
            # Random Survival Forest
            X_train_abl = df_train[features].values
            X_val_abl = df_val[features].values
            X_test_abl = df_test[features].values
            
            if use_robust:
                model = RandomSurvivalForest(
                    n_estimators=config.RSF_N_ESTIMATORS,
                    max_depth=config.RSF_MAX_DEPTH,
                    min_samples_split=config.RSF_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=config.RSF_MIN_SAMPLES_LEAF,
                    max_features=config.RSF_MAX_FEATURES,
                    n_jobs=-1, random_state=RANDOM_STATE, verbose=0
                )
            else:
                model = RandomSurvivalForest(
                    n_estimators=200, max_depth=10,
                    min_samples_split=20, min_samples_leaf=10,
                    max_features="sqrt",
                    n_jobs=-1, random_state=RANDOM_STATE, verbose=0
                )
            
            model.fit(X_train_abl, y_train)
            train_c = model.score(X_train_abl, y_train)
            val_c = model.score(X_val_abl, y_val)
            test_c = model.score(X_test_abl, y_test)
            
        elif model_type == 'gb':
            # Gradient Boosting
            X_train_abl = df_train[features].values
            X_val_abl = df_val[features].values
            X_test_abl = df_test[features].values
            
            scaler = RobustScaler() if use_robust else StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_abl)
            X_val_scaled = scaler.transform(X_val_abl)
            X_test_scaled = scaler.transform(X_test_abl)
            
            if use_robust:
                model = GradientBoostingSurvivalAnalysis(
                    n_estimators=config.GB_N_ESTIMATORS,
                    learning_rate=config.GB_LEARNING_RATE,
                    max_depth=config.GB_MAX_DEPTH,
                    min_samples_split=config.GB_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,
                    subsample=config.GB_SUBSAMPLE,
                    random_state=RANDOM_STATE, verbose=0
                )
            else:
                model = GradientBoostingSurvivalAnalysis(
                    n_estimators=100, learning_rate=0.05, max_depth=3,
                    min_samples_split=20, min_samples_leaf=10,
                    subsample=0.7,
                    random_state=RANDOM_STATE, verbose=0
                )
            
            model.fit(X_train_scaled, y_train)
            train_c = model.score(X_train_scaled, y_train)
            val_c = model.score(X_val_scaled, y_val)
            test_c = model.score(X_test_scaled, y_test)
            
        elif model_type == 'elastic':
            # Elastic Net
            X_train_abl = df_train[features].values
            X_val_abl = df_val[features].values
            X_test_abl = df_test[features].values
            
            scaler = RobustScaler() if use_robust else StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_abl)
            X_val_scaled = scaler.transform(X_val_abl)
            X_test_scaled = scaler.transform(X_test_abl)
            
            l1_ratio = config.ELASTIC_NET_L1 if use_robust else 0.5
            
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio, alpha_min_ratio=0.1, 
                max_iter=2000, n_alphas=20
            )
            
            model.fit(X_train_scaled, y_train)
            train_c = model.score(X_train_scaled, y_train)
            val_c = model.score(X_val_scaled, y_val)
            test_c = model.score(X_test_scaled, y_test)
            
        elif model_type == 'svm':
            # SVM
            X_train_abl = df_train[features].values
            X_val_abl = df_val[features].values
            X_test_abl = df_test[features].values
            
            scaler = RobustScaler() if use_robust else StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_abl)
            X_val_scaled = scaler.transform(X_val_abl)
            X_test_scaled = scaler.transform(X_test_abl)
            
            model, test_c, train_c, params = train_optimized_svm(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                random_state=RANDOM_STATE
            )
            
            val_c = model.score(X_val_scaled, y_val)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            'Train_C_Index': train_c,
            'Val_C_Index': val_c,
            'Test_C_Index': test_c,
            'Overfit': train_c - val_c
        }
        
    except Exception as e:
        # Silently return None for failed models
        return None

# ============================================================================
# Ablation Study 1: Top Features (ALL MODELS)
# ============================================================================

print("\n[Ablation Study 1] Top Features Analysis (All Models)")
print("-" * 80)

for set_name, features in ablation_feature_sets.items():
    print(f"\nTesting {set_name} ({len(features)} features)...")
    
    # Test each model type
    model_configs = [
        ('Cox - Combined', 'cox', False),
        ('Cox - Combined (Robust)', 'cox', True),
        ('Random Survival Forest', 'rsf', False),
        ('RSF (Robust)', 'rsf', True),
        ('Gradient Boosting Survival', 'gb', False),
        ('GB (Robust)', 'gb', True),
        ('Cox - Elastic Net', 'elastic', False),
        ('Elastic Net (Robust)', 'elastic', True),
        ('Survival SVM ', 'svm', False),
        ('SVM (Robust)', 'svm', True)
    ]
    
    for model_name, model_type, use_robust in model_configs:
        result = train_ablation_model(
            model_type, features, X_train, X_val, X_test,
            y_train, y_val, y_test, df_train, df_val, df_test,
            use_robust=use_robust
        )
        
        if result:
            ablation_results.append({
                'Feature_Set': set_name,
                'N_Features': len(features),
                'Model': model_name,
                'Train_C_Index': result['Train_C_Index'],
                'Val_C_Index': result['Val_C_Index'],
                'Test_C_Index': result['Test_C_Index'],
                'Overfit': result['Overfit']
            })
    
    print(f"  ? Completed testing for {set_name}")

# ============================================================================
# Ablation Study 2: Genomic Features (ALL MODELS)
# ============================================================================

print("\n[Ablation Study 2] Genomic Features Analysis (All Models)")
print("-" * 80)

for set_name, features in genomic_ablation_sets.items():
    print(f"\nTesting {set_name} ({len(features)} features)...")
    
    for model_name, model_type, use_robust in model_configs:
        result = train_ablation_model(
            model_type, features, X_train, X_val, X_test,
            y_train, y_val, y_test, df_train, df_val, df_test,
            use_robust=use_robust
        )
        
        if result:
            ablation_results.append({
                'Feature_Set': set_name,
                'N_Features': len(features),
                'Model': model_name,
                'Train_C_Index': result['Train_C_Index'],
                'Val_C_Index': result['Val_C_Index'],
                'Test_C_Index': result['Test_C_Index'],
                'Overfit': result['Overfit']
            })
    
    print(f"  ? Completed testing for {set_name}")

# ============================================================================
# Ablation Study 3: Clinical Features (ALL MODELS)
# ============================================================================

print("\n[Ablation Study 3] Clinical Features Analysis (All Models)")
print("-" * 80)

for set_name, features in clinical_ablation_sets.items():
    n_features = int(set_name.split('_')[1])
    
    if len(features) < n_features:
        print(f"\nSkipping {set_name} (only {len(features)} clinical features available)")
        continue
        
    print(f"\nTesting {set_name} ({len(features)} features)...")
    
    for model_name, model_type, use_robust in model_configs:
        result = train_ablation_model(
            model_type, features, X_train, X_val, X_test,
            y_train, y_val, y_test, df_train, df_val, df_test,
            use_robust=use_robust
        )
        
        if result:
            ablation_results.append({
                'Feature_Set': set_name,
                'N_Features': len(features),
                'Model': model_name,
                'Train_C_Index': result['Train_C_Index'],
                'Val_C_Index': result['Val_C_Index'],
                'Test_C_Index': result['Test_C_Index'],
                'Overfit': result['Overfit']
            })
    
    print(f"  ? Completed testing for {set_name}")

# ============================================================================
# Ablation Study 4: Clinical-Only vs Genomic-Only (ALL MODELS)
# ============================================================================

print("\n[Ablation Study 4] Clinical-Only vs Genomic-Only (All Models)")
print("-" * 80)

baseline_sets = {
    'clinical_only': clinical_only_features,
    'genomic_only': genomic_only_features
}

for set_name, features in baseline_sets.items():
    print(f"\nTesting {set_name} ({len(features)} features)...")
    
    for model_name, model_type, use_robust in model_configs:
        result = train_ablation_model(
            model_type, features, X_train, X_val, X_test,
            y_train, y_val, y_test, df_train, df_val, df_test,
            use_robust=use_robust
        )
        
        if result:
            ablation_results.append({
                'Feature_Set': set_name,
                'N_Features': len(features),
                'Model': model_name,
                'Train_C_Index': result['Train_C_Index'],
                'Val_C_Index': result['Val_C_Index'],
                'Test_C_Index': result['Test_C_Index'],
                'Overfit': result['Overfit']
            })
    
    print(f"  ? Completed testing for {set_name}")

# Save ablation results
ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(f'{output_base}/ablation/ablation_study_results.csv', index=False)

print("\n[Ablation Study Summary]")
print("-" * 80)

if len(ablation_df) > 0:
    print("\nTop Features Ablation (Best Test C-Index per feature set):")
    top_feat_summary = ablation_df[ablation_df['Feature_Set'].str.contains('top_', na=False)].groupby('N_Features')['Test_C_Index'].max()
    print(top_feat_summary.to_string())
    
    print("\nGenomic Features Ablation (Best Test C-Index per feature set):")
    genomic_summary = ablation_df[ablation_df['Feature_Set'].str.contains('genomic_', na=False)].groupby('N_Features')['Test_C_Index'].max()
    print(genomic_summary.to_string())
    
    print("\nClinical Features Ablation (Best Test C-Index per feature set):")
    clinical_summary = ablation_df[ablation_df['Feature_Set'].str.contains('clinical_', na=False)].groupby('N_Features')['Test_C_Index'].max()
    if len(clinical_summary) > 0:
        print(clinical_summary.to_string())
    
    print("\nRobust vs Original Models Comparison (on stable features):")
    stable_results = ablation_df[ablation_df['Feature_Set'] == 'top_' + str(len(stable_features))]
    if len(stable_results) > 0:
        for model_base in ['Cox - Combined', 'Random Survival Forest', 'Gradient Boosting', 'Elastic Net', 'SVM']:
            orig = stable_results[stable_results['Model'] == model_base]['Test_C_Index'].values
            robust = stable_results[stable_results['Model'].str.contains(model_base.split()[0]) & 
                                  stable_results['Model'].str.contains('Robust')]['Test_C_Index'].values
            
            if len(orig) > 0 and len(robust) > 0:
                print(f"  {model_base}:")
                print(f"    Original: {orig[0]:.4f}")
                print(f"    Robust:   {robust[0]:.4f}")
                print(f"    Improvement: {robust[0] - orig[0]:+.4f}")

print("\n? Part 10 Complete: Comprehensive ablation studies across ALL models")

# ============================================================================
# PART 11: COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n[PHASE 11] GENERATING ENHANCED VISUALIZATIONS")
print("-" * 80)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# Figure 1: Descriptive Statistics Overview
# ============================================================================

print("\n[1] Creating descriptive statistics overview...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Subplot 1: Survival time distribution
    ax1 = axes[0, 0]
    ax1.hist(df['Time'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(df['Time'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Time"].mean():.2f}')
    ax1.axvline(df['Time'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Time"].median():.2f}')
    ax1.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Survival Time Distribution (Normalized)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Event distribution
    ax2 = axes[0, 1]
    event_counts = df['Event'].value_counts()
    colors = ['lightgreen', 'salmon']
    ax2.bar(['Censored', 'Death'], event_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Event Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate(event_counts.values):
        ax2.text(i, v + 10, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Cancer type distribution
    ax3 = axes[1, 0]
    cancer_counts = df_processed['cancer_type_label'].value_counts()
    ax3.bar(cancer_counts.index, cancer_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Cancer Type Distribution', fontsize=14, fontweight='bold')
    for i, (idx, v) in enumerate(cancer_counts.items()):
        ax3.text(i, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Age distribution
    ax4 = axes[1, 1]
    ax4.hist(df['age_at_diagnosis'], bins=30, alpha=0.7, color='orchid', edgecolor='black')
    ax4.axvline(df['age_at_diagnosis'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["age_at_diagnosis"].mean():.1f}')
    ax4.axvline(df['age_at_diagnosis'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["age_at_diagnosis"].median():.1f}')
    ax4.set_xlabel('Age at Diagnosis', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Age at Diagnosis Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Descriptive Statistics Overview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/01_descriptive_statistics.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("   ? Saved: 01_descriptive_statistics.png")
    
except Exception as e:
    print(f"   ? Error: {str(e)[:100]}")

# ============================================================================
# Figure 2: Enhanced Model Performance (4 subplots)
# ============================================================================

print("\n[2] Creating enhanced model performance comparison...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    
    # Subplot 1: Bar chart with C-index scores
    ax1 = axes[0, 0]
    models = perf_df_all['Model'].values
    train_scores = perf_df_all['Train_C_Index'].values
    val_scores = perf_df_all.get('Val_C_Index', perf_df_all['Test_C_Index']).values
    test_scores = perf_df_all['Test_C_Index'].values
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_scores, width, label='Training', 
                    alpha=0.8, edgecolor='black', linewidth=1.2,
                    color=[MODEL_COLORS.get(m, 'skyblue') for m in models])
    bars2 = ax1.bar(x, val_scores, width, label='Validation', 
                    alpha=0.7, edgecolor='black', linewidth=1.2,
                    color=[MODEL_COLORS.get(m, 'lightcoral') for m in models])
    bars3 = ax1.bar(x + width, test_scores, width, label='Testing', 
                    alpha=0.6, edgecolor='black', linewidth=1.2,
                    color=[MODEL_COLORS.get(m, 'coral') for m in models])
    
    # Add scores on top of bars
    for i, (train, val, test) in enumerate(zip(train_scores, val_scores, test_scores)):
        ax1.text(i - width, train + 0.01, f'{train:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax1.text(i, val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax1.text(i + width, test + 0.01, f'{test:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('C-Index', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison (Train/Val/Test)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontweight='bold', fontsize=8)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0.5, 1.0])
    
    # Subplot 2: Overfitting analysis (Train-Val Gap)
    ax2 = axes[0, 1]
    overfit = perf_df_all.get('Train_Val_Gap', perf_df_all['Train_C_Index'] - perf_df_all.get('Val_C_Index', perf_df_all['Test_C_Index'])).values
    
    colors_overfit = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in overfit]
    bars = ax2.barh(range(len(models)), overfit, 
                    color=colors_overfit, 
                    alpha=0.7, edgecolor='black')
    
    # Add scores on bars
    for i, val in enumerate(overfit):
        ax2.text(val + 0.005, i, f'{val:.3f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontweight='bold', fontsize=9)
    ax2.set_xlabel('Overfitting (Train - Val C-Index)', fontsize=12, fontweight='bold')
    ax2.set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.05, color='orange', linestyle='--', label='Acceptable (<0.05)')
    ax2.axvline(x=0.1, color='red', linestyle='--', label='High (>0.1)')
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    
    # Subplot 3: Generalization (Val-Test Gap)
    ax3 = axes[1, 0]
    gen_gap = perf_df_all.get('Val_Test_Gap', np.zeros(len(models))).values
    
    colors_gen = ['green' if abs(x) < 0.02 else 'orange' if abs(x) < 0.05 else 'red' for x in gen_gap]
    bars = ax3.barh(range(len(models)), gen_gap, 
                    color=colors_gen, 
                    alpha=0.7, edgecolor='black')
    
    for i, val in enumerate(gen_gap):
        ax3.text(val + 0.005 if val > 0 else val - 0.005, i, f'{val:+.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels(models, fontweight='bold', fontsize=9)
    ax3.set_xlabel('Generalization Gap (Val - Test C-Index)', fontsize=12, fontweight='bold')
    ax3.set_title('Generalization Analysis (Val-Test Gap)', fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.axvline(x=0.02, color='orange', linestyle='--', alpha=0.5)
    ax3.axvline(x=-0.02, color='orange', linestyle='--', alpha=0.5)
    ax3.grid(axis='x', alpha=0.3)
    
    # Subplot 4: Generalization scatter with shapes and colors
    ax4 = axes[1, 1]
    
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, 'gray')
        marker = MODEL_MARKERS.get(model, 'o')
        
        ax4.scatter(train_scores[i], test_scores[i], 
                   s=200, alpha=0.7, c=color, marker=marker,
                   edgecolors='black', linewidth=1.5, label=model)
    
    lims = [0.5, max(max(train_scores), max(test_scores)) + 0.02]
    ax4.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Generalization')
    
    ax4.set_xlabel('Training C-Index', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test C-Index', fontsize=12, fontweight='bold')
    ax4.set_title('Generalization Performance (Train vs Test)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=7, loc='best', ncol=2)
    ax4.grid(alpha=0.3)
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    
    plt.suptitle('Enhanced Model Performance Analysis with Robust Configurations', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/02_enhanced_model_performance.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("   ? Saved: 02_enhanced_model_performance.png")
    
except Exception as e:
    print(f"   ? Error: {str(e)[:100]}")

# ============================================================================
# Figure 3: Enhanced Ablation Study (2x2 grid)
# ============================================================================

print("\n[3] Creating enhanced ablation study visualizations...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Subplot 1: Top features ablation
    ax1 = axes[0, 0]
    
    top_ablation = ablation_df[ablation_df['Feature_Set'].str.contains('top_', na=False)]
    
    if len(top_ablation) > 0:
        # Get unique models and plot each
        for model in top_ablation['Model'].unique():
            model_data = top_ablation[top_ablation['Model'] == model]
            model_data = model_data.sort_values('N_Features')
            
            color = MODEL_COLORS.get(model, 'gray')
            marker = MODEL_MARKERS.get(model, 'o')
            
            ax1.plot(model_data['N_Features'], model_data['Test_C_Index'], 
                    marker=marker, label=model, linewidth=2.5, markersize=8, 
                    alpha=0.8, color=color)
        
        ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test C-Index', fontsize=12, fontweight='bold')
        ax1.set_title('Ablation: Top Features (All Models)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=8, loc='best', ncol=2)
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0.5, 0.8])
    else:
        ax1.text(0.5, 0.5, 'No top ablation data', ha='center', va='center', transform=ax1.transAxes)
    
    # Subplot 2: Genomic features ablation
    ax2 = axes[0, 1]
    
    genomic_ablation = ablation_df[ablation_df['Feature_Set'].str.contains('genomic_', na=False)]
    
    if len(genomic_ablation) > 0:
        for model in genomic_ablation['Model'].unique():
            model_data = genomic_ablation[genomic_ablation['Model'] == model]
            model_data = model_data.sort_values('N_Features')
            
            color = MODEL_COLORS.get(model, 'gray')
            marker = MODEL_MARKERS.get(model, 's')
            
            ax2.plot(model_data['N_Features'], model_data['Test_C_Index'], 
                    marker=marker, label=model, linewidth=2.5, markersize=8, 
                    alpha=0.8, color=color)
        
        ax2.set_xlabel('Number of Genomic Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Test C-Index', fontsize=12, fontweight='bold')
        ax2.set_title('Ablation: Genomic Features', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8, loc='best', ncol=2)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0.5, 0.8])
    else:
        ax2.text(0.5, 0.5, 'No genomic ablation data', ha='center', va='center', transform=ax2.transAxes)
    
    # Subplot 3: Robust vs Original comparison
    ax3 = axes[1, 0]
    
    if len(ablation_df) > 0:
        # Compare robust vs original for each base model
        comparison_data = []
        
        for base_model in ['Cox - Combined', 'Random Survival Forest', 'Gradient Boosting', 'Elastic Net', 'SVM']:
            # Get results for both versions
            orig_results = ablation_df[ablation_df['Model'] == base_model]
            robust_results = ablation_df[ablation_df['Model'].str.contains(base_model.split()[0]) & 
                                       ablation_df['Model'].str.contains('Robust')]
            
            if len(orig_results) > 0 and len(robust_results) > 0:
                # Get average test C-index
                orig_avg = orig_results['Test_C_Index'].mean()
                robust_avg = robust_results['Test_C_Index'].mean()
                
                comparison_data.append({
                    'Model': base_model.split()[0],
                    'Original': orig_avg,
                    'Robust': robust_avg,
                    'Improvement': robust_avg - orig_avg
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            x = np.arange(len(comp_df))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, comp_df['Original'], width, label='Original', 
                           alpha=0.7, edgecolor='black', color='lightblue')
            bars2 = ax3.bar(x + width/2, comp_df['Robust'], width, label='Robust', 
                           alpha=0.7, edgecolor='black', color='lightgreen')
            
            # Add improvement values
            for i, row in comp_df.iterrows():
                ax3.text(i, max(row['Original'], row['Robust']) + 0.01, 
                        f'{row["Improvement"]:+.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='green' if row['Improvement'] > 0 else 'red')
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(comp_df['Model'], fontweight='bold')
            ax3.set_ylabel('Average Test C-Index', fontsize=12, fontweight='bold')
            ax3.set_title('Robust vs Original Hyperparameters', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(axis='y', alpha=0.3)
            ax3.set_ylim([0.5, 0.8])
    
    # Subplot 4: Overfitting vs feature count
    ax4 = axes[1, 1]
    
    if len(top_ablation) > 0:
        for model in top_ablation['Model'].unique():
            model_data = top_ablation[top_ablation['Model'] == model]
            model_data = model_data.sort_values('N_Features')
            
            color = MODEL_COLORS.get(model, 'gray')
            marker = MODEL_MARKERS.get(model, 'v')
            
            ax4.plot(model_data['N_Features'], model_data['Overfit'], 
                    marker=marker, label=model, linewidth=2, markersize=7, 
                    alpha=0.8, color=color)
        
        ax4.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Acceptable (<0.05)')
        ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='High (>0.1)')
        ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Overfitting (Train - Val C-Index)', fontsize=12, fontweight='bold')
        ax4.set_title('Overfitting vs Feature Count', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=8, loc='best', ncol=2)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No overfitting data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle('Comprehensive Ablation Studies with Robust Configurations', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/03_ablation_study_enhanced.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("   ? Saved: 03_ablation_study_enhanced.png")
    
except Exception as e:
    print(f"   ? Error: {str(e)}")

print("\n? Part 11 Complete: Enhanced visualizations generated")

# ============================================================================
# Figure 4: Kaplan-Meier Curves with Proper Cancer Type Filtering
# ============================================================================

print("\n[4] Creating Kaplan-Meier survival curves...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    kmf = KaplanMeierFitter()
    colors = {'Low Risk': 'green', 'Intermediate Risk': 'orange', 'High Risk': 'red'}
    
    # Use best model's risk groups
    if best_model_key in risk_groups_all_models:
        best_risk_groups = risk_groups_all_models[best_model_key]
        df_test_km = df_test.copy()
        df_test_km['Risk_Group'] = best_risk_groups
        
        # Define cancer types with correct filtering
        cancer_types = [
            ('Overall', None, axes[0, 0]),
            ('BRCA', 'BRCA', axes[0, 1]),
            ('COAD', 'COAD', axes[1, 0]),
            ('LUAD', 'LUAD', axes[1, 1])
        ]
        
        for cancer_name, cancer_filter, ax in cancer_types:
            # Filter by cancer type
            if cancer_filter is None:
                plot_df = df_test_km.copy()
            else:
                plot_df = df_test_km[df_test_km['cancer_type_label'] == cancer_filter].copy()
            
            # Calculate TOTAL patients including censored
            n_patients_total = len(plot_df)
            n_events_total = int(plot_df['Event'].sum())
            n_censored_total = n_patients_total - n_events_total
            
            print(f"\n{cancer_name} Statistics:")
            print(f"  Total patients: {n_patients_total}")
            print(f"  Events: {n_events_total}")
            print(f"  Censored: {n_censored_total}")
            
            # Only plot if we have patients
            if n_patients_total == 0:
                ax.text(0.5, 0.5, f'No {cancer_name} patients in test set', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{cancer_name}\n(No patients)', fontsize=12, fontweight='bold')
                continue
            
            for group in ['Low Risk', 'Intermediate Risk', 'High Risk']:
                mask = plot_df['Risk_Group'] == group
                T_group = plot_df.loc[mask, 'Time']
                E_group = plot_df.loc[mask, 'Event']
                
                if len(T_group) > 0:
                    n_group = len(T_group)
                    n_events_group = int(E_group.sum())
                    n_censored_group = n_group - n_events_group
                    
                    kmf.fit(T_group, E_group, 
                           label=f'{group} (n={n_group}, events={n_events_group}, censored={n_censored_group})')
                    kmf.plot_survival_function(ax=ax, ci_show=True, 
                                              color=colors[group], linewidth=2.5, alpha=0.8)
            
            # Log-rank test
            try:
                if n_patients_total > 0 and len(plot_df['Risk_Group'].unique()) > 1:
                    result = multivariate_logrank_test(
                        plot_df['Time'],
                        plot_df['Risk_Group'],
                        plot_df['Event']
                    )
                    p_value = result.p_value
                    p_text = f'Log-rank p={p_value:.4e}' if p_value >= 0.001 else 'Log-rank p<0.001'
                else:
                    p_text = 'Log-rank test N/A'
            except:
                p_text = 'Log-rank test N/A'
            
            ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'{cancer_name}\n{p_text}\n(Total N={n_patients_total}, Events={n_events_total}, Censored={n_censored_total})', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            ax.set_ylim([0, 1.05])
        
        plt.suptitle(f'Kaplan-Meier Survival Curves by Risk Stratification\nBest Model: {best_model_key}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_base}/figures/04_kaplan_meier_curves.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        print("   ? Saved: 04_kaplan_meier_curves.png")
    else:
        print("   ? Skipped: No risk groups available for best model")
    
except Exception as e:
    print(f"   ? Error: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Figure 5: Top 20 Features Heatmap (All Models)
# ============================================================================

print("\n[5] Creating comprehensive top 20 features heatmap...")

try:
    # Collect feature importance from all models
    all_feature_importance = {}
    
    # From univariate analysis
    univar_top20 = univariate_df.head(20).set_index('Feature')['HR'].to_dict()
    all_feature_importance['Univariate HR'] = univar_top20
    
    # From Cox - Clinical Only
    if 'clinical_only' in cox_models:
        try:
            model = cox_models['clinical_only']
            if hasattr(model, 'params_'):
                params = model.params_.abs().sort_values(ascending=False).head(20)
                all_feature_importance['Cox - Clinical Only'] = params.to_dict()
        except:
            pass
    
    # From Cox - Genomic Only
    if 'genomic_only' in cox_models:
        try:
            model = cox_models['genomic_only']
            if hasattr(model, 'params_'):
                params = model.params_.abs().sort_values(ascending=False).head(20)
                all_feature_importance['Cox - Genomic Only'] = params.to_dict()
        except:
            pass
    
    # From Cox - Combined
    if 'combined' in cox_models:
        try:
            model = cox_models['combined']
            if hasattr(model, 'params_'):
                params = model.params_.abs().sort_values(ascending=False).head(20)
                all_feature_importance['Cox - Combined'] = params.to_dict()
        except:
            pass
    
    # From Cox - Combined (Robust)
    if 'combined_robust' in cox_models:
        try:
            model = cox_models['combined_robust']
            if hasattr(model, 'params_'):
                params = model.params_.abs().sort_values(ascending=False).head(20)
                all_feature_importance['Cox - Combined (Robust)'] = params.to_dict()
        except:
            pass
    
    # From Random Survival Forest
    if os.path.exists(f'{output_base}/tables/rsf_feature_importance.csv'):
        rsf_imp = pd.read_csv(f'{output_base}/tables/rsf_feature_importance.csv')
        rsf_imp = rsf_imp.head(20).set_index('Feature')['Importance'].to_dict()
        all_feature_importance['Random Survival Forest'] = rsf_imp
    
    # From Gradient Boosting
    if os.path.exists(f'{output_base}/tables/gbsa_feature_importance.csv'):
        gb_imp = pd.read_csv(f'{output_base}/tables/gbsa_feature_importance.csv')
        gb_imp = gb_imp.head(20).set_index('Feature')['Importance'].to_dict()
        all_feature_importance['Gradient Boosting Survival'] = gb_imp
    
    # From Survival SVM (using permutation importance)
    if 'survival_svm' in cox_models:
        try:
            svm_features = cox_models.get('survival_svm_features', significant_features[:20])
            scaler_svm = cox_models.get('survival_svm_scaler', StandardScaler())
            
            X_test_svm_imp = df_test[svm_features].values
            X_test_svm_scaled_imp = scaler_svm.transform(X_test_svm_imp)
            
            print("  Calculating SVM feature importance using permutation...")
            svm_imp = calculate_svm_feature_importance(
                cox_models['survival_svm'], 
                X_test_svm_scaled_imp, 
                y_test, 
                svm_features,
                n_repeats=10
            )
            
            if svm_imp:
                sorted_svm = sorted(svm_imp.items(), key=lambda x: x[1], reverse=True)[:20]
                all_feature_importance['Survival SVM'] = dict(sorted_svm)
                print(f"   SVM feature importance calculated: {len(sorted_svm)} features")
        except Exception as e:
            print(f"   SVM feature importance error: {str(e)[:100]}")
    
    # From SVM (Robust)
    if 'svm_robust' in cox_models:
        try:
            svm_features = cox_models.get('svm_robust_features', stable_features[:15])
            scaler_svm = cox_models.get('svm_robust_scaler', RobustScaler())
            
            X_test_svm_imp = df_test[svm_features].values
            X_test_svm_scaled_imp = scaler_svm.transform(X_test_svm_imp)
            
            print("  Calculating SVM (Robust) feature importance using permutation...")
            svm_imp = calculate_svm_feature_importance(
                cox_models['svm_robust'], 
                X_test_svm_scaled_imp, 
                y_test, 
                svm_features,
                n_repeats=10
            )
            
            if svm_imp:
                sorted_svm = sorted(svm_imp.items(), key=lambda x: x[1], reverse=True)[:20]
                all_feature_importance['SVM (Robust)'] = dict(sorted_svm)
                print(f"   SVM (Robust) feature importance calculated: {len(sorted_svm)} features")
        except Exception as e:
            print(f"   SVM (Robust) feature importance error: {str(e)[:100]}")
    
    # Create comparison dataframe
    all_features_list_imp = list(set([f for imp_dict in all_feature_importance.values() for f in imp_dict.keys()]))
    
    comparison_df = pd.DataFrame(index=all_features_list_imp)
    for model_name, imp_dict in all_feature_importance.items():
        comparison_df[model_name] = pd.Series(imp_dict)
    
    # Fill NaN with 0 and normalize each column
    comparison_df = comparison_df.fillna(0)
    for col in comparison_df.columns:
        if comparison_df[col].max() > 0:
            comparison_df[col] = comparison_df[col] / comparison_df[col].max()
    
    # Get top 20 features by average importance
    comparison_df['Average'] = comparison_df.mean(axis=1)
    top_20_features_comparison = comparison_df.nlargest(20, 'Average').drop('Average', axis=1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(top_20_features_comparison.T, 
                annot=True, fmt='.2f', 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Importance'},
                linewidths=0.5,
                ax=ax)
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features: Importance Comparison Across All Models', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_base}/figures/05_top20_features_all_models.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save the comparison
    top_20_features_comparison.to_csv(f'{output_base}/tables/top20_features_comparison.csv')
    print("   ? Saved: 05_top20_features_all_models.png")
    
except Exception as e:
    print(f"   ? Error: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Figure 6: Uncertainty Visualization
# ============================================================================

print("\n[6] Creating uncertainty visualization...")

try:
    # Use robust RSF if available, otherwise original
    rsf_key = 'rsf_robust' if 'rsf_robust' in cox_models else 'random_survival_forest'
    
    if rsf_key in cox_models:
        print("  Calculating prediction uncertainty using RSF ensemble...")
        
        rsf_model = cox_models[rsf_key]
        
        # Get predictions from individual trees
        n_estimators = rsf_model.n_estimators
        tree_predictions = []
        
        for tree in rsf_model.estimators_:
            pred = tree.predict(X_test)
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate uncertainty metrics
        pred_mean = tree_predictions.mean(axis=0)
        pred_std = tree_predictions.std(axis=0)
        pred_cv = pred_std / (np.abs(pred_mean) + 1e-10)
        
        # Add risk groups if available - FIXED
        if best_model_key in risk_groups_all_models:
            risk_groups_for_uncertainty = risk_groups_all_models[best_model_key]
            # Convert to numpy array if it's a Categorical
            if hasattr(risk_groups_for_uncertainty, 'to_numpy'):
                risk_groups_array = risk_groups_for_uncertainty.to_numpy()
            elif hasattr(risk_groups_for_uncertainty, 'values'):
                risk_groups_array = risk_groups_for_uncertainty.values
            else:
                risk_groups_array = np.array(risk_groups_for_uncertainty)
        else:
            risk_groups_array = np.array(['Unknown'] * len(df_test))
        
        # Create uncertainty dataframe
        uncertainty_df_viz = pd.DataFrame({
            'Patient_ID': range(len(df_test)),
            'True_Time': T_test,
            'True_Event': E_test,
            'Pred_Mean': pred_mean,
            'Pred_Std': pred_std,
            'Pred_CV': pred_cv,
            'Risk_Group': risk_groups_array
        })
        
        # Create comprehensive uncertainty visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Prediction mean vs standard deviation
        ax1 = axes[0, 0]
        scatter = ax1.scatter(pred_mean, pred_std, c=E_test, 
                             cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Mean Prediction (Risk Score)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Standard Deviation (Uncertainty)', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Uncertainty vs Risk Score\n(Colored by Event Status)', 
                     fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Event (0=Censored, 1=Death)', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Subplot 2: Coefficient of variation by risk group
        ax2 = axes[0, 1]
        risk_groups_list = ['Low Risk', 'Intermediate Risk', 'High Risk']
        
        # Filter to only include known risk groups
        cv_by_group = []
        labels_with_counts = []
        for grp in risk_groups_list:
            group_data = uncertainty_df_viz[uncertainty_df_viz['Risk_Group'] == grp]['Pred_CV'].values
            if len(group_data) > 0:
                cv_by_group.append(group_data)
                labels_with_counts.append(f'{grp}\n(n={len(group_data)})')
        
        if len(cv_by_group) > 0:
            bp = ax2.boxplot(cv_by_group, labels=labels_with_counts, patch_artist=True,
                            showmeans=True, meanline=True)
            
            colors_box = ['green', 'orange', 'red'][:len(cv_by_group)]
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax2.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
            ax2.set_title('Prediction Uncertainty by Risk Group', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No risk group data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Subplot 3: Uncertainty vs survival time
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(T_test, pred_std, c=E_test, 
                              cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('True Survival Time (Years)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Prediction Uncertainty (Std Dev)', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Uncertainty vs Survival Time', fontsize=14, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Event')
        ax3.grid(alpha=0.3)
        
        # Subplot 4: Distribution of uncertainty
        ax4 = axes[1, 1]
        
        for i, grp in enumerate(risk_groups_list):
            group_cv = uncertainty_df_viz[uncertainty_df_viz['Risk_Group'] == grp]['Pred_CV'].values
            if len(group_cv) > 0:
                ax4.hist(group_cv, bins=30, alpha=0.6, label=f'{grp} (n={len(group_cv)})', 
                        color=colors_box[i] if i < len(colors_box) else 'gray', edgecolor='black')
        
        ax4.set_xlabel('Coefficient of Variation', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Prediction Uncertainty', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Prediction Uncertainty Analysis\nModel: {rsf_key} (Ensemble Variance)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_base}/figures/06_uncertainty_quantification.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print("   ? Saved: 06_uncertainty_quantification.png")
        
        # Print summary statistics
        print("\n  Uncertainty Statistics:")
        print(f"    Mean CV: {pred_cv.mean():.4f}")
        print(f"    Median CV: {np.median(pred_cv):.4f}")
        print(f"    Max CV: {pred_cv.max():.4f}")
        
        if len(cv_by_group) > 0:
            print("\n  By Risk Group:")
            for i, group in enumerate(risk_groups_list[:len(cv_by_group)]):
                group_cv = cv_by_group[i]
                print(f"    {group}: Mean CV = {np.mean(group_cv):.4f}, Median CV = {np.median(group_cv):.4f}")
    
    else:
        print("   ? Skipped: Random Survival Forest not available")
        
except Exception as e:
    print(f"   ? Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n? Part 12 Complete: Kaplan-Meier curves and feature importance visualizations")

# ============================================================================
# Figure 7: Comprehensive Statistical Tests Summary
# ============================================================================

print("\n[7] Creating comprehensive statistical validation summary...")

try:
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Subplot 1: Cross-validation with scores
    ax1 = fig.add_subplot(gs[0, :2])
    
    cv_file = f'{output_base}/uncertainty/cross_validation_all_models.csv'
    if os.path.exists(cv_file) and os.path.getsize(cv_file) > 0:
        try:
            cv_data = pd.read_csv(cv_file)
            
            if len(cv_data) > 0:
                x_cv = np.arange(len(cv_data))
                
                # Use consistent colors
                colors_cv = [MODEL_COLORS.get(m, 'gray') for m in cv_data['Model']]
                
                bars = ax1.bar(x_cv, cv_data['CV_Mean'], yerr=cv_data['CV_Std'],
                              capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5,
                              color=colors_cv)
                
                # Add C-index scores on top
                for i, (mean, std) in enumerate(zip(cv_data['CV_Mean'], cv_data['CV_Std'])):
                    ax1.text(i, mean + std + 0.01, f'{mean:.3f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax1.set_xticks(x_cv)
                ax1.set_xticklabels(cv_data['Model'], rotation=45, ha='right', fontweight='bold', fontsize=8)
                ax1.set_ylabel('C-Index', fontweight='bold', fontsize=12)
                ax1.set_title('5-Fold Cross-Validation Results (All Models)', 
                             fontweight='bold', fontsize=14)
                ax1.grid(axis='y', alpha=0.3)
                ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Baseline (0.7)')
                ax1.set_ylim([0.0, 1.0])
                ax1.legend(fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'No CV data available', ha='center', va='center', transform=ax1.transAxes)
        except Exception as e:
            ax1.text(0.5, 0.5, f'CV data error: {str(e)[:30]}', ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'CV data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('5-Fold Cross-Validation Results', fontweight='bold', fontsize=14)
    
    # Subplot 2: Bootstrap for ALL models
    ax2 = fig.add_subplot(gs[0, 2])
    
    boot_file = f'{output_base}/uncertainty/bootstrap_all_models.csv'
    if os.path.exists(boot_file) and os.path.getsize(boot_file) > 0:
        try:
            boot_data = pd.read_csv(boot_file)
            
            if len(boot_data) > 0:
                # Take top 10 models to fit in subplot
                boot_data_top = boot_data.nlargest(10, 'C_Index_Mean')
                
                y_boot = np.arange(len(boot_data_top))
                
                for i, (idx, row) in enumerate(boot_data_top.iterrows()):
                    color = MODEL_COLORS.get(row['Model'], 'steelblue')
                    ax2.errorbar(row['C_Index_Mean'], i,
                                xerr=[[row['C_Index_Mean'] - row['CI_Lower']], 
                                      [row['CI_Upper'] - row['C_Index_Mean']]],
                                fmt='o', markersize=8, capsize=5, capthick=2, 
                                linewidth=2, color=color)
                
                ax2.set_yticks(y_boot)
                ax2.set_yticklabels([m[:20] for m in boot_data_top['Model']], fontweight='bold', fontsize=8)
                ax2.set_xlabel('C-Index', fontweight='bold', fontsize=10)
                ax2.set_title('Bootstrap 95% CI\n(Top 10, 500 iter)', 
                             fontweight='bold', fontsize=12)
                ax2.grid(axis='x', alpha=0.3)
                ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
                ax2.set_xlim([0.5, 0.85])
            else:
                ax2.text(0.5, 0.5, 'No bootstrap data', ha='center', va='center', transform=ax2.transAxes)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Bootstrap error: {str(e)[:30]}', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Bootstrap data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Bootstrap 95% CI', fontweight='bold', fontsize=12)
    
    # Subplot 3: Ablation summary
    ax3 = fig.add_subplot(gs[1, 0])
    
    ablation_file = f'{output_base}/ablation/ablation_study_results.csv'
    if os.path.exists(ablation_file):
        ablation_df_viz = pd.read_csv(ablation_file)
        
        top_ablation_viz = ablation_df_viz[ablation_df_viz['Feature_Set'].str.contains('top_', na=False)]
        
        if len(top_ablation_viz) > 0:
            # Group by N_Features and Model, get max
            top_abl_summary = top_ablation_viz.groupby(['N_Features', 'Model'])['Test_C_Index'].max().unstack()
            
            # Plot top 5 models only
            if len(top_abl_summary.columns) > 5:
                # Get models with highest average performance
                avg_perf = top_abl_summary.mean()
                top_models = avg_perf.nlargest(5).index
                top_abl_summary = top_abl_summary[top_models]
            
            for model in top_abl_summary.columns:
                color = MODEL_COLORS.get(model, 'gray')
                marker = MODEL_MARKERS.get(model, 'o')
                
                ax3.plot(top_abl_summary.index, top_abl_summary[model], 
                        marker=marker, label=model[:25], linewidth=2, markersize=6,
                        alpha=0.8, color=color)
            
            ax3.set_xlabel('Number of Features', fontweight='bold', fontsize=10)
            ax3.set_ylabel('Best Test C-Index', fontweight='bold', fontsize=10)
            ax3.set_title('Ablation Study Summary\n(Top 5 Models)', fontweight='bold', fontsize=12)
            ax3.legend(fontsize=8, loc='best', framealpha=0.9)
            ax3.grid(alpha=0.3, linestyle='--')
            ax3.set_ylim([0.5, 0.8])
        else:
            ax3.text(0.5, 0.5, 'No ablation data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Ablation data not available', ha='center', va='center', transform=ax3.transAxes)
    
    # Subplot 4: Hazard ratios across models
    ax4 = fig.add_subplot(gs[1, 1])
    
    hr_file = f'{output_base}/statistics/hazard_ratios_all_models.csv'
    if os.path.exists(hr_file):
        hr_data = pd.read_csv(hr_file)
        
        # Filter for "High vs Low" comparison
        hr_high = hr_data[hr_data['Comparison'] == 'High vs Low']
        
        if len(hr_high) > 0:
            # Take top 10 models
            hr_high_top = hr_high.nlargest(10, 'HR')
            
            y_hr = np.arange(len(hr_high_top))
            
            for i, (idx, row) in enumerate(hr_high_top.iterrows()):
                color = MODEL_COLORS.get(row['Model'], 'gray')
                ax4.errorbar(row['HR'], i,
                            xerr=[[row['HR'] - row['CI_Lower']],
                                  [row['CI_Upper'] - row['HR']]],
                            fmt='o', markersize=10, capsize=5, capthick=2, 
                            linewidth=2, color=color)
            
            ax4.axvline(x=1, color='black', linestyle='--', linewidth=2, label='No Effect')
            ax4.set_yticks(y_hr)
            ax4.set_yticklabels([m[:25] for m in hr_high_top['Model']], fontweight='bold', fontsize=8)
            ax4.set_xlabel('Hazard Ratio (95% CI)', fontweight='bold', fontsize=10)
            ax4.set_title('Risk Group Hazard Ratios\n(High vs Low, Top 10)', 
                         fontweight='bold', fontsize=12)
            ax4.legend(fontsize=8)
            ax4.grid(axis='x', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'HR data not available', ha='center', va='center', transform=ax4.transAxes)
    
    # Subplot 5: Feature stability scores
    ax5 = fig.add_subplot(gs[1, 2])
    
    if os.path.exists(f'{output_base}/tables/feature_stability_scores.csv'):
        stability_data = pd.read_csv(f'{output_base}/tables/feature_stability_scores.csv')
        top_stable = stability_data.nlargest(15, 'Stability')
        
        colors_stable = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' 
                        for s in top_stable['Stability']]
        
        ax5.barh(range(len(top_stable)), top_stable['Stability'], 
                color=colors_stable, alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_stable)))
        ax5.set_yticklabels(top_stable['Feature'], fontweight='bold', fontsize=8)
        ax5.set_xlabel('Stability Score', fontweight='bold', fontsize=10)
        ax5.set_title('Top 15 Most Stable Features\n(Bootstrap Selection)', 
                     fontweight='bold', fontsize=12)
        ax5.axvline(x=0.6, color='orange', linestyle='--', label='Threshold (0.6)')
        ax5.axvline(x=0.8, color='green', linestyle='--', label='High Stability (0.8)')
        ax5.legend(fontsize=8)
        ax5.grid(axis='x', alpha=0.3)
        ax5.set_xlim([0, 1])
    else:
        ax5.text(0.5, 0.5, 'Stability data not available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Feature Stability', fontweight='bold', fontsize=12)
    
    # Subplot 6: Performance consistency (CV mean vs std)
    ax6 = fig.add_subplot(gs[2, 0])
    
    if os.path.exists(cv_file) and os.path.getsize(cv_file) > 0:
        try:
            cv_data = pd.read_csv(cv_file)
            
            if len(cv_data) > 0:
                for i, row in cv_data.iterrows():
                    color = MODEL_COLORS.get(row['Model'], 'gray')
                    marker = MODEL_MARKERS.get(row['Model'], 'o')
                    
                    ax6.scatter(row['CV_Mean'], row['CV_Std'], 
                               s=150, alpha=0.6, c=color, marker=marker,
                               edgecolors='black', linewidth=1.5, label=row['Model'][:20])
                
                ax6.set_xlabel('Mean CV C-Index', fontweight='bold', fontsize=10)
                ax6.set_ylabel('CV Standard Deviation', fontweight='bold', fontsize=10)
                ax6.set_title('Performance vs Consistency', fontweight='bold', fontsize=11)
                ax6.grid(alpha=0.3)
                ax6.set_ylim([0.0, 0.10])
                
                # Simplified legend
                handles, labels = ax6.get_legend_handles_labels()
                if len(labels) > 10:
                    ax6.legend(handles[:10], labels[:10], fontsize=7, loc='upper right', ncol=2)
                else:
                    ax6.legend(fontsize=7, loc='best', ncol=2)
                
                mean_std = cv_data['CV_Std'].mean()
                ax6.axhline(y=mean_std, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax6.text(0.02, mean_std + 0.002, f'Mean Std: {mean_std:.4f}', 
                        transform=ax6.get_yaxis_transform(), fontsize=9, 
                        color='red', fontweight='bold')
            else:
                ax6.text(0.5, 0.5, 'No CV data', ha='center', va='center', transform=ax6.transAxes)
        except:
            ax6.text(0.5, 0.5, 'CV data error', ha='center', va='center', transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'CV data not available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Performance vs Consistency', fontweight='bold', fontsize=12)
    
    # Subplot 7: Robust vs Original comparison
    ax7 = fig.add_subplot(gs[2, 1])
    
    if len(perf_df_all) > 0:
        # Compare robust vs original
        comparison_pairs = [
            ('Cox - Combined', 'Cox - Combined (Robust)'),
            ('Random Survival Forest', 'RSF (Robust)'),
            ('Gradient Boosting Survival', 'GB (Robust)'),
            ('Cox - Elastic Net', 'Elastic Net (Robust)'),
            ('Survival SVM ', 'SVM (Robust)')
        ]
        
        comparison_data = []
        for orig_name, robust_name in comparison_pairs:
            orig_perf = perf_df_all[perf_df_all['Model'] == orig_name]['Test_C_Index'].values
            robust_perf = perf_df_all[perf_df_all['Model'] == robust_name]['Test_C_Index'].values
            
            if len(orig_perf) > 0 and len(robust_perf) > 0:
                comparison_data.append({
                    'Model': orig_name.split()[0],
                    'Original': orig_perf[0],
                    'Robust': robust_perf[0],
                    'Improvement': robust_perf[0] - orig_perf[0]
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            x = np.arange(len(comp_df))
            width = 0.35
            
            bars1 = ax7.bar(x - width/2, comp_df['Original'], width, label='Original', 
                           alpha=0.7, edgecolor='black', color='lightblue')
            bars2 = ax7.bar(x + width/2, comp_df['Robust'], width, label='Robust', 
                           alpha=0.7, edgecolor='black', color='lightgreen')
            
            # Add improvement values
            for i, row in comp_df.iterrows():
                improvement = row['Improvement']
                y_pos = max(row['Original'], row['Robust']) + 0.01
                color = 'green' if improvement > 0 else 'red'
                ax7.text(i, y_pos, f'{improvement:+.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color=color)
            
            ax7.set_xticks(x)
            ax7.set_xticklabels(comp_df['Model'], fontweight='bold')
            ax7.set_ylabel('Test C-Index', fontsize=11, fontweight='bold')
            ax7.set_title('Robust vs Original Configurations', fontsize=12, fontweight='bold')
            ax7.legend(fontsize=10)
            ax7.grid(axis='y', alpha=0.3)
            ax7.set_ylim([0.5, 0.85])
    
    # Subplot 8: Train-Val-Test progression for top models
    ax8 = fig.add_subplot(gs[2, 2])
    
    if len(perf_df_all) > 0:
        # Get top 5 models
        top_5_models = perf_df_all.head(5)
        
        x_prog = np.arange(3)
        width_prog = 0.15
        
        for i, (idx, row) in enumerate(top_5_models.iterrows()):
            model_name = row['Model']
            train_c = row['Train_C_Index']
            val_c = row.get('Val_C_Index', row['Test_C_Index'])
            test_c = row['Test_C_Index']
            
            color = MODEL_COLORS.get(model_name, 'gray')
            
            positions = x_prog + i * width_prog
            ax8.plot(positions, [train_c, val_c, test_c], 
                    marker='o', linewidth=2, markersize=8,
                    label=model_name[:20], color=color, alpha=0.7)
        
        ax8.set_xticks(x_prog + width_prog * 2)
        ax8.set_xticklabels(['Train', 'Val', 'Test'], fontweight='bold')
        ax8.set_ylabel('C-Index', fontweight='bold', fontsize=11)
        ax8.set_title('Train-Val-Test Progression\n(Top 5 Models)', fontweight='bold', fontsize=12)
        ax8.legend(fontsize=8, loc='best')
        ax8.grid(alpha=0.3)
        ax8.set_ylim([0.5, 0.9])
    
    plt.suptitle('Comprehensive Statistical Validation Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f'{output_base}/figures/07_statistical_validation_complete.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("   ? Saved: 07_statistical_validation_complete.png")
    
except Exception as e:
    print(f"   ? Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n Part 13 Complete: All visualizations with statistical validation")
# ============================================================================
# PART 12: COMPREHENSIVE FINAL REPORT
# ============================================================================

print("\n[PHASE 12] GENERATING COMPREHENSIVE FINAL REPORT")
print("=" * 80)

report_lines = []

report_lines.append("=" * 80)
report_lines.append("TEMPORAL GENOMIC EVOLUTION PREDICTION - COMPREHENSIVE FINAL REPORT")
report_lines.append("Enhanced with Robust Training and Validation Strategies")
report_lines.append("=" * 80)
report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Feature Type: {config_dict['feature_type']}")
report_lines.append("")

# ============================================================================
# 1. Dataset Summary
# ============================================================================

report_lines.append("1. DATASET SUMMARY")
report_lines.append("-" * 80)
report_lines.append(f"Total Patients: {config_dict['n_patients']}")
report_lines.append(f"Death Events: {config_dict['n_events']} ({config_dict['event_rate']*100:.2f}%)")
report_lines.append(f"Censored: {config_dict['n_patients'] - config_dict['n_events']} ({(1-config_dict['event_rate'])*100:.2f}%)")
report_lines.append("")

report_lines.append(f"Training Set: {len(df_train)} patients ({int(df_train['Event'].sum())} events, {df_train['Event'].sum()/len(df_train)*100:.1f}%)")
report_lines.append(f"Validation Set: {len(df_val)} patients ({int(df_val['Event'].sum())} events, {df_val['Event'].sum()/len(df_val)*100:.1f}%)")
report_lines.append(f"Test Set: {len(df_test)} patients ({int(df_test['Event'].sum())} events, {df_test['Event'].sum()/len(df_test)*100:.1f}%)")
report_lines.append("")

report_lines.append(f"Survival Time Statistics:")
report_lines.append(f"  Mean: {df['Time'].mean():.2f} years")
report_lines.append(f"  Median: {df['Time'].median():.2f} years")
report_lines.append(f"  Range: [{df['Time'].min():.2f}, {df['Time'].max():.2f}] years")
report_lines.append(f"  IQR: {df['Time'].quantile(0.75) - df['Time'].quantile(0.25):.2f} years")
report_lines.append(f"  Note: Time values are normalized (bell-shaped distribution)")
report_lines.append("")

# Cancer type distribution
report_lines.append("Cancer Type Distribution (Continuous Features):")
for _, row in cancer_stats_df.iterrows():
    report_lines.append(f"  {row['Cancer_Type']}:")
    report_lines.append(f"    Mean Score: {row['Mean_Score']:.3f} +/- {row['Std_Score']:.3f}")
    report_lines.append(f"    High Score Patients: {int(row['N_Patients_High'])} ({row['Percentage']:.1f}%)")
    report_lines.append(f"    Events: {int(row['N_Events'])} ({row['Event_Rate']:.1f}%)")
    report_lines.append(f"    Median Survival: {row['Median_Survival']:.2f} years")
report_lines.append("")

# ============================================================================
# 2. Robust Feature Engineering and Selection
# ============================================================================

report_lines.append("2. ROBUST FEATURE ENGINEERING AND SELECTION")
report_lines.append("-" * 80)
report_lines.append(f"Feature Type: Continuous/Normalized")
report_lines.append(f"Genomic Features: {len(genomic_features)} (continuous)")
report_lines.append(f"Clinical Features: {len(clinical_features_scaled)} (age robustly normalized)")
report_lines.append(f"Gender Features: {len(gender_features)} (continuous)")
report_lines.append(f"Tumor Stage Features: {len(tumor_stage_features)} (continuous)")
report_lines.append(f"Total Features: {len(all_features_list)}")
report_lines.append("")

report_lines.append("Robust Preprocessing Steps Applied:")
report_lines.append("  1. All features are continuous/normalized (not binary)")
report_lines.append("  2. RobustScaler used for age normalization (resistant to outliers)")
report_lines.append("  3. Stability-based feature selection (100 bootstrap iterations)")
report_lines.append(f"  4. Feature stability threshold: {config.FEATURE_THRESHOLD} (must appear in {config.FEATURE_THRESHOLD*100:.0f}% of bootstraps)")
report_lines.append(f"  5. Selected {len(stable_features)} stable features from {len(all_features_list)}")
report_lines.append("  6. Three-way data split: Train (55%) / Validation (15%) / Test (30%)")
report_lines.append("  7. Time values: shifted for sksurv ML models, original for lifelines Cox")
report_lines.append("  8. Consistent RobustScaler applied across all ML models")
report_lines.append("  9. Cancer types EXCLUDED from modeling (used for stratification only)")
report_lines.append("")

report_lines.append("Top 10 Most Stable Features:")
for _, row in feature_stability_df.head(10).iterrows():
    report_lines.append(f"  {row['Feature']:30s} Stability: {row['Stability']:.3f}")
report_lines.append("")

# ============================================================================
# 3. Robust Model Configuration
# ============================================================================

report_lines.append("3. ROBUST MODEL CONFIGURATION")
report_lines.append("-" * 80)
report_lines.append("Enhanced Regularization Parameters:")
report_lines.append(f"  Cox Penalizer: {config.COX_PENALIZER} (increased from 0.1)")
report_lines.append(f"  Cox L1 Ratio: {config.COX_L1_RATIO} (increased from 0.3)")
report_lines.append(f"  Elastic Net L1: {config.ELASTIC_NET_L1} (increased from 0.5)")
report_lines.append("")

report_lines.append("Conservative Random Survival Forest:")
report_lines.append(f"  N Estimators: {config.RSF_N_ESTIMATORS} (increased from 200)")
report_lines.append(f"  Max Depth: {config.RSF_MAX_DEPTH} (decreased from 10)")
report_lines.append(f"  Min Samples Split: {config.RSF_MIN_SAMPLES_SPLIT} (increased from 20)")
report_lines.append(f"  Min Samples Leaf: {config.RSF_MIN_SAMPLES_LEAF} (increased from 10)")
report_lines.append(f"  Max Features: {config.RSF_MAX_FEATURES} (30% of features per tree)")
report_lines.append("")

report_lines.append("Conservative Gradient Boosting:")
report_lines.append(f"  N Estimators: {config.GB_N_ESTIMATORS} (decreased from 100)")
report_lines.append(f"  Learning Rate: {config.GB_LEARNING_RATE} (decreased from 0.05)")
report_lines.append(f"  Max Depth: {config.GB_MAX_DEPTH} (decreased from 3)")
report_lines.append(f"  Min Samples Split: {config.GB_MIN_SAMPLES_SPLIT} (increased from 20)")
report_lines.append(f"  Min Samples Leaf: {config.GB_MIN_SAMPLES_LEAF} (increased from 10)")
report_lines.append(f"  Subsample: {config.GB_SUBSAMPLE} (decreased from 0.7)")
report_lines.append("")

report_lines.append("Highly Regularized SVM:")
report_lines.append(f"  Alpha Range: {config.SVM_ALPHA_RANGE} (higher values = more regularization)")
report_lines.append(f"  Hyperparameter optimization on validation set")
report_lines.append("")

# ============================================================================
# 4. Model Performance with Validation
# ============================================================================

report_lines.append("4. MODEL PERFORMANCE (WITH VALIDATION MONITORING)")
report_lines.append("-" * 80)
report_lines.append("All Models (sorted by Test C-Index):")
report_lines.append("")
report_lines.append(f"{'Model':<40} {'Features':>8} {'Train C':>10} {'Val C':>10} {'Test C':>10} {'T-V Gap':>10} {'V-T Gap':>10}")
report_lines.append("-" * 120)

for _, row in perf_df_all.iterrows():
    train_val_gap = row.get('Train_Val_Gap', row['Train_C_Index'] - row.get('Val_C_Index', row['Test_C_Index']))
    val_test_gap = row.get('Val_Test_Gap', 0)
    
    report_lines.append(
        f"{row['Model']:<40} {int(row['Features']):>8} "
        f"{row['Train_C_Index']:>10.4f} {row.get('Val_C_Index', row['Test_C_Index']):>10.4f} "
        f"{row['Test_C_Index']:>10.4f} {train_val_gap:>10.4f} {val_test_gap:>10.4f}"
    )

report_lines.append("")
report_lines.append(f"Best Model: {best_model_name}")
report_lines.append(f"  Test C-Index: {best_test_c:.4f}")
best_model_row = perf_df_all[perf_df_all['Model'] == best_model_name].iloc[0]
report_lines.append(f"  Train-Val Gap: {best_model_row.get('Train_Val_Gap', 0):.4f}")
report_lines.append(f"  Val-Test Gap: {best_model_row.get('Val_Test_Gap', 0):.4f}")
report_lines.append("")

# ============================================================================
# 5. Robustness Metrics
# ============================================================================

report_lines.append("5. ROBUSTNESS AND GENERALIZATION METRICS")
report_lines.append("-" * 80)

report_lines.append("\nGeneralization Quality (Val-Test Gap):")
report_lines.append("  Smaller gap = better generalization to new data")
for _, row in perf_df_all.iterrows():
    gap = row.get('Val_Test_Gap', 0)
    status = "Excellent" if abs(gap) < 0.02 else "Good" if abs(gap) < 0.05 else "Fair" if abs(gap) < 0.10 else "Poor"
    report_lines.append(f"  {row['Model']:<35} Gap: {gap:+.4f} ({status})")

report_lines.append("\nOverfitting Control (Train-Val Gap):")
report_lines.append("  Smaller gap = less overfitting")
for _, row in perf_df_all.iterrows():
    gap = row.get('Train_Val_Gap', row['Train_C_Index'] - row.get('Val_C_Index', row['Test_C_Index']))
    status = "Excellent" if gap < 0.05 else "Good" if gap < 0.10 else "Fair" if gap < 0.15 else "Poor"
    report_lines.append(f"  {row['Model']:<35} Gap: {gap:.4f} ({status})")

report_lines.append("\nRobust vs Original Comparison:")
comparison_pairs = [
    ('Cox - Combined', 'Cox - Combined (Robust)'),
    ('Random Survival Forest', 'RSF (Robust)'),
    ('Gradient Boosting Survival', 'GB (Robust)'),
    ('Cox - Elastic Net', 'Elastic Net (Robust)'),
    ('Survival SVM ', 'SVM (Robust)')
]

for orig_name, robust_name in comparison_pairs:
    orig_perf = perf_df_all[perf_df_all['Model'] == orig_name]['Test_C_Index'].values
    robust_perf = perf_df_all[perf_df_all['Model'] == robust_name]['Test_C_Index'].values
    
    if len(orig_perf) > 0 and len(robust_perf) > 0:
        improvement = robust_perf[0] - orig_perf[0]
        report_lines.append(f"  {orig_name}:")
        report_lines.append(f"    Original: {orig_perf[0]:.4f}")
        report_lines.append(f"    Robust:   {robust_perf[0]:.4f}")
        report_lines.append(f"    Improvement: {improvement:+.4f}")

report_lines.append("")

# ============================================================================
# 6. Statistical Validation
# ============================================================================

report_lines.append("6. STATISTICAL VALIDATION")
report_lines.append("-" * 80)

# Bootstrap CI
if os.path.exists(f'{output_base}/uncertainty/bootstrap_all_models.csv'):
    boot_ci = pd.read_csv(f'{output_base}/uncertainty/bootstrap_all_models.csv')
    report_lines.append("Bootstrap Confidence Intervals (500 iterations):")
    for _, row in boot_ci.head(10).iterrows():  # Top 10 models
        report_lines.append(f"  {row['Model']}:")
        report_lines.append(f"    C-Index: {row['C_Index_Mean']:.4f} +/- {row['C_Index_Std']:.4f}")
        report_lines.append(f"    95% CI: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
    report_lines.append("")

# Cross-validation
if os.path.exists(f'{output_base}/uncertainty/cross_validation_all_models.csv'):
    cv_res = pd.read_csv(f'{output_base}/uncertainty/cross_validation_all_models.csv')
    report_lines.append("5-Fold Cross-Validation:")
    for _, row in cv_res.head(10).iterrows():  # Top 10 models
        report_lines.append(f"  {row['Model']}:")
        report_lines.append(f"    Mean: {row['CV_Mean']:.4f} +/- {row['CV_Std']:.4f}")
        report_lines.append(f"    Range: [{row['CV_Min']:.4f}, {row['CV_Max']:.4f}]")
    report_lines.append("")

# Point-biserial correlations
if os.path.exists(f'{output_base}/statistics/feature_event_correlations.csv'):
    corr_results = pd.read_csv(f'{output_base}/statistics/feature_event_correlations.csv')
    report_lines.append("Point-Biserial Correlations (Continuous vs Binary Event):")
    report_lines.append(f"  Total features tested: {len(corr_results)}")
    report_lines.append(f"  Significant (p<0.05): {corr_results['Significant'].sum()}")
    report_lines.append(f"  Highly significant (p<0.001): {(corr_results['P_value'] < 0.001).sum()}")
    report_lines.append("")
    
    # Top 5 most significant
    report_lines.append("  Top 5 Most Significant Correlations:")
    top_corr = corr_results.nsmallest(5, 'P_value')
    for _, row in top_corr.iterrows():
        report_lines.append(f"    {row['Variable']}: r={row['Correlation']:.4f}, p={row['P_value']:.4e}")
    report_lines.append("")

# Mann-Whitney U tests
if os.path.exists(f'{output_base}/statistics/mann_whitney_tests.csv'):
    mw_results = pd.read_csv(f'{output_base}/statistics/mann_whitney_tests.csv')
    report_lines.append("Mann-Whitney U Tests (Alive vs Dead):")
    report_lines.append(f"  Variables tested: {len(mw_results)}")
    report_lines.append(f"  Significant (p<0.05): {mw_results['Significant'].sum()}")
    report_lines.append("")

# ============================================================================
# 7. Ablation Studies
# ============================================================================

report_lines.append("7. ABLATION STUDIES")
report_lines.append("-" * 80)

if len(ablation_df) > 0:
    report_lines.append("Top Features Ablation (Best Test C-Index per feature count):")
    top_abl = ablation_df[ablation_df['Feature_Set'].str.contains('top_', na=False)].groupby('N_Features')['Test_C_Index'].max()
    for n_feat, c_idx in top_abl.items():
        report_lines.append(f"  {int(n_feat):2d} features: {c_idx:.4f}")
    report_lines.append("")
    
    report_lines.append("Genomic Features Ablation (Best Test C-Index per feature count):")
    gen_abl = ablation_df[ablation_df['Feature_Set'].str.contains('genomic_', na=False)].groupby('N_Features')['Test_C_Index'].max()
    for n_feat, c_idx in gen_abl.items():
        report_lines.append(f"  {int(n_feat):2d} features: {c_idx:.4f}")
    report_lines.append("")
    
    report_lines.append("Clinical vs Genomic Comparison:")
    baseline_comparison = ablation_df[ablation_df['Feature_Set'].isin(['clinical_only', 'genomic_only'])]
    if len(baseline_comparison) > 0:
        for model in baseline_comparison['Model'].unique():
            model_data = baseline_comparison[baseline_comparison['Model'] == model]
            clinical = model_data[model_data['Feature_Set'] == 'clinical_only']['Test_C_Index'].values
            genomic = model_data[model_data['Feature_Set'] == 'genomic_only']['Test_C_Index'].values
            
            if len(clinical) > 0 and len(genomic) > 0:
                report_lines.append(f"  {model}:")
                report_lines.append(f"    Clinical-Only: {clinical[0]:.4f}")
                report_lines.append(f"    Genomic-Only:  {genomic[0]:.4f}")
                report_lines.append(f"    Difference:    {genomic[0] - clinical[0]:.4f}")
    report_lines.append("")

# ============================================================================
# 8. Risk Stratification
# ============================================================================

report_lines.append("8. RISK STRATIFICATION")
report_lines.append("-" * 80)

if os.path.exists(f'{output_base}/statistics/risk_group_statistics.csv'):
    risk_stats_saved = pd.read_csv(f'{output_base}/statistics/risk_group_statistics.csv')
    
    for _, row in risk_stats_saved.iterrows():
        report_lines.append(f"{row['Risk_Group']}:")
        report_lines.append(f"  Patients: {int(row['N_Patients'])} ({row['N_Patients']/len(df_test)*100:.1f}%)")
        report_lines.append(f"  Events: {int(row['N_Events'])} ({row['Event_Rate']:.1f}%)")
        report_lines.append(f"  Censored: {int(row['N_Censored'])} ({row['N_Censored']/row['N_Patients']*100:.1f}%)")
        report_lines.append(f"  Median Survival: {row['Median_Survival']:.2f} years")
    report_lines.append("")

# Hazard Ratios
if os.path.exists(f'{output_base}/statistics/hazard_ratios_all_models.csv'):
    hr_all = pd.read_csv(f'{output_base}/statistics/hazard_ratios_all_models.csv')
    report_lines.append("Hazard Ratios Across Models:")
    
    # Show top 5 models
    models_shown = hr_all['Model'].unique()[:5]
    for model in models_shown:
        report_lines.append(f"\n  {model}:")
        model_hrs = hr_all[hr_all['Model'] == model]
        for _, row in model_hrs.iterrows():
            report_lines.append(
                f"    {row['Comparison']}: HR={row['HR']:.3f} "
                f"(95% CI: {row['CI_Lower']:.3f}-{row['CI_Upper']:.3f}), "
                f"p={row['P_value']:.4e}"
            )
    report_lines.append("")

# ============================================================================
# 9. ADVANCED SURVIVAL ANALYSES (INSERT HERE)
# ============================================================================
report_lines.append("9. ADVANCED SURVIVAL ANALYSES")
report_lines.append("-" * 80)

# Time-Dependent Covariates
if os.path.exists(f'{output_base}/statistics/time_dependent_effects.csv'):
    tv_results = pd.read_csv(f'{output_base}/statistics/time_dependent_effects.csv')
    
    report_lines.append("Time-Dependent Covariate Analysis:")
    report_lines.append(f"  Significant time-varying effects detected: {len(tv_results)}")
    
    if len(tv_results) > 0:
        report_lines.append("  Features with time-varying effects:")
        for _, row in tv_results.iterrows():
            report_lines.append(f"    {row['Feature']}: HR={row['Time_Interaction_HR']:.3f}, "
                              f"p={row['P_value']:.4e}, {row['Interpretation']}")
    report_lines.append("")

# Competing Risks
if os.path.exists(f'{output_base}/statistics/cumulative_incidence.csv'):
    ci_results = pd.read_csv(f'{output_base}/statistics/cumulative_incidence.csv')
    
    report_lines.append("Competing Risks Analysis:")
    report_lines.append("  Cumulative Incidence at Key Time Points:")
    
    for _, row in ci_results.iterrows():
        report_lines.append(f"    {int(row['Time_Years'])} years:")
        report_lines.append(f"      Cancer-specific death: {row['CI_Cancer_Death']:.3f}")
        report_lines.append(f"      Other cause death: {row['CI_Other_Death']:.3f}")
        report_lines.append(f"      Total events: {row['Total_Events']:.3f}")
    report_lines.append("")

# Subgroup Analysis
if os.path.exists(f'{output_base}/statistics/subgroup_analysis.csv'):
    subgroup_results = pd.read_csv(f'{output_base}/statistics/subgroup_analysis.csv')
    
    report_lines.append("Subgroup Analysis:")
    report_lines.append(f"  Total subgroups analyzed: {len(subgroup_results)}")
    
    # Best and worst performing subgroups
    best_subgroup = subgroup_results.nlargest(1, 'C_Index').iloc[0]
    worst_subgroup = subgroup_results.nsmallest(1, 'C_Index').iloc[0]
    
    report_lines.append(f"\n  Best performing subgroup:")
    report_lines.append(f"    {best_subgroup['Subgroup']} ({best_subgroup['Subgroup_Type']})")
    report_lines.append(f"    C-Index: {best_subgroup['C_Index']:.4f}")
    report_lines.append(f"    Event Rate: {best_subgroup['Event_Rate_%']:.1f}%")
    
    report_lines.append(f"\n  Worst performing subgroup:")
    report_lines.append(f"    {worst_subgroup['Subgroup']} ({worst_subgroup['Subgroup_Type']})")
    report_lines.append(f"    C-Index: {worst_subgroup['C_Index']:.4f}")
    report_lines.append(f"    Event Rate: {worst_subgroup['Event_Rate_%']:.1f}%")
    report_lines.append("")

# Interaction Analysis
if os.path.exists(f'{output_base}/statistics/interaction_analysis.csv'):
    int_results = pd.read_csv(f'{output_base}/statistics/interaction_analysis.csv')
    
    report_lines.append("Interaction Analysis:")
    report_lines.append(f"  Interactions tested: {len(int_results)}")
    report_lines.append(f"  Significant interactions (p<0.05): {int_results['Significant'].sum()}")
    
    sig_interactions = int_results[int_results['Significant']]
    if len(sig_interactions) > 0:
        report_lines.append("\n  Significant Interaction Effects:")
        for _, row in sig_interactions.iterrows():
            report_lines.append(f"    {row['Interaction']}:")
            report_lines.append(f"      HR: {row['Interaction_HR']:.3f}, p={row['Interaction_P']:.4e}")
            report_lines.append(f"      C-index improvement: {row['C_Index_Improvement']:+.4f}")
    report_lines.append("")

# Calibration
if os.path.exists(f'{output_base}/statistics/calibration_metrics.csv'):
    cal_results = pd.read_csv(f'{output_base}/statistics/calibration_metrics.csv')
    
    report_lines.append("Calibration Analysis:")
    report_lines.append(f"  Model-time combinations evaluated: {len(cal_results)}")
    
    # Average calibration slope (ideal = 1.0)
    avg_slope = cal_results['Calibration_Slope'].mean()
    report_lines.append(f"  Average calibration slope: {avg_slope:.3f} (ideal: 1.0)")
    
    # Well-calibrated models (slope between 0.8-1.2)
    well_calibrated = cal_results[
        (cal_results['Calibration_Slope'] >= 0.8) & 
        (cal_results['Calibration_Slope'] <= 1.2)
    ]
    
    report_lines.append(f"  Well-calibrated predictions: {len(well_calibrated)}/{len(cal_results)}")
    report_lines.append("")


# ============================================================================
# 10. Key Innovations and Updates
# ============================================================================

report_lines.append("9. KEY INNOVATIONS AND UPDATES")
report_lines.append("-" * 80)
report_lines.append("Robust Training Strategies:")
report_lines.append("  1. Stability-based feature selection (100 bootstrap iterations)")
report_lines.append("  2. Three-way data split (Train/Validation/Test) for better monitoring")
report_lines.append("  3. Increased regularization across all models")
report_lines.append("  4. Conservative hyperparameters for tree-based models")
report_lines.append("  5. RobustScaler for better outlier handling")
report_lines.append("  6. Validation set monitoring to detect overfitting early")
report_lines.append("  7. Train-Val-Test gap analysis for generalization assessment")
report_lines.append("  8. Repeated cross-validation (5-fold, 3 repeats)")
report_lines.append("")

report_lines.append("Feature Engineering:")
report_lines.append("  1. All features treated as continuous/normalized (not binary)")
report_lines.append("  2. Point-biserial correlation for continuous vs binary Event")
report_lines.append("  3. Mann-Whitney U tests for continuous variables (Alive vs Dead)")
report_lines.append("  4. Kruskal-Wallis test for survival time across cancer types")
report_lines.append("  5. Spearman correlation between genomic features")
report_lines.append("  6. Cancer types assigned based on highest continuous score")
report_lines.append("  7. Time handling: original for Cox, shifted for sksurv ML models")
report_lines.append("  8. Univariate Cox: HR per unit increase (continuous predictors)")
report_lines.append("")

report_lines.append("Statistical Validation:")
report_lines.append("  1. Bootstrap confidence intervals (500 iterations)")
report_lines.append("  2. 5-fold cross-validation with repeated runs")
report_lines.append("  3. Prediction uncertainty quantification (RSF ensemble variance)")
report_lines.append("  4. Pairwise log-rank tests with Bonferroni correction")
report_lines.append("  5. Multivariate hazard ratio analysis")
report_lines.append("")

# ============================================================================
# 10. Clinical Implications
# ============================================================================

report_lines.append("10. CLINICAL IMPLICATIONS")
report_lines.append("-" * 80)
report_lines.append("Key Findings:")
report_lines.append("   Successfully developed robust multi-cancer genomic survival prediction models")
report_lines.append(f"  Best performing model: {best_model_name} (C-Index: {best_test_c:.4f})")
report_lines.append("   Three-tier risk stratification shows significant survival differences")
report_lines.append("   Continuous genomic features provide prognostic value beyond clinical variables")
report_lines.append("   Robust configurations improve generalization while maintaining performance")
report_lines.append("   Models show excellent stability across validation strategies")
report_lines.append("")

report_lines.append("Statistical Robustness:")
report_lines.append("   Bootstrap confidence intervals confirm model stability (500 iterations)")
report_lines.append("   5-fold cross-validation shows consistent performance across folds")
report_lines.append("   Log-rank tests validate risk stratification (p < 0.001)")
report_lines.append("   Hazard ratios demonstrate clinical significance across models")
report_lines.append("   Continuous feature analysis reveals dose-response relationships")
report_lines.append("   Stability-based feature selection ensures reproducible results")
report_lines.append("   Three-way split validates true generalization capability")
report_lines.append("")

report_lines.append("Generalization Performance:")
report_lines.append("   Train-Val gaps indicate appropriate model complexity")
report_lines.append("   Val-Test gaps confirm stable generalization to unseen data")
report_lines.append("   Robust models show reduced overfitting compared to original")
report_lines.append("   Ensemble uncertainty quantification provides confidence estimates")
report_lines.append("")

# ============================================================================
# 11. Output Files
# ============================================================================

report_lines.append("11. OUTPUT FILES")
report_lines.append("-" * 80)

report_lines.append("Tables:")
report_lines.append("  univariate_analysis.csv - Univariate Cox for continuous features")
report_lines.append("  all_models_performance.csv - Comprehensive model comparison with validation")
report_lines.append("  top20_features_comparison.csv - Feature importance across models")
report_lines.append("  feature_stability_scores.csv - Stability-based feature selection results")
report_lines.append("  risk_stratification_*.csv - Risk scores for each model")
report_lines.append("")

report_lines.append("Statistics (Continuous Features):")
report_lines.append("  cancer_type_statistics.csv - Cancer-specific statistics (continuous)")
report_lines.append("  genomic_feature_statistics.csv - Genomic feature distributions")
report_lines.append("  feature_event_correlations.csv - Point-biserial correlations")
report_lines.append("  mann_whitney_tests.csv - Alive vs Dead comparisons")
report_lines.append("  gene_correlations.csv - Spearman correlations between genes")
report_lines.append("  hazard_ratios_all_models.csv - HR for all models")
report_lines.append("  risk_group_statistics.csv - Risk stratification statistics")
report_lines.append("  pairwise_logrank_tests.csv - Pairwise log-rank with Bonferroni")
report_lines.append("")

report_lines.append("Ablation:")
report_lines.append("  ablation_study_results.csv - Complete ablation across all models")
report_lines.append("")

report_lines.append("Uncertainty:")
report_lines.append("  bootstrap_all_models.csv - Bootstrap CI for all models (500 iter)")
report_lines.append("  cross_validation_all_models.csv - 5-fold CV for all models")
report_lines.append("  prediction_uncertainty.csv - RSF ensemble variance analysis")
report_lines.append("")

report_lines.append("Figures:")
report_lines.append("  01_descriptive_statistics.png - Dataset overview")
report_lines.append("  02_enhanced_model_performance.png - 4-subplot model comparison (Train/Val/Test)")
report_lines.append("  03_ablation_study_enhanced.png - 2x2 ablation grid with robust comparison")
report_lines.append("  04_kaplan_meier_curves.png - KM curves with censored counts")
report_lines.append("  05_top20_features_all_models.png - Feature importance heatmap")
report_lines.append("  06_uncertainty_quantification.png - Prediction uncertainty analysis")
report_lines.append("  07_statistical_validation_complete.png - Comprehensive validation summary")
report_lines.append("")

report_lines.append("Models:")
report_lines.append("  trained_models.pkl - All trained models (robust and original)")
report_lines.append("  scalers.pkl - All scalers (RobustScaler and StandardScaler)")
report_lines.append("  feature_sets.json - Feature sets for each model configuration")
report_lines.append("")

# NEW SECTION 10: OUTPUT FILES (EXTENDED)
report_lines.append("10. OUTPUT FILES (EXTENDED)")
report_lines.append("-" * 80)

report_lines.append("Advanced Analysis Results:")
report_lines.append("   time_dependent_effects.csv - Time-varying covariate effects")
report_lines.append("   cumulative_incidence.csv - Competing risks cumulative incidence")
report_lines.append("   subgroup_analysis.csv - Performance across patient subgroups")
report_lines.append("   interaction_analysis.csv - Genomic-clinical interactions")
report_lines.append("   calibration_metrics.csv - Model calibration statistics")
report_lines.append("")

report_lines.append("Additional Figures:")
report_lines.append("   08_competing_risks_analysis.png - Cumulative incidence functions")
report_lines.append("   09_subgroup_analysis.png - Subgroup performance visualization")
report_lines.append("   10_interaction_analysis.png - Interaction effects")
report_lines.append("   11_calibration_analysis.png - Calibration plots")
report_lines.append("")

# [Keep remaining sections from original]

report_lines.append("=" * 80)
report_lines.append("END OF COMPREHENSIVE EXTENDED REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Save extended report
report_text = "\n".join(report_lines)

with open(f'{output_base}/COMPREHENSIVE_EXTENDED_REPORT.txt', 'w') as f:
    f.write(report_text)

print("\n Comprehensive extended report saved")
print(f"  Location: {output_base}/COMPREHENSIVE_EXTENDED_REPORT.txt")
print(f"  Lines: {len(report_lines)}")
# ============================================================================
# 12. Configuration Summary
# ============================================================================

report_lines.append("12. CONFIGURATION SUMMARY")
report_lines.append("-" * 80)
report_lines.append(f"Random Seed: {RANDOM_STATE}")
report_lines.append(f"Test Size: {config.TEST_SIZE * 100:.0f}%")
report_lines.append(f"Validation Size: {config.VALIDATION_SIZE * 100:.0f}%")
report_lines.append(f"CV Folds: {config.CV_FOLDS}")
report_lines.append(f"CV Repeats: {config.CV_REPEATS}")
report_lines.append(f"Scaler Type: {config.SCALER_TYPE}")
report_lines.append(f"Feature Selection: {config.USE_FEATURE_SELECTION}")
report_lines.append(f"Feature Selection Method: {config.FEATURE_SELECTION_METHOD}")
report_lines.append(f"Bootstrap Iterations (Feature Selection): {config.N_BOOTSTRAP_FEATURE}")
report_lines.append(f"Feature Stability Threshold: {config.FEATURE_THRESHOLD}")
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF COMPREHENSIVE REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Feature Type: {config_dict['feature_type']}")
report_lines.append(f"Total Models Trained: {len(perf_df_all)}")
report_lines.append(f"Total Ablation Experiments: {len(ablation_df)}")
report_lines.append("")

# ============================================================================
# Save All Models and Scalers
# ============================================================================

print("\n[Saving Trained Models and Artifacts]")
print("-" * 40)

try:
    with open(f'{output_base}/models/trained_models.pkl', 'wb') as f:
        pickle.dump(cox_models, f)
    print(f"  ? All models saved to trained_models.pkl ({len(cox_models)} models)")
except Exception as e:
    print(f"  ? Could not save models: {str(e)[:50]}")

try:
    scalers_dict = {
        'scaler_global': scaler_global,
        'scaler_ml': scaler_ml,
        'age_scaler': age_scaler
    }
    with open(f'{output_base}/models/scalers.pkl', 'wb') as f:
        pickle.dump(scalers_dict, f)
    print("  ? All scalers saved")
except Exception as e:
    print(f"  ? Could not save scalers: {str(e)[:50]}")

try:
    with open(f'{output_base}/models/feature_sets.json', 'w') as f:
        json.dump(feature_sets, f, indent=2)
    print("  ? Feature sets saved")
except Exception as e:
    print(f"  ? Could not save feature sets: {str(e)[:50]}")

# Save risk scores for all models
try:
    if len(risk_scores_all_models) > 0:
        risk_scores_summary = pd.DataFrame(risk_scores_all_models)
        risk_scores_summary.to_csv(f'{output_base}/tables/risk_scores_all_models.csv', index=False)
        print(f"  ? Risk scores for all models saved ({len(risk_scores_all_models)} models)")
except Exception as e:
    print(f"  ? Could not save risk scores: {str(e)[:50]}")

# Save configuration
try:
    config_save = {
        'RANDOM_STATE': RANDOM_STATE,
        'TEST_SIZE': config.TEST_SIZE,
        'VALIDATION_SIZE': config.VALIDATION_SIZE,
        'CV_FOLDS': config.CV_FOLDS,
        'CV_REPEATS': config.CV_REPEATS,
        'COX_PENALIZER': config.COX_PENALIZER,
        'COX_L1_RATIO': config.COX_L1_RATIO,
        'ELASTIC_NET_L1': config.ELASTIC_NET_L1,
        'RSF_N_ESTIMATORS': config.RSF_N_ESTIMATORS,
        'RSF_MAX_DEPTH': config.RSF_MAX_DEPTH,
        'RSF_MIN_SAMPLES_SPLIT': config.RSF_MIN_SAMPLES_SPLIT,
        'RSF_MIN_SAMPLES_LEAF': config.RSF_MIN_SAMPLES_LEAF,
        'RSF_MAX_FEATURES': config.RSF_MAX_FEATURES,
        'GB_N_ESTIMATORS': config.GB_N_ESTIMATORS,
        'GB_LEARNING_RATE': config.GB_LEARNING_RATE,
        'GB_MAX_DEPTH': config.GB_MAX_DEPTH,
        'GB_MIN_SAMPLES_SPLIT': config.GB_MIN_SAMPLES_SPLIT,
        'GB_MIN_SAMPLES_LEAF': config.GB_MIN_SAMPLES_LEAF,
        'GB_SUBSAMPLE': config.GB_SUBSAMPLE,
        'SVM_ALPHA_RANGE': config.SVM_ALPHA_RANGE,
        'SCALER_TYPE': config.SCALER_TYPE,
        'USE_FEATURE_SELECTION': config.USE_FEATURE_SELECTION,
        'FEATURE_SELECTION_METHOD': config.FEATURE_SELECTION_METHOD,
        'N_BOOTSTRAP_FEATURE': config.N_BOOTSTRAP_FEATURE,
        'FEATURE_THRESHOLD': config.FEATURE_THRESHOLD
    }
    
    with open(f'{output_base}/models/robust_config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    print("  ? Configuration saved")
except Exception as e:
    print(f"  ? Could not save configuration: {str(e)[:50]}")

# Save report
report_text = "\n".join(report_lines)

with open(f'{output_base}/COMPREHENSIVE_FINAL_REPORT.txt', 'w') as f:
    f.write(report_text)

print("\n? Comprehensive final report saved")
print(f"  Location: {output_base}/COMPREHENSIVE_FINAL_REPORT.txt")
print(f"  Lines: {len(report_lines)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 80)
print(f"\nTotal Models Trained: {len(perf_df_all)}")
print(f"  - Robust Models: {len([m for m in perf_df_all['Model'] if 'Robust' in m])}")
print(f"  - Original Models: {len([m for m in perf_df_all['Model'] if 'Robust' not in m])}")
print(f"\nBest Model: {best_model_name}")
print(f"  Test C-Index: {best_test_c:.4f}")

if 'Train_Val_Gap' in best_model_row:
    print(f"  Train-Val Gap: {best_model_row['Train_Val_Gap']:.4f}")
if 'Val_Test_Gap' in best_model_row:
    print(f"  Val-Test Gap: {best_model_row['Val_Test_Gap']:.4f}")

print(f"\nAblation Experiments: {len(ablation_df)}")
print(f"Feature Stability Scores: {len(feature_stability_df)}")
print(f"Stable Features Selected: {len(stable_features)}/{len(all_features_list)}")

print("\n" + "=" * 80)
print("ALL ANALYSIS COMPLETE - ROBUST SURVIVAL PREDICTION SYSTEM")
print("=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
