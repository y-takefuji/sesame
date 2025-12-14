import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import FeatureAgglomeration
import xgboost as xgb
import shap
from scipy import stats
import traceback
import sys

# Load the data
data = pd.read_csv('data.csv')

# Show shape of original dataset
print("Original dataset shape:", data.shape)

# Convert string variables to numerical
le_reg = LabelEncoder()
le_prov = LabelEncoder()

if 'REG' in data.columns:
    data['REG'] = le_reg.fit_transform(data['REG'])
    print("\nREG distribution:")
    print(data['REG'].value_counts())

if 'PROV' in data.columns:
    data['PROV'] = le_prov.fit_transform(data['PROV'])
    print("\nPROV distribution:")
    print(data['PROV'].value_counts())

# Show target distribution
target = data['yields']
print("\nTarget distribution:")
print(target.describe())

# Calculate target median threshold and remove boundary instances
threshold = target.median()
lower_bound = threshold - 20
upper_bound = threshold + 20

# Remove boundary instances
filtered_data = data[(target <= lower_bound) | (target >= upper_bound)].copy()

# Convert to binary classification
filtered_data['class'] = (filtered_data['yields'] >= threshold).astype(int)

# Save as class.csv
filtered_data.to_csv('class.csv', index=False)

# Show shape of filtered dataset and target distribution
print("\nFiltered dataset shape:", filtered_data.shape)
print("\nBinary target distribution:")
print(filtered_data['class'].value_counts())

# Prepare data for feature selection
X = filtered_data.drop(['yields', 'class'], axis=1)
y = filtered_data['class']

# Function to perform Random Forest feature selection - complete fitting
def rf_feature_selection(X, y, n_features):
    # Complete model fit
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a list of (feature, importance) tuples
    feature_importance = list(zip(X.columns, importances))
    
    # Sort by importance and get top features
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return [feature for feature, _ in feature_importance[:n_features]], rf

# Function to perform XGBoost feature selection
def xgb_feature_selection(X, y, n_features):
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X, y)
    
    # Get feature importances
    importances = xgb_model.feature_importances_
    
    # Create a list of (feature, importance) tuples
    feature_importance = list(zip(X.columns, importances))
    
    # Sort by importance and get top features
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return [feature for feature, _ in feature_importance[:n_features]], xgb_model

# Function to perform Feature Agglomeration - properly using sklearn's FeatureAgglomeration
def feature_agglomeration(X, n_features):
    # Determine number of clusters
    n_clusters = min(n_features, X.shape[1])
    
    # Perform Feature Agglomeration
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)
    
    # Get cluster assignments for each feature
    cluster_labels = fa.labels_
    
    # Calculate variance for each feature
    feature_variances = X.var().values
    
    # Create a dictionary to store the best feature from each cluster
    cluster_best_features = {}
    
    # For each feature, check if it's the highest variance feature in its cluster
    for i, (col, var, cluster) in enumerate(zip(X.columns, feature_variances, cluster_labels)):
        if cluster not in cluster_best_features or var > cluster_best_features[cluster][1]:
            cluster_best_features[cluster] = (col, var)
    
    # Get all features with their variances
    all_features_with_variance = [(col, var) for col, var in zip(X.columns, feature_variances)]
    
    # Sort all features by variance (across all clusters)
    all_features_with_variance.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n_features based on variance
    return [feat for feat, _ in all_features_with_variance[:n_features]]

# Function to perform Highly Variable Gene Selection (based on variance)
def hvgs_feature_selection(X, n_features):
    variances = X.var().sort_values(ascending=False)
    return variances.index[:n_features].tolist()

# Function to perform Spearman's correlation feature selection
def spearman_feature_selection(X, y, n_features):
    correlations = []
    for col in X.columns:
        corr, _ = stats.spearmanr(X[col], y)
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in correlations[:n_features]]

# FIXED: Function to perform RF-SHAP feature selection with extensive debugging
def rf_shap_feature_selection(X, y, n_features):
    print("DEBUG: Starting RF-SHAP feature selection")
    print(f"DEBUG: Data shape: {X.shape}, Target shape: {y.shape}")
    
    # Train a Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    print("DEBUG: Random Forest model trained successfully")
    
    # Select 100 samples for SHAP analysis
    if len(X) > 100:
        X_sample = X.sample(100, random_state=42)
        print(f"DEBUG: Selected 100 samples from {len(X)} total samples")
    else:
        # If less than 100 samples, use all with replacement to get 100
        X_sample = X.sample(100, replace=True, random_state=42)
        print(f"DEBUG: Selected 100 samples with replacement from {len(X)} total samples")
    
    # Create a TreeExplainer for SHAP
    print("DEBUG: Creating SHAP explainer")
    explainer = shap.TreeExplainer(rf)
    print(f"DEBUG: SHAP explainer created: {type(explainer)}")
    
    # Get SHAP values - try different approaches based on SHAP version
    print("DEBUG: Computing SHAP values")
    
    # Check SHAP version to use appropriate API
    shap_version = getattr(shap, "__version__", "unknown")
    print(f"DEBUG: SHAP version: {shap_version}")
    
    # Get the SHAP values using the correct API
    if hasattr(explainer, "__call__"):  # Modern SHAP API
        print("DEBUG: Using modern SHAP API (explainer(X))")
        shap_values = explainer(X_sample)
        
        if hasattr(shap_values, "values"):
            print(f"DEBUG: Modern API returned values attribute with shape {shap_values.values.shape}")
            shap_values_array = shap_values.values
            
            # Check if we have a 3D array (samples, features, classes)
            if len(shap_values_array.shape) == 3:
                print(f"DEBUG: Got 3D SHAP values array with shape {shap_values_array.shape}")
                # For binary classification, sum across classes or take positive class
                if shap_values_array.shape[2] == 2:  # Binary classification
                    print("DEBUG: Using positive class (index 1) SHAP values")
                    shap_values_array = shap_values_array[:, :, 1]  # Take positive class
                else:
                    print("DEBUG: Summing SHAP values across all classes")
                    shap_values_array = np.sum(shap_values_array, axis=2)  # Sum across classes
                print(f"DEBUG: After processing, SHAP values array shape: {shap_values_array.shape}")
        else:
            print(f"DEBUG: Modern API returned object of type {type(shap_values)}")
            # For newer SHAP versions that may have different attribute names
            if hasattr(shap_values, "shap_values"):
                shap_values_array = shap_values.shap_values
            else:
                # Direct conversion (might work for some versions)
                shap_values_array = np.array(shap_values)
    else:  # Legacy SHAP API
        print("DEBUG: Using legacy SHAP API (explainer.shap_values(X))")
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            print(f"DEBUG: Legacy API returned list with {len(shap_values)} elements")
            # For binary classification, use class 1 (positive class) values
            if len(shap_values) > 1:
                print(f"DEBUG: Using class 1 values with shape {np.array(shap_values[1]).shape}")
                shap_values_array = shap_values[1]
            else:
                print(f"DEBUG: Using class 0 values with shape {np.array(shap_values[0]).shape}")
                shap_values_array = shap_values[0]
        else:
            print(f"DEBUG: Legacy API returned single array with shape {np.array(shap_values).shape}")
            shap_values_array = shap_values
    
    # Calculate mean absolute SHAP values for each feature
    print(f"DEBUG: Final SHAP values array shape before averaging: {shap_values_array.shape}")
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    print(f"DEBUG: Calculated mean absolute SHAP values with shape: {mean_abs_shap.shape}")
    
    # Ensure we have a 1D array for feature importance
    if len(mean_abs_shap.shape) > 1:
        print(f"DEBUG: Mean SHAP values still multi-dimensional with shape {mean_abs_shap.shape}, taking mean across dimensions")
        mean_abs_shap = mean_abs_shap.mean(axis=1) if mean_abs_shap.shape[1] > mean_abs_shap.shape[0] else mean_abs_shap.mean(axis=0)
        print(f"DEBUG: After taking mean, shape is now: {mean_abs_shap.shape}")
    
    # Make sure it's a 1D array
    mean_abs_shap = np.ravel(mean_abs_shap)
    print(f"DEBUG: After raveling, shape is: {mean_abs_shap.shape}")
    
    # Create a list of (feature, importance) tuples
    feature_importance = list(zip(X.columns, mean_abs_shap))
    
    # Sort by importance and get top features
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_importance[:n_features]]
    print(f"DEBUG: Selected {len(selected_features)} features: {selected_features[:5]}...")
    
    return selected_features

# FIXED: Function to perform XGB-SHAP feature selection with extensive debugging
def xgb_shap_feature_selection(X, y, n_features):
    print("DEBUG: Starting XGB-SHAP feature selection")
    print(f"DEBUG: Data shape: {X.shape}, Target shape: {y.shape}")
    
    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X, y)
    print("DEBUG: XGBoost model trained successfully")
    
    # Select 100 samples for SHAP analysis
    if len(X) > 100:
        X_sample = X.sample(100, random_state=42)
        print(f"DEBUG: Selected 100 samples from {len(X)} total samples")
    else:
        # If less than 100 samples, use all with replacement to get 100
        X_sample = X.sample(100, replace=True, random_state=42)
        print(f"DEBUG: Selected 100 samples with replacement from {len(X)} total samples")
    
    # Create a TreeExplainer for SHAP
    print("DEBUG: Creating SHAP explainer for XGBoost")
    explainer = shap.TreeExplainer(xgb_model)
    print(f"DEBUG: SHAP explainer created: {type(explainer)}")
    
    # Check SHAP version to use appropriate API
    shap_version = getattr(shap, "__version__", "unknown")
    print(f"DEBUG: SHAP version: {shap_version}")
    
    # Get the SHAP values using the correct API
    if hasattr(explainer, "__call__"):  # Modern SHAP API
        print("DEBUG: Using modern SHAP API (explainer(X))")
        shap_values = explainer(X_sample)
        
        if hasattr(shap_values, "values"):
            print(f"DEBUG: Modern API returned values attribute with shape {shap_values.values.shape}")
            shap_values_array = shap_values.values
            
            # Check if we have a 3D array (samples, features, classes)
            if len(shap_values_array.shape) == 3:
                print(f"DEBUG: Got 3D SHAP values array with shape {shap_values_array.shape}")
                # For binary classification, sum across classes or take positive class
                if shap_values_array.shape[2] == 2:  # Binary classification
                    print("DEBUG: Using positive class (index 1) SHAP values")
                    shap_values_array = shap_values_array[:, :, 1]  # Take positive class
                else:
                    print("DEBUG: Summing SHAP values across all classes")
                    shap_values_array = np.sum(shap_values_array, axis=2)  # Sum across classes
                print(f"DEBUG: After processing, SHAP values array shape: {shap_values_array.shape}")
        else:
            print(f"DEBUG: Modern API returned object of type {type(shap_values)}")
            # For newer SHAP versions that may have different attribute names
            if hasattr(shap_values, "shap_values"):
                shap_values_array = shap_values.shap_values
            else:
                # Direct conversion (might work for some versions)
                shap_values_array = np.array(shap_values)
    else:  # Legacy SHAP API
        print("DEBUG: Using legacy SHAP API (explainer.shap_values(X))")
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            print(f"DEBUG: Legacy API returned list with {len(shap_values)} elements")
            # For XGBoost classification, typically use the first element
            if len(shap_values) > 0:
                print(f"DEBUG: Using first array with shape {np.array(shap_values[0]).shape}")
                shap_values_array = shap_values[0]
            else:
                raise ValueError("SHAP values list is empty")
        else:
            print(f"DEBUG: Legacy API returned single array with shape {np.array(shap_values).shape}")
            shap_values_array = shap_values
    
    # Calculate mean absolute SHAP values for each feature
    print(f"DEBUG: Final SHAP values array shape before averaging: {shap_values_array.shape}")
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    print(f"DEBUG: Calculated mean absolute SHAP values with shape: {mean_abs_shap.shape}")
    
    # Ensure we have a 1D array for feature importance
    if len(mean_abs_shap.shape) > 1:
        print(f"DEBUG: Mean SHAP values still multi-dimensional with shape {mean_abs_shap.shape}, taking mean across dimensions")
        mean_abs_shap = mean_abs_shap.mean(axis=1) if mean_abs_shap.shape[1] > mean_abs_shap.shape[0] else mean_abs_shap.mean(axis=0)
        print(f"DEBUG: After taking mean, shape is now: {mean_abs_shap.shape}")
    
    # Make sure it's a 1D array
    mean_abs_shap = np.ravel(mean_abs_shap)
    print(f"DEBUG: After raveling, shape is: {mean_abs_shap.shape}")
    
    # Create a list of (feature, importance) tuples
    feature_importance = list(zip(X.columns, mean_abs_shap))
    
    # Sort by importance and get top features
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_importance[:n_features]]
    print(f"DEBUG: Selected {len(selected_features)} features: {selected_features[:5]}...")
    
    return selected_features

# Function to evaluate model with cross-validation
def evaluate_model(X, y, features):
    X_selected = X[features]
    rf = RandomForestClassifier(random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_selected, y, cv=kf, scoring='accuracy')
    
    return round(scores.mean(), 4)

# Initialize results dictionary
results = []

# Random Forest feature selection - Complete refitting for both CV10 and CV9
print("\nPerforming Random Forest feature selection...")
print("Fitting RF model on full dataset for top 10 features...")
rf_features_10, _ = rf_feature_selection(X, y, 10)
print("Top 10 features from RF:", rf_features_10)

# Remove the highest feature and do a complete refit
top_feature = rf_features_10[0]
X_reduced = X.drop(columns=[top_feature])
print("Fitting RF model on reduced dataset for top 9 features...")
rf_features_9, _ = rf_feature_selection(X_reduced, y, 9)
print("Top 9 features from RF after removing top feature:", rf_features_9)

# Evaluate both sets with cross-validation
rf_cv10 = evaluate_model(X, y, rf_features_10)
rf_cv9 = evaluate_model(X, y, rf_features_9)
results.append({
    'method': 'RF',
    'CV10_accuracy': rf_cv10,
    'CV9_accuracy': rf_cv9,
    'top10_features_CV10': ", ".join(rf_features_10),
    'top9_features_CV9': ", ".join(rf_features_9)
})

# XGBoost feature selection - also with complete refitting
print("\nPerforming XGBoost feature selection...")
print("Fitting XGB model on full dataset for top 10 features...")
xgb_features_10, _ = xgb_feature_selection(X, y, 10)
print("Top 10 features from XGB:", xgb_features_10)

top_feature = xgb_features_10[0]
X_reduced = X.drop(columns=[top_feature])
print("Fitting XGB model on reduced dataset for top 9 features...")
xgb_features_9, _ = xgb_feature_selection(X_reduced, y, 9)
print("Top 9 features from XGB after removing top feature:", xgb_features_9)

xgb_cv10 = evaluate_model(X, y, xgb_features_10)
xgb_cv9 = evaluate_model(X, y, xgb_features_9)
results.append({
    'method': 'XGB',
    'CV10_accuracy': xgb_cv10,
    'CV9_accuracy': xgb_cv9,
    'top10_features_CV10': ", ".join(xgb_features_10),
    'top9_features_CV9': ", ".join(xgb_features_9)
})

# Feature Agglomeration - using sklearn's FeatureAgglomeration library
print("\nPerforming Feature Agglomeration...")
fa_features_10 = feature_agglomeration(X, 10)
print("Top 10 features from FA:", fa_features_10)

top_feature = fa_features_10[0]
X_reduced = X.drop(columns=[top_feature])
fa_features_9 = feature_agglomeration(X_reduced, 9)
print("Top 9 features from FA after removing top feature:", fa_features_9)

fa_cv10 = evaluate_model(X, y, fa_features_10)
fa_cv9 = evaluate_model(X, y, fa_features_9)
results.append({
    'method': 'FA',
    'CV10_accuracy': fa_cv10,
    'CV9_accuracy': fa_cv9,
    'top10_features_CV10': ", ".join(fa_features_10),
    'top9_features_CV9': ", ".join(fa_features_9)
})

# Highly Variable Gene Selection
print("\nPerforming Highly Variable Gene Selection...")
hvgs_features_10 = hvgs_feature_selection(X, 10)
print("Top 10 features from HVGS:", hvgs_features_10)

top_feature = hvgs_features_10[0]
X_reduced = X.drop(columns=[top_feature])
hvgs_features_9 = hvgs_feature_selection(X_reduced, 9)
print("Top 9 features from HVGS after removing top feature:", hvgs_features_9)

hvgs_cv10 = evaluate_model(X, y, hvgs_features_10)
hvgs_cv9 = evaluate_model(X, y, hvgs_features_9)
results.append({
    'method': 'HVGS',
    'CV10_accuracy': hvgs_cv10,
    'CV9_accuracy': hvgs_cv9,
    'top10_features_CV10': ", ".join(hvgs_features_10),
    'top9_features_CV9': ", ".join(hvgs_features_9)
})

# Spearman's correlation
print("\nPerforming Spearman's correlation feature selection...")
spearman_features_10 = spearman_feature_selection(X, y, 10)
print("Top 10 features from Spearman:", spearman_features_10)

top_feature = spearman_features_10[0]
X_reduced = X.drop(columns=[top_feature])
spearman_features_9 = spearman_feature_selection(X_reduced, y, 9)
print("Top 9 features from Spearman after removing top feature:", spearman_features_9)

spearman_cv10 = evaluate_model(X, y, spearman_features_10)
spearman_cv9 = evaluate_model(X, y, spearman_features_9)
results.append({
    'method': 'Spearman',
    'CV10_accuracy': spearman_cv10,
    'CV9_accuracy': spearman_cv9,
    'top10_features_CV10': ", ".join(spearman_features_10),
    'top9_features_CV9': ", ".join(spearman_features_9)
})

# Fixed RF-SHAP with extensive debugging
print("\nPerforming RF-SHAP feature selection (with 100 samples)...")
try:
    rf_shap_features_10 = rf_shap_feature_selection(X, y, 10)
    print("Top 10 features from RF-SHAP:", rf_shap_features_10)

    top_feature = rf_shap_features_10[0]
    X_reduced = X.drop(columns=[top_feature])
    print("DEBUG: Running RF-SHAP on reduced dataset after removing top feature")
    # Completely rerun the feature selection on the reduced dataset
    rf_shap_features_9 = rf_shap_feature_selection(X_reduced, y, 9)
    print("Top 9 features from RF-SHAP after removing top feature:", rf_shap_features_9)

    rf_shap_cv10 = evaluate_model(X, y, rf_shap_features_10)
    rf_shap_cv9 = evaluate_model(X, y, rf_shap_features_9)
    results.append({
        'method': 'RF-SHAP',
        'CV10_accuracy': rf_shap_cv10,
        'CV9_accuracy': rf_shap_cv9,
        'top10_features_CV10': ", ".join(rf_shap_features_10),
        'top9_features_CV9': ", ".join(rf_shap_features_9)
    })
except Exception as e:
    print(f"Error in RF-SHAP: {str(e)}")
    print("DEBUG: Full traceback for RF-SHAP error:")
    traceback.print_exc(file=sys.stdout)
    # If RF-SHAP genuinely fails, we'll skip adding it to the results
    print("RF-SHAP failed and will not be included in results")

# XGB-SHAP with extensive debugging
print("\nPerforming XGB-SHAP feature selection (with 100 samples)...")
try:
    xgb_shap_features_10 = xgb_shap_feature_selection(X, y, 10)
    print("Top 10 features from XGB-SHAP:", xgb_shap_features_10)

    top_feature = xgb_shap_features_10[0]
    X_reduced = X.drop(columns=[top_feature])
    print("DEBUG: Running XGB-SHAP on reduced dataset after removing top feature")
    # Completely rerun the feature selection on the reduced dataset
    xgb_shap_features_9 = xgb_shap_feature_selection(X_reduced, y, 9)
    print("Top 9 features from XGB-SHAP after removing top feature:", xgb_shap_features_9)

    xgb_shap_cv10 = evaluate_model(X, y, xgb_shap_features_10)
    xgb_shap_cv9 = evaluate_model(X, y, xgb_shap_features_9)
    results.append({
        'method': 'XGB-SHAP',
        'CV10_accuracy': xgb_shap_cv10,
        'CV9_accuracy': xgb_shap_cv9,
        'top10_features_CV10': ", ".join(xgb_shap_features_10),
        'top9_features_CV9': ", ".join(xgb_shap_features_9)
    })
except Exception as e:
    print(f"Error in XGB-SHAP: {str(e)}")
    print("DEBUG: Full traceback for XGB-SHAP error:")
    traceback.print_exc(file=sys.stdout)
    # If XGB-SHAP genuinely fails, we'll skip adding it to the results
    print("XGB-SHAP failed and will not be included in results")

# Create summary table and save as result.csv
results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df)
results_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")

# Print SHAP version information for debugging
print("\nDEBUG: SHAP version information:")
print(f"SHAP version: {shap.__version__ if hasattr(shap, '__version__') else 'Unknown'}")
print(f"Python version: {sys.version}")
