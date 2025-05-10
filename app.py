import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import warnings

# Suppress specific warnings that might be noisy (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')


# 🌈 Streamlit page setup
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f4f6fa; }
        .stButton>button { color: white; background-color: #0099ff; border: none; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; transition: background-color 0.3s ease;}
        .stButton>button:hover { background-color: #007acc; }
        .stSelectbox>div>div { background-color: #ffffff; color: #333; border: 1px solid #ccc; border-radius: 4px; padding: 8px 12px; }
        .stTextInput>div>div>input { border: 1px solid #ccc; border-radius: 4px; padding: 8px 12px; }
        .stTextArea>div>div>textarea { border: 1px solid #ccc; border-radius: 4px; padding: 8px 12px; }
        .css-1d3z3zj { padding-top: 35px; } /* Adjust padding at the top */
    </style>
""", unsafe_allow_html=True)

# 🧾 Title
st.title("📊 Customer Churn Prediction App")
st.markdown("Upload your **training** and **testing** datasets (CSV format) to predict customer churn using multiple ML models.")

# 📂 File upload
# Use st.file_uploader without hardcoded paths. It returns a file-like object.
st.subheader("📂 Upload Datasets")
train_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
test_file = st.file_uploader("Upload Testing Data (CSV)", type="csv")


if not train_file or not test_file:
    st.warning("⚠️ Please upload both training and testing datasets to proceed.")
    st.stop()

# 🧮 Load Data
# Use a try-except block to catch potential errors during file reading
st.subheader("🧩 Data Loading")
try:
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    st.success(f"✅ Successfully loaded Training Data: {df_train.shape[0]} rows, {df_train.shape[1]} columns | Testing Data: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
except Exception as e:
    st.error(f"❌ Error loading data. Please ensure the uploaded files are valid CSVs and not corrupted. Error details: {e}")
    st.stop()

# 🔍 Show sample
with st.expander("🔍 Preview Uploaded Datasets"):
    st.subheader("Training Data Head")
    st.dataframe(df_train.head())
    st.subheader("Testing Data Head")
    st.dataframe(df_test.head())

# 🎯 Detect target column
st.subheader("🎯 Target Variable Detection")
# Look for common churn-related column names, case-insensitive
churn_cols = [col for col in df_train.columns if 'churn' in col.lower()]
if not churn_cols:
    churn_cols = [col for col in df_train.columns if 'attrit' in col.lower() or 'exit' in col.lower()]

churn_column = None
if churn_cols:
    churn_column = churn_cols[0] # Use the first detected column
    st.info(f"🎯 Automatically detected target variable: `{churn_column}`")
else:
    st.warning("❌ Could not automatically detect a churn-related column (looked for 'churn', 'attrit', 'exit').")
    # Allow the user to select the column manually
    all_cols = df_train.columns.tolist()
    churn_column = st.selectbox("Please select the target (churn) column:", all_cols, index=None) # Use index=None for no default selection

if not churn_column:
     st.warning("Please select the target column to proceed.")
     st.stop()

# Ensure the target column exists in both dataframes before proceeding
if churn_column not in df_train.columns or churn_column not in df_test.columns:
     st.error(f"❌ The selected target column '{churn_column}' was not found in both the training and testing datasets.")
     st.stop()

st.success(f"Selected target variable: `{churn_column}`")


# 🛠 Combine for processing
# Add a source column to distinguish between train and test data after combining
df_train['__data_source__'] = 'train'
df_test['__data_source__'] = 'test'
full_data = pd.concat([df_train, df_test], ignore_index=True) # Use ignore_index=True for clean index after concat

# --- Data Preprocessing ---

st.subheader("🔧 Data Preprocessing")

# Identify column types (re-evaluate after potential concat issues if any)
num_cols = full_data.select_dtypes(include=np.number).columns.tolist()
# Exclude the target and source column from categorical columns
cat_cols = full_data.select_dtypes(include='object').columns.drop([churn_column, '__data_source__'], errors='ignore').tolist()

st.write(f"Identified Numeric columns: {num_cols}")
st.write(f"Identified Categorical columns: {cat_cols}")

# 🔧 Missing value handling
st.write("Handling missing values...")
missing_num = full_data[num_cols].isnull().sum()
missing_cat = full_data[cat_cols].isnull().sum()

if missing_num.sum() > 0 or missing_cat.sum() > 0:
    st.write("Missing values detected:")
    if missing_num.sum() > 0:
        st.write("Numeric columns with missing values:")
        st.dataframe(missing_num[missing_num > 0])
    if missing_cat.sum() > 0:
        st.write("Categorical columns with missing values:")
        st.dataframe(missing_cat[missing_cat > 0])

    for col in num_cols:
        if full_data[col].isnull().any():
            median_val = full_data[col].median()
            full_data[col].fillna(median_val, inplace=True)
            #st.write(f"Filled missing values in numeric column `{col}` with median ({median_val:.2f})") # Optional detailed logging

    for col in cat_cols:
        if full_data[col].isnull().any():
            mode_val = full_data[col].mode()
            if not mode_val.empty:
                full_data[col].fillna(mode_val[0], inplace=True)
                #st.write(f"Filled missing values in categorical column `{col}` with mode ({mode_val[0]})") # Optional detailed logging
            #else:
                #st.warning(f"Could not find a mode for categorical column `{col}`. Leaving missing values as is.") # Optional warning

    st.success("Missing value handling complete.")
else:
    st.info("No missing values found in the data.")

# 🧠 Encode target variable if it's categorical
st.subheader("🎯 Target Variable Encoding")
if full_data[churn_column].dtype == 'object' or not pd.api.types.is_numeric_dtype(full_data[churn_column]):
    st.write(f"Encoding target column `{churn_column}`...")
    try:
        # Attempt common mappings first (handling potential whitespace issues)
        full_data[churn_column] = full_data[churn_column].astype(str).str.strip()
        mapping = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}
        # Apply mapping, set unmapped values to NaN initially
        full_data['_encoded_target_'] = full_data[churn_column].map(mapping)

        # If there are still NaN values after mapping, try factorizing the original column
        if full_data['_encoded_target_'].isnull().any():
             st.warning(f"Standard mapping failed for some values in target column `{churn_column}`. Attempting factorize...")
             unique_vals = full_data[churn_column].dropna().unique()
             # Create a mapping from unique string values to integers
             mapping_factorize = {val: i for i, val in enumerate(unique_vals)}
             full_data['_encoded_target_'] = full_data[churn_column].map(mapping_factorize)
             st.info(f"Used factorize encoding for target column `{churn_column}`. Mapping: {mapping_factorize}")


        if full_data['_encoded_target_'].isnull().any() or not pd.api.types.is_numeric_dtype(full_data['_encoded_target_']):
             st.error(f"❌ Failed to encode target column `{churn_column}` into numeric format. Values still non-numeric or NaN.")
             st.write("Unique values in target column after attempted encoding:", full_data['_encoded_target_'].unique())
             st.stop()

        full_data[churn_column] = full_data['_encoded_target_'] # Replace original column with encoded one
        full_data = full_data.drop(columns='_encoded_target_') # Drop temporary column

        st.success(f"Target column `{churn_column}` successfully encoded to numeric.")
        st.write("Encoded target values:", full_data[churn_column].unique())

    except Exception as e:
        st.error(f"❌ An unexpected error occurred while encoding target column `{churn_column}`: {e}")
        st.stop()
else:
    st.info(f"Target column `{churn_column}` is already in a numeric format.")


# 🔁 One-hot encoding for categorical features
st.subheader("🔄 Categorical Feature Encoding")
if cat_cols:
    st.write("Performing one-hot encoding for categorical features...")
    try:
        full_data = pd.get_dummies(full_data, columns=cat_cols, drop_first=True)
        st.success("One-hot encoding complete.")
    except Exception as e:
        st.error(f"❌ Error during one-hot encoding: {e}")
        st.stop()
else:
    st.info("No categorical columns found for one-hot encoding (excluding target and source).")


# 🔙 Split data back into training and testing sets
st.subheader("🔪 Splitting Data")
df_train_clean = full_data[full_data['__data_source__'] == 'train'].drop(columns='__data_source__')
df_test_clean = full_data[full_data['__data_source__'] == 'test'].drop(columns='__data_source__')

# Separate features (X) and target (y)
# Ensure the target column exists in the cleaned dataframes
if churn_column not in df_train_clean.columns or churn_column not in df_test_clean.columns:
     st.error(f"❌ Target column '{churn_column}' not found after preprocessing. This is unexpected.")
     st.stop()

X_train = df_train_clean.drop(columns=[churn_column])
y_train = df_train_clean[churn_column]
X_test = df_test_clean.drop(columns=[churn_column])
y_test = df_test_clean[churn_column]

st.write(f"Cleaned Training Data shape: {X_train.shape}")
st.write(f"Cleaned Testing Data shape: {X_test.shape}")

# Check if shapes match after preprocessing
if X_train.shape[1] != X_test.shape[1]:
    st.error("❌ Feature columns do not match between training and testing sets after preprocessing. This might be due to inconsistent unique values in categorical columns.")
    st.write("Training columns:", X_train.columns.tolist()[:50], "...") # Show first 50 columns
    st.write("Testing columns:", X_test.columns.tolist()[:50], "...") # Show first 50 columns
    # Attempt to align columns - this can sometimes fix shape mismatch from get_dummies
    st.info("Attempting to realign columns between training and testing sets...")
    try:
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        train_only_cols = list(set(X_train.columns) - set(X_test.columns))
        test_only_cols = list(set(X_test.columns) - set(X_train.columns))

        X_train_aligned = X_train[common_cols]
        X_test_aligned = X_test[common_cols]

        # Add missing columns filled with 0
        for col in test_only_cols:
             X_train_aligned[col] = 0
        for col in train_only_cols:
             X_test_aligned[col] = 0

        # Ensure columns are in the same order
        X_test_aligned = X_test_aligned[X_train_aligned.columns]

        X_train = X_train_aligned
        X_test = X_test_aligned
        st.success("Columns realigned successfully.")
        st.write(f"Realigned Training Data shape: {X_train.shape}")
        st.write(f"Realigned Testing Data shape: {X_test.shape}")

        if X_train.shape[1] != X_test.shape[1]:
             st.error("❌ Column realignment failed. Shapes still do not match.")
             st.stop()

    except Exception as e:
        st.error(f"❌ Error during column realignment: {e}")
        st.stop()


# 🔃 Scale numerical features
st.subheader("🔃 Feature Scaling")
try:
    # Identify numerical columns *after* one-hot encoding
    # These will be the original numeric columns + new dummy columns
    numeric_cols_post_encoding = X_train.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols_post_encoding:
        st.warning("No numeric columns found after one-hot encoding. Skipping scaling.")
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[numeric_cols_post_encoding])
        X_test_scaled = scaler.transform(X_test[numeric_cols_post_encoding])

        # Create new dataframes with scaled features, keeping non-numeric columns if any (unlikely after encoding)
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        X_train_processed[numeric_cols_post_encoding] = X_train_scaled
        X_test_processed[numeric_cols_post_encoding] = X_test_scaled

        st.success("Feature scaling complete.")

except Exception as e:
    st.error(f"❌ Error during feature scaling: {e}")
    st.stop()


# 🚀 Run all models
st.subheader("⚙️ Train and Compare Models")
if st.button("🔍 Run Model Comparison"):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, objective='binary:logistic') # Added objective
    }

    results = {}

    # Use processed/scaled data for training and prediction
    X_train_final = X_train_processed
    X_test_final = X_test_processed

    # Check if target variable has more than one unique value in test set for classification metrics
    is_classification = len(y_test.unique()) > 1

    if not is_classification:
        st.warning("The test dataset contains only one unique value for the target variable. Classification metrics (AUC, Classification Report, Confusion Matrix) will not be calculated.")


    for name, model in models.items():
        st.write(f"Training {name}...")
        try:
            # Check if training data has more than one unique value for the target
            if len(y_train.unique()) <= 1:
                st.warning(f"Skipping {name}: Training data contains only one unique value for the target variable.")
                results[name] = None
                continue # Skip to the next model

            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)

            acc = accuracy_score(y_test, y_pred)
            auc_score = None
            y_prob = None

            # Calculate ROC curve and AUC only if it's a classification problem and model has predict_proba
            if is_classification and hasattr(model, 'predict_proba'):
                 y_prob = model.predict_proba(X_test_final)[:, 1]
                 fpr, tpr, _ = roc_curve(y_test, y_prob)
                 auc_score = auc(fpr, tpr)


            results[name] = {
                "model": model,
                "accuracy": acc,
                "auc": auc_score,
                "y_pred": y_pred,
                "y_prob": y_prob # Store probabilities even if AUC isn't calculated
            }
            st.success(f"Finished training {name}.")
        except Exception as e:
            st.error(f"❌ Error training {name}: {e}")
            results[name] = None # Mark this model as failed


    # Filter out failed and skipped models
    successful_results = {k: v for k, v in results.items() if v is not None}

    if not successful_results:
         st.error("All models failed to train or were skipped.")
         st.stop()


    # 📊 Table
    st.subheader("📋 Model Performance")
    perf_data = []
    for name, res in successful_results.items():
        perf_data.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "AUC": f"{res['auc']:.4f}" if res["auc"] is not None else 'N/A' # Format AUC if available
        })
    perf_df = pd.DataFrame(perf_data).sort_values("Accuracy", ascending=False)
    st.dataframe(perf_df.style.background_gradient(cmap="YlGnBu", subset=['Accuracy'] if 'Accuracy' in perf_df.columns else None))


    # 📈 ROC Curves
    st.subheader("📉 ROC Curve Comparison")
    # Plot ROC curves only if AUC was calculated for at least one model
    if is_classification and any(res['auc'] is not None for res in successful_results.values()):
        fig_roc, ax_roc = plt.subplots(figsize=(10, 6)) # Create figure and axes
        for name, res in successful_results.items():
             if res['auc'] is not None: # Only plot if AUC was calculated
                fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})") # Plot on axes

        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve Comparison")
        ax_roc.legend()
        st.pyplot(fig_roc) # Pass the figure object to st.pyplot
        plt.close(fig_roc) # Close the figure to free memory
    elif is_classification:
        st.info("ROC curves cannot be plotted because AUC could not be calculated for any model.")
    else:
        st.info("ROC curves not applicable because the test data contains only one class.")


    # 📥 Download
    # Find the best model based on accuracy (or handle if perf_df is empty)
    if not perf_df.empty:
        # Find the best model name (first row after sorting by accuracy)
        best_model_name = perf_df.iloc[0]["Model"]
        best_model_results = successful_results.get(best_model_name)

        if best_model_results:
            best_preds = best_model_results["y_pred"]

            # Use original df_test to merge predictions, ensuring it has the same index or a reliable key
            # A simple merge might be safer than just copying if indices were reset during concat/split
            # For simplicity here, we assume original df_test index aligns with X_test, y_test
            output_df = df_test.copy()
            # Ensure output_df has the same number of rows as y_test/best_preds
            if len(output_df) == len(best_preds):
                 output_df['Predicted_Churn'] = best_preds
                 st.subheader(f"📥 Download Predictions (Best Model: `{best_model_name}`)")
                 csv = output_df.to_csv(index=False).encode()
                 st.download_button(
                     label="📥 Download CSV with Predictions",
                     data=csv,
                     file_name=f"predicted_churn_results_{best_model_name.replace(' ', '_').lower()}.csv",
                     mime="text/csv"
                 )

                 # Display the output dataframe
                 st.write("Output Data with Predictions (from Test Set):")
                 st.dataframe(output_df.head()) # Show head as dataframe can be large

            else:
                 st.error("Could not merge predictions with original test data due to shape mismatch.")
                 st.write(f"Original Test Data rows: {len(output_df)}")
                 st.write(f"Predicted Churn rows: {len(best_preds)}")


            # Show classification report of the best model.
            st.subheader("📊 Classification Report (Best Model)")
            # Check if classification report is meaningful (needs at least two classes in y_test)
            if is_classification and len(np.unique(best_preds)) > 1:
                 st.text(classification_report(y_test, best_preds))
            else:
                 st.info("Classification report not applicable (test data or predictions contain only one class).")


            # Show confusion matrix of the best model.
            st.subheader("🔢 Confusion Matrix (Best Model)")
            # Check if confusion matrix is meaningful
            if is_classification and len(np.unique(best_preds)) > 1:
                conf_matrix = confusion_matrix(y_test, best_preds)
                fig_cm, ax_cm = plt.subplots() # Create figure and axes
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm) # Plot on axes
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm) # Pass the figure object
                plt.close(fig_cm) # Close the figure
            else:
                st.info("Confusion matrix not applicable (test data or predictions contain only one class).")

            # Feature Importance (for tree-based models)
            st.subheader("🌳 Feature Importance (Best Model)")
            if best_model_name in ["Random Forest", "XGBoost"]:
                try:
                    # Use the trained model object directly from results
                    model_obj = best_model_results["model"]
                    # Ensure the model has feature_importances_ attribute
                    if hasattr(model_obj, 'feature_importances_'):
                         importances = model_obj.feature_importances_
                         # Use the columns from the final processed data used for training
                         feature_names = X_train_final.columns
                         feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                         feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20) # Show top 20

                         if not feature_importance_df.empty:
                             fig_fi, ax_fi = plt.subplots(figsize=(10, 8)) # Adjust figure size for readability
                             sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
                             plt.title('Top 20 Feature Importance')
                             st.pyplot(fig_fi) # Pass the figure object
                             plt.close(fig_fi) # Close the figure
                         else:
                            st.info("Feature importance data is empty.")
                    else:
                         st.info(f"{best_model_name} does not have a 'feature_importances_' attribute.")
                except Exception as e:
                    st.warning(f"Could not display feature importance for {best_model_name}: {e}")
            else:
                st.info("Feature importance is typically displayed for tree-based models like Random Forest or XGBoost.")

        else:
            st.error(f"Could not retrieve results for the best model: {best_model_name}")

    else:
        st.error("No models were successfully trained to determine the best model.")

st.markdown("---")
st.markdown("App created using Streamlit, Pandas, Scikit-learn, Seaborn, Matplotlib, and XGBoost.")