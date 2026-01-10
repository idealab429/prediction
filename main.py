import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

# Try importing SMOTE, handle if not installed
try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False
    st.warning("imbalanced-learn library not found. SMOTE will be skipped. Please install via 'pip install imbalanced-learn'")

# Set Page Config
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Title
st.title("📊 Churn Management Data Analysis App")

# 1. Load Data
@st.cache_data
def load_data():
    # Assuming the file is in the same directory
    df = pd.read_csv("Churn_management.csv")
    return df

try:
    df = load_data()
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("File 'Churn_management.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# Initialize Session State for processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None

# ==========================================
# 1. Data Analysis & Visualization
# ==========================================
st.header("1. Data Analysis & Visualization")

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Target Variable Distribution
st.subheader("Distribution of Target Variable (Exited)")
fig_target, ax_target = plt.subplots(figsize=(6, 4))
sns.countplot(x='Exited', data=df, ax=ax_target, palette='viridis')
st.pyplot(fig_target)

# Interactive Plotting
st.subheader("Variable Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])
with col2:
    x_var = st.selectbox("Select X Variable", df.columns)
with col3:
    y_var = st.selectbox("Select Y Variable (Optional for Hist/Box)", [None] + list(df.columns))

fig_custom, ax_custom = plt.subplots(figsize=(8, 5))

if plot_type == "Histogram":
    sns.histplot(data=df, x=x_var, kde=True, ax=ax_custom)
elif plot_type == "Box Plot":
    sns.boxplot(data=df, x=x_var, y=y_var, ax=ax_custom)
elif plot_type == "Scatter Plot":
    if y_var:
        sns.scatterplot(data=df, x=x_var, y=y_var, hue='Exited', ax=ax_custom)
    else:
        st.warning("Please select a Y variable for Scatter Plot.")
elif plot_type == "Bar Chart":
    if y_var:
        sns.barplot(data=df, x=x_var, y=y_var, ax=ax_custom)
    else:
        # Count plot if no Y provided
        sns.countplot(data=df, x=x_var, ax=ax_custom)
elif plot_type == "Line Chart":
    if y_var:
        sns.lineplot(data=df, x=x_var, y=y_var, ax=ax_custom)
    else:
        st.warning("Please select a Y variable for Line Chart.")

st.pyplot(fig_custom)


# ==========================================
# 2. Data Preprocessing
# ==========================================
st.header("2. Data Preprocessing")

if st.button("Run Data Preprocessing"):
    with st.spinner("Preprocessing data..."):
        df_processed = df.copy()

        # 1. Remove ID columns (irrelevant for modeling)
        drop_cols = ['id', 'CustomerId', 'Surname']
        df_processed = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
        
        # 2. Handling Missing Values (Numeric -> Mean, Categorical -> Mode)
        num_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Separate Target if exists in num_cols (Exited is int)
        if 'Exited' in num_cols:
            num_cols = num_cols.drop('Exited')

        imputer_num = SimpleImputer(strategy='mean')
        df_processed[num_cols] = imputer_num.fit_transform(df_processed[num_cols])
        
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_processed[cat_cols] = imputer_cat.fit_transform(df_processed[cat_cols])

        # 3. Outlier Handling (IQR Method - Clipping)
        for col in num_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

        # 4. One-Hot Encoding
        df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)

        # 5. Feature Scaling (StandardScaler)
        scaler = StandardScaler()
        # Scale all columns except Target
        cols_to_scale = [c for c in df_processed.columns if c != 'Exited']
        df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])

        # Prepare X and y
        X = df_processed.drop('Exited', axis=1)
        y = df_processed['Exited']

        # 6. Class Imbalance Handling (SMOTE)
        if has_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        # 7. Stepwise Selection (Sequential Feature Selector)
        # Using a simple estimator for selection to be faster
        sfs = SequentialFeatureSelector(LogisticRegression(max_iter=1000), 
                                        n_features_to_select='auto', 
                                        direction='forward',
                                        tol=None)
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()]
        X = X[selected_features]
        
        # Save to session state
        st.session_state.processed_data = {'X': X, 'y': y}
        st.success("Preprocessing Complete! Data is ready for modeling.")
        st.write(f"Selected Features: {list(selected_features)}")
        st.write(f"Data Shape after preprocessing: {X.shape}")


# ==========================================
# 3. Data Splitting & Model Setup
# ==========================================
st.header("3. Data Splitting & Model Setup")

if st.session_state.processed_data is not None:
    X = st.session_state.processed_data['X']
    y = st.session_state.processed_data['y']

    # Split Ratio
    split_ratio = st.radio("Select Train/Test Split Ratio", ["7:3", "8:2"])
    test_size = 0.3 if split_ratio == "7:3" else 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    col_dt, col_log = st.columns(2)

    with col_dt:
        st.subheader("Decision Tree Options")
        dt_max_depth = st.slider("Max Depth", 1, 20, 5)
        dt_criterion = st.selectbox("Criterion", ["gini", "entropy"])

    with col_log:
        st.subheader("Logistic Regression Options")
        lr_C = st.slider("C (Inverse of regularization)", 0.01, 10.0, 1.0)
        lr_max_iter = st.number_input("Max Iterations", value=500)

    # Trigger Training
    if st.button("Train & Evaluate Models"):
        
        # Helper function for metrics
        def show_metrics(y_true, y_pred, y_prob, title):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            st.markdown(f"**{title} Metrics**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2f}")
            m2.metric("Precision", f"{prec:.2f}")
            m3.metric("Recall", f"{rec:.2f}")
            m4.metric("F1 Score", f"{f1:.2f}")

            # Plots
            c1, c2 = st.columns(2)
            
            # Confusion Matrix
            with c1:
                st.write("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                st.pyplot(fig_cm)
            
            # ROC Curve
            with c2:
                st.write("ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

        # ==========================================
        # 4. Model Evaluation (Decision Tree)
        # ==========================================
        st.header("4. Model Evaluation (Decision Tree)")
        dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, criterion=dt_criterion, random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
        
        show_metrics(y_test, y_pred_dt, y_prob_dt, "Decision Tree")

        # ==========================================
        # 5. Model Evaluation (Logit Model)
        # ==========================================
        st.header("5. Model Evaluation (Logistic Regression)")
        lr_model = LogisticRegression(C=lr_C, max_iter=int(lr_max_iter), random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

        show_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")

else:
    st.info("Please run 'Data Preprocessing' in Section 2 first.")
