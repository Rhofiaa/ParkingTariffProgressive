import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import folium
from streamlit_folium import folium_static 
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import Search, Fullscreen, MiniMap
from streamlit_option_menu import option_menu
import re

# Streamlit Page Configuration
st.set_page_config(
    page_title="Parking Tariff Analysis Dashboard",
    layout="wide", 
    initial_sidebar_state="expanded",
)


FILE_PATH = 'DataParkir_Fix.xlsx' 

# Translation mappings for internal labels to English display
CLASS_TRANSLATION = {
    'Rendah': 'Low',
    'Sedang': 'Medium',
    'Tinggi': 'High'
}

CLASS_TRANSLATION_REVERSE = {
    'Low': 'Rendah',
    'Medium': 'Sedang',
    'High': 'Tinggi'
} 

# UTILITY FUNCTIONS (Conversion, Time Categories, Tariffs)
def decimal_to_hhmm(decimal_hour):
    """Convert decimal hour (11.875) to HH:MM format (11:52)."""
    hours = int(decimal_hour)
    minutes = int(round((decimal_hour - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"

def parse_time_to_decimal(time_str):
    """Convert time string (H.M, H:M, or H) to decimal hour."""
    try:
        time_str = str(time_str).replace(',', '.').replace(':', '.')
        if '.' in time_str:
            h_str, m_part_str = time_str.split('.', 1)
            h = int(h_str) if h_str else 0
            # Assume H.MM is H hours and MM minutes
            m = int(m_part_str.ljust(2, '0')[:2]) 
            return h + m / 60.0
        else:
            return float(time_str)
    except Exception:
        return np.nan

def convert_hour(x):
    """Convert hour format (e.g., '20.00-22.00') to average decimal hour (e.g., 21.0)."""
    if pd.isna(x) or str(x).strip() in ('-', '', 'nan'):
        return np.nan
    s = str(x).strip()
    try:
        parts = re.split(r'\s*-\s*', s)
        start_time_dec = parse_time_to_decimal(parts[0].strip())
        end_time_dec = parse_time_to_decimal(parts[1].strip()) if len(parts) > 1 else start_time_dec
        # Handle range crossing midnight (22 -> 02)
        if len(parts) > 1 and pd.notna(start_time_dec) and pd.notna(end_time_dec) and end_time_dec < start_time_dec:
            end_time_dec += 24.0
        if pd.isna(start_time_dec) or pd.isna(end_time_dec):
            return np.nan
        return (start_time_dec + end_time_dec) / 2
    except Exception:
        return np.nan

def time_to_decimal_hour(time_obj):
    """Convert datetime.time object (H:M) to decimal hour (H + M/60)."""
    if time_obj is None:
        return np.nan
    return time_obj.hour + time_obj.minute / 60.0

def auto_time_category(hour):
    """Automatically categorize time based on hour."""
    if (hour < 6) or (hour > 22):
        return 'Off-Peak'
    elif (hour >= 9 and hour < 17):
        return 'Peak'
    else:
        return 'Moderate'

# Translation mapping for class labels (Internal: Indonesian -> Display: English)
CLASS_TRANSLATION = {
    'Rendah': 'Low',
    'Sedang': 'Medium',
    'Tinggi': 'High'
}

CLASS_TRANSLATION_REVERSE = {
    'Low': 'Rendah',
    'Medium': 'Sedang',
    'High': 'Tinggi'
}

# Base Tariff Mapping - Support BOTH Indonesian and English keys
tariff_mapping = {
    'Motorcycle': {
        'Low': 1000, 'Medium': 2000, 'High': 3000,  # English keys
        'Rendah': 1000, 'Sedang': 2000, 'Tinggi': 3000  # Indonesian keys
    },
    'Car': {
        'Low': 3000, 'Medium': 4000, 'High': 5000,  # English keys
        'Rendah': 3000, 'Sedang': 4000, 'Tinggi': 5000  # Indonesian keys
    }
}

# NEW FUNCTION FOR PROGRESSIVE TARIFF
def calculate_progressive_tariff(vehicle_type, potential_class, decimal_hour):
    """Apply progressive tariff logic based on potential and hour."""
    base_tariff = tariff_mapping[vehicle_type].get(potential_class, 0)

    # Normalize class label to English for logic (model uses Indonesian)
    class_for_logic = CLASS_TRANSLATION.get(potential_class, potential_class)
    
    # Progressive Logic (Example: Tariff Increase Above 9:00 AM)
    if decimal_hour > 9.0:
        if class_for_logic == 'High':
            return base_tariff + 1000  # E.g., from 3000 to 4000
        elif class_for_logic == 'Medium':
            return base_tariff + 500  # E.g., from 2000 to 2500
        else:
            return base_tariff
    else:
        return base_tariff

# 2. Data Loading and Cleaning (Caching)
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None, None, None, None, None, None
        
    df_raw = df.copy()

    revenue_cols = [
        'Annual Parking Fee Revenue (Weekday – Motorcycles)', 
        'Annual Parking Fee Revenue (Weekday – Cars)',
        'Annual Parking Fee Revenue (Weekend – Motorcycles)', 
        'Annual Parking Fee Revenue (Weekend – Cars)'
    ]
    
    # Check if revenue columns exist
    missing_cols = [c for c in revenue_cols if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in Excel file: {missing_cols}")
        st.info(f"📋 Available columns in file:\n{list(df.columns)}")
        return None, None, None, None, None, None
    
    # Count columns (For Load Charts)
    count_cols = [c for c in df.columns if c.startswith('Number of')]

    # STAGE 1: CLEANING (before split) - only convert data types
    for c in revenue_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Convert Hours (no imputation - will be done after split)
    hour_cols = [c for c in df.columns if 'Hours' in c]
    for col in hour_cols:
        df[col] = df[col].apply(convert_hour)
        # NO imputation yet - will be done in train_models after split

    # Feature engineering and labeling moved to train_models after split
    # But save version for visualization only (use full data)
    df_display = df.copy()
    
    # Calculate Total Revenue ONLY for visualization
    df_display['Total_Revenue_Motorcycle'] = (df_display['Annual Parking Fee Revenue (Weekday – Motorcycles)'] + 
                                               df_display['Annual Parking Fee Revenue (Weekend – Motorcycles)'])
    df_display['Total_Revenue_Car'] = (df_display['Annual Parking Fee Revenue (Weekday – Cars)'] + 
                                        df_display['Annual Parking Fee Revenue (Weekend – Cars)'])
    
    # Imputation for visualization only
    for col in df_display.columns:
        if df_display[col].dtype != 'object':
            if col not in revenue_cols:
                df_display[col] = df_display[col].fillna(df_display[col].median())
        else:
            if col not in ['Location Point', 'Class_Motorcycle', 'Class_Car']:
                df_display[col] = df_display[col].fillna(df_display[col].mode()[0] if len(df_display[col].mode()) > 0 else 'Unknown')

    # Labeling for visualization
    motorcycle_quantiles = None
    car_quantiles = None
    
    try:
        df_display['Class_Motorcycle'] = pd.qcut(df_display['Total_Revenue_Motorcycle'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
        motorcycle_quantiles = df_display['Total_Revenue_Motorcycle'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df_display['Class_Motorcycle'] = pd.cut(df_display['Total_Revenue_Motorcycle'], bins=[-np.inf, df_display['Total_Revenue_Motorcycle'].median(), np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
        motorcycle_quantiles = df_display['Total_Revenue_Motorcycle'].quantile([0.5]).drop_duplicates().sort_values()
        
    try:
        df_display['Class_Car'] = pd.qcut(df_display['Total_Revenue_Car'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
        car_quantiles = df_display['Total_Revenue_Car'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df_display['Class_Car'] = pd.cut(df_display['Total_Revenue_Car'], bins=[-np.inf, df_display['Total_Revenue_Car'].median(), np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
        car_quantiles = df_display['Total_Revenue_Car'].quantile([0.5]).drop_duplicates().sort_values()

    if all(c in df.columns for c in ['Latitude', 'Longitude', 'Location Point']):
        df_spatial = df[['Latitude', 'Longitude', 'Location Point'] + hour_cols + count_cols].copy()
        # Remove rows without location info or point name from df_spatial
        df_spatial['Location Point'] = df_spatial['Location Point'].astype(str).str.strip()
        df_spatial = df_spatial.replace({'Location Point': {'nan': None}})
        df_spatial = df_spatial.dropna(subset=['Location Point', 'Latitude', 'Longitude'])
        df_spatial = df_spatial.reset_index(drop=True)

        # Remove rows without 'Location Point' name
        before_drop = df.shape[0]
        df['Location Point'] = df['Location Point'].astype(str).str.strip()
        df = df.replace({'Location Point': {'nan': None}})
        df = df.dropna(subset=['Location Point']).reset_index(drop=True)
        df_display['Location Point'] = df_display['Location Point'].astype(str).str.strip()
        df_display = df_display.replace({'Location Point': {'nan': None}})
        df_display = df_display.dropna(subset=['Location Point']).reset_index(drop=True)
        after_drop = df.shape[0]
        st.session_state.setdefault('rows_dropped_no_titik', 0)
        st.session_state['rows_dropped_no_titik'] = before_drop - after_drop
    else:
        st.error("Coordinate columns ('Location Point', 'Latitude', 'Longitude') not found.")
        return None, None, None, None, None, None
    
    # Return: df (clean only), df_display (with features for viz), df_spatial, hour_cols, df_raw, quantiles
    return df, df_display, df_spatial, hour_cols, df_raw, {'motorcycle': motorcycle_quantiles, 'car': car_quantiles}


# --- 3. Random Forest Model Training (Caching) ---
@st.cache_resource
def train_models(df, hour_cols):
    # Revenue columns (English names from Excel)
    revenue_cols_motorcycle = ['Annual Parking Fee Revenue (Weekday – Motorcycles)', 'Annual Parking Fee Revenue (Weekend – Motorcycles)']
    revenue_cols_car = ['Annual Parking Fee Revenue (Weekday – Cars)', 'Annual Parking Fee Revenue (Weekend – Cars)']
    
    # Base feature columns (before feature engineering) - English names from Excel
    motorcycle_features_base = ['Number of Motorcycles (Weekday)', 'Number of Motorcycles (Weekend)'] + [c for c in hour_cols if 'Motorcycles' in c]
    car_features_base = ['Number of Cars (Weekday)', 'Number of Cars (Weekend)'] + [c for c in hour_cols if 'Cars' in c]

    def build_model(df_input, feature_cols, revenue_cols, vehicle_type='motorcycle'):
        # STAGE 0: Calculate TEMPORARY labels for stratification
        df_temp = df_input.copy()
        df_temp['Total_Revenue_temp'] = df_temp[revenue_cols[0]] + df_temp[revenue_cols[1]]
        
        try:
            label_temp = pd.qcut(df_temp['Total_Revenue_temp'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
        except:
            label_temp = pd.cut(df_temp['Total_Revenue_temp'], bins=[-np.inf, df_temp['Total_Revenue_temp'].median(), np.inf], labels=['Rendah', 'Tinggi'])
        
        # STAGE 1: SPLIT with STRATIFICATION
        try:
            train_idx, test_idx = train_test_split(
                df_input.index, 
                test_size=0.2, 
                random_state=42,
                stratify=label_temp
            )
        except:
            train_idx, test_idx = train_test_split(
                df_input.index, 
                test_size=0.2, 
                random_state=42
            )
        
        df_train = df_input.loc[train_idx].copy()
        df_test = df_input.loc[test_idx].copy()
        
        # STAGE 2: FEATURE ENGINEERING (separate for train and test)
        df_train['Total_Revenue'] = df_train[revenue_cols[0]] + df_train[revenue_cols[1]]
        df_test['Total_Revenue'] = df_test[revenue_cols[0]] + df_test[revenue_cols[1]]
        
        # STAGE 3: IMPUTATION using TRAIN statistics only
        impute_values = {}
        for col in feature_cols:
            if df_train[col].isna().any():
                impute_values[col] = df_train[col].median()
        
        # Apply imputation
        df_train[feature_cols] = df_train[feature_cols].fillna(impute_values)
        df_test[feature_cols] = df_test[feature_cols].fillna(impute_values)
        
        # STAGE 4: LABELING using TRAIN thresholds only
        le = LabelEncoder()
        quantile_thresholds = None
        
        try:
            # Calculate quantile from TRAIN
            labels = pd.qcut(df_train['Total_Revenue'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
            quantiles = df_train['Total_Revenue'].quantile([0.333, 0.666]).values
            quantile_thresholds = quantiles
            
            # Apply to train
            df_train['Class'] = labels
            
            # Apply train threshold to test
            if len(quantiles) == 2:
                df_test['Class'] = pd.cut(
                    df_test['Total_Revenue'], 
                    bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
                    labels=['Rendah','Sedang','Tinggi']
                )
            else:
                median_val = df_train['Total_Revenue'].median()
                quantile_thresholds = np.array([median_val])
                df_train['Class'] = pd.cut(df_train['Total_Revenue'], bins=[-np.inf, median_val, np.inf], labels=['Rendah', 'Tinggi'])
                df_test['Class'] = pd.cut(df_test['Total_Revenue'], bins=[-np.inf, median_val, np.inf], labels=['Rendah', 'Tinggi'])
        except:
            median_val = df_train['Total_Revenue'].median()
            quantile_thresholds = np.array([median_val])
            df_train['Class'] = pd.cut(df_train['Total_Revenue'], bins=[-np.inf, median_val, np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
            df_test['Class'] = pd.cut(df_test['Total_Revenue'], bins=[-np.inf, median_val, np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
        
        # Encode labels
        y_train = df_train['Class']
        y_test = df_test['Class']
        
        if len(y_train.unique()) <= 1:
            return None, le, pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), np.array([]), pd.DataFrame(), {}, {}, None
        
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        X_train = df_train[feature_cols]
        X_test = df_test[feature_cols]
        
        # STAGE 5: MODEL TRAINING
        # Same hyperparameters for Motorcycle and Car
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=3,
            bootstrap=True,
            random_state=42,
            criterion='gini'
        )
        
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        
        # Metrics for visualization
        X_ref = pd.concat([X_train, X_test]).reset_index(drop=True)
        train_scores = []
        test_scores = []
        tree_counts = []
        
        n_est = model.n_estimators
        
        # Range from 10, 20, ..., 150 (not from 1)
        for n_trees in range(10, n_est+1, 10):
            y_pred_train_prob = np.zeros((len(y_train_enc), len(le.classes_)))
            y_pred_test_prob = np.zeros((len(y_test_enc), len(le.classes_)))
            
            for estimator in model.estimators_[:n_trees]:
                y_pred_train_prob += estimator.predict_proba(X_train)
                y_pred_test_prob += estimator.predict_proba(X_test)
            
            y_pred_train_final = np.argmax(y_pred_train_prob, axis=1)
            y_pred_test_final = np.argmax(y_pred_test_prob, axis=1)
            
            train_acc = np.mean(y_pred_train_final == y_train_enc)
            test_acc = np.mean(y_pred_test_final == y_test_enc)
            
            train_scores.append(train_acc)
            test_scores.append(test_acc)
            tree_counts.append(n_trees)
        
        training_metrics = {
            'tree_counts': tree_counts,
            'train_scores': train_scores,
            'test_scores': test_scores
        }
        
        oob_scores = {'n_estimators': [10, 50, 100, 150]}
        
        return model, le, X_train, X_test, y_train_enc, y_test_enc, y_pred, X_ref, training_metrics, oob_scores, quantile_thresholds
    
    # Build models (clean data without features/labels - will be created in build_model)
    results = {}
    
    # Build motorcycle model
    model_motorcycle, le_motorcycle, X_train_m, X_test_m, y_train_m, y_test_m, y_pred_m, X_ref_m, metrics_m, oob_m, thresh_m = build_model(
        df, motorcycle_features_base, revenue_cols_motorcycle, vehicle_type='motorcycle'
    )

    results['motorcycle'] = {
        'model': model_motorcycle, 'le': le_motorcycle, 'X_train': X_train_m, 'X_test': X_test_m, 'y_train': y_train_m, 
        'y_test': y_test_m, 'y_pred': y_pred_m, 'X_ref': X_ref_m, 'features': motorcycle_features_base, 
        'X_all': df[motorcycle_features_base] if model_motorcycle else pd.DataFrame(),
        'training_metrics': metrics_m, 'oob_scores': oob_m, 'quantile_thresholds': thresh_m
    }

    # Build car model
    model_car, le_car, X_train_c, X_test_c, y_train_c, y_test_c, y_pred_c, X_ref_c, metrics_c, oob_c, thresh_c = build_model(
        df, car_features_base, revenue_cols_car, vehicle_type='car'
    )

    results['car'] = {
        'model': model_car, 'le': le_car, 'X_train': X_train_c, 'X_test': X_test_c, 'y_train': y_train_c, 'y_test': y_test_c, 'y_pred': y_pred_c, 'X_ref': X_ref_c, 'features': car_features_base, 'X_all': df[car_features_base] if model_car else pd.DataFrame(), 'training_metrics': metrics_c, 'oob_scores': oob_c, 'quantile_thresholds': thresh_c
    }
    
    return results

# Prediction Function for Simulation
def predict_single_input(vehicle_type, day, hour_input, count_input, model, le, X_ref, quantile_thresholds, baseline_data=None): 
    """Predict using trained Random Forest."""
    if model is None:
        return "Model Failed", 0.0, pd.Series({"No Model": 0}), {"Error": 1.0}, "No trained model available", None, None

    time_category = auto_time_category(hour_input)
    prefix = vehicle_type
    
    # Use baseline_data if available, otherwise use mean from X_ref
    if baseline_data is not None:
        new_data = pd.DataFrame([baseline_data], columns=X_ref.columns)
    else:
        new_data = pd.DataFrame([X_ref.mean()], columns=X_ref.columns)
    
    # Update vehicle count column according to input (English column name)
    vehicle_plural = 'Motorcycles' if vehicle_type == 'Motorcycle' else 'Cars'
    count_column = f'Number of {vehicle_plural} ({day})'
    if count_column in new_data.columns: 
        new_data[count_column] = count_input
    
    # Update hour column according to category (English column name)
    category_english = {'Off-Peak': 'Off-Peak', 'Moderate': 'Moderate', 'Peak': 'Peak'}[time_category]
    vehicle_plural = 'Motorcycles' if vehicle_type == 'Motorcycle' else 'Cars'
    hour_column_input = f'{category_english} Hours for {vehicle_plural} ({day})'
    if hour_column_input in new_data.columns: 
        new_data[hour_column_input] = hour_input
    
    # REVENUE ESTIMATION based on input
    tariff_wd = 2000 if vehicle_type == 'Motorcycle' else 3000
    tariff_we = 2500 if vehicle_type == 'Motorcycle' else 4000
    
    if day == "Weekday":
        estimated_revenue = count_input * 52 * (5/7) * tariff_wd
    else:
        estimated_revenue = count_input * 52 * (2/7) * tariff_we
    
    # CLASSIFICATION based on threshold
    threshold_class = None
    if quantile_thresholds is not None and len(quantile_thresholds) == 2:
        if estimated_revenue <= quantile_thresholds[0]:
            threshold_class = 'Low'
        elif estimated_revenue <= quantile_thresholds[1]:
            threshold_class = 'Medium'
        else:
            threshold_class = 'High'
    
    hour_explanation = f"Input hour **{hour_input:.2f}** is categorized as **'{time_category}'**."

    try:
        # PREDICTION USING RANDOM FOREST
        pred_encoded = model.predict(new_data)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(new_data)[0]
        confidence = proba[pred_encoded] 
        
        # Feature importance
        global_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        local_gain_calc = (new_data.iloc[0] - X_ref.mean()) * global_importance
        top_gain = local_gain_calc.abs().sort_values(ascending=False).head(3)
        
        proba_dict = dict(zip(le.classes_, proba))
        
        return pred_class, confidence, top_gain, proba_dict, hour_explanation, estimated_revenue, threshold_class
    except Exception as e:
        return f"Prediction Error: {e}", 0.0, pd.Series({"Error": 0}), {"Error": 1.0}, hour_explanation, None, None


# === MODULE 1: DISPLAY DATA TABLE ===
def display_data_table(df_raw, df_processed):
    # Create tabs for raw and processed data
    tab_raw, tab_processed = st.tabs(["📋 Raw Data", "✅ Pre-processed Data"])
    
    with tab_raw:
        st.subheader("Raw Data")
        st.info("Original data from Excel file without cleaning or processing.")
        st.dataframe(df_raw, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Raw Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", df_raw.shape[0])
        with col2:
            st.metric("Number of Columns", df_raw.shape[1])
        with col3:
            st.metric("Total Missing Values", df_raw.isna().sum().sum())
    
    with tab_processed:
        st.subheader("Pre-processed Data (Ready for Modeling)")
        st.info("Data after cleaning, numeric conversion, imputation, and addition of **Total Revenue** and **Potential Class** columns (Classification Target).")
        st.dataframe(df_processed, use_container_width=True)

        st.markdown("---")
        st.subheader("Statistical Summary of Key Columns")
        
        col_m, col_c = st.columns(2)
        
        with col_m:
            st.markdown("#### Motorcycle Total Revenue Statistics")
            st.dataframe(df_processed['Total_Revenue_Motorcycle'].describe().to_frame(), use_container_width=True)
            
        with col_c:
            st.markdown("#### Car Total Revenue Statistics")
            st.dataframe(df_processed['Total_Revenue_Car'].describe().to_frame(), use_container_width=True)


# --- VISUALIZATION SUPPORT FUNCTIONS ---
def plot_load_vs_time(df, hour_cols, count_cols):
    """Plot Scatter/Line Average Vehicle Load at Aggregate Time Points."""
    st.subheader("Average Load (Vehicle Count) vs Time Chart (Aggregate Points)")
    st.info("This chart illustrates the average vehicle count ('Load') correlated with time ranges in your dataset. These points represent times when 'Load' is highest/moderate/lowest.")
    
    df_load = df[hour_cols + count_cols].copy()
    data_points = []
    
    # Motorcycle
    motorcycle_cols = [c for c in df_load.columns if 'Motor' in c]
    for hour_col in [c for c in motorcycle_cols if c.startswith('Jam')]:
        try:
            # Parse format: 'Peak Hours for Motorcycles (Weekday)'
            match = re.search(r'(.*) Hours for Motorcycles \((.*)\)', hour_col)
            if match:
                time_category = match.group(1)
                day = match.group(2)
                count_col = f'Number of Motorcycles ({day})'
            else:
                continue

            if count_col in df_load.columns:
                avg_hour = df_load[hour_col].mean()
                avg_count = df_load[count_col].mean()
                data_points.append({'Time (Decimal Hour)': avg_hour, 'Average Load': avg_count, 'Type': 'Motorcycle - ' + day, 'Time Category': time_category})
        except Exception:
            pass

    # Car
    car_cols = [c for c in df_load.columns if 'Mobil' in c]
    for hour_col in [c for c in car_cols if c.startswith('Jam')]:
        try:
            # Parse format: 'Peak Hours for Cars (Weekday)'
            match = re.search(r'(.*) Hours for Cars \((.*)\)', hour_col)
            if match:
                time_category = match.group(1)
                day = match.group(2)
                count_col = f'Number of Cars ({day})'
            else:
                continue
                
            if count_col in df_load.columns:
                avg_hour = df_load[hour_col].mean()
                avg_count = df_load[count_col].mean()
                data_points.append({'Time (Decimal Hour)': avg_hour, 'Average Load': avg_count, 'Type': 'Car - ' + day, 'Time Category': time_category})
        except Exception:
            pass

    df_plot = pd.DataFrame(data_points)
    
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='Time (Decimal Hour)', y='Average Load', hue='Type', style='Type', s=200, ax=ax)
        sns.lineplot(data=df_plot, x='Time (Decimal Hour)', y='Average Load', hue='Type', legend=False, alpha=0.5, ax=ax)
        ax.set_title('Average Vehicle Load by Time (Aggregate Points)')
        ax.set_xticks(np.arange(0, 25, 3)) 
        ax.set_xlim(0, 24)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for line in df_plot.itertuples():
            ax.annotate(line.Time_Category, (line._1, line._2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
        st.pyplot(fig)
    else:
        st.warning("No Load (Vehicle Count) data available to plot.")


# --- NEW FUNCTION (24-HOUR LINE CHART WITH LOAD CATEGORY BACKGROUND) ---
def plot_load_24_hours(df):
    """
    Create a line plot of average vehicle count per hour over 24 hours 
    with background colored by Load Category.
    """
    st.subheader("24-Hour Average Vehicle Count Line Chart 📉")
    st.info("Load Categories (Off-Peak/Moderate/Peak) are represented by background colors.")
    
    # 1. Aggregate Average Vehicle Count per Hour (Synthetic/Estimated Data)
    # Take average total vehicles as load basis
    count_cols = [c for c in df.columns if c.startswith('Number of')]
    base_load = df[count_cols].mean().mean() / 5 if count_cols and df[count_cols].mean().mean() > 0 else 50
    
    hours = np.arange(24)
    # Synthesize 24-hour trend
    avg_counts_synth = [
        0, 0, 0, 0, 0, 0, 
        base_load * 0.5, base_load * 1.5, base_load * 2.5, 
        base_load * 4, base_load * 5, base_load * 6, base_load * 8, base_load * 7, 
        base_load * 6.5, base_load * 5, base_load * 4.5, 
        base_load * 4, base_load * 3.5, 
        base_load * 3, base_load * 2.5, base_load * 1.5, 
        base_load * 0.5, base_load * 0.2, 
    ]
    
    if max(avg_counts_synth) < 100 and max(avg_counts_synth) > 0:
        avg_counts_synth = [c * (100 / max(avg_counts_synth)) for c in avg_counts_synth]
    elif max(avg_counts_synth) == 0:
        avg_counts_synth = [c + 50 for c in avg_counts_synth]

    df_24h = pd.DataFrame({'Hour': hours, 'Average Vehicle Count': avg_counts_synth})
    df_24h['Load Category'] = df_24h['Hour'].apply(auto_time_category)
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Background Colors (Adjusted)
    colors = {'Off-Peak': "#EEF06A49", 'Peak': "#F4444445", 'Moderate': "#359EDF50"}
    
    # --- FIXED axvspan RANGES ---
    
    # Off-Peak: 00:00 - 06:00
    ax.axvspan(0, 6, color=colors['Off-Peak'], label='Off-Peak', zorder=0)

    # Moderate: 06:00 - 09:00
    ax.axvspan(6, 9.001, color=colors['Moderate'], label='Moderate', zorder=0) 

    # Peak: 09:00 - 17:00
    ax.axvspan(9.001, 17, color=colors['Peak'], label='Peak', zorder=0)

    # Moderate: 17:00 - 22:00
    ax.axvspan(17, 22.001, color=colors['Moderate'], label='Moderate', zorder=0)

    # Off-Peak: 22:00 - 24:00 (Main fix for 22-00 range)
    ax.axvspan(22.001, 24, color=colors['Off-Peak'], label='Off-Peak', zorder=0) 
    
    # Plot Average Vehicle Count Line
    sns.lineplot(data=df_24h, x='Hour', y='Average Vehicle Count', marker='o', color='darkorange', linewidth=2.5, ax=ax)
    
    ax.set_title('Average Vehicle Count per Hour vs Load Category (24 Hours)')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average Vehicle Count (Load)')
    ax.set_xticks(hours[::3])
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6, axis='both')
    
    # --- FIXED LEGEND ---
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['Off-Peak'], label='Off-Peak (00-06 & 22-24)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Peak'], label='Peak (09-17)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Moderate'], label='Moderate (06-09 & 17-22)')
    ]
    ax.legend(handles=legend_elements, title='Load Category', loc='upper right')
    
    st.pyplot(fig)


# Module 2: Visualization (UI/UX Improved)
def display_visualization(df, quantile_boundaries, hour_cols):
    st.header("2️⃣ Data Visualization & Analysis")
    st.markdown("---")
    
    # CHANGED: Add 4th tab for 24-hour line chart
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Revenue Distribution & Classes", "💰 Quantile Boundaries (Rupiah)", "🔗 Average Density", "📉 Load vs Time (24H Line Graph)"])
    
    with tab1:
        st.subheader("Total Revenue Distribution & Potential Category")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Motorcycle Tariff Potential Category Distribution 🏍️")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Revenue_Motorcycle', hue='Class_Motorcycle', palette='viridis', multiple='stack', ax=ax, kde=True)
            ax.set_title('Motorcycle Total Revenue vs Category')
            ax.set_xlabel('Annual Total Revenue')
            st.pyplot(fig)
            with st.expander("View Points Count per Category"):
                st.dataframe(df['Class_Motorcycle'].value_counts().to_frame('Point Count'), use_container_width=True)

        with col2:
            st.markdown("#### Car Tariff Potential Category Distribution 🚗")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Revenue_Car', hue='Class_Car', palette='plasma', multiple='stack', ax=ax, kde=True)
            ax.set_title('Car Total Revenue vs Category')
            ax.set_xlabel('Annual Total Revenue')
            st.pyplot(fig)
            with st.expander("View Points Count per Category"):
                st.dataframe(df['Class_Car'].value_counts().to_frame('Point Count'), use_container_width=True)

    with tab2:
        st.subheader("💰 ANNUAL TOTAL REVENUE QUANTILE BOUNDARIES (Rp) 💰")
        col_m, col_c = st.columns(2)
        
        if quantile_boundaries['motorcycle'] is not None:
            with col_m:
                st.markdown("### Motorcycle Quantile Boundaries 🏍️")
                motorcycle_boundaries = quantile_boundaries['motorcycle']
                if len(motorcycle_boundaries) == 2:
                    # MOTORCYCLE QUANTILE BOUNDARY VISUALIZATION - COLOR ZONES (BEFORE TEXT)
                    fig_motorcycle, ax_motorcycle = plt.subplots(figsize=(12, 2), dpi=80)
                    
                    # Calculate values for visualization
                    min_val = 0
                    max_val = df['Total_Revenue_Motorcycle'].max()
                    b1 = motorcycle_boundaries.iloc[0]
                    b2 = motorcycle_boundaries.iloc[1]
                    
                    # Create stacked bar with 3 zones
                    ax_motorcycle.barh(['Motorcycle'], [b1 - min_val], left=min_val, color='#FF6B6B', label='Low', height=0.5)
                    ax_motorcycle.barh(['Motorcycle'], [b2 - b1], left=b1, color='#FFC93C', label='Medium', height=0.5)
                    ax_motorcycle.barh(['Motorcycle'], [max_val - b2], left=b2, color='#4ECDC4', label='High', height=0.5)
                    
                    # Add boundary lines and text
                    ax_motorcycle.axvline(b1, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                    ax_motorcycle.axvline(b2, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    # Calculate percentage of data in each category
                    cnt_low = (df['Total_Revenue_Motorcycle'] < b1).sum()
                    cnt_medium = ((df['Total_Revenue_Motorcycle'] >= b1) & (df['Total_Revenue_Motorcycle'] < b2)).sum()
                    cnt_high = (df['Total_Revenue_Motorcycle'] >= b2).sum()
                    total = len(df)
                    
                    # Text in center of each zone
                    ax_motorcycle.text((min_val + b1) / 2, 0, f'{cnt_low}\n({100*cnt_low/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_motorcycle.text((b1 + b2) / 2, 0, f'{cnt_medium}\n({100*cnt_medium/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_motorcycle.text((b2 + max_val) / 2, 0, f'{cnt_high}\n({100*cnt_high/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    
                    ax_motorcycle.set_xlabel('Annual Total Revenue (Rp)', fontsize=10, fontweight='bold')
                    ax_motorcycle.set_xlim(min_val, max_val * 1.05)
                    ax_motorcycle.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x/1e6:.0f}M'))
                    ax_motorcycle.legend(loc='upper right', fontsize=9, ncol=3)
                    ax_motorcycle.set_yticks([])
                    plt.tight_layout()
                    st.pyplot(fig_motorcycle, use_container_width=True)
                    
                    # TEXT EXPLANATION AFTER DIAGRAM
                    st.markdown(f"* **Low** : Revenue < **Rp{motorcycle_boundaries.iloc[0]:,.0f}**")
                    st.markdown(f"* **Medium** : **Rp{motorcycle_boundaries.iloc[0]:,.0f}** to **Rp{motorcycle_boundaries.iloc[1]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{motorcycle_boundaries.iloc[1]:,.0f}**")
                    
                elif len(motorcycle_boundaries) == 1:
                    # MOTORCYCLE QUANTILE BOUNDARY VISUALIZATION (2 CATEGORIES) - COLOR ZONES (BEFORE)
                    fig_motorcycle, ax_motorcycle = plt.subplots(figsize=(12, 2), dpi=80)
                    
                    min_val = 0
                    max_val = df['Total_Revenue_Motorcycle'].max()
                    b1 = motorcycle_boundaries.iloc[0]
                    
                    ax_motorcycle.barh(['Motorcycle'], [b1 - min_val], left=min_val, color='#FF6B6B', label='Low', height=0.5)
                    ax_motorcycle.barh(['Motorcycle'], [max_val - b1], left=b1, color='#4ECDC4', label='High', height=0.5)
                    
                    ax_motorcycle.axvline(b1, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    cnt_low = (df['Total_Revenue_Motorcycle'] < b1).sum()
                    cnt_high = (df['Total_Revenue_Motorcycle'] >= b1).sum()
                    total = len(df)
                    
                    ax_motorcycle.text((min_val + b1) / 2, 0, f'{cnt_low}\n({100*cnt_low/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_motorcycle.text((b1 + max_val) / 2, 0, f'{cnt_high}\n({100*cnt_high/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    
                    ax_motorcycle.set_xlabel('Annual Total Revenue (Rp)', fontsize=10, fontweight='bold')
                    ax_motorcycle.set_xlim(min_val, max_val * 1.05)
                    ax_motorcycle.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x/1e6:.0f}M'))
                    ax_motorcycle.legend(loc='upper right', fontsize=9, ncol=2)
                    ax_motorcycle.set_yticks([])
                    plt.tight_layout()
                    st.pyplot(fig_motorcycle, use_container_width=True)
                    
                    # TEXT EXPLANATION AFTER DIAGRAM
                    st.markdown(f"* **Low** : Revenue < **Rp{motorcycle_boundaries.iloc[0]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{motorcycle_boundaries.iloc[0]:,.0f}**")
                else:
                    st.warning("Insufficient data or quantile variation for motorcycles.")
        else:
            col_m.warning("Motorcycle quantile boundaries cannot be calculated.")

        if quantile_boundaries['car'] is not None:
            with col_c:
                st.markdown("### Car Quantile Boundaries 🚗")
                car_boundaries = quantile_boundaries['car']
                if len(car_boundaries) == 2:
                    # CAR QUANTILE BOUNDARY VISUALIZATION - COLOR ZONES (BEFORE TEXT)
                    fig_car, ax_car = plt.subplots(figsize=(12, 2), dpi=80)
                    
                    min_val = 0
                    max_val = df['Total_Revenue_Car'].max()
                    b1 = car_boundaries.iloc[0]
                    b2 = car_boundaries.iloc[1]
                    
                    ax_car.barh(['Car'], [b1 - min_val], left=min_val, color='#FF6B6B', label='Low', height=0.5)
                    ax_car.barh(['Car'], [b2 - b1], left=b1, color='#FFC93C', label='Medium', height=0.5)
                    ax_car.barh(['Car'], [max_val - b2], left=b2, color='#4ECDC4', label='High', height=0.5)
                    
                    ax_car.axvline(b1, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                    ax_car.axvline(b2, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    cnt_low = (df['Total_Revenue_Car'] < b1).sum()
                    cnt_medium = ((df['Total_Revenue_Car'] >= b1) & (df['Total_Revenue_Car'] < b2)).sum()
                    cnt_high = (df['Total_Revenue_Car'] >= b2).sum()
                    total = len(df)
                    
                    ax_car.text((min_val + b1) / 2, 0, f'{cnt_low}\n({100*cnt_low/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_car.text((b1 + b2) / 2, 0, f'{cnt_medium}\n({100*cnt_medium/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_car.text((b2 + max_val) / 2, 0, f'{cnt_high}\n({100*cnt_high/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    
                    ax_car.set_xlabel('Annual Total Revenue (Rp)', fontsize=10, fontweight='bold')
                    ax_car.set_xlim(min_val, max_val * 1.05)
                    ax_car.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x/1e6:.0f}M'))
                    ax_car.legend(loc='upper right', fontsize=9, ncol=3)
                    ax_car.set_yticks([])
                    plt.tight_layout()
                    st.pyplot(fig_car, use_container_width=True)
                    
                    # TEXT EXPLANATION AFTER DIAGRAM
                    st.markdown(f"* **Low** : Revenue < **Rp{car_boundaries.iloc[0]:,.0f}**")
                    st.markdown(f"* **Medium** : **Rp{car_boundaries.iloc[0]:,.0f}** to **Rp{car_boundaries.iloc[1]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{car_boundaries.iloc[1]:,.0f}**")
                    
                elif len(car_boundaries) == 1:
                    # CAR QUANTILE BOUNDARY VISUALIZATION (2 CATEGORIES) - COLOR ZONES (BEFORE)
                    fig_car, ax_car = plt.subplots(figsize=(12, 2), dpi=80)
                    
                    min_val = 0
                    max_val = df['Total_Revenue_Car'].max()
                    b1 = car_boundaries.iloc[0]
                    
                    ax_car.barh(['Car'], [b1 - min_val], left=min_val, color='#FF6B6B', label='Low', height=0.5)
                    ax_car.barh(['Car'], [max_val - b1], left=b1, color='#4ECDC4', label='High', height=0.5)
                    
                    ax_car.axvline(b1, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    cnt_low = (df['Total_Revenue_Car'] < b1).sum()
                    cnt_high = (df['Total_Revenue_Car'] >= b1).sum()
                    total = len(df)
                    
                    ax_car.text((min_val + b1) / 2, 0, f'{cnt_low}\n({100*cnt_low/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    ax_car.text((b1 + max_val) / 2, 0, f'{cnt_high}\n({100*cnt_high/total:.1f}%)', 
                                 ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    
                    ax_car.set_xlabel('Annual Total Revenue (Rp)', fontsize=10, fontweight='bold')
                    ax_car.set_xlim(min_val, max_val * 1.05)
                    ax_car.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x/1e6:.0f}M'))
                    ax_car.legend(loc='upper right', fontsize=9, ncol=2)
                    ax_car.set_yticks([])
                    plt.tight_layout()
                    st.pyplot(fig_car, use_container_width=True)
                    
                    # TEXT EXPLANATION AFTER DIAGRAM
                    st.markdown(f"* **Low** : Revenue < **Rp{car_boundaries.iloc[0]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{car_boundaries.iloc[0]:,.0f}**")
                else:
                    st.warning("Insufficient data or quantile variation for cars.")
        else:
            col_c.warning("Car quantile boundaries cannot be calculated.")

    with tab3:
        # REPLACE: Display new visualization (Bar plot + Heatmap) based on average hours and assumed load
        st.subheader("Density & Time Visualization (Bar Plot + Heatmap)")
        st.info("Visual: Average time (decimal hour) for time categories and estimated load based on average vehicle count.")

        # Hour Features (use processed hour_cols) - English column names
        hour_features = [c for c in df.columns if 'Hours' in c]

        # 1. Calculate Average Decimal Hour for each category (English names)
        avg_hours = {}
        for category in ['Peak', 'Moderate', 'Off-Peak']:
            for vehicle_type in ['Motorcycles', 'Cars']:
                for day in ['Weekday', 'Weekend']:
                    column = f'{category} Hours for {vehicle_type} ({day})'
                    if column in df.columns:
                        avg_hours[column] = df[column].mean()

        # 2. Prepare df_visual DataFrame
        visual_data = []

        for vehicle_type in ['Motorcycles', 'Cars']:
            for day in ['Weekday', 'Weekend']:
                count_column = f'Number of {vehicle_type} ({day})'
                if count_column in df.columns:
                    avg_vehicle_load = df[count_column].mean()
                else:
                    avg_vehicle_load = 0

                # Assume relative density
                load_peak = avg_vehicle_load * 1.0
                load_moderate = avg_vehicle_load * 0.5
                load_off_peak = avg_vehicle_load * 0.2

                # Insert if avg_hours available, otherwise skip
                key_peak = f'Peak Hours for {vehicle_type} ({day})'
                key_moderate = f'Moderate Hours for {vehicle_type} ({day})'
                key_off_peak = f'Off-Peak Hours for {vehicle_type} ({day})'

                if key_peak in avg_hours:
                    visual_data.append({
                        'Vehicle Type': vehicle_type, 'Day': day, 'Category': 'Peak',
                        'Avg_Time': avg_hours[key_peak],
                        'Avg_Load': load_peak
                    })
                if key_moderate in avg_hours:
                    visual_data.append({
                        'Vehicle Type': vehicle_type, 'Day': day, 'Category': 'Moderate',
                        'Avg_Time': avg_hours[key_moderate],
                        'Avg_Load': load_moderate
                    })
                if key_off_peak in avg_hours:
                    visual_data.append({
                        'Vehicle Type': vehicle_type, 'Day': day, 'Category': 'Off-Peak',
                        'Avg_Time': avg_hours[key_off_peak],
                        'Avg_Load': load_off_peak
                    })

        if len(visual_data) == 0:
            st.warning("Insufficient data for new visualization — ensure 'Jam ...' and 'Jumlah ...' columns exist in dataset.")
        else:
            df_visual = pd.DataFrame(visual_data)
            st.write("Sample visualization data:")
            st.dataframe(df_visual.head(), use_container_width=True)

            # --- 2. Bar Plot Visualization ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            df_plot = df_visual.sort_values(by='Avg_Load', ascending=False).copy()
            df_plot['Combined_Category'] = df_plot['Category'].astype(str) + ' (' + df_plot['Day'].astype(str) + ')'

            order_bar = ['Peak (Weekday)', 'Peak (Weekend)', 'Moderate (Weekday)', 'Moderate (Weekend)', 'Off-Peak (Weekday)', 'Off-Peak (Weekend)']

            # Vehicle types already in English from data (Motorcycles/Cars → Motorcycle/Car for display)
            df_plot['Vehicle Type'] = df_plot['Vehicle Type'].replace({'Motorcycles': 'Motorcycle', 'Cars': 'Car'})

            sns.barplot(
                data=df_plot[df_plot['Vehicle Type'] == 'Motorcycle'],
                x='Combined_Category',
                y='Avg_Load',
                order=order_bar,
                palette='viridis',
                ax=axes[0]
            )
            axes[0].set_title('Average Motorcycle Parking Density (By Time Category)')
            axes[0].set_xlabel('Time Category')
            axes[0].set_ylabel('Average Motorcycle Count')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            sns.barplot(
                data=df_plot[df_plot['Vehicle Type'] == 'Car'],
                x='Combined_Category',
                y='Avg_Load',
                order=order_bar,
                palette='magma',
                ax=axes[1]
            )
            axes[1].set_title('Average Car Parking Density (By Time Category)')
            axes[1].set_xlabel('Time Category')
            axes[1].set_ylabel('Average Car Count')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # --- 3. Heatmap Visualization ---
            df_heatmap = df_visual.pivot_table(index=['Day', 'Category'], columns='Vehicle Type', values='Avg_Load')
            order_day = ['Weekday', 'Weekend']
            order_category = ['Peak', 'Moderate', 'Off-Peak']
            idx = pd.MultiIndex.from_product([order_day, order_category], names=['Day', 'Category'])
            df_heatmap = df_heatmap.reindex(idx, fill_value=0)

            plt.figure(figsize=(8, 5))
            df_heatmap_final = df_heatmap.unstack(level=0)
            sns.heatmap(
                df_heatmap_final.T,
                annot=True,
                fmt=".0f",
                cmap="YlGnBu",
                linewidths=.5,
                cbar_kws={'label': 'Average Vehicle Count'}
            )
            plt.title('Parking Density (Load) Comparison Heatmap')
            plt.xlabel('Time Category and Day')
            plt.ylabel('Vehicle Type')
            plt.yticks(rotation=0)
            st.pyplot(plt.gcf())
        
    with tab4:
        st.subheader("24-Hour Line Chart (Alternative)")
        st.info("Synthetic 24-hour line chart illustrating daily trend (alternative visual view).")
        # Call available 24-hour line chart function
        try:
            plot_load_24_hours(df)
        except Exception as e:
            st.warning(f"Failed to display 24-hour chart: {e}")

    with st.expander("Support Explanation (Confidence & Top 1 Contributor)"):
        st.markdown(
            """
            * **Confidence (Support):** Represents the model's probability confidence in the predicted class (e.g., 0.95 means 95% confident).
            * **Top 1 Contributor (Local Gain):** The feature (column) with the greatest influence (*gain*) in predicting the class for a specific *input* row. This proves the model uses relevant features, not just guessing.
            """
        )

# --- Module 3: Modeling (ADDED) ---
def display_modeling(df_processed, models_data):
    st.header("3️⃣ Tariff Potential Classification Modeling (Random Forest)")
    st.markdown("---")
    
    # Random Forest Parameter Details
    with st.expander("⚙️ Random Forest Parameters Used", expanded=False):
        st.markdown("""
        ### Random Forest Model Configuration
        
        **Main Parameters:**
        | Parameter | Value | Explanation |
        |-----------|-------|-----------|
        | **n_estimators** | 150 | Number of decision trees in ensemble (reduced from 200 for efficiency) |
        | **max_depth** | 15 | Maximum depth limit for each tree (prevents overfitting) |
        | **min_samples_split** | 2 | Minimum 2 samples to split a node (sklearn default) |
        | **min_samples_leaf** | 3 | Minimum 3 samples in each leaf (makes decisions more robust) |
        | **bootstrap** | True | Use bootstrap sampling for each tree |
        | **random_state** | 42 | Seed for reproducibility (consistent results) |
        | **criterion** | gini | Metric for measuring split quality (Gini impurity) |
        
        **Why These Parameters?**
        - **150 estimators**: Trade-off between accuracy and training speed
        - **max_depth=15**: Deep enough to capture patterns, but not too deep to memorize noise
        - **min_samples_split=2**: Sklearn default, allows split with minimum 2 samples
        - **min_samples_leaf=3**: Each decision based on minimum 3 locations, not just 1 outlier
        - **bootstrap=True**: Each tree trained with different data subset (sampling with replacement)
        - **criterion=gini**: Uses Gini impurity to select best splits
        
        **Training Process:**
        1. Training data (80% of total) used to train 150 trees
        2. Each tree learns independently with random bootstrap data subset
        3. Final prediction = voting from 150 trees (majority vote)
        4. Evaluation using testing data (20% of total) NOT used during training
        """)
    
    tab_motorcycle, tab_car, tab_training_data, tab_training, tab_tree, tab_recommendations = st.tabs(["🏍️ Motorcycle Model", "🚗 Car Model", "📊 Training Data", "📈 Training Charts", "🌳 Tree Visualization", "📑 Tariff Recommendations"])

    def display_model_results(vehicle_type, data):
        st.subheader(f"{vehicle_type.capitalize()} Model Training Results")
        
        model = data['model']
        y_test = data['y_test']
        y_pred = data['y_pred']
        le = data['le']

        if model is None:
            st.error(f"{vehicle_type.capitalize()} model not trained because target column 'Class_{vehicle_type.capitalize()}' lacks sufficient class variation (nunique <= 1).")
            return
        
        # Training Summary
        st.info(f"""
        ✅ **Training Complete!** Random Forest model with 150 trees has been trained using training data and evaluated with testing data.
        See **"📊 Training Data" tab** for 80:20 split details and **"📈 Training Charts" tab** to view learning curve (how accuracy improves with training).
        """)
            
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Calculate correct predictions (confusion matrix diagonal)
                correct_predictions = np.trace(cm)
                total_predictions = len(y_test)
                
                st.metric(f"{vehicle_type.capitalize()} Model Accuracy", f"{acc*100:.2f} %")
                st.caption(f"📊 Correct Predictions: {correct_predictions} out of {total_predictions} data | Manual: {correct_predictions}/{total_predictions} = {(correct_predictions/total_predictions)*100:.2f}%")
                
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('Actual Class')
                ax.set_xlabel('Predicted Class')
                st.pyplot(fig)
            except ValueError as e:
                st.warning(f"Cannot calculate metrics because predicted classes don't match actual classes: {e}")

        with col2:
            st.markdown("#### Classification Report")
            try:
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1] # Get Precision, Recall, F1-Score
                st.dataframe(report_df, use_container_width=True)
            except ValueError as e:
                st.warning(f"Cannot create Classification Report: {e}")

            st.markdown("#### Feature Importance")
            importance = pd.Series(model.feature_importances_, index=data['features']).sort_values(ascending=False).head(5)
            fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
            bars = sns.barplot(x=importance.values, y=importance.index, ax=ax_imp, palette='magma')
            
            # Add value to each bar
            for i, (idx, value) in enumerate(importance.items()):
                ax_imp.text(value + 0.002, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
            
            ax_imp.set_title(f'Top 5 Feature Importance - {vehicle_type.capitalize()}')
            ax_imp.set_xlabel('Importance Score')
            st.pyplot(fig_imp)
    
    def display_tree_visualization(vehicle_type, data):
        st.subheader(f"Decision Tree Visualization - {vehicle_type.capitalize()}")
        
        model = data['model']
        le = data['le']
        features = data['features']
        
        if model is None:
            st.error(f"{vehicle_type.capitalize()} model not available. Ensure model was successfully trained.")
            return
        
        st.info(f"{vehicle_type.capitalize()} Random Forest has {len(model.estimators_)} decision trees. Visualization below shows 1 sample tree.")
        
        try:
            # Get first tree (index 0) as sample
            sample_tree = model.estimators_[0]
            
            # Create tree visualization with larger size for readability
            fig, ax = plt.subplots(figsize=(25, 15))
            plot_tree(sample_tree, 
                     feature_names=features,
                     class_names=le.classes_,
                     filled=True,
                     rounded=True,
                     fontsize=10,
                     ax=ax)
            
            # Post-process: Convert decimal hours to HH:MM
            import re
            for text_obj in ax.texts:
                text_str = text_obj.get_text()
                if text_str:
                    # Find decimal hour pattern only if preceded by "Jam" (feature name)
                    # Pattern: "Jam ... <= 11.875" or "Jam ... >= 15.5"
                    modified_text = re.sub(
                        r'(Jam [A-Za-z\s]+[<>=]+\s)(\d{1,2}\.\d+)',
                        lambda m: f"{m.group(1)}{decimal_to_hhmm(float(m.group(2)))}" if 0 <= float(m.group(2)) <= 23.999 else m.group(0),
                        text_str
                    )
                    if modified_text != text_str:
                        text_obj.set_text(modified_text)
            
            ax.set_title(f"Sample Decision Tree #1 - {vehicle_type.capitalize()} Model\n(From {len(model.estimators_)} trees in Random Forest)", fontsize=16, fontweight='bold')
            
            st.pyplot(fig, use_container_width=True)
            
            # Display tree statistics
            st.markdown("---")
            st.markdown("#### Tree Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tree Depth", sample_tree.get_depth())
            
            with col2:
                st.metric("Node Count", sample_tree.tree_.node_count)
            
            with col3:
                st.metric("Leaf Count", np.sum(sample_tree.tree_.children_left == -1))
            
            # Other trees information
            st.markdown("---")
            st.markdown("#### Other Trees Information in Random Forest")
            
            tree_stats = []
            for i, tree in enumerate(model.estimators_[:5]):  # Display first 5 trees
                tree_stats.append({
                    'Tree #': i + 1,
                    'Depth': tree.get_depth(),
                    'Node Count': tree.tree_.node_count,
                    'Leaf Count': np.sum(tree.tree_.children_left == -1)
                })
            
            df_tree_stats = pd.DataFrame(tree_stats)
            st.dataframe(df_tree_stats, use_container_width=True)
            
            # TREE EXPLANATION SECTION
            st.markdown("---")
            st.markdown("#### 📚 How to Read Decision Trees")
            
            with st.expander("📝 Note: Tree Structure After Fine-Tuning", expanded=True):
                st.markdown(f"""
                **Model Parameters Used:**
                
                - **n_estimators = 150** (number of trees in ensemble)
                - **max_depth = 15** (maximum tree depth limit)
                - **min_samples_leaf = 3** (minimum samples per decision leaf)
                
                **Effect of These Parameters on Trees:**
                ✅ **Simpler**: Trees cannot grow too deep
                ✅ **More Robust**: Each leaf has minimum 3 samples
                ✅ **Optimal Balance**: Good generalization on new data
                """)
            
            with st.expander("🔍 Tree Structure Explanation", expanded=True):
                st.markdown("""
                **Each Node (Box) on the tree contains the following information:**
                
                1. **Condition/Question** (above line): Question used to split data
                   - Example: `Jumlah Motor Weekday <= 45.5` (is motorcycle weekday count <= 45.5?)
                   
                2. **Gini Index** (Gini value): Measure of disorder or variation in node
                   - Smaller Gini → more homogeneous data (more certain decision)
                   - Larger Gini → more mixed data (still much uncertainty)
                
                3. **Samples**: Number of data entering this node
                   - Example: `samples = 25` means 25 data points in this node
                
                4. **Class Distribution** (color and bar height): Proportion of each class in node
                   - Red, blue, green bars represent different classes (Low, Medium, High)
                   - Larger bar of a color → more data from that class
                
                5. **Node Background Color**:
                   - **Light color** → Majority class in that node
                   - Example: If red dominant = "Low" class
                
                **How to Follow Tree from Root to Leaf:**
                - Start from ROOT NODE (top)
                - If condition TRUE (yes) → Follow left child
                - If condition FALSE (no) → Follow right child
                - Stop at LEAF (bottommost node without branches)
                - Leaf becomes final prediction
                """)
            
            # SPECIFIC EXPLANATION FOR VEHICLE TYPE
            st.markdown("---")
            st.markdown(f"#### 🎯 Specific Explanation for {vehicle_type.capitalize()} Model")
            
            with st.expander(f"📖 Example Trace Path - {vehicle_type.capitalize()} Model", expanded=True):
                if vehicle_type.lower() == 'motorcycle':
                    st.markdown("""
                    **EXAMPLE: Predicting Potential Class for Motorcycle Parking Location**
                    
                    Suppose we have motorcycle parking location data with characteristics:
                    - Motorcycle Weekday Count: 50 vehicles
                    - Motorcycle Weekend Count: 35 vehicles
                    - Motorcycle Peak Weekday Hour: 10.5 hours
                    - Motorcycle Peak Weekend Hour: 9.0 hours
                    
                    **STEPS TO FOLLOW TREE:**
                    
                    1️⃣ **At ROOT Node**
                       - Question: "Number of Motorcycles (Weekday) <= X.X?" 
                       - Our data: 50 vehicles
                       - If 50 ≤ root threshold → Follow LEFT
                       - If 50 > root threshold → Follow RIGHT
                    
                    2️⃣ **At Next Node**
                       - New question appears, e.g.: "Peak Hours for Motorcycles (Weekend) <= Y.Y?"
                       - Evaluate this condition with our data
                       - Choose LEFT or RIGHT based on result
                    
                    3️⃣ **Continue until reaching LEAF**
                       - Leaf displays final predicted class
                       - Example: "value = [5, 2, 18]" means:
                         - 5 samples from "Low" class
                         - 2 samples from "Medium" class
                         - 18 samples from "High" class
                       - **PREDICTION = Class with most values = "High"**
                    
                    **RESULT INTERPRETATION:**
                    - Location with these characteristics predicted to have **HIGH** potential
                    - Meaning: motorcycle parking revenue at this location is estimated high
                    - Recommendation: Apply higher tariff (Rp3000 for motorcycles)
                    """)
                else:
                    st.markdown("""
                    **EXAMPLE: Predicting Potential Class for Car Parking Location**
                    
                    Suppose we have car parking location data with characteristics:
                    - Car Weekday Count: 120 vehicles
                    - Car Weekend Count: 95 vehicles
                    - Car Peak Weekday Hour: 12.5 hours
                    - Car Peak Weekend Hour: 11.0 hours
                    
                    **STEPS TO FOLLOW TREE:**
                    
                    1️⃣ **At ROOT Node**
                       - Question: "Jumlah Mobil Weekday <= X.X?"
                       - Our data: 120 vehicles
                       - If 120 ≤ root threshold → Follow LEFT
                       - If 120 > root threshold → Follow RIGHT
                    
                    2️⃣ **At Next Node**
                       - New question: "Jam Sedang Mobil Weekday <= Y.Y?" or similar
                       - Evaluate condition with our data
                       - Choose LEFT or RIGHT based on result
                    
                    3️⃣ **Continue until reaching LEAF**
                       - Leaf displays final predicted class
                       - Example: "value = [3, 8, 25]" means:
                         - 3 samples from "Low" class
                         - 8 samples from "Medium" class
                         - 25 samples from "High" class
                       - **PREDICTION = Class with most values = "High"**
                    
                    **RESULT INTERPRETATION:**
                    - Location with these characteristics predicted to have **HIGH** potential
                    - Meaning: car parking revenue at this location is estimated high
                    - Recommendation: Apply higher tariff (Rp5000 for cars)
                    """)
            
            # GINI INDEX EXPLANATION
            st.markdown("---")
            with st.expander("📊 Gini Index Explanation & Interpretation", expanded=False):
                st.markdown("""
                **What is Gini Index?**
                
                Gini Index is a measure of **impurity** in data within one node.
                
                **Gini Formula:** 
                Gini = 1 - Σ(p_i)²
                Where p_i is the proportion of class i in the node
                
                **Gini Values:**
                - **Gini = 0**: All data in node is SAME class → Perfect/Pure
                - **Gini = 0.5**: Data WELL MIXED between classes → Uncertain
                - **Gini near 1**: Data very MIXED → Very uncertain
                
                **Examples:**
                1. **Node with 100 data, all "High" class:**
                   - p(High) = 1.0, p(Medium) = 0, p(Low) = 0
                   - Gini = 1 - (1² + 0² + 0²) = 0 ✓ Perfect!
                
                2. **Node with 100 data, 33-33-34 distribution:**
                   - p(each class) ≈ 0.33
                   - Gini = 1 - (0.33² + 0.33² + 0.34²) ≈ 0.666 → Very mixed
                
                **Implications for Trees:**
                - Node with low Gini → CLEAR decision (easy to distinguish classes)
                - Node with high Gini → AMBIGUOUS decision (hard to distinguish)
                - Tree continues splitting until Gini is minimal
                """)
            
            # IMPORTANT FEATURES EXPLANATION
            st.markdown("---")
            with st.expander("🎯 Which Features Are Most Important in This Tree?", expanded=False):
                st.markdown(f"""
                **Feature Used at ROOT Node** is the **MOST IMPORTANT** for prediction!
                
                Why? Because:
                1. Root feature divides data into two most different groups (lowest Gini)
                2. All data MUST pass through root node first
                3. Division at root node has greatest impact on final result
                
                **Decision Tree Strategy:**
                - Tree selects feature that gives **best data separation** at each level
                - Numeric features selected by finding **optimal threshold** (boundary value)
                - Threshold chosen to **minimize Gini** in child nodes
                
                **Practical Implications:**
                If feature "Number of {vehicle_type} (Weekday)" is at root:
                → Weekday vehicle count VERY important in determining tariff potential
                → For better prognosis, focus on weekday vehicle count data
                """)
            
        except Exception as e:
            st.error(f"Error visualizing tree: {e}")
            
    with tab_motorcycle:
        display_model_results('Motorcycle', models_data['motorcycle'])
        
    with tab_car:
        display_model_results('Car', models_data['car'])
    
    with tab_training_data:
        st.subheader("📊 Training vs Testing Data Visualization")
        st.info("This section displays data split for Random Forest model training process")
        
        # General info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df_processed))
        with col2:
            st.metric("Training Data (80%)", int(len(df_processed) * 0.8))
        with col3:
            st.metric("Testing Data (20%)", int(len(df_processed) * 0.2))
        
        st.markdown("---")
        
        # Pie chart visualization
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📈 Train vs Test Data Proportion")
            fig, ax = plt.subplots(figsize=(6, 5))
            sizes = [len(df_processed) * 0.8, len(df_processed) * 0.2]
            labels = ['Training (80%)', 'Testing (20%)']
            colors = ['#2E86AB', '#A23B72']
            explode = (0.05, 0)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
                  shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title('Data Split for Training', fontsize=12, fontweight='bold')
            st.pyplot(fig, use_container_width=True)
        
        with col_right:
            st.markdown("#### 📊 Training Data Distribution per Class")
            
            # Training data for Motorcycle
            df_train_indices = list(range(int(len(df_processed) * 0.8)))
            df_train = df_processed.iloc[df_train_indices]
            
            class_counts_motorcycle = df_train['Class_Motorcycle'].value_counts()
            class_counts_car = df_train['Class_Car'].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            class_counts_motorcycle.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#FFC93C', '#4ECDC4'])
            ax1.set_title('Motorcycle (Training Set)', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            class_counts_car.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#FFC93C', '#4ECDC4'])
            ax2.set_title('Car (Training Set)', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Training data detail table
        st.markdown("#### 📋 Training Data Detail (80% of Total)")
        with st.expander("View training data (click to expand)", expanded=False):
            st.dataframe(df_train, use_container_width=True, height=400)
        
        # Testing data detail table
        st.markdown("#### 📋 Testing Data Detail (20% of Total)")
        df_test_indices = list(range(int(len(df_processed) * 0.8), len(df_processed)))
        df_test = df_processed.iloc[df_test_indices]
        
        with st.expander("View testing data (click to expand)", expanded=False):
            st.dataframe(df_test, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Explanation
        st.markdown("#### 💡 Explanation")
        with st.expander("Why need train-test split?", expanded=True):
            st.markdown("""
            **Train-Test Split** is a technique to objectively evaluate models:
            
            1. **Training Data (80%):** Used to train the model
               - Model learns patterns from this data
               - Used to update model parameters
            
            2. **Testing Data (20%):** Used to evaluate the model
               - Data NEVER seen by model during training
               - Shows model's generalization ability
               - Measures performance on new data
            
            3. **Why 80:20?**
               - Industry standard for balancing data amount and evaluation
               - Sufficient training data to learn patterns
               - Sufficient testing data for reliable evaluation
            
            4. **Stratification:**
               - Data shuffled and split proportionally
               - Ensures class distribution in train and test is same
            """)

    
    with tab_training:
        st.subheader("📈 Model Training Process Visualization")
        st.info("Charts below show how model accuracy improves as number of trees in Random Forest increases.")
        
        def display_training_curves(vehicle_type, data):
            st.markdown(f"### {vehicle_type.capitalize()}")
            
            model = data['model']
            metrics = data.get('training_metrics', {})
            
            if model is None or not metrics:
                st.error(f"Training data not available for {vehicle_type}. Model may not have been successfully trained.")
                return
            
            try:
                tree_counts = metrics.get('tree_counts', [])
                train_scores = metrics.get('train_scores', [])
                test_scores = metrics.get('test_scores', [])
                
                if not tree_counts or not train_scores or not test_scores:
                    st.warning(f"Training metric data empty for {vehicle_type}.")
                    return
                
                # Create dataframe for plotting
                df_metrics = pd.DataFrame({
                    'Tree Count': tree_counts,
                    'Training Accuracy': train_scores,
                    'Testing Accuracy': test_scores
                })
                
                # Plotting with matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(df_metrics['Tree Count'], df_metrics['Training Accuracy'], 
                       marker='o', linewidth=2, label='Training Accuracy', color='#2E86AB')
                ax.plot(df_metrics['Tree Count'], df_metrics['Testing Accuracy'], 
                       marker='s', linewidth=2, label='Testing Accuracy', color='#A23B72')
                
                ax.fill_between(df_metrics['Tree Count'], 
                               df_metrics['Training Accuracy'], 
                               df_metrics['Testing Accuracy'],
                               alpha=0.2, color='gray', label='Gap (Overfitting)')
                
                ax.set_xlabel('Number of Trees in Random Forest', fontsize=12, fontweight='bold')
                ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax.set_title(f'Learning Curve - {vehicle_type.capitalize()} Model\n(Accuracy Improvement with Tree Addition)', 
                           fontsize=13, fontweight='bold')
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_ylim([0, 1.05])
                ax.set_xlim([0, 160])  # Adjusted to 150 trees
                
                st.pyplot(fig, use_container_width=True)
                
                # Display metrics table
                st.markdown("#### Accuracy Table per Tree Count")
                df_display = df_metrics.copy()
                df_display['Training Accuracy'] = (df_display['Training Accuracy'] * 100).round(2).astype(str) + '%'
                df_display['Testing Accuracy'] = (df_display['Testing Accuracy'] * 100).round(2).astype(str) + '%'
                st.dataframe(df_display, use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.markdown("#### 📊 Training Statistics Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    final_train_acc = train_scores[-1]
                    st.metric("Final Training Accuracy", f"{final_train_acc*100:.2f}%")
                
                with col2:
                    final_test_acc = test_scores[-1]
                    st.metric("Final Testing Accuracy", f"{final_test_acc*100:.2f}%")
                
                with col3:
                    gap = final_train_acc - final_test_acc
                    st.metric("Gap (Overfitting)", f"{gap*100:.2f}%", 
                             delta=f"{'⚠️ High' if gap > 0.1 else '✅ Normal' if gap > 0 else '✓ Good'}")
                
                with col4:
                    improvement = train_scores[-1] - train_scores[0]
                    st.metric("Training Improvement", f"{improvement*100:.2f}%")
                
                # Explanation
                st.markdown("---")
                st.markdown("#### 💡 Chart Interpretation")
                
                with st.expander("What does this chart mean?", expanded=True):
                    st.markdown(f"""
                    **Blue Line (Training Accuracy):** Model accuracy on training data
                    - Tends to **rise/stabilize** because model has seen this data
                    - If continues rising → Model still learning
                    - If plateau → Model has converged
                    
                    **Purple Line (Testing Accuracy):** Model accuracy on testing data (unseen)
                    - This is most important for model evaluation
                    - Shows model's generalization ability
                    
                    **Gray Area (Gap):** Difference between Training and Testing
                    - Smaller gap → Better model generalization
                    - Large gap → Indicates **overfitting** (model memorizes training data)
                    - This model: Gap = {final_train_acc - final_test_acc:.4f}
                    
                    **Curve Trend:**
                    - Both curves rise together → Model learning well ✅
                    - Testing rises, Training stable → Normal curve ✅
                    - Testing falls when Training rises → Overfitting ⚠️
                    
                    **Optimal Tree Count:**
                    - Usually seen from highest testing accuracy
                    - For {vehicle_type}: **{tree_counts[test_scores.index(max(test_scores))]} trees** gives best accuracy
                    - But 150 trees is already good enough (bias-variance trade-off)
                    """)
            
            except Exception as e:
                st.error(f"Error visualizing {vehicle_type} training: {e}")
        
        col_motorcycle, col_car = st.columns(2)
        
        with col_motorcycle:
            display_training_curves('Motorcycle', models_data['motorcycle'])
        
        with col_car:
            display_training_curves('Car', models_data['car'])
    
    with tab_tree:
        st.markdown("### 🏍️ Motorcycle Tree")
        display_tree_visualization('Motorcycle', models_data['motorcycle'])
        
        st.markdown("---")
        
        st.markdown("### 🚗 Car Tree")
        display_tree_visualization('Car', models_data['car'])

    with tab_recommendations:
        st.subheader("📑 Progressive Tariff Policy Recommendations Table")
        st.info("Displays tariff potential classification results for each parking point based on trained model.")
        
        model_motorcycle = models_data['motorcycle']['model']
        le_motorcycle = models_data['motorcycle']['le']
        features_motorcycle = models_data['motorcycle']['features']
        X_all_motorcycle = models_data['motorcycle']['X_all']
        
        model_car = models_data['car']['model']
        le_car = models_data['car']['le']
        features_car = models_data['car']['features']
        X_all_car = models_data['car']['X_all']
        
        if model_motorcycle is None or model_car is None:
            st.warning("Models not available or could not be trained. Ensure Modeling page has run with sufficient data.")
        else:
            try:
                # Predict for all data
                y_pred_m_enc = model_motorcycle.predict(X_all_motorcycle)
                df_result = pd.DataFrame(X_all_motorcycle).reset_index(drop=True)
                
                # Add Parking Point column from original df_processed
                df_result.insert(0, 'Parking Point', df_processed['Location Point'].values)
                
                # Get Indonesian class labels from model
                motorcycle_classes_raw = le_motorcycle.inverse_transform(y_pred_m_enc)
                
                # Translate to English for display
                df_result['Motorcycle Potential (Display)'] = [CLASS_TRANSLATION.get(c, c) for c in motorcycle_classes_raw]
                
                # Get base tariff using Indonesian labels
                df_result['Motorcycle Base Tariff'] = [f"Rp{tariff_mapping['Motorcycle'].get(c, 0):,.0f}" for c in motorcycle_classes_raw]
                
                # Get progressive tariff (using peak hour assumption of 12:00)
                df_result['Motorcycle Progressive Tariff'] = [
                    f"Rp{calculate_progressive_tariff('Motorcycle', c, 12.0):,.0f}" 
                    for c in motorcycle_classes_raw
                ]
                
                y_pred_c_enc = model_car.predict(X_all_car)
                car_classes_raw = le_car.inverse_transform(y_pred_c_enc)
                
                # Translate to English for display
                df_result['Car Potential (Display)'] = [CLASS_TRANSLATION.get(c, c) for c in car_classes_raw]
                
                # Get base tariff using Indonesian labels
                df_result['Car Base Tariff'] = [f"Rp{tariff_mapping['Car'].get(c, 0):,.0f}" for c in car_classes_raw]
                
                # Get progressive tariff (using peak hour assumption of 12:00)
                df_result['Car Progressive Tariff'] = [
                    f"Rp{calculate_progressive_tariff('Car', c, 12.0):,.0f}" 
                    for c in car_classes_raw
                ]
                
                # Output columns with English display
                output_columns = ['Parking Point', 'Motorcycle Potential (Display)', 'Motorcycle Base Tariff', 
                               'Motorcycle Progressive Tariff', 'Car Potential (Display)', 'Car Base Tariff', 
                               'Car Progressive Tariff']
                
                st.markdown("### Tariff Recommendations Summary (First 10 Rows)")
                st.dataframe(df_result[output_columns].head(10), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Classification Distribution Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Motorcycle Distribution")
                    motorcycle_dist = df_result['Motorcycle Potential (Display)'].value_counts()
                    st.bar_chart(motorcycle_dist)
                
                with col2:
                    st.markdown("#### Car Distribution")
                    car_dist = df_result['Car Potential (Display)'].value_counts()
                    st.bar_chart(car_dist)
                
                st.markdown("---")
                st.markdown("### Complete Recommendations Table")
                st.dataframe(df_result[output_columns], use_container_width=True)
                
                # Download option
                csv = df_result[output_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="Parking_Tariff_Recommendations.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Failed to create recommendations: {e}")


# Module 4: Map & Simulation (Changed: Dropdown for Location Selection)
def display_map_and_simulation(df_long, map_center, models_data, df_spatial):
    st.header("4️⃣ Map & Progressive Tariff Simulation")
    st.markdown("---")
    
    st.subheader("Static Tariff Potential Prediction Map")
    st.info("Select map style: Satellite or StreetMap. Point colors standardized to focus on popup and simulation.")

    # Default map: OpenStreetMap (Additional TileLayer still included in map)
    m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')

    # Add additional TileLayer so users can switch in LayerControl if desired
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri Satellite', attr='Esri').add_to(m)
    
    FIXED_COLOR = 'darkblue'

    fg_all = folium.FeatureGroup(name='All Parking Points', show=True)
    features_search = []
    
    # 1. CircleMarker (Round Points, Single Color) - VISUAL
    for index, row in df_spatial.iterrows():
        point = row['Location Point']
        lat, lon = row['Latitude'], row['Longitude']
        
        # Get static classification result from df_processed (already in df_long)
        motorcycle_data = df_long[(df_long['titik'] == point) & (df_long['jenis_kendaraan'] == 'Motorcycle')]
        car_data = df_long[(df_long['titik'] == point) & (df_long['jenis_kendaraan'] == 'Car')]

        # Get first row (should be only one)
        motorcycle_row = motorcycle_data.iloc[0] if not motorcycle_data.empty else None
        car_row = car_data.iloc[0] if not car_data.empty else None

        # Handle missing data case - translate Indonesian class labels to English for display
        motorcycle_class_raw = motorcycle_row['kategori_load'] if motorcycle_row is not None else 'N/A'
        motorcycle_class_display = CLASS_TRANSLATION.get(motorcycle_class_raw, motorcycle_class_raw)
        motorcycle_tariff = int(motorcycle_row['prediksi_tarif']) if motorcycle_row is not None else 0
        
        car_class_raw = car_row['kategori_load'] if car_row is not None else 'N/A'
        car_class_display = CLASS_TRANSLATION.get(car_class_raw, car_class_raw)
        car_tariff = int(car_row['prediksi_tarif']) if car_row is not None else 0

        popup_html = f"""
        <div style="font-size:13px; font-family:sans-serif;">
            <b>Parking Point:</b> {point}<br>
            <b>Coordinates:</b> {lat:.4f}, {lon:.4f}<hr>
            <b>Motorcycle:</b> {motorcycle_class_display} Potential (Base Tariff: Rp{motorcycle_tariff:,})<br>
            <b>Car:</b> {car_class_display} Potential (Base Tariff: Rp{car_tariff:,})<br>
        </div>
        """

        marker = folium.CircleMarker(
            location=[lat, lon], 
            radius=6, 
            color=FIXED_COLOR, 
            fill=True, 
            fill_color=FIXED_COLOR, 
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300), 
            tooltip=point 
        )
        marker.add_to(fg_all)

        features_search.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {"name": point}, 
        })

    fg_all.add_to(m)
    geojson_layer_search = folium.GeoJson(
        {"type": "FeatureCollection", "features": features_search},
        name="Search Points (Hidden)",
        style_function=lambda x: {'opacity': 0, 'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
    ).add_to(m)

    Search(
        layer=geojson_layer_search, 
        search_label="name", 
        placeholder="Search parking point name...", 
        collapsed=False, 
        position="topleft", 
        geom_type="Point",
    ).add_to(m)
    
    # Layer Control and other Plugins
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add Fullscreen and MiniMap for better UX
    Fullscreen(position='topright').add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    
    folium_static(m, width=None, height=550)

    # --- Simulation Section (Dynamic Input from Dropdown) ---
    st.subheader("Tariff Potential Prediction Simulation (What-If Analysis)")
    st.markdown("**1. Select Parking Location** (Static Aggregate Location Input will be used as *Default*)")
    
    # Location Dropdown Selection (Replaces map click)
    selected_point = st.selectbox(
        "Select Parking Point for Simulation:", 
        df_spatial['Location Point'].unique().tolist(), 
        key='sim_point_select'
    )
    
    # Detect location change and reset input to new location default
    if 'last_selected_point' not in st.session_state:
        st.session_state.last_selected_point = selected_point
    
    if st.session_state.last_selected_point != selected_point:
        # Location changed, reset all inputs to new location default
        st.session_state.last_selected_point = selected_point
        st.session_state.sim_vehicle_type_override = 'Motorcycle'
        st.session_state.sim_day_override = 'Weekday'
        st.session_state.sim_hour_override = None
        st.session_state.sim_count_override = None
    
    # Get aggregate data from selected point
    default_data = df_spatial[df_spatial['Location Point'] == selected_point].iloc[0]
    
    # Determine default Peak Hour (take average from Weekday Motorcycle)
    default_hour_val = default_data.get('Peak Hours for Motorcycles (Weekday)', 9.0)

    with st.expander(f"⚙️ Set Dynamic Scenario for {selected_point} (Progressive Tariff) ⚙️"):
        # Provide more informative default information
        st.markdown(f"**Default Location Aggregate Data:**")
        st.markdown(f"* Motorcycle WD Count: **{default_data.get('Number of Motorcycles (Weekday)', 0):.0f}** units, Car WD: **{default_data.get('Number of Cars (Weekday)', 0):.0f}** units")
        st.markdown(f"* Motorcycle WD Peak Hour: **{default_hour_val:.2f}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Determine vehicle type and day (can be overridden by session state)
        vehicle_type_default = st.session_state.get('sim_vehicle_type_override', 'Motorcycle')
        day_default = st.session_state.get('sim_day_override', 'Weekday')
        
        with col1: 
            vehicle_type = st.selectbox("Vehicle Type", ['Motorcycle', 'Car'], 
                               index=['Motorcycle', 'Car'].index(vehicle_type_default),
                               key='sim_vehicle_type', help="Select vehicle type to simulate.")
        with col2: 
            day = st.selectbox("Day", ['Weekday', 'Weekend'], 
                              index=['Weekday', 'Weekend'].index(day_default),
                              key='sim_day', help="Select whether weekday or weekend.")
        
        # Update default based on currently selected vehicle type and day
        vehicle_plural = 'Motorcycles' if vehicle_type == 'Motorcycle' else 'Cars'
        hour_key = f'Peak Hours for {vehicle_plural} ({day})'
        count_key = f'Number of {vehicle_plural} ({day})'
        default_hour_current = default_data.get(hour_key, 9.0)
        default_count_current = default_data.get(count_key, 100)
        
        with col3: 
            # Use default hour value based on vehicle type and day
            hour_for_time_input = st.session_state.get('sim_hour_override', default_hour_current)
            if hour_for_time_input is None:
                hour_for_time_input = default_hour_current
            try:
                time_obj_default = datetime.time(int(hour_for_time_input // 1), int((hour_for_time_input % 1) * 60))
            except (ValueError, TypeError):
                time_obj_default = datetime.time(9, 0) # Fallback to 9 AM
                
            time_obj = st.time_input(
                "Hour (HH:MM)", 
                value=time_obj_default, 
                step=datetime.timedelta(minutes=1), # Allow per-minute input
                key=f'sim_hour_time_{selected_point}_{vehicle_type}_{day}',  # Unique key per location/type/day
                help="Parking time (24-hour format)."
            )
            decimal_hour_input = time_to_decimal_hour(time_obj) 
            st.caption(f"Model Hour Value: **{decimal_hour_input:.2f}**") 
            
        with col4: 
            # Use vehicle count at selected location as default
            count_for_input = st.session_state.get('sim_count_override', default_count_current)
            if count_for_input is None:
                count_for_input = default_count_current
            count_input = st.number_input(f"{vehicle_type} Count (Estimate)", 
                                          min_value=1, max_value=500, 
                                          value=int(count_for_input), 
                                          key=f'sim_count_{selected_point}_{vehicle_type}_{day}',  # Unique key per location/type/day
                                          help=f"Estimated {vehicle_type} count parking at that hour.")
        with col5: 
            st.markdown("<br>", unsafe_allow_html=True) 
            submitted = st.button("Predict Result 🚀", key='sim_submit', type='primary')

        if submitted:
            data = models_data['motorcycle'] if vehicle_type == 'Motorcycle' else models_data['car']
            
            # Prepare baseline data from selected location (default_data)
            # Convert to format matching model features - need to map back to Indonesian column names
            baseline_row = default_data[data['features']].to_dict() if all(f in default_data.index for f in data['features']) else None
            
            # Auto-standardize all count columns for this type with input count
            if baseline_row is not None:
                vehicle_plural = 'Motorcycles' if vehicle_type == 'Motorcycle' else 'Cars'
                count_cols = [c for c in baseline_row.keys() if c.startswith(f'Number of {vehicle_plural}')]
                for c in count_cols:
                    baseline_row[c] = count_input

            # Call Random Forest prediction function
            pred_class, confidence, top_gain, proba_dict, hour_explanation, estimated_revenue, threshold_class = predict_single_input(
                vehicle_type, day, decimal_hour_input, count_input, 
                data['model'], data['le'], data['X_ref'], data['quantile_thresholds'],
                baseline_data=baseline_row
            )
            
            if not isinstance(pred_class, str) or "Error" in pred_class or "Model Failed" in pred_class:
                st.error(f"Simulation Failed: {pred_class}. Ensure model is trained (Check Modeling page).")
            else:
                # Translate Indonesian class label to English for display
                pred_class_display = CLASS_TRANSLATION.get(pred_class, pred_class)
                
                # >>> Apply Progressive Tariff Logic
                recommended_base_tariff = tariff_mapping[vehicle_type].get(pred_class, 0)
                recommended_progressive_tariff = calculate_progressive_tariff(vehicle_type, pred_class, decimal_hour_input)
                
                st.markdown("---")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("RF Model Prediction", f"{pred_class_display.upper()} Potential", delta=f"Confidence: {confidence:.3f}")
                col_res2.metric("Base Tariff Recommendation", f"Rp{recommended_base_tariff:,}", delta=f"Class: {pred_class_display}")
                col_res3.metric("PROGRESSIVE Tariff", f"Rp{recommended_progressive_tariff:,}", delta=f"+Rp{recommended_progressive_tariff - recommended_base_tariff:,}")
                
                st.markdown("---")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**Time Logic Explanation:**")
                    st.info(hour_explanation)
                    
                    st.markdown("**Top 3 Contributors (Local Gain):**")
                    if isinstance(top_gain, pd.Series):
                        for f in top_gain.index: st.markdown(f"- **{f}** (Main prediction driver)")
                    st.caption(f"All Class Probabilities: {proba_dict}")

                with col_info2:
                    st.markdown("**Progressive Logic Applied:**")
                    st.warning(f"If **Hour > 9.00**, **{pred_class_display}** Tariff increased by **Rp{recommended_progressive_tariff - recommended_base_tariff:,}** from base tariff.")


# =================================================================
# === MAIN APPLICATION EXECUTION ===
# =================================================================

# Load Data and Models
df_clean, df_processed, df_spatial, hour_cols, df_raw, quantile_boundaries = load_and_preprocess_data(FILE_PATH)
if df_processed is None: 
    st.error(f"Failed to load or process data. Ensure file '{FILE_PATH}' exists with correct format and contains spatial columns (Location Point, Latitude, Longitude).")
    st.stop()

models_data = train_models(df_clean, hour_cols)  # Pass clean data (no features/labels yet)

# --- Static Prediction for Map ---
df_long = pd.DataFrame()
try:
    # Ensure models trained before predicting
    if models_data['motorcycle']['model']:
        df_processed['Pred_Class_Motorcycle'] = models_data['motorcycle']['le'].inverse_transform(models_data['motorcycle']['model'].predict(models_data['motorcycle']['X_all']))
    else:
        df_processed['Pred_Class_Motorcycle'] = df_processed['Class_Motorcycle'] # Fallback to raw data class
        
    if models_data['car']['model']:
        df_processed['Pred_Class_Car'] = models_data['car']['le'].inverse_transform(models_data['car']['model'].predict(models_data['car']['X_all']))
    else:
        df_processed['Pred_Class_Car'] = df_processed['Class_Car'] # Fallback to raw data class

    df_mapping = df_spatial.dropna(subset=['Latitude', 'Longitude'])

    df_motorcycle_map = df_mapping.copy()
    df_motorcycle_map['jenis_kendaraan'] = 'Motorcycle'
    df_motorcycle_map['kategori_load'] = df_processed['Pred_Class_Motorcycle']
    df_motorcycle_map['prediksi_tarif'] = df_processed['Pred_Class_Motorcycle'].apply(lambda x: tariff_mapping['Motorcycle'].get(x, 0))
    df_motorcycle_map.rename(columns={'Location Point': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_car_map = df_mapping.copy()
    df_car_map['jenis_kendaraan'] = 'Car'
    df_car_map['kategori_load'] = df_processed['Pred_Class_Car']
    df_car_map['prediksi_tarif'] = df_processed['Pred_Class_Car'].apply(lambda x: tariff_mapping['Car'].get(x, 0))
    df_car_map.rename(columns={'Location Point': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_long = pd.concat([df_motorcycle_map, df_car_map], ignore_index=True).dropna(subset=['latitude', 'longitude']) 

    if not df_long.empty:
        map_center = [df_long['latitude'].mean(), df_long['longitude'].mean()]
    else:
        map_center = [-7.4168, 109.2155] # Default Banyumas Coordinates
        
except Exception as e:
    st.error(f"Error preparing map/static prediction data: {e}")
    map_center = [-7.4168, 109.2155] # Default Banyumas Coordinates

st.title("🅿️ Parking Tariff Potential Analysis")
st.caption("Dashboard for Random Forest-based parking tariff potential classification modeling.")
st.markdown("---")

# --- Main Sidebar Navigation (USING OPTION MENU WITH STYLE) ---
with st.sidebar:
    st.markdown("---")
    
    page = option_menu(
        menu_title="Analytics Dashboard 📊", 
        options=["Data Table", "Visualization", "Modeling", "Map & Simulation"],
        icons=["table", "bar-chart", "calculator", "geo-alt"], 
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# --- Display Logic ---
if page == "Data Table":
    display_data_table(df_raw, df_processed)

elif page == "Visualization":
    display_visualization(df_processed, quantile_boundaries, hour_cols)

elif page == "Modeling":
    display_modeling(df_processed, models_data)

elif page == "Map & Simulation":
    display_map_and_simulation(df_long, map_center, models_data, df_spatial)
