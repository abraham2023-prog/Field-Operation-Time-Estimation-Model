import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="🚜 Field Operation Time Estimator",
    page_icon="🚜",
    layout="wide"
)

# Initialize session state with sample data if not exists
if 'field_data' not in st.session_state:
    sample_data = {
        'field_size': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'soil_type': ['clay', 'sandy', 'loam', 'clay', 'sandy', 'loam', 'clay', 'sandy', 'loam', 'clay', 'sandy', 'loam', 'clay'],
        'implement': ['plow', 'harrow', 'planter', 'plow', 'harrow', 'planter', 'plow', 'harrow', 'planter', 'plow', 'harrow', 'planter', 'plow'],
        'horsepower': [100, 120, 140, 100, 120, 140, 100, 120, 140, 100, 120, 140, 100],
        'operation_type': ['plowing', 'harrowing', 'planting', 'plowing', 'harrowing', 'planting', 
                          'plowing', 'harrowing', 'planting', 'plowing', 'harrowing', 'planting', 'plowing'],
        'operation_time': [2.5, 2.8, 3.5, 3.1, 3.4, 4.2, 3.7, 4.0, 4.9, 4.3, 4.6, 5.6, 5.0]
    }
    st.session_state.field_data = pd.DataFrame(sample_data)

# Preprocessing functions
def preprocess_data(df):
    """Preprocess the data for model training"""
    df_processed = df.copy()
    
    # One-hot encode categorical variables
    categorical_cols = ['soil_type', 'implement', 'operation_type']
    for col in categorical_cols:
        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
        df_processed = pd.concat([df_processed, dummies], axis=1)
        df_processed.drop(col, axis=1, inplace=True)
    
    return df_processed

# Model training functions
def train_models(X, y):
    """Train both linear regression and random forest models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    return {
        'Linear Regression': {'model': lr_model, 'mae': lr_mae, 'r2': lr_r2},
        'Random Forest': {'model': rf_model, 'mae': rf_mae, 'r2': rf_r2}
    }, X_test, y_test

# Dashboard Header
st.title("🚜 Field Operation Time Estimation Model")
st.markdown("### Estimate how long a machine will take to complete tasks based on field and machine parameters")

# Sidebar Controls
with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection
    model_type = st.radio(
        "Select Model Type",
        ["Linear Regression", "Random Forest"],
        help="Choose which regression model to use for predictions"
    )
    
    # Data management
    st.subheader("Data Management")
    
    # Upload new data
    uploaded_file = st.file_uploader("Upload Field Data (CSV)", type=['csv'])
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            required_cols = ['field_size', 'soil_type', 'implement', 'horsepower', 'operation_type', 'operation_time']
            if all(col in new_data.columns for col in required_cols):
                st.session_state.field_data = new_data
                st.success("Field data uploaded successfully!")
            else:
                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Download current data
    st.download_button(
        "📥 Download Current Data",
        st.session_state.field_data.to_csv(index=False),
        file_name="field_operation_data.csv"
    )
    
    # Add new data point
    with st.expander("Add New Data Point"):
        with st.form("new_data_form"):
            new_field_size = st.number_input("Field Size (ha)", min_value=0.1, value=10.0, step=0.5)
            new_soil_type = st.selectbox("Soil Type", ["clay", "sandy", "loam"])
            new_implement = st.selectbox("Implement", ["plow", "harrow", "planter", "sprayer"])
            new_horsepower = st.slider("Machine Horsepower", min_value=50, max_value=300, value=100)
            new_operation_type = st.selectbox("Operation Type", ["plowing", "harrowing", "planting", "spraying"])
            new_operation_time = st.number_input("Actual Operation Time (hours)", min_value=0.1, value=2.5, step=0.1)
            
            if st.form_submit_button("Add to Dataset"):
                new_row = {
                    'field_size': new_field_size,
                    'soil_type': new_soil_type,
                    'implement': new_implement,
                    'horsepower': new_horsepower,
                    'operation_type': new_operation_type,
                    'operation_time': new_operation_time
                }
                st.session_state.field_data = st.session_state.field_data.append(new_row, ignore_index=True)
                st.success("New data point added!")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Estimation", "Data", "Model Performance", "Visualization"])

with tab1:
    st.header("Operation Time Estimation")
    
    # Input form for prediction
    col1, col2 = st.columns(2)
    
    with col1:
        field_size = st.number_input("Field Size (ha)", min_value=0.1, value=10.0, step=0.5, key="pred_field_size")
        soil_type = st.selectbox("Soil Type", ["clay", "sandy", "loam"], key="pred_soil_type")
        implement = st.selectbox("Implement", ["plow", "harrow", "planter", "sprayer"], key="pred_implement")
    
    with col2:
        horsepower = st.slider("Machine Horsepower", min_value=50, max_value=300, value=100, key="pred_horsepower")
        operation_type = st.selectbox("Operation Type", ["plowing", "harrowing", "planting", "spraying"], key="pred_operation_type")
    
    # Prepare data for model training
    df_processed = preprocess_data(st.session_state.field_data)
    X = df_processed.drop('operation_time', axis=1)
    y = df_processed['operation_time']
    
    # Train models
    models, X_test, y_test = train_models(X, y)
    
    # Prepare input for prediction
    input_data = {
        'field_size': field_size,
        'horsepower': horsepower
    }
    
    # Add one-hot encoded columns
    for col in X.columns:
        if col.startswith('soil_type_'):
            input_data[col] = 1 if col == f'soil_type_{soil_type}' else 0
        elif col.startswith('implement_'):
            input_data[col] = 1 if col == f'implement_{implement}' else 0
        elif col.startswith('operation_type_'):
            input_data[col] = 1 if col == f'operation_type_{operation_type}' else 0
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns are in the same order as training data
    input_df = input_df[X.columns]
    
    # Make prediction
    if st.button("Estimate Operation Time", type="primary"):
        selected_model = models[model_type]['model']
        prediction = selected_model.predict(input_df)[0]
        
        st.metric("Estimated Operation Time", f"{prediction:.2f} hours")
        
        # Show confidence interval (simplified)
        mae = models[model_type]['mae']
        st.info(f"Estimation range: {max(0.1, prediction - mae):.2f} to {prediction + mae:.2f} hours")
        
        # Show model performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model MAE", f"{mae:.2f} hours")
        with col2:
            st.metric("Model R² Score", f"{models[model_type]['r2']:.3f}")

with tab2:
    st.header("Field Operation Data")
    
    st.dataframe(st.session_state.field_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Summary")
        st.write(st.session_state.field_data.describe())
    
    with col2:
        st.subheader("Data Information")
        st.write(f"Total records: {len(st.session_state.field_data)}")
        
        # Count by soil type
        soil_counts = st.session_state.field_data['soil_type'].value_counts()
        st.write("**Records by Soil Type:**")
        for soil, count in soil_counts.items():
            st.write(f"- {soil}: {count}")
        
        # Count by operation type
        op_counts = st.session_state.field_data['operation_type'].value_counts()
        st.write("**Records by Operation Type:**")
        for op, count in op_counts.items():
            st.write(f"- {op}: {count}")

with tab3:
    st.header("Model Performance Comparison")
    
    # Prepare data for model training
    df_processed = preprocess_data(st.session_state.field_data)
    X = df_processed.drop('operation_time', axis=1)
    y = df_processed['operation_time']
    
    # Train models
    models, X_test, y_test = train_models(X, y)
    
    # Display performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Linear Regression")
        st.metric("Mean Absolute Error", f"{models['Linear Regression']['mae']:.2f} hours")
        st.metric("R² Score", f"{models['Linear Regression']['r2']:.3f}")
    
    with col2:
        st.subheader("Random Forest")
        st.metric("Mean Absolute Error", f"{models['Random Forest']['mae']:.2f} hours")
        st.metric("R² Score", f"{models['Random Forest']['r2']:.3f}")
    
    # Create comparison chart
    model_names = list(models.keys())
    mae_values = [models[model]['mae'] for model in model_names]
    r2_values = [models[model]['r2'] for model in model_names]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names,
        y=mae_values,
        name='MAE (hours)',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=model_names,
        y=r2_values,
        name='R² Score',
        marker_color='lightseagreen',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis=dict(title='Model'),
        yaxis=dict(title='MAE (hours)', side='left'),
        yaxis2=dict(title='R² Score', side='right', overlaying='y'),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted values
    st.subheader("Actual vs Predicted Values")
    
    # Get predictions for both models
    lr_pred = models['Linear Regression']['model'].predict(X_test)
    rf_pred = models['Random Forest']['model'].predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=lr_pred,
        mode='markers',
        name='Linear Regression',
        marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=y_test, y=rf_pred,
        mode='markers',
        name='Random Forest',
        marker=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Operation Time',
        xaxis_title='Actual Time (hours)',
        yaxis_title='Predicted Time (hours)',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Data Visualization")
    
    # Create various visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Operation time by field size
        fig = px.scatter(
            st.session_state.field_data,
            x='field_size',
            y='operation_time',
            color='operation_type',
            size='horsepower',
            title='Operation Time by Field Size',
            labels={'field_size': 'Field Size (ha)', 'operation_time': 'Operation Time (hours)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Operation time by soil type
        fig = px.box(
            st.session_state.field_data,
            x='soil_type',
            y='operation_time',
            color='operation_type',
            title='Operation Time by Soil Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Operation time by implement
        fig = px.violin(
            st.session_state.field_data,
            x='implement',
            y='operation_time',
            color='operation_type',
            title='Operation Time by Implement',
            box=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Horsepower distribution by operation type
        fig = px.histogram(
            st.session_state.field_data,
            x='horsepower',
            color='operation_type',
            title='Horsepower Distribution by Operation Type',
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Relationship Analysis")
    fig = px.scatter_3d(
        st.session_state.field_data,
        x='field_size',
        y='horsepower',
        z='operation_time',
        color='operation_type',
        symbol='soil_type',
        title='Field Size, Horsepower vs Operation Time'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🚜 Field Operation Time Estimation Model")
st.markdown("This tool helps farmers and agricultural professionals estimate how long field operations will take based on various parameters.")
