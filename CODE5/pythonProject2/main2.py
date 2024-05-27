# Import libraries
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="US Freight Commodity Flow Survey (CFS) Model Choice Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r'D:\UTK\Project_Han\Project_5\Code5\pythonProject2\modified2_cfs_majbah_train.csv')
    return df

df = load_data()



# State abbreviations mapping
state_abbreviations = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

# Encode categorical variables
@st.cache_data
def encode_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Encode the entire dataset for training the model
df_encoded, label_encoders = encode_data(df.copy())

# Define features and target variable for the entire dataset
X = df_encoded.drop(columns=['MODE'])
y = df_encoded['MODE']


# Remap mode labels to be continuous integers starting from 0
unique_modes = sorted(y.unique())
mode_mapping = {mode: i for i, mode in enumerate(unique_modes)}
y = y.map(mode_mapping)

mode_name_mapping = {
    0: "For-hire truck",
    1: "Privately owned truck",
    2: "Parcel Service",
    3: "Air",
    4: "Water and Other modes"
}

# Train model function
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(y.unique()), eval_metric='mlogloss', use_label_encoder=False)
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    importance = best_model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df[importance_df['Importance'] >= 0.001]
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    accuracy = grid_search.best_score_

    return best_model, importance_df, accuracy, X_test, y_test

best_model, importance_df, accuracy, X_test, y_test  = train_model(X, y)


# Sidebar
with st.sidebar:
    st.title('🚚 US Freight Model Choice Dashboard')

    # Select model (only one available currently)
    model_choice = st.selectbox('Select Model', ['XGBoost'])

    # Let the state name ranks for A-Z
    sorted_orig_states = sorted(df['orig_state_full'].dropna().astype(str).unique())
    sorted_dest_states = sorted(df['dest_state_full'].dropna().astype(str).unique())

    orig_state_full = st.selectbox('Select Origin State', sorted_orig_states)
    dest_state_full = st.selectbox('Select Destination State', sorted_dest_states)

# Filter data based on origin and destination states
filtered_df = df[(df['orig_state_full'] == orig_state_full) & (df['dest_state_full'] == dest_state_full)]

# Dashboard main panel
st.title("US Freight Model Choice Dashboard")

# Column layout
col1, col2 = st.columns(2)

with col1:
    st.header("Feature Importance")
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    st.header("Model Accuracy and Evaluation")
    st.write(f"Accuracy: {accuracy:.4f}")

    # Model evaluation results
    y_pred = best_model.predict(X_test)
    non_zero_indices = y_test != 0

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
    r2 = r2_score(y_test, y_pred)

    # Create a DataFrame to display the results in a table format
    results = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²'],
        'Value': [mse, rmse, mae, f'{mape:.4f}%', r2]
    }).reset_index(drop=True)

    st.table(results)

with col2:
    # Calculate utility values and probabilities
    if not filtered_df.empty:
        st.header("Utility Function Values")

        # Encode filtered data
        filtered_df_encoded, _ = encode_data(filtered_df.copy())

        # Calculate utility values for each mode
        utility_values = []
        modes_present = []
        for mode in mode_mapping.values():
            mode_data = filtered_df_encoded[filtered_df_encoded['MODE'] == mode]
            if not mode_data.empty:
                X_mode = mode_data[importance_df['Feature']]
                X_mode = X_mode.apply(pd.to_numeric, errors='coerce').fillna(0)
                utility_value = np.dot(X_mode, importance_df['Importance'].values).mean()
                utility_values.append(utility_value)
                modes_present.append(mode)

        # Only consider present modes
        modes_present_names = [mode_name_mapping[mode] for mode in modes_present]

        # Calculate probabilities for each mode
        sum_utility_values = np.sum(utility_values)
        probabilities = utility_values / sum_utility_values

        # Display utility values
        utility_df = pd.DataFrame({
            'Mode': modes_present_names,
            'Utility Value': utility_values
        })

        st.dataframe(utility_df)

        # Display probabilities
        probability_df = pd.DataFrame({
            'Mode': modes_present_names,
            'Probability': probabilities
        })

        st.header("Mode Choice Probabilities")
        fig_pie = go.Figure(data=[go.Pie(labels=probability_df['Mode'], values=probability_df['Probability'])])
        fig_pie.update_layout(title_text='Probabilities for the Selected Route')
        st.plotly_chart(fig_pie)
    else:
        st.write("No data available for selected states.")


with st.expander('About', expanded=True):
    st.write('''
        - Data from: [U.S. Commodity Flow Survey](https://www.census.gov/programs-surveys/cfs.html).
        - **NAICS**: NAICS is an abbreviation for North American Industry Classification System.
        - **Freight Metrics by NAICS**: The total value and total tonnage is calculated according to the guidance.
        [2017 Commodity Flow Survey (CFS) Public Use File (PUF) Data Users Guide ](https://www2.census.gov/programs-surveys/cfs/datasets/2017/cfs_2017_puf_users_guide.pdf)
    ''')