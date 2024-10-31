import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(layout="wide")

def stationary(series):
    result = adfuller(series.dropna())  # Drop NaNs for the test
    return 'stationary' if result[1] < 0.05 else 'not stationary'

# Title for the Streamlit app
st.title("Random Forest ML model for Index Prediction")

# Data fetching and preprocessing
end_date = datetime.now()
start_date = end_date - timedelta(days=1*365)
ticker = "^NSEI"

stock = yf.Ticker(ticker)
historical_data = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval='1d')

# Calculate signals and indicators
historical_data['future_returns'] = historical_data['Close'].pct_change().shift(-1)
historical_data['signal'] = np.where(historical_data['future_returns'] > 0, 1, 0)
historical_data['rsi'] = ta.rsi(historical_data['Close'], length=14)
historical_data['adx'] = ta.adx(historical_data['High'], historical_data['Low'], historical_data['Open'])['ADX_14']
historical_data['sma'] = historical_data['Close'].rolling(window=14).mean()
historical_data['corr'] = historical_data['Close'].rolling(window=14).corr(historical_data['sma'])
historical_data['pct_change'] = historical_data['Close'].pct_change()
historical_data['volatility'] = historical_data['pct_change'].rolling(14).std() * 100

# Clean data
historical_data.fillna(method='ffill', inplace=True)
historical_data.dropna(inplace=True)

# Prepare features
y = historical_data[['signal']].copy()
X = historical_data[['Open', 'High', 'Low', 'Close', 'pct_change', 'rsi', 'adx', 'sma', 'corr', 'volatility']].copy()

# Drop non-stationary features
for col in X.columns:
    if stationary(historical_data[col]) == 'not stationary':
        X.drop(columns=[col], axis=1, inplace=True)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=False)
rf_model = RandomForestClassifier(n_estimators=3, max_features=3, max_depth=2, random_state=4)
rf_model.fit(X_train, y_train['signal'])

# Make predictions
y_pred = rf_model.predict(X_test)
accuracy_data = (y_pred == y_test['signal'].values)
accuracy_percentage = round(100 * accuracy_data.sum() / len(accuracy_data), 2)

# Create masks for plotting
green_mask = (y_test['signal'].values == 1) & (y_pred == 1)
red_mask = (y_test['signal'].values == 0) & (y_pred == 1)

# Plotting Charts
fig = go.Figure()

# Add main price line
fig.add_trace(
    go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    )
)

# Add correct predictions
fig.add_trace(
    go.Scatter(
        x=y_test.index[green_mask],
        y=historical_data.loc[y_test.index[green_mask], 'Close'],
        mode='markers',
        name='Correct Prediction (Buy Signal)',
        marker=dict(symbol='circle', color='green', size=8, line=dict(width=1.5, color='darkgreen'))
    )
)

# Add false predictions
fig.add_trace(
    go.Scatter(
        x=y_test.index[red_mask],
        y=historical_data.loc[y_test.index[red_mask], 'Close'],
        mode='markers',
        name='False Prediction (Buy Signal)',
        marker=dict(symbol='circle', color='red', size=8, line=dict(width=1.5, color='darkred'))
    )
)

# Update layout with better proportions and accuracy text below Date
fig.update_layout(
    height=700,
    title={
        'text': "Close Price with Correct and Incorrect Buy Predictions",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        range=[historical_data.index.min(), historical_data.index.max()],
        domain=[0.05, 0.95]
    ),
    yaxis=dict(
        title='Close Price',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        tickformat=',.0f',
        range=[historical_data['Close'].min() * 0.99, historical_data['Close'].max() * 1.01]
    ),
    legend=dict(
        x=0.5,
        y=1.12,
        xanchor='center',
        yanchor='top',
        orientation='h'
    ),
    annotations=[
        dict(
            text=f"Model Accuracy: {accuracy_percentage}%",
            xref='paper',
            yref='paper',
            x=0.5,
            y=-0.20,
            showarrow=False,
            font=dict(size=24, color='black')
        )
    ],
    margin=dict(
        t=120,
        l=50,
        r=50,
        b=100
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    showlegend=True,
    hovermode='x unified'
)

# Display plots in Streamlit with full width
st.plotly_chart(fig, use_container_width=True)

# Create a centered container for the heatmap
st.markdown("<h3 style='text-align: center;'>Feature Correlation Heatmap</h3>", unsafe_allow_html=True)

# Create three columns to center the heatmap
col1, col2, col3 = st.columns([1,2,1])

with col2:
    # Create correlation heatmap with adjusted size and style
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), 
                annot=True, 
                cmap='coolwarm', 
                ax=ax,
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': .8})
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display the heatmap in the center column
    st.pyplot(fig_corr)

# Add footer
st.markdown(
    """
    <div style='text-align: center; color: black; font-size: 24px; padding-top: 20px;'>
       Developed by Anupam Kabade, +91 9008816799
    </div>
    """,
    unsafe_allow_html=True
)
