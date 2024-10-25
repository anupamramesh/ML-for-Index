import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib as ta
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def stationary(series):
    result = adfuller(series)
    return 'stationary' if result[1] < 0.05 else 'not stationary'



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
historical_data['volatility'] = historical_data.rolling(14, min_periods=14)['pct_change'].std()*100

# Clean data
historical_data.fillna(method='ffill', inplace=True)
historical_data.dropna(inplace=True)

# Prepare features
y = historical_data[['signal']].copy()
X = historical_data[['Open','High','Low','Close','pct_change', 'rsi', 'adx', 'sma','corr', 'volatility']].copy()

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
accuracy_data = (y_pred == y_test['signal'])
accuracy_percentage = round(100 * accuracy_data.sum() / len(accuracy_data), 2)

# Create masks for plotting
green_mask = (y_test['signal'] == 1) & (y_pred == 1)
red_mask = (y_test['signal'] == 0) & (y_pred == 1)

# Create figure
# Create figure with adjusted dimensions
fig = make_subplots(rows=2, cols=1, 
                    row_heights=[0.90, 0.10],  # Adjusted ratio
                    vertical_spacing=0.20)      # Increased spacing
# Add main price line
fig.add_trace(
    go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1
)

# Add correct predictions
fig.add_trace(
    go.Scatter(
        x=y_test.index[green_mask],
        y=historical_data.loc[y_test.index[green_mask], 'Close'],
        mode='markers',
        name='Correct Prediction (Buy Signal)',
        marker=dict(symbol='circle', color='green', size=10, line=dict(width=2, color='darkgreen'))
    ),
    row=1, col=1
)

# Add false predictions
fig.add_trace(
    go.Scatter(
        x=y_test.index[red_mask],
        y=historical_data.loc[y_test.index[red_mask], 'Close'],
        mode='markers',
        name='False Prediction (Buy Signal)',
        marker=dict(symbol='circle', color='red', size=10, line=dict(width=2, color='darkred'))
    ),
    row=1, col=1
)

# Add accuracy text
accuracy_text = f"Model Accuracy: {accuracy_percentage}%"
fig.add_trace(
    go.Scatter(
        x=[historical_data.index[len(historical_data)//2]],
        y=[0.5],
        mode='text',
        text=[accuracy_text],
        textfont=dict(size=16, color='black'),
        showlegend=False,
    ),
    row=2, col=1
)

# Update layout with new dimensions and centering
fig.update_layout(
    title={
        'text': "Close Price with Correct and Incorrect Buy Predictions",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend_title='Legend',
    legend={
        'x': 0.5,
        'y': 1.15,
        'xanchor': 'center',
        'yanchor': 'top',
        'orientation': 'h'
    },
    hovermode='x unified',
    width=1000,          # Increased width
    height=700,          # Adjusted height
    margin=dict(
        t=150,           # Top margin
        l=50,            # Left margin
        r=50,            # Right margin
        b=100            # Increased bottom margin
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    showlegend=True
)

# Center the main chart
fig.update_xaxes(
    showgrid=True,
    gridcolor='rgba(128, 128, 128, 0.2)',
    range=[pd.to_datetime('2024-08-01'), pd.to_datetime('2024-10-31')],
    row=1, col=1,
    domain=[0.1, 0.9]    # Centers the chart horizontally
)

fig.update_yaxes(
    showgrid=True,
    gridcolor='rgba(128, 128, 128, 0.2)',
    tickformat=',.0f',
    row=1, col=1,
    range=[historical_data['Close'].min() * 0.98, historical_data['Close'].max() * 1.02]  # Add padding to y-axis
)

# Hide axes for accuracy text and position it better
fig.update_xaxes(visible=False, row=2, col=1)
fig.update_yaxes(visible=False, row=2, col=1)

# Update accuracy text position
accuracy_text = f"Model Accuracy: {accuracy_percentage}%"
fig.add_trace(
    go.Scatter(
        x=[historical_data.index[len(historical_data)//2]],
        y=[0.5],
        mode='text',
        text=[accuracy_text],
        textfont=dict(size=16, color='black'),
        showlegend=False,
    ),
    row=2, col=1
)

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 16px;'>
       Developed by Anupam Kabade
    </div>
    """,
    unsafe_allow_html=True
    )


# Streamlit display
st.title("Buy Signal for Index by ML")
st.plotly_chart(fig, use_container_width=True)  # Makes chart responsive to container width
