import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
import seaborn as sns
import matplotlib.colors as clrs
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = "{:,.2f}".format

def stationary(series):
  """Function to check if the series is stationary or not.
  """
  result = adfuller(series)
  if(result[1] < 0.05):
    return 'stationary'
  else:
    return 'not stationary'


def get_pair_above_threshold(X, threshold):
  """Function to return the pairs with correlation above threshold.
  """
  # Calculate the correlation matrix
  correl = X.corr()

  # Unstack the matrix
  correl = correl.abs().unstack()

  # Recurring & redundant pair
  pairs_to_drop = set()
  cols = X.corr().columns
  for i in range(0, X.corr().shape[1]):
    for j in range(0, i+1):
      pairs_to_drop.add((cols[i], cols[j]))

  # Drop the recurring & redundant pair
  correl = correl.drop(labels=pairs_to_drop).sort_values(ascending=False)
  return correl[correl > threshold].index


end_date = datetime.now()
start_date = end_date - timedelta(days=1*365)

ticker = "^NSEI"

stock = yf.Ticker(ticker)
historical_data = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval='1d')

historical_data['future_returns'] = historical_data['Close'].pct_change().shift(-1)
historical_data['signal'] = np.where(historical_data['future_returns'] > 0, 1, 0)

#RSI
historical_data['rsi'] = ta.rsi(historical_data['Close'], length=14)

#ADX
historical_data['adx'] = ta.adx(historical_data['High'], historical_data['Low'], historical_data['Open'])['ADX_14']


#SMA
historical_data['sma'] = historical_data['Close'].rolling(window=14).mean()

#CORR
historical_data['corr'] = historical_data['Close'].rolling(window=14).corr(historical_data['sma'])

#PCT_CHANGE
historical_data['pct_change'] = historical_data['Close'].pct_change()

#VOLATILITY
historical_data['volatility'] = historical_data.rolling(14, min_periods=14)['pct_change'].std()*100


#print(historical_data.tail(3))

# Drop the missing values
#historical_data.dropna(inplace=True)

historical_data.fillna(method='ffill', inplace=True)
historical_data.dropna(inplace=True)

# Target
y = historical_data[['signal']].copy()
# Features
X = historical_data[['Open','High','Low','Close','pct_change', 'rsi', 'adx', 'sma','corr', 'volatility']].copy()
i=1

nrows = int(np.ceil(X.shape[1] / 3))  # 3 plots per row

#page 35

for col in X.columns:
  if stationary(historical_data[col]) == 'not stationary':
    #print('%s is not stationary. Dropping it.' % col)
    X.drop(columns=[col], axis=1, inplace=True)

#print(X)


#Display the Final Features
list(X.columns)

"""train:test by 80%:20%"""


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=False)


rf_model = RandomForestClassifier(n_estimators=3, max_features=3, max_depth=2, random_state=4)

rf_model.fit(X_train, y_train['signal'])
"The model is trained"

unseen_data_single_day = X_test.head(1)

unseen_data_single_day.T

single_day_prediction = rf_model.predict(unseen_data_single_day)

single_day_prediction

y_pred = rf_model.predict(X_test)


accuracy_data = (y_pred == y_test['signal'])

accuracy_percentage = round(100 * accuracy_data.sum() / len(accuracy_data), 2)

print(f"The accuracy is {accuracy_percentage}%.")


green_mask = (y_test['signal'] == 1) & (y_pred == 1)
red_mask = (y_test['signal'] == 0) & (y_pred == 1)


# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15], vertical_spacing=0.1)

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

# Update layout
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
    width=900,
    height=800,  # Increased height to accommodate accuracy score
    margin=dict(t=150, l=50, r=50, b=50),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    showlegend=True
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridcolor='rgba(128, 128, 128, 0.2)',
    range=[pd.to_datetime('2024-08-01'), pd.to_datetime('2024-10-31')],
    row=1, col=1
)

fig.update_yaxes(
    showgrid=True,
    gridcolor='rgba(128, 128, 128, 0.2)',
    tickformat=',.0f',
    row=1, col=1
)

# Hide axes for accuracy text
fig.update_xaxes(visible=False, row=2, col=1)
fig.update_yaxes(visible=False, row=2, col=1)

fig.show()
