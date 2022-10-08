" The following program is used to show various forecasting methods and takes stock return rates, real GDP, real consumption, and real investment of the US as examples. " 


### Enviornment
from arch import arch_model
import ffn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.tsa as tsa
from scipy.stats import norm, kstest
import yfinance as yf
import matplotlib.pyplot as plt
%matplotlib inline



### Autoregressive Moving Average (ARMA) Model
# Estimate the index return rate of S&P 500 by ARMA.

## Get the index data from Yahoo Finance.
Data = ffn.get("^GSPC", start = '2015-01-01', end = '2022-09-30')
## Sketch the index rate.
Data.plot.line(grid = True)

## Calculate the return rates of S&P 500 index.
Data["Return Rate"] = Data['gspc'].pct_change()
Data.dropna(inplace = True)

## Run Augmented Dickey-Fuller unit root test.
# The test can be used to test for a unit root in a univariate process in the presence of serial correlation.
# The 1st element in the result is the test statistic (adf); the 2nd is p-value; the 3rd is number of lags used; the 4th is number of observations; the 5th dict is critical values for the test statistic at the 1 %, 5 %, and 10 % levels; the last one is the maximized information criterion if autolag is not None.
sm.tsa.stattools.adfuller(Data["Return Rate"].dropna())

## Draw the histogram to show the return rates.
Data["Return Rate"].plot.hist(density = True, bins = 100)

## Run Kolmogorov-Smirnov test for goodness of fit.
loc, scale = norm.fit(Data["Return Rate"])
n = norm(loc = loc, scale = scale)
print( kstest(Data["Return Rate"].dropna(), n.cdf) )

## Plot the autocorrelation function. Plots lags on the horizontal and the correlations on vertical axis. The shaped area is the confidence interval (95% default) and the bar exceeded the shaped area represent autocorrelation significantly.
smg.tsaplots.plot_acf(Data["Return Rate"], lags = 20)
## Plot the partial autocorrelation function.
smg.tsaplots.plot_pacf(Data["Return Rate"], lags = 20)

## Compute information criteria for many ARMA models.
tsa.stattools.arma_order_select_ic(Data["Return Rate"], ic = ['aic', 'bic'], trend = 'nc')

## Run ARMA(2, 2) model on the return rates and fit it using exact maximum likelihood via Kalman filter.
_result = sm.tsa.ARMA(Data["Return Rate"].dropna(), (2, 2)).fit(disp = False)
print(_result.summary())

## Plot forecasts of the time-series data under the regressed result of ARMA model.
plt.figure(figsize=(6,10))
_result.plot_predict(start = 1800, end = 1970)



### Generalized Autoregressive Conditional Heteroskedasticity (GARCH) Model
## Get the price data of S&P 500 from Yahoo Finance.
Data = yf.download("^gspc", start = "2022-01-01", end = "2022-09-30")
## Calculate the return rates of the index.
ReturnRates = Data["Close"].pct_change().dropna()

## Show the return rates and the squared return rates that are employed in GARCH model.
ReturnRates.plot(grid = True)
(ReturnRates ** 2).plot(grid = True)

## Run the GARCH(1, 1) model on the return rates and show the result table.
_model = arch_model(ReturnRates * 100, vol = 'garch', p = 1, o = 0, q = 1, dist = 'Normal')
_result = _model.fit()
print(_result.summary())

## Plot the standardized residuals and the conditional volatility of the GARCH model run.
fig = _result.plot(annualize = 'D')
fig.set_size_inches(12, 6)

## Run the GARCH(2, 1) model on the return rates and show the result table.
_model2 = arch_model(ReturnRates * 100, vol = 'garch', p = 2, o = 0, q = 1, dist = 'Normal')
_result2 = _model2.fit()
print(_result2.summary())

## Plot the standardized residuals and the conditional volatility of GARCH(2, 1).
fig = _result2.plot(annualize = 'D')
fig.set_size_inches(8, 6)

## Show the forecasted variance.
_yhat = _result2.forecast(horizon = 10)
plt.plot(_yhat.variance.values[-1, :])



### Vector Autoregression (VAR)
## Download the macroeconomic data of the US from pandas database.
Data = sm.datasets.macrodata.load_pandas().data
## Extract the time data.
Dates = Data[['year', 'quarter']].astype(int).astype(str)
## Generate quarter data.
Quarterly = Dates["year"] + "Q" + Dates["quarter"]

## Convert the quarter data to datatime format.
Quarterly = tsa.base.datetools.dates_from_str(Quarterly)
## Extract the data of real GDP, real comsumption, and real investment.
Data = Data[['realgdp','realcons','realinv']]
## Employ the timestamp data as the index of the macroeconomic data to be investigated.
Data.index = pd.DatetimeIndex(Quarterly)

## Calculate the percentage rate of change of the macroeconomic data.
Percent = np.log(Data).diff().dropna()

## Fit VAR(1) process and do lag order selection.
_model = tsa.api.VAR(Percent)
_results = _model.fit(1)
_results.summary()

## Plot the results of the macroeconomic data under the VAR(1) process.
_results.plot()

## Choose the order(p) of the VAR model based on each of the available information criteria with the lowest scores attained and with maximum 15 lag orders.
# The available information criteria include AIC, BIC, FPE, and HQIC in this case.
_result_all = _model.select_order(15)
## Demonstrate the result.
print(_result_all.summary())

## Choose the order(p) of the VAR model using the lowest Bayesian information criterion with maximum 15 lag orders.
_results = _model.fit(maxlags = 15, ic = "bic")

## Get the order of the VAR process.
lagOrder = _results.k_ar
## Produce linear minimum MSE forecasts for desired number of steps ahead, using the values with the order chosen by the lowest BIC score.
_results.forecast(Percent.values[-lagOrder:], 5)

## Plot the forecasting of the macrodata, real GDP, real comsumption, and real investment, with 10 periods.
_results.plot_forecast(10)
