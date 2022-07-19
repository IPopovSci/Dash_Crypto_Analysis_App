# Crypto_Dash_Analysis_App

A project to help inspect and visualize crypto and stock data.

## Table of contents

* [Introduction](#Introduction)
* [Technologies](#Technologies)
* [Setup](#setup)
* [Features](#features)
* [Author](#Author)

### Introduction
![alt text](screenshots/screenshot.png "First app page")

This project is created to help visualize various crypto data, and to analyze performance of classic machine learning
algorithms when applied to financial time series.

You can explore this app here: https://dashboard-financial-data.herokuapp.com
Due to computational constrains, most methods on "Fitted Backtest" page won't work

### Technologies and libraries

* Anaconda 3
* Python 3.9
* Pandas
* Plotly Dash
* Numpy
* Scikit-learn


### Setup

To set up the project, please clone the repository.
The app can be run by launching main.py

### Features

* Plotting of OHLCV candles
* Calculation and display of asset volatility
* Data gaps and drawdown analysis
* Technical analysis feature correlation matrix
* Linear and polynomial fit (Of n degrees) of features vs target returns for correlation determination
* Regression using OLS,polynomial,Random Forest and Gradient Boost to predict future returns
* Visual representation of regression returns vs buy and hold strategy
* Backtest simulation of various regressors with variable bet size and initial capital
* Computation of Information Coefficient of predicted returns

### Project Status
This project might be expanded upon if the need arises.
Currently, the regression methods are not properly tuned to give the best results.
Some additional functionality could be added in the future.

### Author

Created by Ivan Popov

