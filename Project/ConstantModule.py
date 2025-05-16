Dataset_Link = "https://github.com/furkhansuhail/ProjectData/raw/refs/heads/main/MultipleLinearRegressionDataset/Advertising_MLR.csv"
import urllib.request as request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Importing the train_test_split function from the scikit-learn library's model_selection module. This function is used to split the input data into training and testing sets.
from sklearn.model_selection import train_test_split
#Importing mean_squared_error from sklearn.metrics
from sklearn.metrics import mean_squared_error

from yellowbrick.regressor import PredictionError, ResidualsPlot

#Importing PCA from the "sklearn.metrics" library.
from sklearn.decomposition import PCA
