# Multiple-Linear-Regression
Multiple Linear Regression is a supervised learning algorithm used to predict a continuous target variable using two or more independent variables (features).
It's an extension of simple linear regression, which uses only one feature.
Linear Regression is a useful tool for predicting a quantitative response.


Prediction using: 
    1. Simple Linear Regression 
    2. Multiple Linear Regression

# 1. Simple Linear Regression
Simple linear regression has only one x and one y variable. It is an approach for predicting a 
quantitative response using a single feature.
It establishes the relationship between two variables using a straight line. Linear regression 
attempts to draw a line that comes closest to the data by finding the slope and intercept that 
define the line and minimize regression errors.

## Formula: Y = β0 + β1X + e

    Y = Dependent variable / Target variable

    β0 = Intercept of the regression line 
    
    β1 = Slope of the regression lime which tells whether the line is increasing or decreasing
    
    X = Independent variable / Predictor variable
    
    e = Error
    
    Equation: Sales = β0 + β1X + TV

# 2. Multiple Linear Regression
Multiple linear regression has one y and two or more x variables. 
It is an extension of Simple Linear regression as it takes more than one predictor variable to 
predict the response variable.

Multiple Linear Regression is one of the important regression algorithms which models the linear
relationship between a single dependent continuous variable and more than one independent variable.
Assumptions for Multiple Linear Regression: 
1. A linear relationship should exist between the Target and predictor variables. 
2. The regression residuals must be normally distributed.
3. MLR assumes little or no multicollinearity (correlation between the independent variable) in data.

# Formula: Y = β0 + β1X1 + β2X2 + β3X3 + ... + βnXn + e
    
    Y = Dependent variable / Target variable
    
    β0 = Intercept of the regression line 
    
    β1, β2,..βn = Slope of the regression lime which tells whether the line is increasing or decreasing
    
    X1, X2,..Xn = Independent variables / Predictor variables
    
    e = Error
    
    Equation: Sales = β0 + (β1 * TV) + (β2 * Radio) + (β3 * Newspaper)

_______________________________________________________________________________________________________________________
