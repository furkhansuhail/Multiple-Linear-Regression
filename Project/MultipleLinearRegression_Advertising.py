from dataclasses import asdict

from ConstantModule import *


# This code is to download dataset from my github repo and save it in Dataset Folder

# Step 2: Data ingestion configuration using @dataclass
# ✅ Update the config class to include all needed fields
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list  # Can be removed if not used


# ✅ Properly create the config object
config = DataIngestionConfig(
    root_dir=Path("Dataset"),  # <-- Folder where file is saved
    source_URL=Dataset_Link,
    local_data_file=Path("Dataset/Advertising_MLR.csv"),  # <-- CSV file name
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]  # Can be removed or filled if necessary
)


def download_project_file(source_URL, local_data_file):
    # Create the parent directory if it doesn't exist
    local_data_file.parent.mkdir(parents=True, exist_ok=True)

    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(
            url=source_URL,
            filename=local_data_file
        )
        print(f"✅ File downloaded and saved to: {file_path}")



class MultiLinearRegression:
    def __init__(self):
        # Downloading Dataset from github and saving it in dataset directory
        # download_project_file(config.source_URL, config.local_data_file)
        self.Dataset = pd.read_csv("Dataset/Advertising_MLR.csv")
        self.ModuleDriver()

    def ModuleDriver(self):
        self.ExploratoryDataAnalysis()
        self.DataCleaning()
        # self.DataVisualization()


    def ExploratoryDataAnalysis(self):
        # Checks the datatypes of the columns
        print(self.Dataset.dtypes)
        # Displaying the numbe of rows and column of the dataframe
        print("The data has {} rows and {} columns.".format(self.Dataset.shape[0], self.Dataset.shape[1]))

        # Checking data from the dataset inside Influencer column
        print(self.Dataset['Influencer'].unique())

        # Creating new column named new_school to store school's data in binary form converting the existing data
        self.Dataset['InfluencerNew'] = self.Dataset['Influencer'].map({'Mega': 3, 'Micro': 2, 'Nano': 1, 'Macro': 0})
        # Shows the data inside new influencer column
        print(self.Dataset['InfluencerNew'])

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the sum of data inside each column in the array above for ad_data
                print('Sum of', col, '=', self.Dataset[col].sum())
                print("_____________________")

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the mean of data inside each column in the array above for ad_data
                print('Mean of', col, '=', self.Dataset[col].mean())
                print("_____________________")

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the median of data inside each column in the array above for ad_data
                print('Median of', col, '=', self.Dataset[col].median())
                print("_____________________")

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the standard deviation of data inside each column in the array above for ad_data
                print('Standard Deviation of', col, '=', self.Dataset[col].std())
                print("_____________________")

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the maximum of data inside each column in the array above for ad_data
                print('Maximum of', col, '=', self.Dataset[col].max())
                print("_____________________")

        # Using for loop in column of ad_data dataset
        for col in self.Dataset:
            # If condition for "TV", "Radio", "Social Media", "InfluencerNew", "Sales" columns
            if col in ["TV", "Radio", "Social Media", "InfluencerNew", "Sales"]:
                # Printing the column mentioned in the array above
                print(col)
                # Printing the minimum of data inside each column in the array above for ad_data
                print('Minimum of', col, '=', self.Dataset[col].min())
                print("_____________________")

        # Checking statistical measure of all continuous data
        print(self.Dataset.describe())

    def DataCleaning(self):

        # Displaying original dataframe using ad_data variable
        print(self.Dataset.head())
        # shift column 'InfluencerNew' to third position

        third_column = self.Dataset.pop('InfluencerNew')
        # insert column using insert(position,column_name,third_column) function
        self.Dataset.insert(3, 'InfluencerNew', third_column)
        # Displaying final data after shifling column dataframe using ad_data variable
        print("Final DataFrame")
        print(self.Dataset.head())

        # Looking for Missing Values
        print(self.Dataset.isnull().sum())

        """
        whether to drop only the missing values and keep the data in the set, or to eliminate the feature
        (the entire column) wholesale because there are so many missing datapoints that it isn’t fit for analysis.
         Or, Inputting values with Pandas or NumBy standards which is also a form of inputting missing data
        """

        # replace missing values with the median.
        medSales = self.Dataset['Sales'].median()
        print(medSales)
        self.Dataset['Sales'] = self.Dataset['Sales'].fillna(medSales)

        # replace missing values with the median.
        medRadio = self.Dataset['Radio'].median()
        print(medRadio)
        self.Dataset['Radio'] = self.Dataset['Radio'].fillna(medRadio)

        # replace missing values with the median.
        medSocial = self.Dataset['Social Media'].median()
        print(medSocial)
        self.Dataset['Social Media'] = self.Dataset['Social Media'].fillna(medSocial)

        # Removing the not readable string data as a new data column is introduced
        self.Dataset.drop(['Influencer'], axis=1)

        # Checking for empty values
        print(self.Dataset.isnull().sum())

        # Checking duplicated data
        print(self.Dataset.duplicated())

        # checking for outliers
        self.Dataset.iloc[:, :].boxplot(figsize=[20, 8])
        plt.subplots_adjust(bottom=0.25)
        plt.show()

        # Outlier Analysis - Horizontal Boxplots
        fig, axs = plt.subplots(4, figsize=(7, 6))

        plt1 = sns.boxplot(x=self.Dataset['TV'], ax=axs[0])
        plt2 = sns.boxplot(x=self.Dataset['Radio'], ax=axs[1])
        plt3 = sns.boxplot(x=self.Dataset['Social Media'], ax=axs[2])
        plt4 = sns.boxplot(x=self.Dataset['InfluencerNew'], ax=axs[3])

        axs[0].set_title("TV")
        axs[1].set_title("Radio")
        axs[2].set_title("Social Media")
        axs[3].set_title("Influencer New")

        plt.tight_layout()
        plt.show()

        remove = ['Social Media', 'Influencer']
        clean_data = self.Dataset.drop(remove, axis=1)
        print(clean_data.head(10))

        # Identify Dependent and Independent Variables

        # Displaying the correlation between the dataset
        print(clean_data.corr())

        # Splitting Into Teaining and Testing Database

        # Independent Variable
        X = clean_data.iloc[:, :-2]
        # Dependent variable
        y = clean_data['Sales']
        self.DataVisualization(X, y, clean_data)

    def DataVisualization(self, X, y, clean_data):

        # Visualizing Independent Data
        sns.histplot(X, kde=True)
        plt.show()

        # Visualizing dependent Data
        sns.histplot(y, kde=True)
        plt.show()

        sns.boxplot(self.Dataset['Sales'])
        plt.show()

        # Let's see how Sales are related with other variables using scatter plot.
        sns.pairplot(self.Dataset, x_vars=['TV', 'Radio', 'Social Media', 'InfluencerNew'], y_vars='Sales', height=4,
                     aspect=1, kind='scatter')
        plt.show()

        # Let's see how Sales are related with other variables using scatter plot.
        sns.pairplot(self.Dataset, x_vars=['TV', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
        plt.show()

        plt.rcParams["figure.figsize"] = (20, 10)
        sns.pairplot(self.Dataset, hue='Sales')

        plt.rcParams["figure.figsize"] = (20, 10)
        sns.pairplot(self.Dataset, hue='Sales')

        #  Correlation for train set
        plt.rcParams["figure.figsize"] = (20, 10)
        sns.heatmap(self.Dataset[['TV', 'Radio', 'Social Media', 'InfluencerNew', 'Sales']].corr(), annot=True)
        plt.show()
        # The heatmap displays the correlations between each pair of variables as color-coded cells,
        # where darker colors indicate stronger correlations. This is a useful way to visualize the relationships
        # between the variables and to identify any highly correlated pairs of variables.

        TV = self.Dataset['TV'].to_numpy()
        Radio = self.Dataset['Radio'].to_numpy()
        Sales = self.Dataset['Sales'].to_numpy()

        from mpl_toolkits.mplot3d import Axes3D
        # Ploting the scores as scatter plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(TV, Radio, Sales, color='#ef1234')
        plt.show()

        sns.pairplot(self.Dataset, x_vars=['TV', 'Radio'], y_vars='Sales', height=7, aspect=0.7);
        plt.show()
        self.ModelDevelopment(TV, Radio, Sales, clean_data)

    # Define the cost function
    def cost_function(self, X, Y, W):
        """ Parameters:
        This function finds the Mean Square Error.
        Input parameters:
          X: Feature Matrix or independent variable
          Y: Target Matrix or dependent variable
          W: Weight Matrix
        Output Parameters:
          J: accumulated mean square error.
        """
        m = len(Y)  # len of data in your datasets

        J = np.sum((X.dot(W) - Y) ** 2) / (2 * m)
        return J

    def gradient_descent(self, X, Y, B, alpha, iterations):
        cost_history = [0] * iterations
        m = len(Y)

        for iteration in range(iterations):
            # Hypothesis Values
            Y_pred = X.dot(B)
            # Difference b/w Hypothesis and Actual Y
            loss = Y_pred - Y
            # Gradient Calculation
            dw = (X.T.dot(loss)) / (m)
            # Changing Values of B using Gradient
            W_update = B - alpha * dw
            # New Cost Value
            cost = self.cost_function(X, Y, W_update)
            cost_history[iteration] = cost

        return W_update, cost_history

    def ModelDevelopment(self, TV, Radio, Sales, clean_data):
        # initializing Matrices which act as an Container to hold our Data.
        x0 = np.ones(len(TV))
        X2 = np.array([TV, Radio]).T
        W = np.array([0, 0])
        Y2 = np.array(Sales)

        inital_cost = self.cost_function(X2, Y2, W)
        print(inital_cost)

        # Optimization Algorithm
        # 100000 Iterations
        alpha = 0.0001  # Learning Rate.
        new_weights, cost_history = self.gradient_descent(X2, Y2, W, alpha, 100000)

        # New Values of
        print("New Value")
        print(new_weights)

        # Final Cost of our Iterations.
        print("Final Cost of iteration", cost_history[-1])


        # create X and y
        independent = ['TV', 'Radio']
        X1 = self.Dataset[independent]
        y1 = self.Dataset.Sales

        # instantiate and fit
        lm1 = LinearRegression()
        lm1.fit(X1, y1)

        # print the coefficients
        print(lm1.intercept_)
        print(lm1.coef_)

        # pair the feature names with the coefficients
        list(zip(independent, lm1.coef_))

        lm2 = LinearRegression().fit(X1[['TV']], y1)
        lm2_preds = lm2.predict(X1[['TV']])

        print("R^2: ", r2_score(y1, lm2_preds))

        lm3 = LinearRegression().fit(X1[['TV', 'Radio']], y1)
        lm3_preds = lm3.predict(X1[['TV', 'Radio']])

        print("R^2: ", r2_score(y1, lm3_preds))

        # Train-Test Split
        # Now I need to split the variable into training and testing sets. To perform this by importing train_test_split from the sklearn.model_selection library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset
        # from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, train_size = 0.7, test_size = 0.3, random_state = 100)

        # Let's now take a look at the train dataset
        # print(X_train.head())
        # print(y_train.head())

        # Add a constant to get an intercept
        X_train_sm = sm.add_constant(X_train)

        # Fit the resgression line using 'OLS'
        lr = sm.OLS(y_train, X_train_sm).fit()

        # Print the parameters, i.e. the intercept and the slope of the regression line fitted
        print(lr.params)
        print(lr.summary())
        # To evaluate the performance of the model that was trained using the gradient descent algorithm
        Y_pred = X2.dot(new_weights)
        # The rmse and r2 functions are then called with Y2 and Y_pred as inputs.
        print(self.rmse(Y2, Y_pred))
        print(self.r2(Y2, Y_pred))

        y_train_pred = lr.predict(X_train_sm)
        res = (y_train - y_train_pred)

        fig = plt.figure()
        sns.histplot (res, bins = 15) # , Kde = True
        fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading
        plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
        plt.show()

        # Looking for patterns in the residuals
        plt.scatter(y_train,res)
        plt.show()

        # Defining the variables
        X1 = self.Dataset[['TV']]
        y1 = self.Dataset.Sales
        # The input parameters for the function are X (the features), y (the target variable), train_size (the proportion of the data to be used for training), test_size (the proportion of the data to be used for testing), and random_state (the random seed for generating a random number).
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=1)
        # Defining  the linear regression
        lm4 = LinearRegression()
        lm4.fit(X_train, y_train)
        lm4_preds = lm4.predict(X_test)
        # Displaying RMSE and R^2
        print("RMSE :", np.sqrt(mean_squared_error(y_test, lm4_preds)))
        print("R^2: ", r2_score(y_test, lm4_preds))

        # Defining the variables
        X1 = self.Dataset[['TV', 'Radio']]
        y1 = self.Dataset.Sales
        # The input parameters for the function are X (the features), y (the target variable), train_size (the proportion of the data to be used for training), test_size (the proportion of the data to be used for testing), and random_state (the random seed for generating a random number).
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=1)
        # Defining  the linear regression
        lm5 = LinearRegression()
        lm5.fit(X_train, y_train)
        lm5_preds = lm5.predict(X_test)
        # Displaying RMSE and R^2
        print("RMSE :", np.sqrt(mean_squared_error(y_test, lm5_preds)))
        print("R^2: ", r2_score(y_test, lm5_preds))


        visualizer = PredictionError(lm5)
        # Fit the training data to the visualizer
        visualizer.fit(X_train, y_train)
        # Evaluate the model on the test data
        visualizer.score(X_test, y_test)
        visualizer.poof()

        # Fit the training data to the visualizer
        visualizer = ResidualsPlot(lm5)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.poof()

        # Dimensionality Reduction
        # Displaying the dataframe
        print("DataFrame:")
        print(clean_data)

        # Performing a principal component analysis (PCA) on the input data and then comparing the performance of a linear regression model fit on both the original data and the PCA transformed data.
        # Create an instance of PCA with two components
        pca = PCA(n_components=2)
        # Fit the PCA model to the training data
        X_train_pca = pca.fit_transform(X_train)
        # Transform the test data using the fitted PCA model
        X_test_pca = pca.transform(X_test)
        # Create a linear regression model
        regressor_pca = LinearRegression()
        # Fit the linear regression model to the transformed training data
        regressor_pca.fit(X_train_pca, y_train)
        # Predict the output of the transformed test data
        y_pred_pca = regressor_pca.predict(X_test_pca)

        # Evaluate the model using the mean squared error metric
        mse_pca = self.rmse(y_test, y_pred_pca)
        print("Mean Squared Error (PCA):", mse_pca)


        # Conclusion
        ##Calculate the root mean squared error (RMSE) between the test data and the predictions from the multiple regression model
        rmse_mlr = np.sqrt(mean_squared_error(y_test, lm5_preds))
        print("RMSE from Multiple Regression: ", rmse_mlr)

        # Compare the evaluation from step 4
        if rmse_mlr < mse_pca:
            print("The model without PCA performed better.")
        if rmse_mlr > mse_pca:
            print("The model with PCA performed better.")
        else:
            print("The model both with PCA and Multiple Regression performed better.")

        # Evaluating on the test data
        r2_pca = r2_score(y_test, y_pred_pca)

        # comparing the performance of two linear regression models from step 4
        if r2_score(y_test, lm5_preds) < r2_pca:
            print("The model without PCA performed better.")
        if r2_score(y_test, lm5_preds) > r2_pca:
            print("The model with PCA performed better.")
        else:
            print("The model both with PCA and Multiple Regression performed better.")

        # Returns the shape (dimensions) of the array X and printing the result as  tuple with two values, representing the number of rows and columns in X, respectively.
        print(X1.shape)

        # Returning '.shape' attribute as a tuple with the number of rows as the first value and the number of columns as the second value.
        print(X_train.shape)

        # Returning the shape of the training data after it has been transformed into principal components using the PCA method
        print(X_train_pca.shape)

        # Retrieving the shape of numpy array
        print(X_test_pca.shape)

        """
        In conclusion, regression is a statistical technique used to analyze the relationship between two or
        more variables. In this case, it is being used to analyze the effect of different types of advertising on
        sales. Multiple linear regression is a type of regression that uses multiple independent variables to 
        predict a dependent variable, in this case sales. By using this technique, businesses can better understand
        the relationship between the different types of advertising and the sales, and use this information to 
        predict sales and allocate their advertising budget accordingly.
        """

        # Model Evaluation - RMSE
    # Defining rmse (root mean squared error) function calculates the root mean squared error between the actual target variables Y and the predicted target variables Y_pred.
    def rmse(self, Y, Y_pred):
        """
        This Function calculates the Root Mean Squres.
        Input Arguments:
          Y: Array of actual(Target) Dependent Varaibles.
          Y_pred: Array of predeicted Dependent Varaibles.
        Output Arguments:
          rmse: Root Mean Square.
        """
        # Calculating the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
        return rmse

    # Model Evaluation - R2
    # Defining r2 (R-squared error) function to calculate the coefficient of determinatio which is a measure of how well the model fits the data.
    def r2(self, Y, Y_pred):
        """
         This Function calculates the R Squared Error.
        Input Arguments:
          Y: Array of actual(Target) Dependent Varaibles.
          Y_pred: Array of predeicted Dependent Varaibles.
        Output Arguments:
          rsquared: R Squared Error.
          """
        mean_y = np.mean(Y)
        # The variable "ss_tot" is the total sum of squares of the difference between Y and the mean of Y.
        ss_tot = sum((Y - mean_y) ** 2)
        # The variable "ss_res" is the residual sum of squares, which is the sum of squares of the difference between Y and the predicted target variable Y_pred.
        ss_res = sum((Y - Y_pred) ** 2)
        # R-squared error of a linear regression model
        r2 = (ss_res / ss_tot) - 1
        return r2



MultiLinearRegressionObj = MultiLinearRegression()














# #
# class MultiLinearRegression:
#     def __init__(self):
#         # Downloading Dataset from github and saving it in dataset directory
#         download_project_file(config.source_URL, config.local_data_file)
#         self.Dataset = pd.read_csv("Dataset/Advertising_MLR.csv")
#         print(self.Dataset.info())
#         print(self.Dataset.head())
#
#         print(self.ModelDriver())
#
#
#     def ModelDriver(self):
#         print(self.Dataset.info())
#         self.DataCleanup_EDA()
#         # self.LinearModelDevelopment()
#         self.MultipleModelDevelopment()
#
#
#     def DataCleanup_EDA(self):
#         print(self.Dataset.shape)
#         print(self.Dataset.isna().sum())
#         # Checking Outliers
#         # fig, axs = plt.subplots(3, figsize=(5, 5))
#         # plt1 = sns.boxplot(self.Dataset['TV'], ax=axs[0])
#         # plt2 = sns.boxplot(self.Dataset['Newspaper'], ax=axs[1])
#         # plt3 = sns.boxplot(self.Dataset['Radio'], ax=axs[2])
#         # plt.tight_layout()
#         # plt.show()
#         sns.displot(self.Dataset['Sales'])
#         plt.show()
#
#         sns.pairplot(self.Dataset, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
#         plt.show()
#         sns.heatmap(self.Dataset.corr(), annot=True)
#         plt.show()
#
#
#     def LinearModelDevelopment(self):
#         # Setting the value for X and Y
#         x = self.Dataset[['TV']]
#         y = self.Dataset['Sales']
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
#         slr = LinearRegression()
#         slr.fit(x_train, y_train)
#         # Printing the model coefficients
#         print('Intercept: ', slr.intercept_)
#         print('Coefficient:', slr.coef_)
#         print('Regression Equation: Sales = 6.948 + 0.054 * TV')
#         # Line of best fit
#         plt.scatter(x_train, y_train)
#         plt.plot(x_train, 6.948 + 0.054 * x_train, 'r')
#         plt.show()
#
#         # Prediction of Test and Training set result
#         y_pred_slr = slr.predict(x_test)
#         x_pred_slr = slr.predict(x_train)
#         print("Prediction for test set: {}".format(y_pred_slr))
#
#         # Actual value and the predicted value
#         slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
#         print(slr_diff)
#
#         # Predict for any value
#         slr.predict([[56]])
#         # print the R-squared value for the model
#         from sklearn.metrics import accuracy_score
#         print('R squared value of the model: {:.2f}'.format(slr.score(x, y) * 100))
#
#         # 0 means the model is perfect. Therefore the value should be as close to 0 as possible
#         meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
#         meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
#         rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))
#
#         print('Mean Absolute Error:', meanAbErr)
#         print('Mean Square Error:', meanSqErr)
#         print('Root Mean Square Error:', rootMeanSqErr)
#
#     def MultipleModelDevelopment(self):
#         # Setting the value for X and Y
#         x = self.Dataset[['TV', 'Radio', 'Newspaper']]
#         y = self.Dataset['Sales']
#
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
#         mlr = LinearRegression()
#         mlr.fit(x_train, y_train)
#
#         # Printing the model coefficients
#         print(mlr.intercept_)
#         # pair the feature names with the coefficients
#         list(zip(x, mlr.coef_))
#
#         # Predicting the Test and Train set result
#         y_pred_mlr = mlr.predict(x_test)
#         x_pred_mlr = mlr.predict(x_train)
#
#
#         print("Prediction for test set: {}".format(y_pred_mlr))
#
#         # Actual value and the predicted value
#         mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
#         print(mlr_diff)
#
#         # Predict for any value
#         mlr.predict([[56, 55, 67]])
#
#         # print the R-squared value for the model
#         print('R squared value of the model: {:.2f}'.format(mlr.score(x, y) * 100))
#
#         # 0 means the model is perfect. Therefore the value should be as close to 0 as possible
#         meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
#         meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
#         rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
#
#         print('Mean Absolute Error:', meanAbErr)
#         print('Mean Square Error:', meanSqErr)
#         print('Root Mean Square Error:', rootMeanSqErr)
#
# MultiLinearRegressionObj = MultiLinearRegression()
#
