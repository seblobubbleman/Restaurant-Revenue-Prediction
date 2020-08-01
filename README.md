# Restaurant-Revenue-Prediction  Project Overview 
* Created a mathematical model to increase the effectiveness of investments in new restaurant sites.
* Explored categorical and numerical relationship to revenue using graphical models. 
* Conducted feature engineering on the data to help best predict the model. 
* Used Linear, Ridge, Lasso, KNeighbors, and Support Vector Regression to predict revenue.
* Choose Support Vector Regression as model.

## Code and Resources Used 
**Python Version:** 3.7
**Packages:** numpy, sklearn, matplotlib, seaborn, datetime

## Exploratory Data Analysis
* I looked at the distributions of the data and the value counts for the various variables. Below are a few highlights from the tables.
![](https://github.com/seblobubbleman/Restaurant-Revenue-Prediction/blob/master/image%204%20.png)
![](https://github.com/seblobubbleman/Restaurant-Revenue-Prediction/blob/master/image%203%20.png)

## Data Cleaning
I made the following changes and created the following variables:
* Transformed 'date open' into to the numerical days from open
* Dropped 'date open' and added 'days from open' column
* Got the cluster centers of cites and classified cities of each data instance to one of the centers
* Calculated KMeans by using city's mean over each 'p-variable'
* Converted unknown cities in test data to clusters based on KMeans
* Transformed the categorical column 'City Group' to 'CG'
* Transformed the categorical column 'Type' to 'T'
* Used MinMaxScaler to scale all input features to between 0 and 1
* Log transformed test revenue to improve performance 

## Model Building 
I tried five different models and evaluated them using R-squared.
* **Multiple Linear Regression** – Baseline for the model
* **Ridge Regression** - Because of the sparse data from the categorical variables, I thought a normalized regression like lasso would be effective.
* **Lasso Regression** – Again, with the sparsity associated with the data, I thought that this would be a good fit.
* **KNeighbors** - Just curious, but I realize that with the large database it is not likely to be a good fit
* **Support Vector Rgression** -

## Model Performance
* **Multiple Linear Regression**: -0.240
* **Ridge Regression**: 0.177
* **Lasso Regression**: -0.033
* **KNeighbors**: 0.048
* **Support Vector Regression**: 0.436

## Conclusion
Based on the model performance, I choose the Support Vector Regression as my model. 
