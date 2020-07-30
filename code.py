import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import shapiro
import seaborn as sns
import time
from datetime import datetime as dt
import numpy
from sklearn import cluster
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

train = pd.read_csv( '/kaggle/input/restaurant-revenue-prediction/train.csv.zip')
test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

num_train = train.shape[0]
num_test = test.shape[0]

test.columns

train.describe()

train.head()

train.columns

data.info()

# Analyzing Categorical Variables

df_cat = train[['Open Date', 'City', 'City Group', 'Type']]
df_num =train[['P1', 'P2', 'P3', 'P4',
       'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
       'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
       'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
       'P36', 'P37', 'revenue']]

fig, ax = plt.subplots(3, 1, figsize=(40, 30))
for variable, subplot in zip(df_cat, ax.flatten()):
    df_2 = train[[variable,'revenue']].groupby(variable).revenue.sum().reset_index()
    df_2.columns = [variable,'total_revenue']
    sns.barplot(x=variable, y='total_revenue', data=df_2 , ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)

# Analyzing Numerical Variables

num=train.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['revenue'], ascending=False).head(1), cmap='Blues')
plt.title(" Numerical features correlation with the Revenue", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)


plt.show()

print(train['revenue'].describe())
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    train['revenue'], norm_hist=False, kde=True
).set(xlabel='revenue', ylabel='P(revenue)');

# Analyzing Relationships Between Numerical Variables and the target ['revneue']

fig, ax = plt.subplots(10, 4, figsize=(30, 35))
for variable, subplot in zip(df_num, ax.flatten()):
    sns.regplot(x=train[variable], y=train['revenue'], ax=subplot)

plt.figure(figsize=(40,20))
sns.heatmap(train.corr(),annot=True)

Num=numcorr['revenue'].sort_values(ascending=False).head(10).to_frame()
cm = sns.light_palette("red", as_cmap=True)
s = Num.style.background_gradient(cmap=cm)
print(s)

num2=data.select_dtypes(exclude='object')
numcorr2=num2.corr()

# Plotting mean of P-variables over each city helps us see which P-variables are highly related to City
# since we are given that one class of P-variables is geographical attributes.
distinct_cities = train.loc[:, "City"].unique()

# Get the mean of each p-variable for each city
means = []
for col in train.columns[5:42]:
    temp = []
    for city in distinct_cities:
        temp.append(train.loc[train.City == city, col].mean())
    means.append(temp)

# Construct data frame for plotting
city_pvars = pd.DataFrame(columns=["city_var", "means"])
for i in range(37):
    for j in range(len(distinct_cities)):
        city_pvars.loc[i + 37 * j] = ["P" + str(i + 1), means[i][j]]
# print(city_pvars)
# Plot boxplot
plt.rcParams['figure.figsize'] = (18.0, 6.0)
sns.boxplot(x="city_var", y="means", data=city_pvars)

# Feature Engineering

# K Means treatment for city
def adjust_cities(data, train, k):
    # As found by box plot of each city's mean over each p-var
    relevant_pvars = ["P1", "P2", "P11", "P19", "P20", "P23", "P30"]
    train = train.loc[:, relevant_pvars]

    # Optimal k is 20 as found by DB-Index plot
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(train)

    # Get the cluster centers and classify city of each data instance to one of the centers
    data['City Cluster'] = kmeans.predict(data.loc[:, relevant_pvars])
    del data["City"]

    return data


def one_hot_ecoding(data, col, pref):
    # One hot encode City Group
    data = data.join(pd.get_dummies(data[col], prefix=pref))
    # Since only n-1 columns are needed to binarize n categories, drop one of the new columns.
    # And drop the original columns.
    data = data.drop([col], axis=1)
    return data

# train
all_diff = []
for date in data["Open Date"]:
    diff = dt.now() - dt.strptime(date, "%m/%d/%Y")
    all_diff.append(int(diff.days/1000))

data['Days_from_open'] = pd.Series(all_diff)
print(data.head())

data = data.drop('Open Date', axis=1)

# Convert unknown cities in test data to clusters based on known cities using KMeans
data = adjust_cities(data, train, 20)
data = one_hot_ecoding(data,'City Group',"CG")
data = one_hot_ecoding(data,'Type',"T")

# Count distinct values for each column in Data frame
data.apply(lambda x: len(x.unique()))

# Scale all input features to between 0 and 1.
min_max_scaler = MinMaxScaler()
data = pd.DataFrame(data=min_max_scaler.fit_transform(data),
                    columns=data.columns, index=data.index)

# Revenue Distribution of Train Set
# Check distribution of revenue and log(revenue) (Other Transformation could be Sqrt Transformation)
plt.rcParams['figure.figsize'] = (16.0, 6.0)
pvalue_before = shapiro(train["revenue"])[1]
pvalue_after = shapiro(np.log(train["revenue"]))[1]
graph_data = pd.DataFrame(
        {
            ("Revenue\n P-value:" + str(pvalue_before)) : train["revenue"],
            ("Log(Revenue)\n P-value:" + str(pvalue_after)) : np.log(train["revenue"])
        }
    )
graph_data.hist()

# Shapiro Wilks test for normality
# log transform revenue as it is approximately normal. If this distribution for revenue holds in the test set,
# log transforming the variable before training models will improve performance vastly.
# However, we cannot be completely certain that this distribution will hold in the test set.
train["revenue"] = np.log(train["revenue"])

# Split into train and test datasets
num_train = train.shape[0]
num_test = test.shape[0]
print(num_train, num_test)

train_processed = data[:num_train]
test_processed = data[num_train:]
# check the shapes
print("Train :",train.shape)
print("Test:",test.shape)

test_processed.head()

X_train_final=train_processed
y_train_final=train['revenue']

# splitting the dataset as training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final)

# building the model

linreg = LinearRegression()
linreg.fit(X_train, y_train)

#Accuracy

print("R-Squared Value for Training Set: {:.3f}".format(linreg.score(X_train, y_train)))
print("R-Squared Value for Test Set: {:.3f}".format(linreg.score(X_test, y_test)))

# KNeighborsRegressor
knnreg = KNeighborsRegressor(n_neighbors = 2)
knnreg.fit(X_train, y_train)

print('R-squared train score: {:.3f}'.format(knnreg.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(knnreg.score(X_test, y_test)))

# Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)

print('R-squared score (training): {:.3f}'.format(ridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(ridge.score(X_test, y_test)))

# Lasso
lasso = Lasso(max_iter = 10000)
lasso.fit(X_train, y_train)

print('R-squared score (training): {:.3f}'.format(lasso.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(lasso.score(X_test, y_test)))

lasso = Lasso(alpha=100, max_iter = 10000)
lasso.fit(train_processed, train['revenue'])
results = lasso.predict(test_processed)
results_2= np.exp(results)
print(results_2)

#SRV
svr = SVR(C=1, epsilon=0.1)
svr.fit(train_processed,train['revenue'])
results_svm = svr.predict(test_processed)
results_svm_exp = np.exp(results_svm)
print(results_svm_exp)

# outputting submission
output = pd.DataFrame({'Id': test.Id, 'Prediction': results_svm_exp })

output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
print(output)