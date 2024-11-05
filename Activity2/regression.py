# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(train["SalePrice"])
plt.show()

# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(x='CentralAir', y='SalePrice', data=train)
plt.title('Sale Price Distribution by Central Air')
plt.xlabel('Central Air (Y/N)')
plt.ylabel('Sale Price')
plt.show()

#  SalePrice distribution w.r.t OverallQual
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.title('Sale Price Distribution by Overall Quality')
plt.xlabel('Overall Quality (1-10)')
plt.ylabel('Sale Price')
plt.show()

#  SalePrice distribution w.r.t BldgType
plt.figure(figsize=(10, 6))
sns.boxplot(x='BldgType', y='SalePrice', data=train)
plt.title('Sale Price Distribution by Building Type')
plt.xlabel('Building Type')
plt.ylabel('Sale Price')
plt.show()

#  SalePrice distribution w.r.t other features (Optional)
# For example, SalePrice vs. YearBuilt
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train)
plt.title('Sale Price vs. Year Built')
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.show()

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood

#  SalePrice distribution w.r.t YearBuilt
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train)
plt.title('Sale Price vs. Year Built')
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.show()

#  SalePrice distribution w.r.t Neighborhood
plt.figure(figsize=(12, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
plt.title('Sale Price Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.show()

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state = 2020)
rf_reg.fit(dummy_train_x, dummy_train_y)

print("Randon Forest Traing Set Performance")
print(evaluate(rf_reg, dummy_train_x, dummy_train_y))

print("Randon Forest Traing Set Performance")
print(evaluate(rf_reg, dummy_test_x, dummy_test_y))


# %%
