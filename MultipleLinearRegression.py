import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)

# verify successful load with some randomly selected records
print(df.sample(5))

# Drop categoricals and any unseless columns
df = df.drop(
    [
        "MODELYEAR",
        "MAKE",
        "MODEL",
        "VEHICLECLASS",
        "TRANSMISSION",
        "FUELTYPE",
    ],
    axis=1,
)

print(df.corr())

df = df.drop(
    [
        "CYLINDERS",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
    ],
    axis=1,
)
print(df.head(9))

axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("right")

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

X = df.iloc[:, [0, 1]].to_numpy()
y = df.iloc[:, [2]].to_numpy

std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42
)

# create a model object
regressor = linear_model.LinearRegression()

# train the model in the training data
regressor.fit(X_train, y_train)

# Print the coefficients
coef_ = regressor.coef_
intercept_ = regressor.intercept_

print("Coefficients: ", coef_)
print("Intercept: ", intercept_)
