import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics


car_dataset = pd.read_csv("car data.csv")

car_dataset.replace(
    {
        'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
        'Seller_Type': {'Dealer': 0, 'Individual': 1},
        'Transmission': {'Manual': 0, 'Automatic': 1}
    },
    inplace=True
)

X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=2
)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

for feature, coef in zip(X.columns, lin_reg_model.coef_):
    print(feature, "->", coef)

training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("Linear Regression - Training R squared error:", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.show()

test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("Linear Regression - Test R squared error:", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.show()

lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train, Y_train)

for feature, coef in zip(X.columns, lasso_reg_model.coef_):
    print(feature, "->", coef)

training_data_prediction = lasso_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("Lasso Regression - Training R squared error:", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression: Actual vs Predicted Prices")
plt.show()

test_data_prediction = lasso_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("Lasso Regression - Test R squared error:", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression: Actual vs Predicted Prices")
plt.show()
