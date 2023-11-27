import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
age = np.random.uniform(18, 65, 100)
health_costs = 5000 + 300 * age + np.random.normal(0, 1000, 100)

# Reshape age to a 2D array
age = age.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(age, health_costs, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error on the test set
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Plot the data and the regression line
plt.scatter(age, health_costs, label='Actual Data')
plt.plot(age, model.predict(age), color='red', linewidth=3, label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Health Costs')
plt.title('Linear Regression Health Costs Calculator')
plt.legend()
plt.show()

# Ask for user input to predict health costs for a specific age
user_age = float(input('Enter the age for health costs prediction: '))
predicted_health_costs = model.predict([[user_age]])
print(f'Predicted Health Costs for Age {user_age}: ${predicted_health_costs[0]:.2f}')
