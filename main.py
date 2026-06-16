import numpy as np
from sklearn.linear_model import LinearRegression

# Years of experience
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

# Salary ($)
salary = np.array([30000, 35000, 40000, 50000, 55000, 60000, 70000, 75000])

# Train model
model = LinearRegression()
model.fit(experience, salary)

# Predict salary
years = float(input("Enter years of experience: "))
predicted_salary = model.predict([[years]])

print(f"Predicted salary: ${predicted_salary[0]:,.2f}")