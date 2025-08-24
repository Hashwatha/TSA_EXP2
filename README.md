# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Developed by: Hashwatha M
### Date: 19/08/2025
## AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CarDekho dataset
file_path = "cardekho.csv"   # <-- update file path
data = pd.read_csv(file_path)

# Check dataset columns
print(data.head())

# Assuming dataset has columns: "year" and "selling_price"
# Aggregate yearly sales (sum of selling_price per year)
yearly_sales = data.groupby("year")["selling_price"].sum().reset_index()

# Rename columns to fit trend estimation format
yearly_sales.rename(columns={"year": "Year", "selling_price": "Sales"}, inplace=True)

# Extract values
years = yearly_sales["Year"].tolist()
sales = yearly_sales["Sales"].tolist()

# Prepare values for linear regression
X = [i - (len(years) // 2) for i in range(len(years))]
x2 = [i**2 for i in X]
xy = [i * j for i, j in zip(X, sales)]

n = len(years)
b = (n * sum(xy) - sum(sales) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(sales) - b * sum(X)) / n
linear_trend = [a + b * Xi for Xi in X]

# Polynomial Trend Estimation (Degree 2)
x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i * j for i, j in zip(x2, sales)]

coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(sales), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * Xi + c_poly * (Xi**2) for Xi in X]

```
```
# Display trend equations
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(years, sales, 'bo-', label="Actual Sales")
plt.plot(years, linear_trend, 'k--', label="Linear Trend")

plt.title("CarDekho Sales Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Total Sales (Selling Price Sum)")
plt.legend()
plt.grid(True)
plt.show()

```
```
# Display trend equations
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(years, sales, 'bo-', label="Actual Sales")
plt.plot(years, poly_trend, 'r-', label="Polynomial Trend")

plt.title("CarDekho Sales Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Total Sales (Selling Price Sum)")
plt.legend()
plt.grid(True)
plt.show()

```
## OUTPUT

<img width="836" height="457" alt="image" src="https://github.com/user-attachments/assets/bcfd0b01-2493-4ccc-86b4-96da4a5c4457" />

### A - LINEAR TREND ESTIMATION

<img width="1243" height="706" alt="image" src="https://github.com/user-attachments/assets/0bc4268e-5ed8-463e-b886-922645e78e41" />

### B- POLYNOMIAL TREND ESTIMATION

<img width="1267" height="717" alt="image" src="https://github.com/user-attachments/assets/7192b618-5959-4df0-8be0-5073071150b1" />

## RESULT:

Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
