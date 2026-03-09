import csv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# зчитування даних з CSV
# -----------------------------
def read_data(filename):
    x = []
    y = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))

    return np.array(x), np.array(y)

# -----------------------------
# таблиця розділених різниць
# -----------------------------
def divided_differences(x, y):
    n = len(x)
    coef = np.zeros((n, n))
    coef[:,0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])

    return coef

# -----------------------------
# поліном Ньютона
# -----------------------------
def newton_polynomial(x, coef, value):
    n = len(x)
    result = coef[0][0]

    product = 1.0

    for i in range(1, n):
        product *= (value - x[i-1])
        result += coef[0][i] * product

    return result

# -----------------------------
# похибка
# -----------------------------
def calculate_error(real, approx):
    return abs(real - approx)

# -----------------------------
# головна програма
# -----------------------------
x, y = read_data("data.csv")

coef = divided_differences(x, y)

# прогноз для 600 RPS
prediction = newton_polynomial(x, coef, 600)

print("Прогноз CPU при 600 RPS =", prediction)

# -----------------------------
# побудова графіків
# -----------------------------

x_graph = np.linspace(min(x), max(x), 100)
y_graph = []

for val in x_graph:
    y_graph.append(newton_polynomial(x, coef, val))

# графік CPU(RPS)
plt.figure()
plt.scatter(x, y)
plt.plot(x_graph, y_graph)
plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.title("Інтерполяція CPU(RPS) методом Ньютона")
plt.show()

# -----------------------------
# графік похибки
# -----------------------------
errors = []

for i in range(len(x)):
    approx = newton_polynomial(x, coef, x[i])
    errors.append(calculate_error(y[i], approx))

plt.figure()
plt.plot(x, errors)
plt.xlabel("RPS")
plt.ylabel("Error")
plt.title("Похибка інтерполяції")
plt.show()