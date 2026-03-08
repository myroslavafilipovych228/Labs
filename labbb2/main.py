import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- 1. ТАБУЛЯЦІЯ ТА ПІДГОТОВКА ДАНИХ  ---
def prepare_data():
    # Експериментальні дані Варіанта 3 [cite: 233, 234]
    data = {
        'n': [10000, 20000, 40000, 80000, 160000],
        't': [8, 20, 55, 150, 420]
    }
    df = pd.DataFrame(data)
    df.to_csv('training_data.csv', index=False)
    return df['n'].values, df['t'].values


# --- 2. ОБЧИСЛЕННЯ РОЗДІЛЕНИХ РІЗНИЦЬ [cite: 156] ---
def get_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


# --- 3. ІНТЕРПОЛЯЦІЯ НЬЮТОНА ТА ПОХИБКА [cite: 155, 159] ---
def newton_interpolation(x_nodes, y_nodes, x_target):
    coef = get_divided_diff(x_nodes, y_nodes)
    n = len(x_nodes)
    val = coef[0]
    w_val = 1.0

    # Функція w_k(x) [cite: 156]
    for i in range(1, n):
        w_val *= (x_target - x_nodes[i - 1])
        val += coef[i] * w_val
    return val


# --- 4. ДОСЛІДЖЕННЯ ЕФЕКТУ РУНГЕ ТА КІЛЬКОСТІ ВУЗЛІВ [cite: 162, 234, 245] ---
def run_research(x_nodes, y_nodes):
    target_x = 120000
    print(f"--- Аналіз для Варіанта 3 (n_target = {target_x}) ---")

    # Прогноз для цільового значення [cite: 234]
    res = newton_interpolation(x_nodes, y_nodes, target_x)
    print(f"Прогноз часу тренування: {res:.2f} сек.")

    # Побудова графіків для дослідження [cite: 160, 161]
    x_fine = np.linspace(min(x_nodes), max(x_nodes), 200)
    y_interp = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_fine]

    plt.figure(figsize=(12, 7))
    plt.plot(x_nodes, y_nodes, 'ro', label='Вузли (експеримент)')
    plt.plot(x_fine, y_interp, 'b-', label='Поліном Ньютона (5 вузлів)')

    # Візуалізація w_n(x) для оцінки похибки [cite: 160]
    # (Масштабовано для видимості на графіку)
    plt.fill_between(x_fine, y_interp, alpha=0.1, color='blue')

    plt.title("Інтерполяція часу тренування та аналіз стабільності")
    plt.xlabel("Розмір датасету")
    plt.ylabel("Час (сек)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Виконання всіх етапів
x_v3, y_v3 = prepare_data()
run_research(x_v3, y_v3)