import csv
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# зчитування даних з CSV
# -----------------------------
def read_data(filename):
    x = []
    y = []

    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)

            # Отримуємо назви колонок
            fieldnames = reader.fieldnames
            print(f"Колонки у файлі: {fieldnames}")

            # Визначаємо, які колонки використовувати
            if fieldnames:
                # Шукаємо колонку для RPS (x)
                x_col = None
                possible_x_names = ['RPS', 'rps', 'Rps', 'n', 'x', 'tasks', 'Objects', 'Dataset size']
                for name in possible_x_names:
                    if name in fieldnames:
                        x_col = name
                        break

                # Шукаємо колонку для CPU (y)
                y_col = None
                possible_y_names = ['CPU', 'cpu', 'Cpu', 't', 'y', 'Cost', 'FPS', 'Train time']
                for name in possible_y_names:
                    if name in fieldnames:
                        y_col = name
                        break

                # Якщо знайшли колонки, читаємо дані
                if x_col and y_col:
                    for row in reader:
                        if row[x_col] and row[y_col]:  # перевіряємо, що значення не пусті
                            x.append(float(row[x_col].strip()))
                            y.append(float(row[y_col].strip()))
                    print(f"Зчитано {len(x)} рядків з файлу")
                else:
                    print(f"Не знайдено потрібних колонок. Знайдені колонки: {fieldnames}")
                    return None, None
            else:
                print("Файл не містить колонок")
                return None, None

    except FileNotFoundError:
        print(f"Файл {filename} не знайдено.")
        return None, None
    except Exception as e:
        print(f"Помилка при читанні файлу: {e}")
        return None, None

    return np.array(x), np.array(y)


# -----------------------------
# таблиця розділених різниць
# -----------------------------
def divided_differences(x, y):
    n = len(x)
    coef = np.zeros((n, n))
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef


# -----------------------------
# поліном Ньютона
# -----------------------------
def newton_polynomial(x, coef, value):
    n = len(x)
    result = coef[0][0]

    product = 1.0

    for i in range(1, n):
        product *= (value - x[i - 1])
        result += coef[0][i] * product

    return result


# -----------------------------
# похибка
# -----------------------------
def calculate_error(real, approx):
    return abs(real - approx)


# -----------------------------
# створення тестового CSV файлу
# -----------------------------
def create_sample_csv(filename='data.csv'):
    """Створює приклад CSV файлу з даними варіанту 2"""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['RPS', 'CPU'])
        writer.writerow([50, 20])
        writer.writerow([100, 35])
        writer.writerow([200, 60])
        writer.writerow([400, 110])
        writer.writerow([800, 210])
    print(f"Файл {filename} створено з тестовими даними.")


# -----------------------------
# головна програма
# -----------------------------
def main():
    # Спочатку пробуємо створити тестовий файл
    create_sample_csv('data.csv')

    # Читаємо дані
    x, y = read_data("data.csv")

    # Якщо читання не вдалося, використовуємо тестові дані
    if x is None or y is None or len(x) == 0:
        print("Використовую тестові дані з варіанту 2.")
        x = np.array([50, 100, 200, 400, 800])
        y = np.array([20, 35, 60, 110, 210])

    print("\nДані для інтерполяції:")
    print("RPS:", x)
    print("CPU:", y)
# Сортуємо дані за x (якщо потрібно)
    if len(x) > 1:
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

    coef = divided_differences(x, y)

    # прогноз для 600 RPS
    prediction = newton_polynomial(x, coef, 600)

    print("\n" + "=" * 50)
    print(f"ПРОГНОЗ: При 600 RPS очікуване завантаження CPU = {prediction:.2f}%")
    print("=" * 50 + "\n")

    # -----------------------------
    # Виведення таблиці розділених різниць
    # -----------------------------
    print("ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ:")
    print("-" * 80)
    print(f"{'i':<3} {'x_i':<8} {'f[x_i]':<10} {'f[1]':<12} {'f[2]':<12} {'f[3]':<12} {'f[4]':<12}")
    print("-" * 80)

    for i in range(len(x)):
        row = f"{i:<3} {x[i]:<8.0f} {coef[i][0]:<10.2f}"
        for j in range(1, len(x) - i):
            row += f" {coef[i][j]:<12.4f}"
        print(row)
    print("-" * 80 + "\n")

    # -----------------------------
    # побудова графіків
    # -----------------------------
    # Створюємо фігуру з двома підграфіками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Графік 1: Інтерполяція CPU(RPS)
    x_graph = np.linspace(min(x) - 50, max(x) + 50, 500)
    y_graph = []

    for val in x_graph:
        y_graph.append(newton_polynomial(x, coef, val))

    ax1.scatter(x, y, color='red', s=100, label='Експериментальні дані', zorder=5)
    ax1.plot(x_graph, y_graph, 'b-', linewidth=2, label='Інтерполяція Ньютона')

    # Додаємо точку прогнозу
    ax1.scatter([600], [prediction], color='green', s=150, marker='D',
                label=f'Прогноз: {prediction:.1f}%', zorder=6)

    ax1.set_xlabel("RPS (запитів/сек)", fontsize=12)
    ax1.set_ylabel("CPU (%)", fontsize=12)
    ax1.set_title("Інтерполяційна модель CPU(RPS)", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Графік 2: Похибка інтерполяції
    errors = []
    x_errors = []

    for i in range(len(x)):
        approx = newton_polynomial(x, coef, x[i])
        error = calculate_error(y[i], approx)
        errors.append(error)
        x_errors.append(x[i])
        print(f"У вузлі x={x[i]}: точне={y[i]}, наближене={approx:.2f}, похибка={error:.6f}")

    ax2.plot(x_errors, errors, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel("RPS (запитів/сек)", fontsize=12)
    ax2.set_ylabel("Похибка", fontsize=12)
    ax2.set_title("Похибка інтерполяції у вузлах", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('interpolation_results.png', dpi=150)
    plt.show()

    print("\nГрафіки збережено у файл 'interpolation_results.png'")

    # -----------------------------
    # Додатковий аналіз для 600 RPS
    # -----------------------------
    print("\n" + "=" * 50)
    print("ДОДАТКОВИЙ АНАЛІЗ:")
    print("=" * 50)
    print(f"Прогнозоване значення CPU при 600 RPS: {prediction:.2f}%")
    print(f"Це знаходиться між значеннями при 400 RPS ({y[3]}%) та 800 RPS ({y[4]}%)")
    print(f"Відносне збільшення: {((prediction - y[3]) / (800 - 400) * 100):.1f}% на 100 RPS")


if __name__ == "__main__":
    main()