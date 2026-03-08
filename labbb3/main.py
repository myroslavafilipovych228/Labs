import numpy as np
import matplotlib.pyplot as plt
import csv


# -------------------------------
# ПРОСТІ ФУНКЦІЇ ДЛЯ МНК
# -------------------------------

def zachesty_dani_z_csv(nazva_fajlu):
    """Зчитує місяці та температури з CSV файлу"""
    misyatsi = []
    temperatury = []

    try:
        with open(nazva_fajlu, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # пропускаємо заголовок
            for row in reader:
                if len(row) >= 2:
                    misyatsi.append(float(row[0]))
                    temperatury.append(float(row[1]))
        print(f"Дані успішно зчитано з файлу {nazva_fajlu}")
    except:
        # Якщо файл не знайдено, створюємо тестові дані
        print("Файл не знайдено. Використовую тестові дані...")
        misyatsi = list(range(1, 25))
        temperatury = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
                       -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]

        # Створюємо файл з тестовими даними
        with open(nazva_fajlu, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Month', 'Temp'])
            for m, t in zip(misyatsi, temperatury):
                writer.writerow([m, t])
        print(f"Створено файл {nazva_fajlu} з тестовими даними")

    return np.array(misyatsi), np.array(temperatury)


# -------------------------------
# ФУНКЦІЇ ДЛЯ МЕТОДУ НАЙМЕНШИХ КВАДРАТІВ
# -------------------------------

def formuvaty_matrytsyu(x, m):
    """Створює матрицю коефіцієнтів для системи рівнянь МНК"""
    n = m + 1
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            suma = 0
            for k in range(len(x)):
                suma += x[k] ** (i + j)
            A[i, j] = suma

    return A


def formuvaty_vektor(x, y, m):
    """Створює вектор правих частин для системи рівнянь МНК"""
    n = m + 1
    b = np.zeros(n)

    for i in range(n):
        suma = 0
        for k in range(len(x)):
            suma += y[k] * (x[k] ** i)
        b[i] = suma

    return b


# -------------------------------
# МЕТОД ГАУСА (СПРОЩЕНИЙ)
# -------------------------------

def rozvyazaty_gausom(A, b):
    """
    Розв'язує систему лінійних рівнянь методом Гауса
    Повертає вектор невідомих x
    """
    n = len(b)

    # Перетворюємо вхідні дані в тип float
    A = A.astype(float)
    b = b.astype(float)

    # ПРЯМИЙ ХІД
    for k in range(n):
        # Пошук головного елемента
        max_row = k
        max_val = abs(A[k, k])
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        # Міняємо рядки місцями якщо знайшли більший елемент
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # Нормалізуємо рядок k
        pivot = A[k, k]
        if abs(pivot) < 1e-12:
            print(f"Увага: малий головний елемент на кроці {k}")
            pivot = 1e-12 if pivot >= 0 else -1e-12

        # Виключаємо x_k з інших рівнянь
        for i in range(k + 1, n):
            koef = A[i, k] / pivot
            for j in range(k, n):
                A[i, j] -= koef * A[k, j]
            b[i] -= koef * b[k]

    # ЗВОРОТНІЙ ХІД
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = b[i]
        for j in range(i + 1, n):
            suma -= A[i, j] * x[j]
        x[i] = suma / A[i, i]

    return x


# -------------------------------
# ОБЧИСЛЕННЯ ЗНАЧЕНЬ ПОЛІНОМА
# -------------------------------

def obchyslyty_polinom(x, koef):
    """
    Обчислює значення полінома в точці x
    """
    rezultat = 0
    for i, k in enumerate(koef):
        rezultat += k * (x ** i)
    return rezultat


def obchyslyty_polinom_masyv(x_masyv, koef):
    """
    Обчислює значення полінома для масиву точок
    """
    rezultat = []
    for x in x_masyv:
        rezultat.append(obchyslyty_polinom(x, koef))
    return np.array(rezultat)


# -------------------------------
# ОБЧИСЛЕННЯ ПОХИБОК
# -------------------------------

def obchyslyty_dyspersiyu(y_realni, y_nablyzheni):
    """
    Обчислює дисперсію: sqrt( sum((y_real - y_nabl)^2) / n )
    """
    n = len(y_realni)
    suma_kvadrativ = 0
    for i in range(n):
        suma_kvadrativ += (y_realni[i] - y_nablyzheni[i]) ** 2
    return np.sqrt(suma_kvadrativ / n)


def obchyslyty_pohybku(y_realni, y_nablyzheni):
    """
    Обчислює похибку в кожній точці: |y_real - y_nabl|
    """
    pohibky = []
    for i in range(len(y_realni)):
        pohibky.append(abs(y_realni[i] - y_nablyzheni[i]))
    return np.array("pohybky")


# -------------------------------
# ГОЛОВНА ПРОГРАМА
# -------------------------------

def main():
    print("=" * 60)
    print("ЛАБОРАТОРНА РОБОТА №4")
    print("Метод найменших квадратів")
    print("=" * 60)

    # 1. Зчитуємо дані
    x, y = zachesty_dani_z_csv("temperatures.csv")
    n = len(x)
    print(f"\n1. Зчитано {n} точок даних")
    print(f"   Місяці: від {int(x[0])} до {int(x[-1])}")
    print(f"   Температури: від {min(y):.1f}°C до {max(y):.1f}°C")

    # 2. Аналізуємо поліноми різних степенів
    max_m = min(10, n - 2)  # степінь має бути менше кількості точок
    dyspersiyi = []
    vsi_koeficienty = []

    print(f"\n2. Обчислення для степенів m = 1...{max_m}")
    print("-" * 50)
    print("   m   |   Дисперсія")
    print("-" * 50)

    for m in range(1, max_m + 1):
        # Створюємо систему рівнянь
        A = formuvaty_matrytsyu(x, m)
        b = formuvaty_vektor(x, y, m)

        # Розв'язуємо систему
        koef = rozvyazaty_gausom(A, b)
        vsi_koeficienty.append(koef)

        # Обчислюємо дисперсію
        y_nabl = obchyslyty_polinom_masyv(x, koef)
        dysp = obchyslyty_dyspersiyu(y, y_nabl)
        dyspersiyi.append(dysp)

        print(f"   {m:2d}   |   {dysp:.8f}")

    print("-" * 50)

    # 3. Знаходимо оптимальний степінь
    optimalny_m = 1
    minimalna_dysp = dyspersiyi[0]

    for i in range(1, len(dyspersiyi)):
        if dyspersiyi[i] < minimalna_dysp:
            minimalna_dysp = dyspersiyi[i]
            optimalny_m = i + 1

    optimalni_koef = vsi_koeficienty[optimalny_m - 1]

    print(f"\n3. ОПТИМАЛЬНИЙ СТУПІНЬ: m = {optimalny_m}")
    print(f"   Мінімальна дисперсія: {minimalna_dysp:.8f}")

    # 4. Прогноз на 3 місяці
    x_prognoz = np.array([25, 26, 27])
    y_prognoz = obchyslyty_polinom_masyv(x_prognoz, optimalni_koef)

    print(f"\n4. ПРОГНОЗ НА 3 МІСЯЦІ:")
    for i in range(3):
        print(f"   Місяць {int(x_prognoz[i])}: {y_prognoz[i]:.2f}°C")

    # 5. СТВОРЕННЯ ГРАФІКІВ
    plt.figure(figsize=(15, 12))

    # Графік 1: Залежність дисперсії від степеня
    plt.subplot(2, 3, 1)
    m_znachennya = list(range(1, max_m + 1))
    plt.plot(m_znachennya, dyspersiyi, 'bo-', linewidth=2, markersize=8)
    plt.plot(optimalny_m, minimalna_dysp, 'r*', markersize=15, label=f'Оптимум m={optimalny_m}')
    plt.xlabel('Степінь полінома (m)')
    plt.ylabel('Дисперсія')
    plt.title('Залежність дисперсії від степеня')
    plt.grid(True)
    plt.legend()

    # Графік 2: Апроксимація оптимальним поліномом
    plt.subplot(2, 3, 2)
    y_opt_nabl = obchyslyty_polinom_masyv(x, optimalni_koef)
    plt.plot(x, y, 'bo', label='Фактичні дані', markersize=5)
    plt.plot(x, y_opt_nabl, 'r-', linewidth=2, label=f'Поліном m={optimalny_m}')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.title('Апроксимація даних')
    plt.grid(True)
    plt.legend()

    # Графік 3: Прогноз
    plt.subplot(2, 3, 3)
    # Дані для плавної лінії
    x_plot = np.linspace(1, 27, 200)
    y_plot = obchyslyty_polinom_masyv(x_plot, optimalni_koef)

    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Поліном')
    plt.plot(x, y, 'bo', markersize=4, label='Історичні дані')
    plt.plot(x_prognoz, y_prognoz, 'r--', linewidth=2, marker='s', markersize=8, label='Прогноз')
    plt.axvline(x=24, color='gray', linestyle=':')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.title('Прогноз на 3 місяці')
    plt.grid(True)
    plt.legend()

    # Графік 4: Похибки для різних поліномів
    plt.subplot(2, 3, 4)

    # Створюємо багато точок для плавних графіків
    x_plot_detal = np.linspace(x[0], x[-1], 200)

    # Вибираємо декілька степенів для показу
    pokazaty_m = []
    if 1 <= max_m:
        pokazaty_m.append(1)
    if 3 <= max_m:
        pokazaty_m.append(3)
    if 5 <= max_m:
        pokazaty_m.append(5)
    if optimalny_m not in pokazaty_m:
        pokazaty_m.append(optimalny_m)
    if max_m not in pokazaty_m:
        pokazaty_m.append(max_m)

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for idx, m in enumerate(pokazaty_m):
        if m <= max_m:
            koef_m = vsi_koeficienty[m - 1]
            y_m = obchyslyty_polinom_masyv(x_plot_detal, koef_m)

            # Інтерполюємо фактичні дані
            y_real_interp = []
            for x_val in x_plot_detal:
                # Проста лінійна інтерполяція
                for i in range(len(x) - 1):
                    if x[i] <= x_val <= x[i + 1]:
                        t = (x_val - x[i]) / (x[i + 1] - x[i])
                        y_interp = y[i] + t * (y[i + 1] - y[i])
                        y_real_interp.append(y_interp)
                        break
                if x_val > x[-1]:
                    y_real_interp.append(y[-1])
                elif x_val < x[0]:
                    y_real_interp.append(y[0])

            y_real_interp = np.array(y_real_interp)

            # Обчислюємо похибки
            pohibky = []
            for i in range(len(x_plot_detal)):
                pohibky.append(abs(y_real_interp[i] - y_m[i]))

            if m == optimalny_m:
                plt.plot(x_plot_detal, pohibky, color='red', linewidth=2.5, label=f'm={m} (opt)')
            else:
                plt.plot(x_plot_detal, pohibky, color=colors[idx % len(colors)],
                         linestyle='--', linewidth=1.5, alpha=0.7, label=f'm={m}')

    plt.xlabel('Місяць')
    plt.ylabel('Похибка |f(x)-φ(x)|')
    plt.title('Похибки апроксимації')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Графік 5: Детальна похибка оптимального полінома
    plt.subplot(2, 3, 5)

    # Обчислюємо похибку для оптимального полінома
    y_opt_plot = obchyslyty_polinom_masyv(x_plot_detal, optimalni_koef)

    # Інтерполюємо фактичні дані
    y_real_interp = []
    for x_val in x_plot_detal:
        for i in range(len(x) - 1):
            if x[i] <= x_val <= x[i + 1]:
                t = (x_val - x[i]) / (x[i + 1] - x[i])
                y_interp = y[i] + t * (y[i + 1] - y[i])
                y_real_interp.append(y_interp)
                break
        if x_val > x[-1]:
            y_real_interp.append(y[-1])
        elif x_val < x[0]:
            y_real_interp.append(y[0])

    y_real_interp = np.array(y_real_interp)

    opt_pohybky = []
    for i in range(len(x_plot_detal)):
        opt_pohybky.append(abs(y_real_interp[i] - y_opt_plot[i]))

    plt.plot(x_plot_detal, opt_pohybky, 'g-', linewidth=2, label=f'Похибка m={optimalny_m}')

    # Похибки у вузлах
    pohibky_vuzly = []
    for i in range(len(x)):
        pohibky_vuzly.append(abs(y[i] - y_opt_nabl[i]))

    plt.plot(x, pohibky_vuzly, 'ro', markersize=4, label='У вузлах')
    plt.xlabel('Місяць')
    plt.ylabel('Похибка')
    plt.title(f'Похибка оптимального полінома')
    plt.grid(True)
    plt.legend()

    # Графік 6: Порівняння поліномів
    plt.subplot(2, 3, 6)

    for idx, m in enumerate(pokazaty_m[:4]):  # показуємо не більше 4 ліній
        if m <= max_m:
            koef_m = vsi_koeficienty[m - 1]
            y_m = obchyslyty_polinom_masyv(x_plot, koef_m)
            if m == optimalny_m:
                plt.plot(x_plot, y_m, 'r-', linewidth=2.5, label=f'm={m} (opt)')
            else:
                plt.plot(x_plot, y_m, '--', linewidth=1.5, alpha=0.7, label=f'm={m}')

    plt.plot(x, y, 'ko', markersize=3, label='Дані', alpha=0.5)
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.title('Порівняння поліномів')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 6. Зберігаємо результати
    with open('rezultaty_lab4.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТИ ЛАБОРАТОРНОЇ РОБОТИ №4\n")
        f.write("=" * 50 + "\n\n")

        f.write("Дисперсія для різних степенів:\n")
        f.write("-" * 30 + "\n")
        for i in range(max_m):
            f.write(f"m = {i + 1:2d}: {dyspersiyi[i]:.8f}\n")

        f.write(f"\nОптимальний степінь: m = {optimalny_m}\n")
        f.write(f"Мінімальна дисперсія: {minimalna_dysp:.8f}\n\n")

        f.write("Коефіцієнти оптимального полінома:\n")
        for i, k in enumerate(optimalni_koef):
            f.write(f"a{i} = {k:.8f}\n")

        f.write(f"\nПрогноз на 3 місяці:\n")
        for i in range(3):
            f.write(f"Місяць {int(x_prognoz[i])}: {y_prognoz[i]:.2f}°C\n")

    print(f"\n5. Результати збережено у файл 'rezultaty_lab4.txt'")
    print("\n" + "=" * 60)
    print("РОБОТА ЗАВЕРШЕНА")
    print("=" * 60)


# Запуск програми
if __name__ == "__main__":
    main()