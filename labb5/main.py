import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Визначення функції навантаження
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# 2. Параметри та точне значення
a, b = 0, 24
I0, _ = quad(f, a, b)

# 3. Метод Сімпсона
def simpson_method(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S

# --- ОБЧИСЛЕННЯ ---

# Пошук N_opt для точності 1e-12
N_opt = 10
while abs(simpson_method(f, a, b, N_opt) - I0) > 1e-12 and N_opt < 5000:
    N_opt += 2

# Розрахунок похибок для графіка (пункт 4)
Ns = np.arange(10, 501, 10)
errors = [abs(simpson_method(f, a, b, n) - I0) for n in Ns]

# Методи підвищення точності (пункти 6-7)
N0 = int(N_opt / 10)
N0 = N0 + (8 - N0 % 8) if N0 % 8 != 0 else N0
I_N0 = simpson_method(f, a, b, N0)
I_N0_half = simpson_method(f, a, b, N0 // 2)
I_Runge = I_N0 + (I_N0 - I_N0_half) / 15

# Адаптивний алгоритм (пункт 9) з фіксацією точок розбиття
eval_points = []
def adaptive_simpson(f, a, b, eps, whole):
    mid = (a + b) / 2
    eval_points.append(mid) # зберігаємо точку для графіка
    left = simpson_method(f, a, mid, 2)
    right = simpson_method(f, mid, b, 2)
    if abs(left + right - whole) <= 15 * eps:
        return left + right + (left + right - whole) / 15
    return adaptive_simpson(f, a, mid, eps/2, left) + \
           adaptive_simpson(f, mid, b, eps/2, right)

I_adaptive = adaptive_simpson(f, a, b, 1e-12, simpson_method(f, a, b, 2))

# --- ПОБУДОВА ГРАФІКІВ ---

# Створюємо фігуру з трьома підграфіками
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
plt.subplots_adjust(hspace=0.4)

# Графік 1: Підінтегральна функція
x_plot = np.linspace(a, b, 1000)
ax1.plot(x_plot, f(x_plot), 'b-', linewidth=2)
ax1.fill_between(x_plot, f(x_plot), color='skyblue', alpha=0.3)
ax1.set_title('Графік підінтегральної функції f(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True)

# Графік 2: Залежність похибки від N (логарифмічна шкала)
ax2.semilogy(Ns, errors, 'r-o', markersize=4, label='Похибка Сімпсона')
ax2.axhline(y=1e-12, color='g', linestyle='--', label='Поріг 1e-12')
ax2.set_title('Дослідження точності: Залежність похибки від N')
ax2.set_xlabel('Число розбиття N')
ax2.set_ylabel('log(Error)')
ax2.legend()
ax2.grid(True, which="both", ls="-")

# Графік 3: Робота адаптивного алгоритму (де він ставив точки)
ax3.plot(x_plot, f(x_plot), 'gray', alpha=0.5)
ax3.vlines(eval_points, ymin=40, ymax=80, colors='red', alpha=0.5, linestyles='dotted', label='Точки адаптації')
ax3.set_title('Візуалізація адаптивного алгоритму (густота розбиття)')
ax3.set_xlabel('x')
ax3.legend()
ax3.grid(True)

plt.show()

# Вивід основних даних в консоль
print(f"Результати:")
print(f"Точне значення: {I0:.12f}")
print(f"N_opt знайдено: {N_opt}")
print(f"Похибка Рунге-Ромберга при N0={N0}: {abs(I_Runge - I0):.2e}")
print(f"Похибка адаптивного методу: {abs(I_adaptive - I0):.2e}")