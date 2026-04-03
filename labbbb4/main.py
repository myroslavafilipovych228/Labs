import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Задана функція вологості ґрунту
# -------------------------------------------------
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

# Аналітична похідна
def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Центральна різниця
def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

# -------------------------------------------------
# Розрахункова частина (Консольний вивід)
# -------------------------------------------------
t0 = 1
exact = dM_exact(t0)

print("=== 1. Аналітичне рішення ===")
print("Точне значення похідної:", exact)

print("\n=== 2. Дослідження залежності похибки від кроку ===")
h_values = np.logspace(-10, 0, 50)
errors = []
for h_val in h_values:
    errors.append(abs(exact - central_diff(M, t0, h_val)))

best_idx = np.argmin(errors)
best_h = h_values[best_idx]
best_error = errors[best_idx]
print(f"Оптимальний крок: {best_h:.2e}")
print(f"Досягнута точність: {best_error:.2e}")

print("\n=== 3. Прийнятий крок ===")
h = best_h
print(f"h = {h:.2e}")

print("\n=== 4. Значення похідної для двох кроків ===")
D_h = central_diff(M, t0, h)
D_h2 = central_diff(M, t0, h / 2)
print(f"D(h) = {D_h:.8f}")
print(f"D(h/2) = {D_h2:.8f}")

print("\n=== 5. Похибка при кроці h ===")
error_h = abs(exact - D_h)
print(f"error = {error_h:.2e}")

print("\n=== 6. Метод Рунге–Ромберга ===")
p = 2
D_RR = D_h2 + (D_h2 - D_h) / (2**p - 1)
error_RR = abs(exact - D_RR)
print(f"Уточнене значення: {D_RR:.8f}")
print(f"Похибка: {error_RR:.2e}")

print("\n=== 7. Метод Ейткена ===")
D_h4 = central_diff(M, t0, h / 4)
D_Aitken = D_h - ((D_h2 - D_h)**2) / (D_h4 - 2 * D_h2 + D_h)
error_Aitken = abs(exact - D_Aitken)
p_est = np.log(abs((D_h4 - D_h2) / (D_h2 - D_h))) / np.log(2)
print(f"Уточнене значення: {D_Aitken:.8f}")
print(f"Похибка: {error_Aitken:.2e}")
print(f"Оцінка порядку точності: {p_est:.4f}")

print("\n=== 8. Аналіз режиму поливу ===")
if D_Aitken < 0:
    print("Вологість зменшується → потрібно увімкнути полив")
else:
    print("Вологість достатня → полив не потрібен")

# -------------------------------------------------
# Візуалізація (Тільки потрібні графіки)
# -------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Функція та похідна
t_plot = np.linspace(0, 10, 500)
axes[0, 0].plot(t_plot, M(t_plot), 'b-', label='M(t) (Вологість)')
axes[0, 0].plot(t_plot, dM_exact(t_plot), 'r-', label="M'(t) (Похідна)")
axes[0, 0].axvline(t0, color='green', linestyle='--', label=f't={t0}')
axes[0, 0].set_title('Графік функції та її похідної')
axes[0, 0].legend()

# 2. Похибка від кроку h
axes[0, 1].loglog(h_values, errors, 'k-')
axes[0, 1].scatter([best_h], [best_error], color='red', label=f'Опт. h={best_h:.1e}')
axes[0, 1].set_xlabel('Крок h')
axes[0, 1].set_ylabel('Похибка')
axes[0, 1].set_title('Залежність похибки від кроку h')
axes[0, 1].legend()

# Дані для порівняння методів
methods = ['Центральна (h)', 'Центральна (h/2)', 'Рунге-Ромберг', 'Ейткен']
errs_comp = [error_h, abs(exact - D_h2), error_RR, error_Aitken]
colors = ['skyblue', 'lightblue', 'lightgreen', 'salmon']

# 3. Значення похідної за методами
axes[1, 0].bar(methods, [D_h, D_h2, D_RR, D_Aitken], color=colors)
axes[1, 0].axhline(y=exact, color='red', linestyle='--', label='Точне знач.')
axes[1, 0].set_title('Порівняння значень похідної')
axes[1, 0].set_ylim(exact - 0.1, exact + 0.1) # фокус на значенні
axes[1, 0].legend()

# 4. Похибки методів
axes[1, 1].bar(methods, errs_comp, color=colors)
axes[1, 1].set_yscale('log') # логарифмічна шкала, щоб бачити малі похибки
axes[1, 1].set_title('Порівняння похибок методів (log шкала)')
for i, v in enumerate(errs_comp):
    axes[1, 1].text(i, v, f'{v:.1e}', ha='center', va='bottom')

plt.tight_layout()
plt.show()