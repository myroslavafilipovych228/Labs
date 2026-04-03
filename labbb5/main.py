import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ============================================================
# Функція навантаження на сервер
# ============================================================
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24  # межі інтегрування

# ============================================================
# 1. Графік функції
# ============================================================
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, 'b-', linewidth=2,
         label=r'$f(x)=50+20\sin\!\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('graph1_function.png', dpi=150)
plt.show()

# ============================================================
# 2. Точне значення інтегралу (scipy)
# ============================================================
I0, _ = integrate.quad(f, a, b)
print(f"=== 2. Точне значення інтегралу ===")
print(f"I0 = {I0:.10f}\n")

# ============================================================
# 3. Складова формула Сімпсона
# ============================================================
def simpson(func, a, b, N):
    """Складова формула Сімпсона з N рівних частин (N парне)."""
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    result = y[0] + y[N]
    result += 4 * np.sum(y[1:N:2])   # непарні індекси
    result += 2 * np.sum(y[2:N-1:2]) # парні індекси (крім 0 і N)
    return result * h / 3

# ============================================================
# 4. Залежність похибки від числа розбиттів N = 10..1000
# ============================================================
N_values = np.arange(10, 1002, 2)  # тільки парні
errors = np.array([abs(simpson(f, a, b, int(N)) - I0) for N in N_values])

# Знаходимо N_opt — найменша похибка
idx_opt = np.argmin(errors)
N_opt = int(N_values[idx_opt])
eps_opt = errors[idx_opt]
epsopt = abs(simpson(f, a, b, N_opt) - I0)

print(f"=== 4. Дослідження залежності точності від N ===")
print(f"N_opt = {N_opt}")
print(f"epsopt = {epsopt:.6e}\n")

plt.figure(figsize=(10, 5))
plt.semilogy(N_values, errors, 'g-', linewidth=1.5)
plt.axvline(N_opt, color='r', linestyle='--', label=f'N_opt={N_opt}')
plt.axhline(1e-12, color='orange', linestyle=':', label='ε = 1e-12')
plt.title('Залежність похибки формули Сімпсона від N')
plt.xlabel('N (число розбиттів)')
plt.ylabel('|I(N) – I₀|')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('graph2_error_vs_N.png', dpi=150)
plt.show()

# ============================================================
# 5. Похибка при N0 ~ N_opt/10, кратне 8
# ============================================================
# N0 ~ N_opt/10, але кратне 8 і не менше 16 для коректної демонстрації методів
N0_raw = N_opt // 10
N0 = max(16, (N0_raw // 8) * 8)
# Якщо N0 < 16 — встановлюємо вручну розумне значення
if N0 < 16:
    N0 = 40  # кратне 8, дає наочне порівняння методів
eps0 = abs(simpson(f, a, b, N0) - I0)

print(f"=== 5. Похибка при N0 ===")
print(f"N0 = {N0}")
print(f"eps0 = eps(N0) = {eps0:.6e}\n")

# ============================================================
# 6. Метод Рунге-Ромберга при N0
# ============================================================
I_N0   = simpson(f, a, b, N0)
I_N0h2 = simpson(f, a, b, N0 // 2)

I_R = I_N0 + (I_N0 - I_N0h2) / 15.0   # p=4 для Сімпсона
epsR = abs(I_R - I0)

print(f"=== 6. Метод Рунге-Ромберга ===")
print(f"I(N0)   = {I_N0:.10f}")
print(f"I(N0/2) = {I_N0h2:.10f}")
print(f"I_R     = {I_R:.10f}")
print(f"epsR    = {epsR:.6e}\n")

# ============================================================
# 7. Метод Ейткена при N0, N0/2, N0/4
# ============================================================
N_a  = N0
N_b  = N0 // 2
N_c  = N0 // 4
if N_b < 2: N_b = 2
if N_c < 2: N_c = 2

I_a = simpson(f, a, b, N_a)
I_b = simpson(f, a, b, N_b)
I_c = simpson(f, a, b, N_c)

# Оцінка порядку
denom_p = (I_c - I_b) / (I_b - I_a)
if denom_p > 0 and denom_p != 1:
    p_est = np.log(abs((I_c - I_b) / (I_b - I_a))) / np.log(2)
else:
    p_est = 4.0  # теоретичний для Сімпсона

# Уточнене значення
q = 2 ** p_est
num   = I_a * I_c - I_b ** 2
denom = I_a - 2 * I_b + I_c
I_Aitken = num / denom if abs(denom) > 1e-30 else I_a
epsA = abs(I_Aitken - I0)

print(f"=== 7. Метод Ейткена ===")
print(f"I(N0)    = {I_a:.10f}")
print(f"I(N0/2)  = {I_b:.10f}")
print(f"I(N0/4)  = {I_c:.10f}")
print(f"p        = {p_est:.4f}")
print(f"I_Aitken = {I_Aitken:.10f}")
print(f"epsA     = {epsA:.6e}\n")

# ============================================================
# 8. Порівняльний графік похибок методів
# ============================================================
methods = ['Сімпсон\n(N0)', 'Рунге-\nРомберг', 'Ейткен']
errs    = [eps0, epsR, epsA]
colors  = ['steelblue', 'darkorange', 'green']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(methods, errs, color=colors, width=0.5)
ax.set_yscale('log')
ax.set_title('Порівняння похибок методів чисельного інтегрування')
ax.set_ylabel('Абсолютна похибка (лог. шкала)')
ax.set_xlabel('Метод')
for bar, val in zip(bars, errs):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
            f'{val:.2e}', ha='center', va='bottom', fontsize=9)
ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('graph3_methods_comparison.png', dpi=150)
plt.show()

# ============================================================
# 9. Адаптивний алгоритм Сімпсона
# ============================================================
def adaptive_simpson(func, a, b, tol, max_depth=50):
    """Адаптивний алгоритм на основі формули Сімпсона."""
    call_count = [0]

    def simp(a, b):
        c = (a + b) / 2
        h = b - a
        fa, fc, fb = func(a), func(c), func(b)
        call_count[0] += 3
        return h / 6 * (fa + 4 * fc + fb), fa, fc, fb

    def recursive(a, b, tol, whole, fa, fc, fb, depth):
        c = (a + b) / 2
        fd = func((a + c) / 2)
        fe = func((c + b) / 2)
        call_count[0] += 2
        left  = (c - a) / 6 * (fa + 4 * fd + fc)
        right = (b - c) / 6 * (fc + 4 * fe + fb)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 15 * tol:
            return left + right + delta / 15
        return (recursive(a, c, tol / 2, left,  fa, fd, fc, depth + 1) +
                recursive(c, b, tol / 2, right, fc, fe, fb, depth + 1))

    whole, fa, fc, fb = simp(a, b)
    result = recursive(a, b, tol, whole, fa, fc, fb, 0)
    return result, call_count[0]

print("=== 9. Адаптивний алгоритм ===")
tol_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
print(f"{'tol':>12} | {'I_adapt':>18} | {'похибка':>12} | {'викликів f':>12}")
print("-" * 62)
for tol in tol_values:
    I_ad, ncalls = adaptive_simpson(f, a, b, tol)
    err_ad = abs(I_ad - I0)
    print(f"{tol:>12.0e} | {I_ad:>18.10f} | {err_ad:>12.4e} | {ncalls:>12}")

print("\n=== Підсумок ===")
print(f"I0       = {I0:.10f}  (точне значення)")
print(f"N_opt    = {N_opt},  epsopt = {epsopt:.4e}")
print(f"N0       = {N0},  eps0   = {eps0:.4e}")
print(f"I_R      = {I_R:.10f},  epsR   = {epsR:.4e}")
print(f"I_Aitken = {I_Aitken:.10f},  epsA   = {epsA:.4e}")
print(f"p_Aitken = {p_est:.4f}")