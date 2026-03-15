import requests
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Запит до Open Elevation API
# ==========================================================

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]

n = len(results)
print("Кількість вузлів:", n)

# ==========================================================
# 2. Табуляція та запис у файл
# ==========================================================

with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Elevation\n")
    for i, p in enumerate(results):
        line = f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}\n"
        print(line.strip())
        f.write(line)

# ==========================================================
# 3. Кумулятивна відстань
# ==========================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)

# ==========================================================
# 4. Метод прогонки (Thomas algorithm)
# ==========================================================

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i]*c_[i-1]
        c_[i] = c[i] / temp if i < n-1 else 0
        d_[i] = (d[i] - a[i]*d_[i-1]) / temp

    x = np.zeros(n)
    x[-1] = d_[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]

    return x

# ==========================================================
# 5. Натуральний кубічний сплайн
# ==========================================================

def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(1, n-1):
        a[i] = h[i-1]
        b[i] = 2*(h[i-1] + h[i])
        c[i] = h[i]
        d[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    b[0] = b[-1] = 1
    d[0] = d[-1] = 0

    M = thomas_algorithm(a, b, c, d)

    print("\nКоефіцієнти c (другі похідні M[i]):")
    for i, val in enumerate(M):
        print(f"c[{i}] = {val:.6f}")
    # ============================================================

    return M

M = cubic_spline(distances, elevations)

# ==========================================================
# 6. Обчислення значень сплайна
# ==========================================================

def spline_value(x, y, M, x_val):
    for i in range(len(x)-1):
        if x[i] <= x_val <= x[i+1]:
            h = x[i+1] - x[i]
            A = (x[i+1] - x_val)/h
            B = (x_val - x[i])/h

            return (A*y[i] + B*y[i+1] +
                    ((A**3 - A)*M[i] + (B**3 - B)*M[i+1])*(h**2)/6)

xx = np.linspace(distances[0], distances[-1], 500)
yy = np.array([spline_value(distances, elevations, M, x) for x in xx])

# ==========================================================
# 7. Функція для k вузлів
# ==========================================================

def build_spline_subset(k):
    idx = np.linspace(0, n-1, k, dtype=int)
    x_sub = distances[idx]
    y_sub = elevations[idx]

    M_sub = cubic_spline(x_sub, y_sub)
    yy_sub = np.array([spline_value(x_sub, y_sub, M_sub, x) for x in xx])

    return yy_sub

yy_10 = build_spline_subset(10)
yy_15 = build_spline_subset(15)
yy_20 = build_spline_subset(20)

# ==========================================================
# 8. Графік впливу кількості вузлів
# ==========================================================

plt.figure(figsize=(10,6))
plt.plot(xx, yy, label="21 вузол (еталон)")
plt.plot(xx, yy_10, '--', label="10 вузлів")
plt.plot(xx, yy_15, '--', label="15 вузлів")
plt.plot(xx, yy_20, '--', label="20 вузлів")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Вплив кількості вузлів")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# 9. Похибка апроксимації
# ==========================================================

error_10 = np.abs(yy - yy_10)
error_15 = np.abs(yy - yy_15)
error_20 = np.abs(yy - yy_20)

plt.figure(figsize=(10,6))
plt.plot(xx, error_10, label="Похибка (10 вузлів)")
plt.plot(xx, error_15, label="Похибка (15 вузлів)")
plt.plot(xx, error_20, label="Похибка (20 вузлів)")
plt.xlabel("Відстань (м)")
plt.ylabel("Абсолютна похибка (м)")
plt.title("Похибка апроксимації")
plt.legend()
plt.grid()
plt.show()

print("\nМаксимальна похибка:")
print("10 вузлів:", np.max(error_10))
print("15 вузлів:", np.max(error_15))
print("20 вузлів:", np.max(error_20))

# ==========================================================
# 10. Характеристики маршруту
# ==========================================================

print("\nЗагальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))

print("Сумарний набір висоти (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)

# ==========================================================
# 11. Аналіз градієнта
# ==========================================================

grad = np.gradient(yy, xx) * 100

print("\nМаксимальний підйом (%):", np.max(grad))
print("Максимальний спуск (%):", np.min(grad))
print("Середній градієнт (%):", np.mean(np.abs(grad)))

# ==========================================================
# 12. Енергія підйому (80 кг)
# ==========================================================

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("\nМеханічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)
print("Енергія (ккал):", energy/4184)