import numpy as np
import random
from typing import Tuple, List, Callable


def generate_matrix_with_diagonal_dominance(n: int, filename_a: str, filename_b: str, solution: List[float] = None):
    """
    Генерує матрицю A з діагональним переважанням розмірності n × n.
    Задає розв'язок системи (за замовчуванням всі x = 2.5) та обчислює вектор вільних членів b.
    Записує матрицю A та вектор b у текстові файли.
    """
    if solution is None:
        solution = [2.5] * n

    # Генеруємо випадкову матрицю з діагональним переважанням
    A = np.zeros((n, n))

    for i in range(n):
        # Генеруємо випадкові значення для рядка
        row_sum = 0
        for j in range(n):
            if i != j:
                A[i, j] = random.uniform(-10, 10)
                row_sum += abs(A[i, j])

        # Діагональний елемент має бути більшим за суму інших
        A[i, i] = row_sum + random.uniform(1, 5)

    # Обчислюємо вектор вільних членів b = A * x
    b = A @ np.array(solution)

    # Записуємо матрицю A у файл
    with open(filename_a, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{A[i, j]:.6f} ")
            f.write("\n")

    # Записуємо вектор b у файл
    with open(filename_b, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(f"{b[i]:.6f}\n")

    print(f"Матрицю A збережено у файл: {filename_a}")
    print(f"Вектор b збережено у файл: {filename_b}")
    print(f"Розмірність матриці: {n}×{n}")

    return A, b, solution


def read_matrix_from_file(filename: str) -> np.ndarray:
    """Зчитує матрицю A з текстового файлу."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        A = np.zeros((n, n))
        for i in range(n):
            row = list(map(float, lines[i + 1].strip().split()))
            A[i, :] = row[:n]
    return A


def read_vector_from_file(filename: str) -> np.ndarray:
    """Зчитує вектор b з текстового файлу."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        b = np.zeros(n)
        for i in range(n):
            b[i] = float(lines[i + 1].strip())
    return b


def matrix_vector_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Обчислює добуток матриці на вектор."""
    return A @ x


def vector_norm(vec: np.ndarray, norm_type: str = 'inf') -> float:
    """
    Обчислює норму вектора.
    norm_type: '1' - перша норма, '2' - евклідова норма, 'inf' - нескінченна норма
    """
    if norm_type == '1':
        return np.sum(np.abs(vec))
    elif norm_type == '2':
        return np.sqrt(np.sum(vec ** 2))
    else:  # 'inf' - норма Чебишева
        return np.max(np.abs(vec))


def matrix_norm(A: np.ndarray, norm_type: str = '1') -> float:
    """
    Обчислює норму матриці.
    norm_type: '1' - max за стовпцями, 'inf' - max за рядками
    """
    if norm_type == '1':
        return np.max(np.sum(np.abs(A), axis=0))
    elif norm_type == 'inf':
        return np.max(np.sum(np.abs(A), axis=1))
    else:
        return np.sqrt(np.sum(A ** 2))


def simple_iteration_method(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                            tau: float, eps: float, max_iter: int = 100000) -> Tuple[np.ndarray, int, List[float]]:
    """
    Метод простої ітерації для розв'язку СЛАР.
    Повертає: (розв'язок, кількість ітерацій, історія норм нев'язки)
    """
    n = len(b)
    x = x0.copy()
    C = np.eye(n) - tau * A
    d = tau * b
    residuals = []

    for k in range(max_iter):
        x_new = C @ x + d
        # Перевірка збіжності за нормою різниці наближень
        diff_norm = vector_norm(x_new - x)
        # Нев'язка: r = A*x_new - b
        residual = matrix_vector_product(A, x_new) - b
        residual_norm = vector_norm(residual)
        residuals.append(residual_norm)

        if diff_norm < eps or residual_norm < eps:
            return x_new, k + 1, residuals

        x = x_new

    return x, max_iter, residuals


def jacobi_method(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                  eps: float, max_iter: int = 100000) -> Tuple[np.ndarray, int, List[float]]:
    """
    Метод Якобі для розв'язку СЛАР.
    Повертає: (розв'язок, кількість ітерацій, історія норм нев'язки)
    """
    n = len(b)
    x = x0.copy()
    residuals = []

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError(f"Нульовий діагональний елемент на позиції {i}")
            sum_ax = 0
            for j in range(n):
                if j != i:
                    sum_ax += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_ax) / A[i, i]

        # Перевірка збіжності
        diff_norm = vector_norm(x_new - x)
        residual = matrix_vector_product(A, x_new) - b
        residual_norm = vector_norm(residual)
        residuals.append(residual_norm)

        if diff_norm < eps or residual_norm < eps:
            return x_new, k + 1, residuals

        x = x_new

    return x, max_iter, residuals


def seidel_method(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                  eps: float, max_iter: int = 100000) -> Tuple[np.ndarray, int, List[float]]:
    """
    Метод Зейделя (Гауса-Зейделя) для розв'язку СЛАР.
    Повертає: (розв'язок, кількість ітерацій, історія норм нев'язки)
    """
    n = len(b)
    x = x0.copy()
    residuals = []

    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError(f"Нульовий діагональний елемент на позиції {i}")
            sum1 = 0  # Сума з новими значеннями (j < i)
            sum2 = 0  # Сума зі старими значеннями (j > i)
            for j in range(i):
                sum1 += A[i, j] * x_new[j]
            for j in range(i + 1, n):
                sum2 += A[i, j] * x[j]
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Перевірка збіжності
        diff_norm = vector_norm(x_new - x)
        residual = matrix_vector_product(A, x_new) - b
        residual_norm = vector_norm(residual)
        residuals.append(residual_norm)

        if diff_norm < eps or residual_norm < eps:
            return x_new, k + 1, residuals

        x = x_new

    return x, max_iter, residuals


def check_diagonal_dominance(A: np.ndarray) -> bool:
    """Перевіряє, чи має матриця діагональне переважання."""
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True


def main():
    """Головна функція для виконання лабораторної роботи."""

    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА № 7")
    print("Ітераційні методи розв'язку систем лінійних алгебраїчних рівнянь")
    print("=" * 70)

    # Параметри
    n = 100  # Розмірність матриці
    eps = 1e-14  # Точність
    x0 = np.ones(n)  # Початкове наближення X(0) = 1.0
    true_solution = [2.5] * n  # Істинний розв'язок

    # Файли для збереження
    matrix_file = "matrix_A.txt"
    vector_file = "vector_b.txt"

    # 1. Генеруємо матрицю з діагональним переважанням та вектор b
    print("\n1. Генерація матриці A з діагональним переважанням...")
    A, b, solution = generate_matrix_with_diagonal_dominance(
        n, matrix_file, vector_file, true_solution
    )

    # Перевірка діагонального переважання
    if check_diagonal_dominance(A):
        print("✓ Матриця має діагональне переважання")
    else:
        print("⚠ Увага: Матриця не має діагонального переважання")

    # 2. Зчитуємо матрицю та вектор з файлів для перевірки
    print("\n2. Перевірка зчитування з файлів...")
    A_read = read_matrix_from_file(matrix_file)
    b_read = read_vector_from_file(vector_file)

    # Перевірка коректності зчитування
    if np.allclose(A, A_read) and np.allclose(b, b_read):
        print("✓ Матрицю та вектор успішно зчитано з файлів")
    else:
        print("✗ Помилка при зчитуванні з файлів")

    # Обчислення норм
    print("\n3. Обчислення норм...")
    norm_inf_A = matrix_norm(A, 'inf')
    norm_1_A = matrix_norm(A, '1')
    norm_2_A = matrix_norm(A, '2')
    print(f"Норма матриці A (1-норма): {norm_1_A:.6f}")
    print(f"Норма матриці A (нескінченна): {norm_inf_A:.6f}")
    print(f"Норма матриці A (евклідова): {norm_2_A:.6f}")

    # 4. Розв'язання СЛАР різними методами
    print("\n" + "=" * 70)
    print("РОЗВ'ЯЗАННЯ СЛАР ІТЕРАЦІЙНИМИ МЕТОДАМИ")
    print(f"Точність: ε = {eps}")
    print(f"Початкове наближення: x(0) = 1.0")
    print("=" * 70)

    # Визначення оптимального параметра τ для методу простої ітерації
    # τ має бути в межах (0, 2/λ_max), де λ_max - максимальне власне значення
    # Наближено використовуємо τ = 1 / norm(A, 'inf')
    tau = 1.0 / (matrix_norm(A, 'inf') + 0.1)
    print(f"\nПараметр τ для методу простої ітерації: {tau:.6f}")

    # Метод простої ітерації
    print("\n" + "-" * 50)
    print("МЕТОД ПРОСТОЇ ІТЕРАЦІЇ")
    print("-" * 50)
    try:
        x_simple, iter_simple, residuals_simple = simple_iteration_method(
            A, b, x0, tau, eps
        )
        error_simple = vector_norm(x_simple - np.array(true_solution))
        print(f"Кількість ітерацій: {iter_simple}")
        print(f"Похибка відносно істинного розв'язку: {error_simple:.6e}")
        print(f"Нев'язка: {residuals_simple[-1]:.6e}")
    except Exception as e:
        print(f"Помилка: {e}")
        iter_simple = -1

    # Метод Якобі
    print("\n" + "-" * 50)
    print("МЕТОД ЯКОБІ")
    print("-" * 50)
    try:
        x_jacobi, iter_jacobi, residuals_jacobi = jacobi_method(A, b, x0, eps)
        error_jacobi = vector_norm(x_jacobi - np.array(true_solution))
        print(f"Кількість ітерацій: {iter_jacobi}")
        print(f"Похибка відносно істинного розв'язку: {error_jacobi:.6e}")
        print(f"Нев'язка: {residuals_jacobi[-1]:.6e}")
    except Exception as e:
        print(f"Помилка: {e}")
        iter_jacobi = -1

    # Метод Зейделя
    print("\n" + "-" * 50)
    print("МЕТОД ЗЕЙДЕЛЯ (ГАУСА-ЗЕЙДЕЛЯ)")
    print("-" * 50)
    try:
        x_seidel, iter_seidel, residuals_seidel = seidel_method(A, b, x0, eps)
        error_seidel = vector_norm(x_seidel - np.array(true_solution))
        print(f"Кількість ітерацій: {iter_seidel}")
        print(f"Похибка відносно істинного розв'язку: {error_seidel:.6e}")
        print(f"Нев'язка: {residuals_seidel[-1]:.6e}")
    except Exception as e:
        print(f"Помилка: {e}")
        iter_seidel = -1

    # 5. Порівняння методів та запис результатів у файл
    print("\n" + "=" * 70)
    print("ПОРІВНЯННЯ МЕТОДІВ")
    print("=" * 70)

    results = []
    results.append(("Метод простої ітерації", iter_simple))
    results.append(("Метод Якобі", iter_jacobi))
    results.append(("Метод Зейделя", iter_seidel))

    print(f"\n{'Метод':<25} {'Кількість ітерацій':<20}")
    print("-" * 45)
    for name, iters in results:
        if iters > 0:
            print(f"{name:<25} {iters:<20}")
        else:
            print(f"{name:<25} {'не збігся':<20}")

    # Запис результатів у файл
    results_file = "results_lab7.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ЛАБОРАТОРНА РОБОТА № 7\n")
        f.write("Ітераційні методи розв'язку систем лінійних алгебраїчних рівнянь\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Розмірність матриці: {n}×{n}\n")
        f.write(f"Точність: ε = {eps}\n")
        f.write(f"Початкове наближення: x(0) = 1.0\n")
        f.write(f"Істинний розв'язок: x_i = 2.5\n\n")

        f.write("НОРМИ МАТРИЦІ:\n")
        f.write(f"1-норма: {norm_1_A:.6f}\n")
        f.write(f"Нескінченна норма: {norm_inf_A:.6f}\n")
        f.write(f"Евклідова норма: {norm_2_A:.6f}\n\n")

        f.write("РЕЗУЛЬТАТИ РОЗВ'ЯЗАННЯ:\n")
        f.write("-" * 70 + "\n")

        if iter_simple > 0:
            f.write(f"\nМетод простої ітерації:\n")
            f.write(f"  Кількість ітерацій: {iter_simple}\n")
            f.write(f"  Кінцева нев'язка: {residuals_simple[-1]:.6e}\n")
            f.write(f"  Похибка: {error_simple:.6e}\n")
            f.write(f"  Параметр τ: {tau:.6f}\n")

        if iter_jacobi > 0:
            f.write(f"\nМетод Якобі:\n")
            f.write(f"  Кількість ітерацій: {iter_jacobi}\n")
            f.write(f"  Кінцева нев'язка: {residuals_jacobi[-1]:.6e}\n")
            f.write(f"  Похибка: {error_jacobi:.6e}\n")

        if iter_seidel > 0:
            f.write(f"\nМетод Зейделя:\n")
            f.write(f"  Кількість ітерацій: {iter_seidel}\n")
            f.write(f"  Кінцева нев'язка: {residuals_seidel[-1]:.6e}\n")
            f.write(f"  Похибка: {error_seidel:.6e}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ПОРІВНЯЛЬНА ТАБЛИЦЯ:\n")
        f.write(f"{'Метод':<25} {'Кількість ітерацій':<20}\n")
        f.write("-" * 45 + "\n")
        for name, iters in results:
            if iters > 0:
                f.write(f"{name:<25} {iters:<20}\n")
            else:
                f.write(f"{name:<25} {'не збігся':<20}\n")

    print(f"\nРезультати збережено у файл: {results_file}")

    # Виведення перших кількох значень розв'язку
    print("\n" + "=" * 70)
    print("ПЕРШІ 10 ЗНАЧЕНЬ РОЗВ'ЯЗКУ (для порівняння з істинним x=2.5):")
    print("-" * 70)
    print(f"{'i':<5} {'Істинний':<12} {'Проста ітерація':<18} {'Якобі':<12} {'Зейдель':<12}")
    print("-" * 70)
    for i in range(min(10, n)):
        true_val = true_solution[i]
        simple_val = x_simple[i] if iter_simple > 0 else 0
        jacobi_val = x_jacobi[i] if iter_jacobi > 0 else 0
        seidel_val = x_seidel[i] if iter_seidel > 0 else 0
        print(f"{i + 1:<5} {true_val:<12.6f} {simple_val:<18.6f} {jacobi_val:<12.6f} {seidel_val:<12.6f}")

    print("\n" + "=" * 70)
    print("ЛАБОРАТОРНУ РОБОТУ ВИКОНАНО УСПІШНО")
    print("=" * 70)


if __name__ == "__main__":
    main()