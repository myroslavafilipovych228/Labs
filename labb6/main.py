import numpy as np
import random

N = 100
EPS0 = 1e-14


def generate_matrix_and_vector():
    """Генерує матрицю A та вектор B"""
    random.seed(42)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i][j] = random.uniform(-10, 10)

    X = np.full(N, 2.5)

    B = A @ X

    # Записуємо у файли
    np.savetxt('matrix_A_2.txt', A)
    np.savetxt('vector_B.txt', B)

    return A, B


def lu_decomposition(A):
    """LU-розклад матриці A"""
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1.0

    # Обчислення L та U
    for k in range(n):
        for i in range(k, n):
            sum_val = 0
            for j in range(k):
                sum_val += L[i][j] * U[j][k]
            L[i][k] = A[i][k] - sum_val

        for i in range(k + 1, n):
            sum_val = 0
            for j in range(k):
                sum_val += L[k][j] * U[j][i]
            U[k][i] = (A[k][i] - sum_val) / L[k][k]

    return L, U


def solve_lu(L, U, B):
    """Розв'язання СЛАР через LU-розклад"""
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)

    # Прямий хід: LZ = B
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * Z[j]
        Z[i] = (B[i] - sum_val) / L[i][i]

    # Зворотній хід: UX = Z
    for i in range(n - 1, -1, -1):
        sum_val = 0
        for j in range(i + 1, n):
            sum_val += U[i][j] * X[j]
        X[i] = Z[i] - sum_val

    return X


def matrix_vector_product(A, X):
    """Множення матриці на вектор"""
    return A @ X


def vector_norm(V):
    """Обчислення норми вектора (максимальна норма)"""
    return np.max(np.abs(V))


def solve_with_iterative_refinement(A, L, U, B, X, eps):
    """Ітераційне уточнення розв'язку"""
    iter_count = 0

    print("   Iteration process:")

    while True:
        # Обчислюємо нев'язку R = B - A*X
        AX = matrix_vector_product(A, X)
        R = B - AX

        # Розв'язуємо A*DX = R
        DX = solve_lu(L, U, R)

        # Уточнюємо X
        X = X + DX

        # Обчислюємо норми
        norm_dx = vector_norm(DX)
        AX_new = matrix_vector_product(A, X)
        norm_residual = vector_norm(B - AX_new)

        iter_count += 1

        print(f"   Iteration {iter_count}: ||DX|| = {norm_dx:.2e}, max|AX-B| = {norm_residual:.2e}")

        # Перевірка умов зупинки
        if norm_dx <= eps or norm_residual <= eps:
            break

        if iter_count > 100:
            print("   Maximum number of iterations reached!")
            break

    print(f"\n   Final accuracy: ||DX|| = {norm_dx:.2e}, max|AX-B| = {norm_residual:.2e}")
    return X, iter_count


def main():
    print("=" * 50)
    print("Laboratory work #6")
    print("LU decomposition and iterative refinement")
    print(f"Matrix size: {N} x {N}")
    print("=" * 50)
    print()

    # 1. Генерація матриці та вектора
    print("1. Generating matrix A and vector B...")
    A, B = generate_matrix_and_vector()
    print("   Matrix A saved to 'matrix_A.txt'")
    print("   Vector B saved to 'vector_B.txt'")
    print()

    # 2. LU-розклад
    print("2. Performing LU decomposition...")
    L, U = lu_decomposition(A)

    # Запис LU-розкладу у файл
    with open('LU_decomposition.txt', 'w') as f:
        f.write("========== L (Lower triangular matrix) ==========\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{L[i][j]:f} ")
            f.write("\n")
        f.write("\n========== U (Upper triangular matrix) ==========\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{U[i][j]:f} ")
            f.write("\n")
    print("   LU decomposition saved to 'LU_decomposition.txt'")
    print()

    # 3. Розв'язання СЛАР
    print("3. Solving system AX = B...")
    X = solve_lu(L, U, B)

    print("   Solution (first 10 components):")
    for i in range(min(10, N)):
        print(f"   X[{i + 1}] = {X[i]:.10f}")
    if N > 10:
        print("   ...")
    print()

    # 4. Оцінка точності
    print("4. Evaluating accuracy...")
    AX = matrix_vector_product(A, X)
    max_error = vector_norm(AX - B)
    print(f"   Maximum error |AX - B| = {max_error:.2e}")
    print()

    # 5. Ітераційне уточнення
    print("5. Iterative refinement (eps = 1e-14)...")
    X_refined, iter_count = solve_with_iterative_refinement(A, L, U, B, X.copy(), EPS0)

    print()
    print("=" * 50)
    print("Work completed. Results saved to files:")
    print("- matrix_A.txt")
    print("- vector_B.txt")
    print("- LU_decomposition.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()