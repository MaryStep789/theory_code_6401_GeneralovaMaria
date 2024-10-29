import random

# Функция для создания единичной матрицы заданного размера
def create_identity(size):
    # Создаём квадратную матрицу с 1 на главной диагонали и 0 в остальных позициях
    return [[1 if row == col else 0 for col in range(size)] for row in range(size)]

# Функция для горизонтального объединения двух матриц
def merge_horizontally(matrix_a, matrix_b):
    # К каждой строке первой матрицы добавляем соответствующую строку второй матрицы
    return [row_a + row_b for row_a, row_b in zip(matrix_a, matrix_b)]

# Функция для вертикального объединения двух матриц
def merge_vertically(matrix_a, matrix_b):
    # Добавляем строки второй матрицы к строкам первой
    return matrix_a + matrix_b

# Генерация вспомогательных векторов для создания матриц
def create_support_vectors(n, k):
    vectors = []
    vector = [0] * (n - k)
    # Генерация набора векторов с использованием побитового переключения
    while len(vectors) < k:
        for i in reversed(range(len(vector))):
            if vector[i] == 0:
                vector[i] = 1
                vectors.append(vector[:])
                break
            else:
                vector[i] = 0
    return vectors

# Генерация базовой матрицы Хэмминга
def generate_hamming_base(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return merge_horizontally(create_identity(k), create_support_vectors(n, k))

# Генерация проверочной матрицы
def generate_check_matrix(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return merge_vertically(create_support_vectors(n, k), create_identity(n - k))

# Создание таблицы синдромов для коррекции ошибок
def build_syndrome_map(check_matrix):
    syndrome_map = {}
    for i in range(len(check_matrix[0])):
        error_vect = [0] * len(check_matrix[0])
        error_vect[i] = 1
        syndrome = matrix_vector_multiplication(error_vect, check_matrix)
        syndrome_map[tuple(syndrome)] = error_vect
    return syndrome_map

# Функция для умножения вектора на матрицу
def matrix_vector_multiplication(vector, matrix):
    return [sum(v * m for v, m in zip(vector, col)) % 2 for col in zip(*matrix)]

# Генерация расширенной матрицы Хэмминга
def generate_extended_hamming(r):
    base_matrix = generate_hamming_base(r)
    for row in base_matrix:
        parity_bit = sum(row) % 2
        row.append(parity_bit)
    return base_matrix

# Генерация расширенной проверочной матрицы
def generate_extended_check_matrix(r):
    check_matrix = generate_check_matrix(r)
    extra_row = [0] * len(check_matrix[0])
    check_matrix.append(extra_row)
    for row in check_matrix:
        row.append(1)
    return check_matrix

# Создание случайного вектора ошибок
def create_random_error(length, error_count):
    error_vector = [0] * length
    error_positions = random.sample(range(length), error_count)
    for pos in error_positions:
        error_vector[pos] = 1
    return error_vector

# Функция для коррекции ошибок в полученном слове
def correct_errors(check_matrix, received_code):
    syndrome = matrix_vector_multiplication(received_code, check_matrix)
    syndrome_map = build_syndrome_map(check_matrix)
    if tuple(syndrome) in syndrome_map:
        error_vect = syndrome_map[tuple(syndrome)]
        corrected_code = [(bit + err) % 2 for bit, err in zip(received_code, error_vect)]
        return corrected_code, syndrome
    return received_code, syndrome

# Основная функция для тестирования кодов Хэмминга
def hamming_code_analysis(r, extended=False):
    if extended:
        g_matrix = generate_extended_hamming(r)
        check_matrix = generate_extended_check_matrix(r)
        max_errors = 4
        print("\nАнализ расширенного кода Хэмминга")
    else:
        g_matrix = generate_hamming_base(r)
        check_matrix = generate_check_matrix(r)
        max_errors = 3
        print("\nАнализ стандартного кода Хэмминга")

    print("\nПорождающая матрица G:")
    for row in g_matrix:
        print(row)

    print("\nПроверочная матрица H:")
    for row in check_matrix:
        print(row)

    codewords = [list(col) for col in zip(*g_matrix)]
    chosen_codeword = codewords[random.randint(0, len(codewords) - 1)]
    print("\nВыбранное кодовое слово:")
    print(chosen_codeword)

    # Проверка на количество ошибок от 1 до max_errors
    for error_count in range(1, max_errors + 1):
        if error_count > len(chosen_codeword):
            break
        print(f"\nАнализ с количеством ошибок: {error_count}")

        error_vect = create_random_error(len(chosen_codeword), error_count)
        print(f"Вектор ошибок: {error_vect}")

        received_code = [(bit + err) % 2 for bit, err in zip(chosen_codeword, error_vect)]
        print(f"Полученное кодовое слово с ошибками: {received_code}")

        corrected_code, syndrome = correct_errors(check_matrix, received_code)
        print(f"Синдром ошибки: {syndrome}")
        print(f"Исправленное кодовое слово: {corrected_code}")

        verification_syndrome = matrix_vector_multiplication(corrected_code, check_matrix)
        print(f"Синдром после коррекции (ожидается [0,...,0]): {verification_syndrome}")

# Примеры использования функций
# Анализ стандартного кода Хэмминга для одно-, двух- и трехкратных ошибок
hamming_code_analysis(2)
hamming_code_analysis(3)
hamming_code_analysis(4)

# Анализ расширенного кода Хэмминга для одно-, двух-, трех- и четырехкратных ошибок
hamming_code_analysis(2, extended=True)
hamming_code_analysis(3, extended=True)
hamming_code_analysis(4, extended=True)
