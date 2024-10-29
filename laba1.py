import numpy as np
from itertools import combinations

# Задание 1.1: Функция приведения матрицы к ступенчатому виду (REF)
def matrix_ref(matrix):
    matrix_copy = np.array(matrix)
    rows, columns = matrix_copy.shape
    pivot = 0
    for row in range(rows):
        if pivot >= columns:
            return matrix_copy
        row_index = row
        while matrix_copy[row_index, pivot] == 0:
            row_index += 1
            if row_index == rows:
                row_index = row
                pivot += 1
                if pivot == columns:
                    return matrix_copy
        # Меняем строки местами
        matrix_copy[[row_index, row]] = matrix_copy[[row, row_index]]
        for row_below in range(row + 1, rows):
            if matrix_copy[row_below, pivot] != 0:
                matrix_copy[row_below] = (matrix_copy[row_below] + matrix_copy[row]) % 2
        pivot += 1
    return matrix_copy

# Задание 1.2: Функция приведения матрицы к приведённому ступенчатому виду (RREF)
def matrix_rref(mat):
    ref_matrix = matrix_ref(mat)
    num_rows, num_columns = ref_matrix.shape

    for row in range(num_rows - 1, -1, -1):
        pivot_col = np.argmax(ref_matrix[row] != 0)
        if ref_matrix[row, pivot_col] != 0:
            for upper_row in range(row - 1, -1, -1):
                if ref_matrix[upper_row, pivot_col] != 0:
                    ref_matrix[upper_row] = (ref_matrix[upper_row] + ref_matrix[row]) % 2
    while not any(ref_matrix[num_rows - 1]):
        ref_matrix = ref_matrix[:-1, :]
        num_rows -= 1
    return ref_matrix

# Задание 1.3: Ведущие столбцы и создание сокращённой матрицы
def find_pivot_columns(matrix):
    pivots = []
    for row in range(len(matrix)):
        for index, value in enumerate(matrix[row]):
            if value == 1:
                pivots.append(index)
                break
    return pivots

# Удаление ведущих столбцов
def remove_pivot_columns(matrix, pivot_columns):
    matrix_copy = np.array(matrix)
    reduced_matrix = np.delete(matrix_copy, pivot_columns, axis=1)
    return reduced_matrix

# Задание 1.3.4: Формирование проверочной матрицы H
def create_check_matrix(reduced, pivot_columns, num_columns):
    num_rows = np.shape(reduced)[1]
    check_matrix = np.zeros((num_columns, num_rows), dtype=int)
    identity_matrix = np.eye(6, dtype=int)

    check_matrix[pivot_columns, :] = reduced
    non_pivots = [idx for idx in range(num_columns) if idx not in pivot_columns]
    check_matrix[non_pivots, :] = identity_matrix
    return check_matrix

# Основная функция выполнения всех шагов лабораторной работы
def CodingProcedure(input_matrix):
    ref_form = matrix_rref(input_matrix)
    print("RREF матрица G* =")
    print(ref_form)

    pivot_columns = find_pivot_columns(ref_form)
    print(f"Пивоты = {pivot_columns}")

    reduced_form = remove_pivot_columns(ref_form, pivot_columns)
    print("Сокращённая матрица X =")
    print(reduced_form)

    num_columns = np.shape(input_matrix)[1]
    check_matrix = create_check_matrix(reduced_form, pivot_columns, num_columns)
    print("Проверочная матрица H =")
    print(check_matrix)

    return check_matrix

# Генерация всех кодовых слов через сложение строк
def codewords_by_sum(G_matrix):
    num_rows = G_matrix.shape[0]
    codeword_set = set()

    for row_count in range(1, num_rows + 1):
        for row_comb in combinations(range(num_rows), row_count):
            summed_word = np.bitwise_xor.reduce(G_matrix[list(row_comb)], axis=0)
            codeword_set.add(tuple(summed_word))

    codeword_set.add(tuple(np.zeros(G_matrix.shape[1], dtype=int)))
    return np.array(list(codeword_set))

# Генерация кодовых слов умножением двоичных слов на G
def codewords_by_multiplication(G_matrix):
    row_count = G_matrix.shape[0]
    col_count = G_matrix.shape[1]
    codewords_list = []

    for i in range(2 ** row_count):
        binary_sequence = np.array(list(np.binary_repr(i, row_count)), dtype=int)
        codeword = np.dot(binary_sequence, G_matrix) % 2
        codewords_list.append(codeword)

    return np.array(codewords_list)

# Проверка кодового слова с помощью матрицы H
def verify_codeword(codeword, check_matrix):
    return np.dot(codeword, check_matrix) % 2

# Вычисление кодового расстояния
def calculate_min_distance(codewords):
    min_dist = float('inf')

    for index_i in range(len(codewords)):
        for index_j in range(index_i + 1, len(codewords)):
            hamming_dist = np.sum(np.bitwise_xor(codewords[index_i], codewords[index_j]))
            if hamming_dist > 0:
                min_dist = min(min_dist, hamming_dist)
    return min_dist

# Выполнение всех шагов с ошибками
def LinearCodeWithErrors(input_matrix):
    ref_form = matrix_rref(input_matrix)
    pivot_columns = find_pivot_columns(ref_form)
    reduced_form = remove_pivot_columns(ref_form, pivot_columns)
    num_columns = np.shape(input_matrix)[1]
    check_matrix = create_check_matrix(reduced_form, pivot_columns, num_columns)

    print("G* (RREF матрица) =")
    print(ref_form)
    print(f"lead = {pivot_columns}")
    print("Сокращённая матрица X =")
    print(reduced_form)
    print("Проверочная матрица H =")
    print(check_matrix)

    # Генерация кодовых слов через сложение строк
    codewords_1 = codewords_by_sum(ref_form)
    print("Все кодовые слова (способ 1):")
    print(codewords_1)

    # Генерация кодовых слов умножением двоичных слов на G
    codewords_2 = codewords_by_multiplication(ref_form)
    print("Все кодовые слова (способ 2):")
    print(codewords_2)

    assert set(map(tuple, codewords_1)) == set(map(tuple, codewords_2)), "Наборы кодовых слов не совпадают!"

    for codeword in codewords_1:
        result = verify_codeword(codeword, check_matrix)
        assert np.all(result == 0), f"Ошибка: кодовое слово {codeword} не прошло проверку матрицей H"

    print("Все кодовые слова прошли проверку матрицей H.")

    # Вычисление кодового расстояния
    min_distance = calculate_min_distance(codewords_1)
    error_tolerance = (min_distance - 1) // 2 if min_distance > 1 else 1
    print(f"Кодовое расстояние d = {min_distance}")
    print(f"Кратность обнаруживаемой ошибки t = {error_tolerance}")

    # Проверка ошибки кратности t
    error_pattern1 = np.zeros(num_columns, dtype=int)
    error_pattern1[2] = 1
    codeword_v = codewords_1[4]
    codeword_with_error1 = (codeword_v + error_pattern1) % 2
    print(f"codeword_v + error1 = {codeword_with_error1}")
    print(f"(codeword_v + error1)@H = {verify_codeword(codeword_with_error1, check_matrix)} - ошибка")

    # Проверка ошибки кратности t+1
    error_pattern2 = np.zeros(num_columns, dtype=int)
    error_pattern2[6] = 1
    error_pattern2[9] = 1
    codeword_with_error2 = (codeword_v + error_pattern2) % 2
    print(f"codeword_v + error2 = {codeword_with_error2}")
    print(f"(codeword_v + error2)@H = {verify_codeword(codeword_with_error2, check_matrix)} - без ошибки")

    return check_matrix

# Пример использования
matrix = ([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
[0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
[0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
)
result = LinearCodeWithErrors(matrix)