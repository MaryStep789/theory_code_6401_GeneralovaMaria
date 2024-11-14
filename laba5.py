import numpy as np
import math
from itertools import combinations, product

# Генерация всех возможных двоичных последовательностей заданной длины с использованием list comprehension
def create_binary_sequences(length):
    return [seq for seq in product([0, 1], repeat=length)]

# Вычисление значения функции с учётом индексов
def compute_parity(binary_vector, indices):
    return np.prod([(binary_vector[i] + 1) % 2 for i in indices])

# Создание вектора для заданных индексов (добавлен вариант с вычислением вектора через генератор)
def generate_vector(indices, num_columns):
    return np.array([compute_parity(seq, indices) for seq in create_binary_sequences(num_columns)], dtype=int) if indices else np.ones(2 ** num_columns, dtype=int)

# Генерация всех подмножеств индексов для заданного размера
def get_all_combinations(num_cols, max_size):
    return [subset for size in range(max_size + 1) for subset in combinations(range(num_cols), size)]

# Расчёт количества строк для матрицы Рида-Маллера
def calculate_matrix_size(r, m):
    return sum(math.comb(m, i) for i in range(r + 1))

# Построение матрицы Рида-Маллера
def build_reed_muller_matrix(r, m):
    matrix_size = calculate_matrix_size(r, m)
    matrix = np.zeros((matrix_size, 2 ** m), dtype=int)
    for i, index_set in enumerate(get_all_combinations(m, r)):
        matrix[i] = generate_vector(index_set, m)
    return matrix

# Сортировка подмножеств индексов для декодирования
def sort_for_decoding(m, r):
    combinations_list = list(combinations(range(m), r))
    combinations_list.sort(key=len)
    return np.array(combinations_list)

# Построение вектора H для заданных индексов
def create_check_vector_H(indices, m):
    return [binary_vector for binary_vector in create_binary_sequences(m) if compute_parity(binary_vector, indices) == 1]

# Нахождение дополнения индекса
def find_complement(indices, m):
    return [i for i in range(m) if i not in indices]

# Вычисление значения функции с учётом вектора t
def calculate_parity_with_t(binary_vector, indices, t_vector):
    return np.prod([(binary_vector[j] + t_vector[j] + 1) % 2 for j in indices])

# Формирование вектора с учётом вектора t
def generate_vector_with_t(indices, m, t_vector):
    return np.array([calculate_parity_with_t(seq, indices, t_vector) for seq in create_binary_sequences(m)], dtype=int) if indices else np.ones(2 ** m, dtype=int)

# Алгоритм декодирования с использованием метода большинства
def majority_decoding(received_word, r, m, matrix_size):
    word = np.copy(received_word)
    decoded_vector = np.zeros(matrix_size, dtype=int)
    max_limit = 2 ** (m - r - 1) - 1
    index = 0

    for i in range(r, -1, -1):
        for indices in sort_for_decoding(m, i):
            max_count = 2 ** (m - i - 1)
            count_zero, count_one = 0, 0
            complement = find_complement(indices, m)

            for t in create_check_vector_H(indices, m):
                V = generate_vector_with_t(complement, m, t)
                checksum = np.dot(word, V) % 2
                count_zero += (checksum == 0)
                count_one += (checksum == 1)

            if count_zero > max_limit and count_one > max_limit:
                return None

            if count_zero > max_count:
                decoded_vector[index] = 0
            elif count_one > max_count:
                decoded_vector[index] = 1
                word = (word + generate_vector(indices, m)) % 2
            index += 1

    return decoded_vector

# Генерация слова с ошибками для эксперимента
def generate_noisy_word_with_errors(G_matrix, num_errors):
    u = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    print("Исходное сообщение:", u)
    encoded_word = np.dot(u, G_matrix) % 2
    error_positions = np.random.choice(len(encoded_word), size=num_errors, replace=False)
    encoded_word[error_positions] = (encoded_word[error_positions] + 1) % 2
    return encoded_word

# Эксперимент с одной ошибкой
def experiment_with_one_error(G_matrix):
    noisy_word = generate_noisy_word_with_errors(G_matrix, 1)
    print("Сообщение с одной ошибкой:", noisy_word)
    decoded_word = majority_decoding(noisy_word, 2, 4, len(G_matrix))
    if decoded_word is None:
        print("\nНе удалось исправить сообщение, требуется повторная отправка")
    else:
        print("Исправленное сообщение:", decoded_word)
        result = np.dot(decoded_word, G_matrix) % 2
        print("Результат исправленного сообщения:", result)

# Эксперимент с двумя ошибками
def experiment_with_two_errors(G_matrix):
    noisy_word = generate_noisy_word_with_errors(G_matrix, 2)
    print("Сообщение с двумя ошибками:", noisy_word)
    decoded_word = majority_decoding(noisy_word, 2, 4, len(G_matrix))
    if decoded_word is None:
        print("\nНе удалось исправить сообщение, требуется повторная отправка")
    else:
        print("Исправленное сообщение:", decoded_word)
        result = np.dot(decoded_word, G_matrix) % 2
        print("Результат исправленного сообщения:", result)

# Главная программа для запуска экспериментов
def main():
    G_matrix = build_reed_muller_matrix(2, 4)
    print("Порождающая матрица G:\n", G_matrix)
    experiment_with_one_error(G_matrix)
    experiment_with_two_errors(G_matrix)

# Запуск программы
if __name__ == '__main__':
    main()
