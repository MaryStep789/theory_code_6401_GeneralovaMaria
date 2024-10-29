import numpy as np

# Функция для генерации порождающей матрицы для кодирования
def create_generator_matrix(info_bits, code_length, custom_matrix=None):
    # Создаём единичную матрицу для информационных битов
    identity_matrix = np.eye(info_bits, dtype=int)
    # Если предоставлена пользовательская матрица, используем её
    if custom_matrix is None:
        encoding_matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    else:
        encoding_matrix = np.array(custom_matrix)
    # Объединяем матрицы для получения порождающей матрицы
    generator_matrix = np.hstack((identity_matrix, encoding_matrix))
    return generator_matrix

# Функция для создания проверочной матрицы
def create_parity_check_matrix(encoding_matrix):
    # Создаём единичную матрицу, соответствующую размеру дополнительной части
    identity_parity = np.eye(encoding_matrix.shape[1], dtype=int)
    # Объединяем транспонированную матрицу и единичную матрицу для получения H
    parity_check_matrix = np.hstack((encoding_matrix.T, identity_parity))
    return parity_check_matrix

# Генерация синдромов ошибок
def generate_error_syndromes(parity_matrix):
    error_syndromes = {}
    # Перебор всех возможных одноразрядных ошибок
    for i in range(parity_matrix.shape[1]):
        error_vector = np.zeros(parity_matrix.shape[1], dtype=int)
        error_vector[i] = 1
        # Рассчёт синдрома текущей ошибки
        syndrome = np.dot(parity_matrix, error_vector) % 2
        error_syndromes[tuple(syndrome)] = error_vector
    return error_syndromes

# Генерация кодового слова из информационного вектора
def encode_information(data_bits, generator_matrix):
    return np.dot(data_bits, generator_matrix) % 2

# Функция для внесения ошибок в кодовое слово
def inject_error(codeword_bits, error_positions):
    for pos in error_positions:
        codeword_bits[pos] ^= 1  # Инвертируем бит для введения ошибки
    return codeword_bits

# Рассчёт синдрома для принятого слова
def compute_syndrome(received_bits, parity_matrix):
    return np.dot(parity_matrix, received_bits) % 2

# Исправление ошибки на основе синдрома
def fix_errors(received_bits, syndrome, error_syndromes):
    if tuple(syndrome) in error_syndromes:
        error_vector = error_syndromes[tuple(syndrome)]
        corrected_bits = (received_bits + error_vector) % 2
        return corrected_bits
    return received_bits

# Основная функция, выполняющая все этапы кодирования, декодирования и исправления
def main():
    print("=== Лабораторная работа по теории кодирования ===\n")

    # Часть 1: Пример для (7, 4, 3) кода
    info_bits, code_length = 4, 7
    print("Часть 1: Генерация матриц для кода (7, 4, 3)")
    generator_matrix = create_generator_matrix(info_bits, code_length)
    print("Порождающая матрица:\n", generator_matrix)
    
    parity_matrix = create_parity_check_matrix(generator_matrix[:, info_bits:])
    print("\nПроверочная матрица:\n", parity_matrix)
    
    error_syndromes = generate_error_syndromes(parity_matrix)
    print("\nСиндромы для одноразрядных ошибок:")
    for syndrome, error in error_syndromes.items():
        print(f"Синдром {syndrome}: Ошибка {error}")
    
    # Пример кодирования и внесения одноразрядной ошибки
    data_bits = np.array([1, 0, 1, 1])
    codeword = encode_information(data_bits, generator_matrix)
    print("\nКодовое слово:", codeword)
    
    received_with_error = inject_error(codeword.copy(), [2])
    print("Кодовое слово с ошибкой в одном разряде:", received_with_error)
    
    syndrome = compute_syndrome(received_with_error, parity_matrix)
    print("Синдром ошибки:", syndrome)
    
    corrected_codeword = fix_errors(received_with_error, syndrome, error_syndromes)
    print("Исправленное слово:", corrected_codeword)
    print("Совпадает с исходным:", np.array_equal(corrected_codeword, codeword))
    
    # Пример для двукратной ошибки
    received_with_double_error = inject_error(codeword.copy(), [1, 5])
    print("\nКодовое слово с двукратной ошибкой:", received_with_double_error)
    
    syndrome_double = compute_syndrome(received_with_double_error, parity_matrix)
    print("Синдром для двукратной ошибки:", syndrome_double)
    
    corrected_double_error = fix_errors(received_with_double_error, syndrome_double, error_syndromes)
    print("Попытка исправления двукратной ошибки:", corrected_double_error)
    print("Результат отличается от исходного:", not np.array_equal(corrected_double_error, codeword))

main()
