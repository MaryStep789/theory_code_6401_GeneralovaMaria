import numpy as np
import random

# Кодирование сообщения с использованием порождающего многочлена
def encode_data(input_message, generator_poly):
    """
    Кодирует сообщение с использованием порождающего многочлена.
    :param input_message: массив битов исходного сообщения
    :param generator_poly: массив коэффициентов порождающего многочлена
    :return: закодированное сообщение (кодовое слово)
    """
    return np.polymul(input_message, generator_poly) % 2

# Внесение ошибок в сообщение
def add_random_errors(codeword, num_errors):
    """
    Вносит случайные ошибки в кодовое слово.
    :param codeword: массив битов кодового слова
    :param num_errors: количество ошибок
    :return: искажённое кодовое слово
    """
    length = len(codeword)
    error_positions = random.sample(range(length), num_errors)
    print(f"Позиции ошибок: {error_positions}")
    for position in error_positions:
        codeword[position] ^= 1  # Инвертируем бит
    return codeword

# Внесение пакета ошибок в сообщение
def add_error_burst(codeword, burst_length):
    """
    Вносит пакет ошибок в сообщение.
    :param codeword: массив битов кодового слова
    :param burst_length: длина пакета ошибок
    :return: искажённое кодовое слово
    """
    length = len(codeword)
    start_index = random.randint(0, length - burst_length)
    for i in range(burst_length):
        codeword[(start_index + i) % length] ^= 1  # Инвертируем биты в пакете
    print(f"Пакет ошибок: от позиции {start_index} до {(start_index + burst_length - 1) % length}")
    return codeword

# Проверка, является ли синдром допустимой ошибкой
def validate_error_pattern(syndrome, max_error_size):
    """
    Проверяет, может ли данный синдром быть исправляемой ошибкой.
    :param syndrome: массив битов синдрома
    :param max_error_size: максимальное количество ошибок, которые можно исправить
    :return: True, если синдром соответствует исправляемой ошибке, иначе False
    """
    trimmed_syndrome = np.trim_zeros(syndrome, 'f')
    trimmed_syndrome = np.trim_zeros(trimmed_syndrome, 'b')
    return 0 < len(trimmed_syndrome) <= max_error_size

# Декодирование сообщения
def decode_data(received_word, generator_poly, max_errors, burst_mode):
    """
    Декодирует полученное сообщение, исправляя ошибки.
    :param received_word: искажённое кодовое слово
    :param generator_poly: массив коэффициентов порождающего многочлена
    :param max_errors: максимальное количество исправляемых ошибок
    :param burst_mode: True, если используется пакетный режим
    :return: декодированное сообщение или None, если декодировать не удалось
    """
    n = len(received_word)
    syndrome = np.polydiv(received_word, generator_poly)[1] % 2  # Синдром

    for shift in range(n):
        error_poly = np.zeros(n, dtype=int)
        error_poly[n - shift - 1] = 1
        shifted_syndrome = np.polymul(syndrome, error_poly) % 2

        syndrome_remainder = np.polydiv(shifted_syndrome, generator_poly)[1] % 2

        if burst_mode:
            if validate_error_pattern(syndrome_remainder, max_errors):
                error_correction = np.polymul(error_poly, syndrome_remainder) % 2
                corrected_codeword = np.polyadd(error_correction, received_word) % 2
                return np.array(np.polydiv(corrected_codeword, generator_poly)[0] % 2).astype(int)
        else:
            if sum(syndrome_remainder) <= max_errors:
                error_correction = np.polymul(error_poly, syndrome_remainder) % 2
                corrected_codeword = np.polyadd(error_correction, received_word) % 2
                return np.array(np.polydiv(corrected_codeword, generator_poly)[0] % 2).astype(int)
    return None

# Исследование кода (7,4)
def explore_code_7_4():
    """
    Исследует код (7,4) с однобитными ошибками.
    """
    print("-------------------------------\nИсследование кода (7,4)\n")
    generator_poly = np.array([1, 1, 0, 1])  # Порождающий многочлен
    max_errors = 1

    for error_count in range(1, 4):
        original_message = np.array([1, 0, 1, 0])
        print(f"Исходное сообщение: {original_message}")
        encoded_message = encode_data(original_message, generator_poly)
        print(f"Закодированное сообщение: {encoded_message}")
        corrupted_message = add_random_errors(encoded_message.copy(), error_count)
        print(f"Сообщение с ошибками: {corrupted_message}")
        decoded_message = decode_data(corrupted_message, generator_poly, max_errors, burst_mode=False)
        print(f"Декодированное сообщение: {decoded_message}")
        if np.array_equal(original_message, decoded_message):
            print("Сообщение успешно декодировано.\n")
        else:
            print("Декодирование не удалось.\n")

# Исследование кода (15,9)
def explore_code_15_9():
    """
    Исследует код (15,9) с пакетными ошибками.
    """
    print("-------------------------------\nИсследование кода (15,9)\n")
    generator_poly = np.array([1, 0, 0, 1, 1, 1, 1])  # Порождающий многочлен
    max_errors = 3

    for burst_length in range(1, 5):
        original_message = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])
        print(f"Исходное сообщение: {original_message}")
        encoded_message = encode_data(original_message, generator_poly)
        print(f"Закодированное сообщение: {encoded_message}")
        corrupted_message = add_error_burst(encoded_message.copy(), burst_length)
        print(f"Сообщение с пакетом ошибок: {corrupted_message}")
        decoded_message = decode_data(corrupted_message, generator_poly, max_errors, burst_mode=True)
        print(f"Декодированное сообщение: {decoded_message}")
        if np.array_equal(original_message, decoded_message):
            print("Сообщение успешно декодировано.\n")
        else:
            print("Декодирование не удалось.\n")

# Основная часть программы
if __name__ == '__main__':
    explore_code_7_4()
    explore_code_15_9()
