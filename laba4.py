import numpy as np
import random

# Расширенный код Голея (24, 12, 8)
code_matrix = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
])

# 4.1 Функция формирования порождающей и проверочной матриц для расширенного кода Голея
def generate_G_H(matrix):
    G_matrix = np.hstack((np.eye(12, dtype=int), matrix))
    H_matrix = np.vstack((np.eye(12, dtype=int), matrix))
    return G_matrix, H_matrix

# 4.2 Функция для симуляции ошибок в кодовом слове
def simulate_errors(codeword, G_matrix, error_count):
    encrypted_word = codeword @ G_matrix % 2
    print(f"\nИсходное слово: {codeword}")
    print(f"Отправленное слово: {encrypted_word}")
    
    error_positions = random.sample(range(encrypted_word.shape[0]), error_count)
    error_vector = np.zeros(encrypted_word.shape[0], dtype=int)
    for pos in error_positions:
        error_vector[pos] = 1
    
    word_with_errors = (encrypted_word + error_vector) % 2
    print(f"Слово с {error_count} ошибками: {word_with_errors}")
    return word_with_errors

# 4.3 Функция для поиска ошибок в полученном слове и их исправления
def detect_and_correct_errors(word_with_errors, H_matrix, code_matrix):
    syndrome = word_with_errors @ H_matrix % 2
    correction = None
    
    if sum(syndrome) <= 3:
        correction = np.array(syndrome)
        correction = np.hstack((correction, np.zeros(len(syndrome), dtype=int)))
    else:
        for i in range(len(code_matrix)):
            temp_syndrome = (syndrome + code_matrix[i]) % 2
            if sum(temp_syndrome) <= 2:
                error_index = np.zeros(len(syndrome), dtype=int)
                error_index[i] = 1
                correction = np.hstack((temp_syndrome, error_index))
    
    if correction is None:
        syndrome_B = syndrome @ code_matrix % 2
        if sum(syndrome_B) <= 3:
            correction = np.hstack((np.zeros(len(syndrome), dtype=int), syndrome_B))
        else:
            for i in range(len(code_matrix)):
                temp_syndrome = (syndrome_B + code_matrix[i]) % 2
                if sum(temp_syndrome) <= 2:
                    error_index = np.zeros(len(syndrome), dtype=int)
                    error_index[i] = 1
                    correction = np.hstack((error_index, temp_syndrome))
    
    return correction

# 4.4 Функция для исправления ошибок
def fix_errors(original_word, word_with_errors, H_matrix, code_matrix, G_matrix):
    correction = detect_and_correct_errors(word_with_errors, H_matrix, code_matrix)
    
    if correction is None:
        print("Ошибка обнаружена, исправить невозможно!")
        return
    
    corrected_word = (word_with_errors + correction) % 2
    print("Исправленное отправленное сообщение:", corrected_word)
    
    expected_word = original_word @ G_matrix % 2
    if not np.array_equal(expected_word, corrected_word):
        print("Сообщение было декодировано с ошибкой!")

# 4.1 - 4.2 Исследование расширенного кода Голея
def golay_code_research():
    print("-------------------------------\n Часть 1")
    
    G_matrix, H_matrix = generate_G_H(code_matrix)
    print(f"G:\n{G_matrix}\nH:\n{H_matrix}")
    
    codeword = np.array([i % 2 for i in range(len(G_matrix))])
    
    for error_count in range(5):
        word_with_errors = simulate_errors(codeword, G_matrix, error_count)
        fix_errors(codeword, word_with_errors, H_matrix, code_matrix, G_matrix)
        print('')

# 4.5 Порождающая и проверочная матрицы для кода Рида-Маллера
def generate_RM_G(r, m):
    if 0 < r < m:
        upper_left = generate_RM_G(r, m - 1)
        lower_right = generate_RM_G(r - 1, m - 1)
        return np.hstack([np.
vstack([upper_left, np.zeros((len(lower_right), len(upper_left.T)), int)]), 
                          np.vstack([upper_left, lower_right])])
    elif r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        upper = generate_RM_G(m - 1, m)
        lower = np.zeros((1, 2 ** m), dtype=int)
        lower[0][len(lower.T) - 1] = 1
        return np.vstack([upper, lower])

def generate_RM_H(i, m):
    H_matrix = np.array([[1, 1], [1, -1]])
    result = np.kron(np.eye(2 ** (m - i)), H_matrix)
    result = np.kron(result, np.eye(2 ** (i - 1)))
    return result

# 4.4 Исследование кода Рида-Маллера
def reed_muller_code_research(word, G_matrix, error_count, m):
    word_with_errors = simulate_errors(word, G_matrix, error_count)
    
    for i in range(len(word_with_errors)):
        if word_with_errors[i] == 0:
            word_with_errors[i] = -1
    
    w_t = [word_with_errors @ generate_RM_H(1, m)]
    for i in range(2, m + 1):
        w_t.append(w_t[-1] @ generate_RM_H(i, m))
    
    max_value = w_t[0][0]
    index = -1
    for i in range(len(w_t)):
        for j in range(len(w_t[i])):
            if abs(w_t[i][j]) > abs(max_value):
                index = j
                max_value = w_t[i][j]
    
    counter = 0
    for i in range(len(w_t)):
        for j in range(len(w_t[i])):
            if abs(w_t[i][j]) == abs(max_value):
                counter += 1
            if counter > 1:
                print("Исправить ошибку невозможно.\n")
                return
    
    corrected_word = list(map(int, list(('{' + f'0:0{m}b' + '}').format(index))))
    corrected_word.append(1 if max_value > 0 else 0)
    print(f"Исправленное сообщение: {np.array(corrected_word[::-1])}")


# 4.3 - 4.5 Исследование кода Рида-Маллера
def reed_muller_research():
    print("-------------------------------\n Часть 2")
    
    m = 3
    print(f"\nПорождающая матрица Рида-Маллера (1,3): \n{generate_RM_G(1, m)}\n")
    codeword = np.array([i % 2 for i in range(len(generate_RM_G(1, m)))])
    
    # Исследование для одно- и двукратных ошибок
    for error_count in range(1, 3):
        reed_muller_code_research(codeword, generate_RM_G(1, m), error_count, m)

    m = 4
    print(f"\nПорождающая матрица Рида-Маллера (1,4): \n{generate_RM_G(1, m)}\n")
    codeword = np.array([i % 2 for i in range(len(generate_RM_G(1, m)))])
    
    # Исследование для одно-, двух-, трёх- и четырёхкратных ошибок
    for error_count in range(1, 5):
        reed_muller_code_research(codeword, generate_RM_G(1, m), error_count, m)

# Главная функция для запуска исследования
if __name__ == '__main__':
    golay_code_research()
    reed_muller_research()