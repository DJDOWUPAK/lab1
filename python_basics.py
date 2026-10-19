"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set('aeiouAEIOU')
    return sum(1 for ch in text if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    d = 2
    n = number
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    result = ""
    for prime in sorted(factors.keys()):
        power = factors[prime]
        if power == 1:
            result += f"({prime})"
        else:
            result += f"({prime}**{power})"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 1
    total = 0
    while total < cube_count:
        total += k * k
        if total == cube_count:
            return k
        k += 1
    return "it is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    n = len(digits)
    mid = n // 2
    if n % 2 == 0:
        left = digits[:mid - 1]
        right = digits[mid + 1:]
    else:
        left = digits[:mid]
        right = digits[mid + 1:]
    return sum(left) == sum(right)


# ==================== ТЕСТЫ ====================

def run_tests():
    print("=== Задача 1: count_vowels ===")
    # Примеры из условия
    assert count_vowels(TextInput("hello")) == 2  # e, o
    assert count_vowels(TextInput("AEIOU")) == 5  # все гласные
    assert count_vowels(TextInput("bcdfg")) == 0  # нет гласных
    # Граничные случаи
    assert count_vowels(TextInput("")) == 0  # пустая строка
    assert count_vowels(TextInput("a")) == 1  # одна гласная
    assert count_vowels(TextInput("AaEeIiOoUu")) == 10  # все гласные в разных регистрах
    assert count_vowels(TextInput("12345")) == 0  # цифры
    print("Все тесты пройдены\n")

    print("=== Задача 2: has_unique_characters ===")
    # Примеры из условия
    assert has_unique_characters(TextInput("abc")) == True
    assert has_unique_characters(TextInput("aab")) == False
    # Граничные случаи
    assert has_unique_characters(TextInput("")) == True  # пустая строка
    assert has_unique_characters(TextInput("a")) == True  # один символ
    assert has_unique_characters(TextInput("abcdefghijklmnopqrstuvwxyz")) == True  # все уникальные
    assert has_unique_characters(TextInput("aa")) == False  # два одинаковых
    print("Все тесты пройдены\n")

    print("=== Задача 3: count_one_bits ===")
    # Примеры из условия
    assert count_one_bits(PositiveIntegerInput(1)) == 1  # 1 в двоичной
    assert count_one_bits(PositiveIntegerInput(7)) == 3  # 111 в двоичной
    assert count_one_bits(PositiveIntegerInput(8)) == 1  # 1000 в двоичной
    # Граничные случаи
    assert count_one_bits(PositiveIntegerInput(0)) == 0  # 0 в двоичной
    assert count_one_bits(PositiveIntegerInput(255)) == 8  # 11111111
    assert count_one_bits(PositiveIntegerInput(1024)) == 1  # 10000000000
    print("Все тесты пройдены\n")

    print("=== Задача 4: multiplicative_persistence ===")
    # Примеры из условия
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3  # 3*9=27, 2*7=14, 1*4=4
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0  # уже однозначное
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4  # 9*9*9=729, 7*2*9=126, 1*2*6=12, 1*2=2
    # Граничные случаи
    assert multiplicative_persistence(PositiveIntegerInput(1)) == 0  # однозначное
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1  # 1*0=0
    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2  # 2*5=10, 1*0=0
    print("Все тесты пройдены\n")

    print("=== Задача 5: mse ===")
    # Примеры из условия
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0  # одинаковые векторы
    assert mse(VectorPairInput([1, 2, 3], [4, 5, 6])) == 9.0  # (9+9+9)/3
    # Граничные случаи
    assert mse(VectorPairInput([0], [0])) == 0.0  # один элемент
    assert mse(VectorPairInput([1], [2])) == 1.0  # один элемент, разница 1
    assert mse(VectorPairInput([1, 1], [2, 2])) == 1.0  # (1+1)/2
    print("Все тесты пройдены\n")

    print("=== Задача 6: prime_factorization ===")
    # Примеры из условия
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(7)) == "(7)"  # простое число
    # Граничные случаи
    assert prime_factorization(PositiveIntegerInput(1)) == ""  # 1 не имеет простых множителей
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"  # простое число
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"
    print("Все тесты пройдены\n")

    print("=== Задача 7: pyramid ===")
    # Примеры из условия
    assert pyramid(PositiveIntegerInput(1)) == 1  # 1^2
    assert pyramid(PositiveIntegerInput(5)) == 2  # 1^2 + 2^2
    assert pyramid(PositiveIntegerInput(14)) == 3  # 1^2 + 2^2 + 3^2
    assert pyramid(PositiveIntegerInput(30)) == 4  # 1^2 + 2^2 + 3^2 + 4^2
    # Граничные случаи
    assert pyramid(PositiveIntegerInput(2)) == "it is impossible"  # нельзя представить
    assert pyramid(PositiveIntegerInput(3)) == "it is impossible"
    assert pyramid(PositiveIntegerInput(55)) == 5  # 1+4+9+16+25
    print("Все тесты пройдены\n")

    print("=== Задача 8: is_balanced_number ===")
    # Примеры из условия
    assert is_balanced_number(PositiveIntegerInput(1234006)) == True  # 1+2+3 = 0+0+6
    assert is_balanced_number(PositiveIntegerInput(123456)) == False  # 1+2 != 5+6
    # Граничные случаи
    assert is_balanced_number(PositiveIntegerInput(1)) == True  # одна цифра, суммы пустые
    assert is_balanced_number(PositiveIntegerInput(11)) == True  # 1 = 1
    assert is_balanced_number(PositiveIntegerInput(123)) == False  # 1 != 3
    assert is_balanced_number(PositiveIntegerInput(12321)) == True  # 1+2 = 2+1
    print(" Все тесты пройдены\n")

    print("Все тесты успешно пройдены!")


if __name__ == "__main__":
    run_tests()
