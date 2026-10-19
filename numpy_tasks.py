"""Заготовки задач на NumPy."""

import numpy as np
import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros_like(vectors[0], dtype=float)
    for A, x in zip(matrices, vectors):
        result = result + A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(float)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(matrix[:, j])) for j in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_variances = matrix.var(axis=1)
    column_variances = matrix.var(axis=0)

    plot_histograms(matrix, row_means, column_means)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


def plot_histograms(matrix, row_means, column_means):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(row_means, bins=20, color='blue', alpha=0.7)
    axes[0].set_title('Гистограмма средних по строкам')
    axes[0].set_xlabel('Среднее значение')
    axes[1].hist(column_means, bins=20, color='green', alpha=0.7)
    axes[1].set_title('Гистограмма средних по столбцам')
    axes[1].set_xlabel('Среднее значение')
    plt.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    result = np.empty((rows, columns), dtype=float)
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                result[i, j] = first
            else:
                result[i, j] = second
    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = np.array(data.shape_color), np.array(data.background_color)

    image = np.tile(background_color, (image_height, image_width, 1)).astype(np.uint8)

    cy, cx = image_height // 2, image_width // 2
    y1 = max(0, cy - height // 2)
    y2 = min(image_height, cy + height // 2)
    x1 = max(0, cx - width // 2)
    x2 = min(image_width, cx + width // 2)

    image[y1:y2, x1:x2] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    a, b = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = np.array(data.shape_color), np.array(data.background_color)

    image = np.tile(background_color, (image_height, image_width, 1)).astype(np.uint8)

    cy, cx = image_height / 2, image_width / 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - cx) ** 2 / a ** 2) + ((y - cy) ** 2 / b ** 2) <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    series = np.array(data.values, dtype=float)
    p = data.window

    mean = float(np.mean(series))
    variance = float(np.var(series))
    std = float(np.std(series))

    local_maxima_indices = []
    local_minima_indices = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            local_maxima_indices.append(i)
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            local_minima_indices.append(i)

    moving_average = np.array([
        np.mean(series[i:i + p]) for i in range(len(series) - p + 1)
    ])

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.array(data.labels, dtype=int)
    if data.class_count is not None:
        num_classes = data.class_count
    else:
        num_classes = int(labels.max()) + 1 if len(labels) > 0 else 0

    result = np.zeros((len(labels), num_classes), dtype=int)
    for i, label in enumerate(labels):
        if 0 <= label < num_classes:
            result[i, label] = 1
    return result


# ==================== ТЕСТЫ ====================

def run_tests():
    print("=== Задача 1: sum_prod ===")
    A1 = np.array([[1, 2], [3, 4]])
    A2 = np.array([[5, 6], [7, 8]])
    x1 = np.array([[1], [0]])
    x2 = np.array([[0], [1]])
    result = sum_prod(MatrixVectorBatchInput([A1, A2], [x1, x2]))
    expected = np.array([[1], [3]]) + np.array([[6], [8]])
    assert np.allclose(result, expected), f"Ожидалось {expected}, получено {result}"
    print("Тест пройден\n")

    print("=== Задача 2: binarize ===")
    matrix = np.array([[1, 5, 3], [2, 8, 0]])
    result = binarize(BinarizeInput(matrix, 3))
    expected = np.array([[0, 1, 0], [0, 1, 0]])
    assert np.array_equal(result, expected), f"Ожидалось {expected}, получено {result}"
    assert np.array_equal(matrix, np.array([[1, 5, 3], [2, 8, 0]]))
    print("Тест пройден\n")

    print("=== Задача 3: unique_rows и unique_columns ===")
    matrix = np.array([[1, 2, 2, 3], [4, 4, 5, 5]])
    rows_result = unique_rows(MatrixInput(matrix))
    assert rows_result == [[1, 2, 3], [4, 5]], f"Получено {rows_result}"
    cols_result = unique_columns(MatrixInput(matrix))
    assert cols_result == [[1, 4], [2, 4], [2, 5], [3, 5]], f"Получено {cols_result}"
    print("Тест пройден\n")

    print("=== Задача 4: matrix_statistics ===")
    stats = matrix_statistics(RandomMatrixInput(rows=100, columns=50, mean=0, std=1, seed=42))
    assert stats.matrix.shape == (100, 50)
    assert len(stats.row_means) == 100
    assert len(stats.column_means) == 50
    assert len(stats.row_variances) == 100
    assert len(stats.column_variances) == 50
    assert abs(stats.column_means.mean()) < 0.3
    print("Тест пройден\n")

    print("=== Задача 5: chess ===")
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(result, expected), f"Получено:\n{result}"
    print("Тест пройден\n")

    print("=== Задача 6: draw_rectangle и draw_ellipse ===")
    rect = draw_rectangle(RectangleInput(
        width=4, height=4, image_height=10, image_width=10,
        shape_color=(255, 0, 0), background_color=(0, 0, 0)
    ))
    assert rect.shape == (10, 10, 3)
    assert rect[5, 5, 0] == 255
    assert rect[0, 0, 0] == 0

    ellipse = draw_ellipse(EllipseInput(
        semi_axis_x=3, semi_axis_y=3, image_height=10, image_width=10,
        shape_color=(0, 255, 0), background_color=(0, 0, 0)
    ))
    assert ellipse.shape == (10, 10, 3)
    assert ellipse[5, 5, 1] == 255
    print("Тест пройден\n")

    print("=== Задача 7: analyze_time_series ===")
    series = [1, 3, 2, 5, 4, 6, 3]
    stats = analyze_time_series(TimeSeriesInput(values=series, window=3))
    assert abs(stats.mean - 24/7) < 1e-6
    assert stats.local_maxima_indices == [1, 3, 5], f"Получено {stats.local_maxima_indices}"
    assert stats.local_minima_indices == [2, 4], f"Получено {stats.local_minima_indices}"
    assert len(stats.moving_average) == len(series) - 3 + 1
    print("Тест пройден\n")

    print("=== Задача 8: one_hot ===")
    labels = [0, 2, 3, 0]
    result = one_hot(OneHotInput(labels=labels, class_count=None))
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert np.array_equal(result, expected), f"Получено:\n{result}"

    result2 = one_hot(OneHotInput(labels=[0, 1, 2], class_count=5))
    assert result2.shape == (3, 5)
    assert result2[0, 0] == 1 and result2[1, 1] == 1 and result2[2, 2] == 1
    print("Тест пройден\n")

    print("Все тесты успешно пройдены!")


if __name__ == "__main__":
    run_tests()
