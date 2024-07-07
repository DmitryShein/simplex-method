import numpy as np

# Коэффициенты целевой функции
c = np.array([80, 55])

# Коэффициенты ограничений
A = np.array([
    [6, 2],
    [2, 2]
])

# Правая часть ограничений
b = np.array([240, 100])

# Инициализация базисных переменных
num_constraints, num_variables = A.shape
slack_variables = np.eye(num_constraints)
tableau = np.hstack((A, slack_variables, b.reshape(-1, 1)))
tableau = np.vstack((tableau, np.hstack((c, np.zeros(num_constraints + 1)))))

# Симплекс-метод
def simplex(tableau):
    while np.any(tableau[-1, :-1] > 0):
        # Выбор входящей переменной
        pivot_col = np.argmax(tableau[-1, :-1])
        
        # Выбор исходящей переменной
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        
        # Обновление таблицы
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_element
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
    
    # Получение оптимальных значений переменных
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        if np.sum(tableau[i, :num_variables] == 1) == 1:
            col = np.argmax(tableau[i, :num_variables])
            solution[col] = tableau[i, -1]
    
    return solution, tableau[-1, -1]

# Решение задачи
solution, max_profit = simplex(tableau)
max_profit = max_profit * (-1)
print(f"Количество столов для производства: {solution[0]:.2f}")
print(f"Количество стульев для производства: {solution[1]:.2f}")
print(f"Максимальная прибыль: {max_profit:.2f} долларов")
