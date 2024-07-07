![image](https://github.com/DmitryShein/simplex-method/assets/108419757/807a9bb8-a520-44de-84e4-1258949b6100)# simplex-method

Постановка задачи
Компания "Мебельный завод" производит столы и стулья.

Для производства одного стола требуется 6 часа работы на станке и 2 часа ручного труда.
Для производства одного стула требуется 2 часа работы на станке и 2 часа ручного труда.

Ресурсы, доступные для производства:

Всего доступно 240 часов работы на станке.
Всего доступно 100 часов ручного труда.

Прибыль:

Прибыль от продажи одного стола составляет 80 долларов.
Прибыль от продажи одного стула составляет 55 долларов.

Нужно определить, сколько столов и стульев нужно произвести, чтобы максимизировать общую прибыль.

Формулировка задачи линейного программирования
Обозначим:

x1 — количество произведенных столов
𝑥2 — количество произведенных стульев

Целевая функция:
Maximize 𝑍 = 80 𝑥1 + 55 𝑥2

Ограничения:
6 x1 + 2 x2 ≤ 240 (часы работы на станке)
2 x1 + 2 x2 ≤ 100 (часы ручного труда)
​x1 ≥ 0, x2 ≥ 0 (неотрицательные переменные)
