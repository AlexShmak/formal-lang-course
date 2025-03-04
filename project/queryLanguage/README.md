# Query Language

Язык запросов к графам, реализованный с помощью ANTLRv4

---

## `Query Language Type Inferencer`

### `QueryLanguageTypeInferenceVisitor.py`

Представляет собой механизм вывода типов, гарантирующий корректность построения запросов. Реализован с помощью класса
`QueryLanguageVisitor`, который генерируется ANTLRv4.

#### Особенности

- **Вывод типов**: Определяет и проверяет типы переменных и выражений, включая графы, рёбра, вершины, числа, множества и
  автоматы.
- **Контекстно-зависимый анализ**: Обеспечивает правильное использование переменных и операций, генерируя исключения для
  некорректных комбинаций типов или переопределений переменных.
- **Работа с регулярными выражениями и графами**: Поддерживает проверку типов для операций, связанных с регулярными
  выражениями и графовыми структурами.

#### Основные методы

- **`visitDeclare`**: Объявляет переменную как тип графа.
- **`visitAdd`**: Проверяет и обрабатывает добавление рёбер или вершин в граф.
- **`visitRemove`**: Обрабатывает удаление рёбер, вершин и множеств вершин из графа.
- **`visitBind`**: Назначает тип переменной на основе выражения.
- **`visitRegexp`**: Выполняет вывод типов для регулярных выражений, поддерживая такие операции, как конкатенация,
  объединение и повторение.
- **`visitSelect`**: Проверяет и определяет типы в конструкциях `Select`, включая корректность использования переменных
  и условий.
- **`visitVar`**: Проверяет тип переменной и генерирует исключение, если она не определена.

### `QueryLanguageType.py`

Файл содержит определения всех типов, поддерживаемых языком запросов.
Типы включают:

- `EDGE`: Рёбра графа.
- `NUM`: Числовые значения.
- `CHAR`: Символы.
- `GRAPH`: Графы.
- `FA`: Конечные автоматы.
- `RSM`: Регулярные грамматики (RSM).
- `SET`: Множества чисел.
- `PAIR_SET`: Множества пар чисел.
- `RANGE`: Диапазоны.
- `UNKNOWN`: Неопределённые или некорректные типы.

### `typechecker.py`

Этот модуль предоставляет функцию `typing_program`, которая выполняет проверку типов для заданной программы на языке
запросов:

- **`program_to_tree`**: Преобразует текст программы в дерево синтаксического разбора и проверяет его корректность.
- **`QueryLanguageTypeInferenceVisitor`**: Используется для проверки типов в дереве синтаксического разбора.
- Возвращает `True`, если проверка прошла успешно, и `False`, если обнаружены ошибки.

#### Пример использования

```python
from project.queryLanguage.typing.typechecker import typing_program

example_program = """example program"""

well_typed = typing_program(example_program)
```

---

## `Query Language Interpreter`

### `QueryLanguageInterpreterVisitor.py`

#### Основные методы:

- `visitDeclare`: Инициализация переменной.
- `visitBind`: Привязка выражения к переменной.
- `visitRegexp`: Обработка регулярных выражений (объединение, пересечение, повторение).
- `visitSelect`: Обработка запросов на выборку путей в графе с использованием контекстно-свободных путей.
- `visitAdd`: Добавление рёбер и вершин в граф.
- `visitRemove`: Удаление рёбер и вершин из графа.
- `visitSet_expr`, `visitEdge_expr`: Обработка выражений для множества и рёбер.
- `visitVar_filter`: Обработка фильтров для переменных.
- `visitVar`, `visitNum`, `visitChar`: Обработка переменных, чисел и символов.

### `utils.py`

Этот файл содержит вспомогательные функции для работы с регулярными выражениями и конечными автоматами:

#### Основные функции:

- `nfa_from_char`: Создаёт ε-NFA из символа.
- `nfa_from_var`: Создаёт ε-NFA из переменной.
- `intersect`: Пересечение двух ε-NFA.
- `concatenate`: Конкатенация двух ε-NFA.
- `union`: Объединение двух ε-NFA.
- `repeat`: Повторение ε-NFA заданное количество раз.
- `kleene`: Применение звезды Клини к ε-NFA.
- `repeat_range`: Повторение ε-NFA с диапазоном повторений.
- `group`: Группировка ε-NFA.
- `build_rsm`: Строит рекурсивный автомат для регулярного выражения

#### Этапы работы интерпретатора

1. Инициализация и переменные `self.__variables` для хранения переменных и `self.__results` для хранения результатов.
2. Запросы к графам обрабатываются в методе `visitSelect`:
    - Извлекаются переменные и фильтры для графа.
    - Строится запрос в виде рекурсивного автомата (`RSM`), который используется для выполнения контекстно-свободного
      поиска путей с помощью метода `tensor_based_cfpq`.
3. На основе полученных результатов определяется, какие вершины должны быть возвращены.

#### Пример использования

```python
from project.queryLanguage.interpreter.interpreter import exec_program

example_program = """example program"""

query_results = exec_program(example_program)
```
