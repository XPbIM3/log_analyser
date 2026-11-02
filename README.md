# log_analyser

Анализ логов TunerStudio/Speeduino (`.msl`) и тюна (`.msq`) для коррекции VE/AFR таблиц.

Проект переписывается с нуля. Старая реализация лежит в `legacy/` как референс.

## Структура
- `log_analyser/axis.py`: оси таблиц `RpmAxis` (кратность 50) и `KpaAxis` (кратность 2), кратность можно переопределить.
- `log_analyser/table.py`: `Table` (оси x/y, значения `len(y) x len(x)`, тип `TableKind`, подпись).
- `log_analyser/tune.py`: `Tune.load("CurrentTune.msq")`; `tune.table("veTable")` собирает таблицу с осями по реестру `SPEEDUINO_TABLES`.
- `log_analyser/datalog.py`: `Log.load("x.msl")` — DataFrame лога; AFR/Lambda по умолчанию сдвигаются на 1 строку вверх (ответ смеси на параметры строки). `Log.load_many("input/*.msl")`, `Log.concat`, `log.where(...)`, `log.with_corr("AFR Target" | afr_table)`.
- `log_analyser/cell.py`: `cell_points(...)` / `plot_cell(log, x_axis, y_axis, rpm, kpa, what="CORR", vs="RPM", path="cell.png")`.
- `log_analyser/table_file.py`: `export_table(table, "out.table")` / `read_table(...)` — формат TunerStudio `.table`.

## Разработка
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
```
