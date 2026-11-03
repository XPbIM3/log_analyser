# log_analyser

Анализ логов TunerStudio/Speeduino (`.msl`) и тюна (`.msq`) для коррекции VE/AFR таблиц.

Проект переписывается с нуля. Старая реализация лежит в `legacy/` как референс.

## Структура
- `log_analyser/axis.py`: оси таблиц `RpmAxis` (кратность 50) и `KpaAxis` (кратность 2), кратность можно переопределить.
- `log_analyser/table.py`: `Table` (оси x/y, значения `len(y) x len(x)`, тип `TableKind`, подпись).
- `log_analyser/tune.py`: `Tune.load("CurrentTune.msq")`; `tune.table("veTable")` собирает таблицу с осями по реестру `SPEEDUINO_TABLES`.
- `log_analyser/datalog.py`: `Log.load("x.msl")` — DataFrame лога; AFR/Lambda по умолчанию сдвигаются на 1 строку вверх (ответ смеси на параметры строки). `Log.load_many("input/*.msl")`, `Log.concat`, `log.where(...)`, `log.with_corr("AFR Target" | afr_table)`.
- `log_analyser/cell.py`: `cell_points(...)` / `plot_cell(log, x_axis, y_axis, rpm, kpa, what="CORR", vs="RPM", path="cell.png")`.
- `log_analyser/afr_target.py`: `AfrFill(lean, rich, kpa_lean, kpa_rich, bias)` + `afr_target_table(rpm, kpa, fill)` — AFR target: плато lean до `kpa_lean`, плато rich от `kpa_rich`, между ними S-переход (smootherstep), вдоль RPM постоянно.
- `log_analyser/ui/`: tkinter-интерфейс (ядро его не импортирует).
- `log_analyser/table_file.py`: `export_table(table, "out.table")` / `read_table(...)` — формат TunerStudio `.table`.

## Разработка
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
```

## UI
```
.venv/bin/python -m log_analyser.ui
```
Окно создания AFR target таблицы: оси RPM и kPa (min / max / points / gamma), заполнение AFR вдоль нагрузки, живой просмотр таблицы и кривой, экспорт в `.table`.
На Linux/WSL нужен tkinter: `sudo apt install python3-tk`.
