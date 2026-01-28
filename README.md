# Симулятор настольной игры: драфт + поле 5x5 + автопаттерны

Этот репозиторий содержит консольный симулятор партии карточной игры:
- колода 90 уникальных карт (rank 1..6, color 1..5, shape 1..3)
- драфт по 3 карты: 1 себе, 1 следующему игроку (слева), 1 в сброс
- поле игрока 5x5 с защищенной зоной (центр 3x3)
- паттерны A-J с автосрабатыванием, очисткой и опциональными наградами
- цель симулятора: собрать статистику и подобрать баланс VP за паттерны

Правила игры и баланс VP задаются конфигом:
- config/game_rules_config_v0_1.json

## Требования

- Python 3.11 или 3.12
- PyCharm (подойдет Community)
- Windows/macOS/Linux

## Быстрый старт

1. Создай и активируй виртуальное окружение
   - PyCharm: New Project -> Pure Python -> New venv
   - или вручную:
     - python -m venv .venv
     - macOS/Linux: source .venv/bin/activate
     - Windows: .venv\Scripts\activate

2. Установи зависимости (минимум)
   - pip install --upgrade pip
   - pip install tqdm

Опционально для удобной аналитики:
- pip install numpy pandas

3. Запусти пример (когда появится код)
- python -m src.main --config config/game_rules_config_v0_1.json --players 3 --games 10000 --seed 123 --agents random random random

## Структура проекта

project_root/
- config/
  - game_rules_config_v0_1.json
- src/
  - main.py
  - game_engine.py
  - pattern_engine.py
  - reward_engine.py
  - config_loader.py
  - models.py
  - simulator.py
  - reports.py
  - agents/
- tests/
- README.md
- TECHSPEC.md

## Что будет в отчетах

После серии симуляций программа должна уметь выводить:
- средние и медианные VP по агентам
- winrate по агентам
- частоты паттернов A-J
- частоты применений наград RWD1..RWD10 и частоты отказов от награды
- среднюю длину партии в ходах
- причины окончания партии (END1, END2)

## Полезные заметки

- Конфиг является единственным источником правды для правил и VP.
- При изменении VP создавай новую версию конфига (например game_rules_config_v0_2.json), чтобы результаты были сравнимыми.
