# ДЗ-3. Логистическая регрессия и Streamlit

Математические методы анализа данных. Прахов Данил, БПИ246.

**Приложение опубликовано в Streamlit Community Cloud:**
[Вино в подборку — открыть приложение](https://mmda-hw3-dp-mfvysmmaj7arqxg5dmxvi5.streamlit.app).

[Ноутбук с решением](<HW3_MMDA, Prakhov Danil.ipynb>) · [Код приложения](wine_app/app.py)

## Структура

- Ноутбук — реализация логистической регрессии, метрики классификации и дисбаланс классов.
- `Data/wine_quality.csv` — исходный датасет Wine Quality.
- `wine_app/` — Streamlit-приложение, оформление и сохранённая модель.
- `.streamlit/config.toml` — настройки темы для запуска из корня репозитория.
- `requirements-hw3.txt` — зависимости для выполнения ноутбука.

## Оформление

Светлая и тёмная темы переключаются через меню **⋮** в правом верхнем углу приложения.

## Запуск приложения

Python **3.14.8**. Команды выполняются из корня репозитория:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```

Описание вкладок, результатов и кэширования: [wine_app/README.md](wine_app/README.md).

Для выполнения ноутбука установите `python -m pip install -r requirements-hw3.txt`
и выберите полученное окружение как ядро Jupyter. Данные загружаются локально из `Data/`.

## Развёртывание

Приложение связано с веткой `main` этого репозитория.
Точка входа — `wine_app/app.py`, версия Python в Community Cloud — `3.14`.
Дополнительные секреты не требуются.
