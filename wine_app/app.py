"""ДЗ-3: оценка вина и исследование качества логистической регрессии."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Вино в подборку', page_icon='🍷', layout='wide')
st.html('<style>' + (BASE_DIR / 'style.css').read_text(encoding='utf-8') + '</style>')

WINE = '#A64D60'
INK = '#542B38'
ROSE = '#D8A5AD'
PAPER = '#FFFEFC'
GRID = '#EADDDD'

@st.cache_resource
def load_model(path, revision):
    # Один экземпляр модели на сервер; только читаем его, не переобучаем.
    return joblib.load(path)


@st.cache_data
def load_test(path, revision):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X, model_revision):
    # Порог сюда не передаю: при его изменении predict_proba не нужен.
    return _model.predict_proba(X)[:, 1]


def validate_batch(frame, features):
    """Возвращаю признаки в порядке обучения либо понятную ошибку."""
    if not frame.columns.is_unique:
        raise ValueError('Названия столбцов не должны повторяться.')
    missing = sorted(set(features) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(features))
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if extra:
        raise ValueError('Лишние столбцы: ' + ', '.join(extra) + '. Нужен CSV без target.')
    if frame.empty:
        raise ValueError('В файле нет строк с винами.')
    try:
        values = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все признаки должны быть числами. Проверьте разделитель CSV.') from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Заполните пропуски и уберите бесконечные значения.')
    if not values['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен равняться 0 (белое) или 1 (красное).')
    return values


try:
    config = json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))
    model_revision = (BASE_DIR / 'model.joblib').stat().st_mtime_ns
    model = load_model(str(BASE_DIR / 'model.joblib'), model_revision)
    data = load_test(str(BASE_DIR / 'test.csv'), (BASE_DIR / 'test.csv').stat().st_mtime_ns)
except (OSError, ValueError) as exc:
    st.error(f'Не удалось загрузить модель или данные: {exc}')
    st.stop()

features = config['features']
X_test = data.loc[:, features]
y_test = data['target'].to_numpy()
scores = test_proba(model, X_test, model_revision)

LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный SO₂',
    'total_sulfur_dioxide': 'Общий SO₂',
    'density': 'Плотность', 'pH': 'pH', 'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь', 'is_red': 'Красное вино',
}
FORMATS = {'density': '%.5f', 'chlorides': '%.3f',
           'volatile_acidity': '%.3f', 'citric_acid': '%.3f',
           'free_sulfur_dioxide': '%.1f', 'total_sulfur_dioxide': '%.1f'}
STEPS = {'density': 0.0001, 'chlorides': 0.001, 'volatile_acidity': 0.001,
         'citric_acid': 0.001, 'free_sulfur_dioxide': 1.0, 'total_sulfur_dioxide': 1.0,
         'alcohol': 0.1, 'pH': 0.01}

GLASS = '''<svg class="wine-mark" viewBox="0 0 80 100" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg" fill="none">
  <path d="M19 12h42l3 27c2 18-8 30-24 30S14 57 16 39l3-27Z" fill="#F1DCDF"/>
  <path d="M17 40c12-5 20 6 32 1 6-3 10-3 15-2 2 18-8 30-24 30S15 57 17 40Z" fill="#BD7383"/>
  <path d="M19 12h42l3 27c2 18-8 30-24 30S14 57 16 39l3-27ZM40 69v22M25 92h30"
        stroke="#884254" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="m24 21-1 15" stroke="#FFFEFC" stroke-width="3" stroke-linecap="round"/>
</svg>'''

st.html(f'''<header class="wine-header">{GLASS}<div>
  <div class="wine-kicker">Заметки сомелье</div>
  <h1>Вино в подборку</h1>
  <p>От лабораторных показателей к следующей дегустации.</p>
</div></header>''')


def reset_threshold():
    st.session_state['threshold'] = float(config['threshold'])


with st.sidebar:
    st.html('<div class="wine-kicker">Маленькая винная карта</div>'
            '<div class="sidebar-brand">Правила отбора</div>')
    st.caption('Насколько строгой будет подборка?')
    threshold = st.slider('Порог рекомендации', min_value=0.0, max_value=1.0,
                          value=float(config['threshold']), step=0.001,
                          format='%.3f', key='threshold')
    st.button('Вернуть выбранный порог', on_click=reset_threshold, key='reset_threshold',
              type='tertiary')
    st.caption(f"Порог из валидации: {config['threshold']:.3f}")
    st.divider()
    st.markdown('**Выше порог — строже отбор.**')
    st.write('В подборку попадёт меньше вин. При снижении порога выбор станет шире, '
             'но возрастёт риск включить посредственное вино.')
    st.divider()
    st.html('<div class="wine-kicker">Что считаем хорошим вином</div>'
            '<div class="sidebar-brand">7 баллов и выше</div>'
            '<p class="sidebar-note">Модель оценивает вероятность такой экспертной оценки. '
            'Окончательное решение остаётся за сомелье.</p>')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])


# Одинаковые шрифты и цвета для всех графиков. rc_context не меняет глобальные настройки.
def chart_style():
    return plt.rc_context({
        'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': '#785F65', 'ytick.color': '#785F65',
        'axes.edgecolor': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.alpha': 0.65,
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
        'legend.frameon': False,
    })


with single_tab:
    form_column, result_column = st.columns([2.2, 1], gap='large')
    with form_column:
        with st.container(key='wine_form'):
            st.subheader('Паспорт вина')
            st.caption('Введите результаты лабораторного анализа.')
            values = {'is_red': int(st.toggle('Красное вино', value=False))}
            columns = st.columns(3, gap='medium')
            numeric_features = [feature for feature in features if feature != 'is_red']
            for i, feature in enumerate(numeric_features):
                with columns[i % 3]:
                    series = X_test[feature]
                    low, high, median = map(float, [series.min(), series.max(), series.median()])
                    values[feature] = st.number_input(
                        LABELS[feature], min_value=low, max_value=high,
                        value=median, step=STEPS.get(feature, 0.01),
                        format=FORMATS.get(feature, '%.2f'),
                        help=f'{feature} · диапазон в данных: {low:g}–{high:g}',
                        key=f'feature_{feature}')
            st.caption('Начальные значения — медианы тестовой выборки. Диапазоны взяты из тех же данных.')
    probability = float(model.predict_proba(pd.DataFrame([values], columns=features))[0, 1])
    with result_column:
        with st.container(key='wine_result'):
            st.html('<div class="wine-kicker">Результат оценки</div>')
            st.subheader('Красное вино' if values['is_red'] else 'Белое вино')
            st.metric('Вероятность хорошего вина', f'{probability:.1%}')
            st.progress(probability)
            selected = probability >= threshold
            decision = 'Включаем в подборку' if selected else 'Пока вне подборки'
            css_class = 'result-label selected' if selected else 'result-label'
            st.html(f'<span class="{css_class}">{decision}</span>')
            comparison = 'не ниже' if selected else 'ниже'
            st.html(f'<p class="result-copy">Вероятность {probability:.1%} {comparison} '
                    f'порога {threshold:.1%}.</p>')
            st.divider()
            st.caption('Измените показатели вина или порог слева — оценка обновится автоматически.')

with metrics_tab:
    st.subheader('Как меняется подборка')
    st.caption(f'Отложенная выборка · {len(y_test):,} вин'.replace(',', ' '))
    pred = scores >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    columns = st.columns(3, gap='medium')
    for col, name, value, explanation in zip(
        columns, ['Precision', 'Recall', 'F1'], [precision, recall, f1],
        ['Доля хороших вин среди выбранных', 'Доля найденных хороших вин', 'Баланс точности и полноты']):
        col.metric(name, f'{value:.3f}', help=explanation, border=True)
    st.caption('Повышение порога сужает подборку и снижает охват. '
               'Precision при этом не обязательно растёт на каждом шаге.')
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    matrix_col, summary_col = st.columns([1.15, 1], gap='medium')
    with matrix_col, st.container(key='matrix_panel'):
        st.markdown('**Матрица ошибок**')
        with chart_style():
            fig, ax = plt.subplots(figsize=(4.8, 3.5))
            cmap = LinearSegmentedColormap.from_list('wine', ['#FAEEEE', '#A64D60', '#6E3042'])
            metrics.ConfusionMatrixDisplay.from_predictions(
                y_test, pred, labels=[0, 1], display_labels=['Остальные', 'Хорошие'],
                ax=ax, cmap=cmap, colorbar=False)
            ax.set_xlabel('Прогноз модели'); ax.set_ylabel('Истинный класс')
            ax.grid(False)
            fig.tight_layout(); st.pyplot(fig, alt='Матрица ошибок на тестовой выборке'); plt.close(fig)
    with summary_col, st.container(key='selection_panel'):
        st.markdown('**Что попадёт в подборку**')
        st.metric('Отобрано вин', f'{int(pred.sum())} из {len(y_test)}')
        st.write(f'**{tp}** хороших вин найдено.')
        st.write(f'**{fp}** посредственных вин попало в подборку.')
        st.write(f'**{fn}** хороших вин осталось за её пределами.')
        st.divider()
        st.caption('Метрики на этой вкладке показывают последствия выбранного порога. '
                   'Исходный порог подобран по отдельной валидационной выборке.')
    roc_col, pr_col = st.columns(2, gap='medium')
    with roc_col, st.container(key='roc_panel'):
        st.markdown('**ROC-кривая**')
        st.caption(f'AUC · {metrics.roc_auc_score(y_test, scores):.3f}')
        with chart_style():
            fig, ax = plt.subplots(figsize=(5, 3.7))
            fpr_curve, tpr_curve, _ = metrics.roc_curve(y_test, scores)
            ax.plot(fpr_curve, tpr_curve, color=WINE, label='Модель', lw=2)
            ax.scatter(fp / (fp + tn), recall, color=INK, s=45, zorder=4, label=f'Порог {threshold:.3f}')
            ax.plot([0, 1], [0, 1], '--', color=ROSE, lw=1)
            ax.set_xlabel('Доля ложных срабатываний'); ax.set_ylabel('Полнота (recall)')
            ax.grid(alpha=0.4); ax.legend(loc='lower right')
            fig.tight_layout(); st.pyplot(fig, alt='ROC-кривая с точкой текущего порога'); plt.close(fig)
    with pr_col, st.container(key='pr_panel'):
        st.markdown('**Precision–Recall**')
        st.caption(f'Average precision · {metrics.average_precision_score(y_test, scores):.3f}')
        with chart_style():
            fig, ax = plt.subplots(figsize=(5, 3.7))
            pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, scores)
            ax.plot(pr_recall, pr_precision, color=WINE, label='Модель', lw=2)
            # При пустой подборке использую крайнюю точку PR-кривой (0, 1).
            curve_precision = precision if pred.any() else 1.0
            ax.scatter(recall, curve_precision, color=INK, s=45, zorder=4, label=f'Порог {threshold:.3f}')
            ax.axhline(y_test.mean(), ls='--', color=ROSE, lw=1, label='Случайный отбор')
            ax.set_xlabel('Полнота (recall)'); ax.set_ylabel('Точность (precision)')
            ax.grid(alpha=0.4); ax.legend(loc='upper right')
            fig.tight_layout(); st.pyplot(fig, alt='PR-кривая с точкой текущего порога'); plt.close(fig)
    if not pred.any():
        st.caption('Подборка пуста: precision не определена. В метриках принят 0, '
                   'а на PR-кривой показана её крайняя точка (0, 1).')

with batch_tab:
    st.subheader('Оценить целую коллекцию')
    st.caption('CSV с 12 признаками вина, без столбца target. Порог общий для всех вкладок.')
    with st.container(key='upload_panel'):
        uploaded = st.file_uploader('Загрузите файл с винами', type=['csv'])
        st.download_button('Пример CSV · 5 вин', X_test.head(5).to_csv(index=False).encode('utf-8'),
                           'wine_example.csv', 'text/csv')
    if uploaded is not None:
        try:
            batch = validate_batch(pd.read_csv(uploaded), features)
            result = batch.copy()
            result['probability'] = model.predict_proba(batch)[:, 1]
            result['selected'] = (result['probability'] >= threshold).astype(int)
            st.write(f"В подборку попадает **{int(result['selected'].sum())} из {len(result)}** вин.")
            st.dataframe(result, hide_index=True, column_config={
                'probability': st.column_config.NumberColumn('Вероятность', format='%.3f'),
                'selected': st.column_config.NumberColumn('В подборке (1 — да)', format='%d'),
            })
            st.download_button('Скачать результаты', result.to_csv(index=False).encode('utf-8'),
                               'wine_predictions.csv', 'text/csv', type='primary')
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as exc:
            st.error(f'Не удалось обработать CSV: {exc}')

with weights_tab:
    st.subheader('Что влияет на оценку')
    st.caption('Коэффициенты логистической регрессии после масштабирования признаков.')
    coefficients = pd.Series(model[-1].coef_.ravel(), index=features).sort_values()
    chart_col, note_col = st.columns([2, 1], gap='large')
    with chart_col, st.container(key='coefficients_panel'):
        with chart_style():
            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.barh([LABELS[name] for name in coefficients.index], coefficients.values,
                    color=np.where(coefficients >= 0, WINE, '#D4A5AA'), height=0.65)
            ax.axvline(0, color='#B99CA4', lw=1)
            ax.set_xlabel('Коэффициент при стандартизированном признаке')
            ax.grid(axis='x', alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig, alt='Коэффициенты модели, отсортированные по значению'); plt.close(fig)
    with note_col:
        st.markdown('**Сильнее повышают оценку**')
        for feature in coefficients[coefficients > 0].nlargest(3).index:
            st.write('· ' + LABELS[feature])
        st.divider()
        st.markdown('**Сильнее снижают оценку**')
        for feature in coefficients[coefficients < 0].nsmallest(3).index:
            st.write('· ' + LABELS[feature])
        st.divider()
        st.caption('Знак показывает направление связи при фиксированных остальных признаках. '
                   'Это не доказательство причинного влияния.')
    with st.expander('Как читать коэффициенты'):
        st.write('Коэффициент показывает изменение log-odds хорошего вина при росте признака '
                 'на одно стандартное отклонение. Признаки могут быть связаны между собой: '
                 'например, сахар, плотность и алкоголь. Поэтому коэффициент описывает '
                 'вклад признака при фиксированных остальных показателях.')

st.html('<footer class="wine-footer">Wine Quality · Математические методы анализа данных · '
        '<a href="https://github.com/danilprakh0v/mmda-hw3-dp" target="_blank" rel="noopener noreferrer">'
        'Код и исследование ↗</a></footer>')
