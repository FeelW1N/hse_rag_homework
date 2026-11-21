# ДЗ 3. RAG по финансовой отчётности

Валерий Убушаев. Ветка для проверки: `valeriy_ubushaev_main`.

## Как работает решение

PyMuPDF извлекает текст с сохранением документа и номера страницы PDF. Страницы без текстового слоя распознаются Tesseract (`rus+eng`, 300 dpi); OpenCV выравнивает сканы и удаляет линии таблиц. В индекс попадают только 55 отчётов; вопросы и эталонные ответы в нём не используются.

Поиск сочетает эмбеддинги сервиса курса и BM25 с русским стеммингом. Фильтры учитывают организацию, год и вид отчётности. Модель `deepseek-flash` получает найденные страницы, формирует ответ и выбирает страницы, подтверждающие его. Если информации недостаточно, она сама запрашивает дополнительный поиск; разрешён один такой шаг.

Библиотеки: PyMuPDF, NumPy, rank-bm25, snowballstemmer, requests, python-dotenv, openpyxl. LangSmith не подключён.

## Установка и подготовка

Команды выполняются из корня репозитория. Нужен Python 3.9–3.13. Для повторной оценки приложен готовый индекс; Tesseract нужен только для его пересборки.

```bash
git clone -b valeriy_ubushaev_main https://github.com/FeelW1N/hse_rag_homework.git
cd hse_rag_homework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполните в `.env` свои `EMB_KEY` и `LLM_API_KEY`. Ключи в Git не добавляются.

Для повторения оценки используйте тот же индекс:

```bash
python -m zipfile -e prepared_index.zip artifacts
python run_eval.py
```

В архиве только текст отчётов, метаданные и векторы. Ответов на вопросы и ключей в нём нет.

## Пересборка индекса из PDF

Этот шаг необязателен при использовании готового индекса. На macOS установите OCR:

```bash
brew install tesseract tesseract-lang
export TESSDATA_PREFIX="$(brew --prefix)/share/tessdata"
```

На Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng`. При необходимости задайте `TESSDATA_PREFIX` — путь к каталогу с файлами `rus.traineddata` и `eng.traineddata`.

```bash
curl -L https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip -o data/reports.zip
python -m zipfile -e data/reports.zip data/reports
python -m rag.build --extract-only
python -m rag.ocr
python -m rag.build
```

Извлечение, OCR и эмбеддинги сохраняются в `artifacts/`. Построение индекса соблюдает ограничения сервиса; повторный запуск использует готовые эмбеддинги. После изменения PDF удалите `artifacts/` и подготовьте индекс заново.

## Проверка и оценка

```bash
python -m pytest -q
python run_eval.py
```

Оценка проходит все 141 вопрос из `data/test.jsonl`. Функции, модель и промпт судьи из `eval/eval_functions.py` не изменены. Результат — `test_results.xlsx`: лист «Итоги» с четырьмя средними и лист «По вопросам» с десятью колонками по шаблону.

Время включает поиск и генерацию, но не вызов судьи и не начальную загрузку индекса. Ответы и эмбеддинги вопросов не кешируются. Задержки внешнего API могут влиять на повторный замер.

Прогресс сохраняется после каждого вопроса в `runs/test.jsonl`. Повтор той же команды продолжает незаконченный прогон. Для независимой повторной проверки:

```bash
python run_eval.py --checkpoint runs/recheck.jsonl --output recheck.xlsx
```

Для проверки на train используйте `--dataset data/train.jsonl --checkpoint runs/train.jsonl --output train_results.xlsx`. Параметр `--limit` позволяет сделать короткий пробный запуск.

## Результаты

Оценены все **141 вопрос** из test. Итоговый файл — [test_results.xlsx](test_results.xlsx).

- `context_recall`: **0,826714**.
- `context_precision`: **0,748818**.
- `answer_correctness`: **0,872340** — 123 ответа из 141 засчитаны судьёй.
- `time`: **7,971835 с** в среднем на вопрос.

Прогон завершён 22 ноября 2026 года. После сбоя доступа к API он был продолжен с сохранённого места; код RAG, индекс и джадж не менялись. Каталог `runs/` не публикуется: на новой машине оценка начнётся с первого вопроса.

[Исходное условие задания](ASSIGNMENT.md).
