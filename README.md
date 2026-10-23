# Issue Code Locator

Учебный проект: по ссылке на **GitHub issue** находит в репозитории **код, который, скорее всего, нужно читать или править**, и выдаёт разработчику отчёт в JSON.

Это результат первых двух этапов дорожной карты (01.10–28.10): пайплайн от ссылки на issue до JSON-отчёта с VectorDB и двумя агентами.

## Как это работает

```
Issue URL
   │
   ▼
GitHub API ──► Issue (заголовок, текст, метки) + Repository (owner/name/commit)
   │
   ▼
Ticket Classification Agent ──► тип issue, краткое описание, ключевые слова,
   │                            поисковые запросы, упомянутые файлы   (= метаданные)
   │
   │            Repository Ingestion ──► Chunking ──► Embeddings ──► VectorDB (ChromaDB)
   │            (скачать .py файлы)      (ast: функции,                       ▲
   │                                      классы, методы)                     │
   ▼                                                                          │
Code Retriever Agent ◄────────────────────────── поиск по запросам ───────────┘
   │
   ▼
JSON Report (затронутые файлы, фрагменты с номерами строк и ссылками на GitHub)
```

| Этап | Что делает | Модуль |
|---|---|---|
| Разбор ссылки | `https://github.com/owner/repo/issues/N` → owner, repo, номер | `github_client.py` |
| GitHub API | issue и SHA последнего коммита; репозиторий качается **одним zip-архивом**, из него берутся только `.py` | `github_client.py` |
| Chunking | через `ast` код режется на функции, методы и классы; в метаданных — путь, имя символа, строки. Большие классы делятся на методы, длинные функции — на окна с перекрытием; файлы, которые не парсятся, режутся по строкам | `chunker.py` |
| Embeddings + VectorDB | фрагменты → векторы → ChromaDB (локальная папка). Индекс привязан к SHA коммита: повторный запуск по тому же репозиторию не качает и не индексирует его заново | `embeddings.py`, `vector_store.py` |
| Ticket Classification Agent | LLM определяет тип issue и формулирует запросы к коду. Если LLM недоступна, работают эвристики (правила + извлечение идентификаторов и путей из текста и traceback) | `agents/classifier.py` |
| Code Retriever Agent | ищет по нескольким запросам, объединяет результаты (Reciprocal Rank Fusion, взвешенный близостью), группирует по файлам, поднимает файлы, упомянутые в issue. Если LLM доступна, она дополнительно проверяет кандидатов и пишет причину по каждому | `agents/retriever.py` |
| JSON Report | метаданные + список файлов с фрагментами и ссылками на нужные строки | `pipeline.py` |

## Установка

Нужен Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

## Настройка (`.env`)

Все переменные необязательны.

| Переменная | Назначение |
|---|---|
| `GITHUB_TOKEN` | Токен GitHub. Без него лимит 60 запросов в час (для запуска хватает, но быстро кончается при отладке). Для приватных репозиториев обязателен |
| `LLM_API_KEY` | Ключ LLM. **Без него проект работает без LLM** (эвристики + векторный поиск) |
| `LLM_BASE_URL`, `LLM_MODEL` | Любой OpenAI-совместимый API. По умолчанию DeepSeek: `https://api.deepseek.com`, `deepseek-chat` |
| `CHROMA_DIR` | Папка с векторной БД, по умолчанию `.chroma` |
| `EMBEDDER` | `default` — модель all-MiniLM-L6-v2 (при первом запуске скачивается ~80 МБ), `hash` — лёгкий вариант без нейросети для отладки |

**Про LLM.** Используется один универсальный клиент, поэтому провайдера можно менять только настройками:

- **DeepSeek** — значения по умолчанию, нужен только `LLM_API_KEY`.
- **Ollama** (локально и бесплатно): `LLM_API_KEY=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:7b`.
- **GigaChat** использует собственную OAuth-авторизацию и сертификаты Минцифры, поэтому напрямую в этот клиент не подключается. Нужен прокси с OpenAI-совместимым интерфейсом либо отдельный клиент в `llm.py` (достаточно реализовать метод `chat_json`).

## Запуск

```bash
python -m issue_locator https://github.com/psf/requests/issues/6000
```

Параметры:

```
--top-k N        сколько файлов вернуть (по умолчанию 5)
--out FILE       сохранить отчёт в файл вместо вывода в консоль
--reindex        игнорировать кэш и переиндексировать репозиторий
--no-llm         не обращаться к LLM
--embedder X     default | hash (переопределяет .env)
-v               подробные логи
```

Из кода:

```python
from issue_locator.pipeline import Pipeline

report = Pipeline().run("https://github.com/owner/repo/issues/123", top_k=5)
print(report["affected_files"][0]["path"])
```

## Формат отчёта

Ниже сокращённый пример. Он получен на исходниках библиотеки `requests` с оффлайн-эмбеддером и без LLM. Выбран `rebuild_auth`, потому что issue описывает именно эту ситуацию.

```json
{
  "metadata": {
    "issue": {
      "url": "https://github.com/psf/requests/issues/1",
      "number": 1,
      "title": "Redirect to a different host keeps the Authorization header",
      "body": "When a request is redirected to another host, ...",
      "labels": ["bug"],
      "state": "open"
    },
    "repository": {
      "owner": "psf", "name": "requests",
      "url": "https://github.com/psf/requests",
      "commit": "<sha последнего коммита>"
    },
    "classification": {
      "issue_type": "bug",
      "summary": "Redirect to a different host keeps the Authorization header",
      "keywords": ["rebuild_auth", "Authorization", "redirect", "host"],
      "search_queries": ["Redirect to a different host keeps the Authorization header",
                         "rebuild_auth Authorization redirect different host keeps"],
      "mentioned_files": [],
      "source": "heuristic"
    }
  },
  "affected_files": [
    {
      "path": "requests/sessions.py",
      "score": 1.0,
      "url": "https://github.com/psf/requests/blob/<sha>/requests/sessions.py",
      "reason": "Semantic match: SessionRedirectMixin.rebuild_auth, ... (best similarity 0.25)",
      "llm_relevant": null,
      "chunks": [
        {
          "symbol": "SessionRedirectMixin.rebuild_auth",
          "kind": "method",
          "start_line": 309,
          "end_line": 332,
          "similarity": 0.246,
          "url": "https://github.com/psf/requests/blob/<sha>/requests/sessions.py#L309-L332",
          "snippet": "    def rebuild_auth(\n        ..."
        }
      ]
    }
  ],
  "run": {
    "generated_at": "2026-10-21T00:00:00+00:00",
    "queries": ["..."],
    "index": "fresh",
    "chunks_in_index": 303,
    "embedder": "hash",
    "llm_classification": false,
    "llm_rerank": false
  }
}
```

Пояснения к полям:

- `metadata` — то, что Classification Agent передаёт Code Retriever Agent (формат `IssueMetadata` в `models.py`).
- `score` — относительная оценка файла: `1.0` у лучшего, у остальных доля от него. Это ранжирование, а не вероятность.
- `similarity` — косинусная близость лучшего запроса к фрагменту.
- `llm_relevant` — вердикт LLM по файлу (`true` / `false`) или `null`, если LLM не использовалась. Файлы с `false` идут в конец списка, но не удаляются.
- `index` — `fresh` (репозиторий проиндексирован сейчас) или `cached` (взят готовый индекс этого коммита).

## Структура проекта

```
issue_locator/
  cli.py            командная строка
  pipeline.py       сборка всех этапов
  github_client.py  GitHub API, парсинг ссылки, извлечение .py из архива
  chunker.py        разбиение кода на фрагменты (ast)
  embeddings.py     эмбеддеры (default / hash)
  vector_store.py   обёртка над ChromaDB
  llm.py            клиент OpenAI-совместимого API
  models.py         модели данных и формат метаданных
  config.py         настройки из .env
  agents/
    classifier.py   Ticket Classification Agent
    retriever.py    Code Retriever Agent
tests/              pytest, без сети и ключей
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

Тесты не ходят в сеть: GitHub и LLM подменены заглушками (`tests/fakes.py`), эмбеддер — `hash`. Покрыты разбор ссылок, фильтрация файлов архива, чанкинг, индексация и кэш, оба агента (с LLM и без неё, включая сбой LLM), весь пайплайн и CLI.

## Ограничения (текущая стадия)

- Анализируются **только Python-файлы** (до 200 КБ каждый, до 3000 файлов, архив до 100 МБ — настраивается).
- Код берётся с **последнего коммита** default-ветки, а не с момента создания issue. Если код уже менялся, результат может отличаться.
- Нейросетевой эмбеддер `default` в тестах не запускается (модель скачивается из интернета). При первом реальном запуске нужен доступ в сеть, на CPU индексация большого репозитория занимает минуты. Эмбеддер обрезает длинные фрагменты (около 256 токенов), поэтому смысл очень больших функций учитывается по началу.
- Эмбеддер `hash` ищет по совпадению слов, а не по смыслу, и нужен для отладки.
- Работа с реальным GitHub API и LLM-провайдером в автоматических тестах не проверяется: они подменены заглушками.
- Метрики качества поиска пока не измерялись (это задача следующих этапов).

## Что дальше (по плану)

- Search Agent (поиск по GitHub Issues и Stack Overflow) и Critic Agent (проверка выводов Code Retriever Agent).
- Summary Agent и итоговый отчёт, метрики и демонстрационные примеры.
- UI на Streamlit и оформление сервиса в виде API.
