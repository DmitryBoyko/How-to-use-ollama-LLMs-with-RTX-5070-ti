# Docker Ollama (только GPU) + CLI (REPL + smoke)

Этот проект запускает **Ollama в Docker со строго обязательным NVIDIA GPU** и даёт два CLI:

- `scripts/ollama-repl.ps1` / `scripts/ollama-smoke.ps1`: PowerShell CLI (Windows)
- `scripts/ollama_cli.py`: Python CLI (Windows/Linux) — доведён до того же поведения

## Требования (Windows)

- Установлены **NVIDIA GPU + драйвер**.
- **Docker Desktop** с backend на WSL2.
- Включена поддержка GPU в Docker (Docker Desktop + WSL2 + NVIDIA integration).

Если GPU недоступен, контейнер `ollama` должен **не стартовать** (CPU fallback не допускается).

## Запуск стека

Из `c:\docker-ollama`:

```powershell
docker compose up -d
docker compose logs -f ollama-init
```

Сервис `ollama-init` на первом запуске скачивает `qwen2.5:14b-instruct-q4_K_M` в постоянный volume.

Также в стеке есть `gpu-metrics` (CUDA base image): он нужен, чтобы получать загрузку GPU через `nvidia-smi`, даже если на хосте `nvidia-smi` не доступен в PATH.

## Быстрые проверки

```powershell
docker compose ps
docker compose exec -T ollama ollama list
```

## Рекомендуемые модели (универсальные 7–14B, «комфортный класс» для RTX 5070 Ti 16GB)

Дальше ориентируемся **только** на универсальные модели 7–14B в квантизации **`q4_K_M`** (или эквивалентных тегах), которые разумно и стабильно работают в 16GB VRAM.

- `qwen2.5:7b-instruct-q4_K_M`
- `qwen2.5:14b-instruct-q4_K_M`
- `mistral:7b-instruct-q4_K_M` (альтернатива: `mistral:7b-instruct-v0.3-q4_K_M`)
- `llama3.1:8b-instruct-q4_K_M`
- `gemma2:9b-instruct-q4_K_M`
- `gemma3:12b-it-q4_K_M`
- `phi4:14b-q4_K_M`
- `qwen2.5-coder:7b-instruct-q4_K_M` (код)
- `qwen2.5-coder:14b-instruct-q4_K_M` (код)

## Запуск интерактивного CLI (REPL)

### PowerShell (Windows)

```powershell
.\scripts\ollama-repl.ps1
```

### Python (Windows/Linux)

Это кроссплатформенный CLI, который работает и на Windows, и на Linux.

### Запуск REPL

```powershell
python .\scripts\ollama_cli.py repl --host http://localhost:11435 --gpu-index 0
```

Команды:

- `/exit`
- `/delete` (удаление установленной модели по номеру)
- `/model` (сменить модель: снова показать меню)

### Smoke (один вопрос → ответ → выход)

```powershell
python .\scripts\ollama_cli.py smoke --host http://localhost:11435 --gpu-index 0 --question "Дай краткое описание квантовой механики."
```

### Smoke (PowerShell, Windows)

```powershell
.\scripts\ollama-smoke.ps1 -ChooseModel -Question "Ответь строго ASCII: OK"
```

### Удалить модель (через API)

```powershell
python .\scripts\ollama_cli.py smoke --host http://localhost:11435 --delete-model --model "qwen2.5:14b-instruct-q4_K_M"
```

Как это работает:

- При запуске REPL один раз выводится меню рекомендованных моделей (7–14B) и выбор по номеру.
- Для смены модели используй команду `/model`.
- Если модели нет локально — скачивается через `POST /api/pull` (прогресс показывается в stderr).
- Во время генерации % GPU показывается в stderr (через `nvidia-smi` или `docker exec gpu-metrics ...`).

### Показ процента GPU (опционально)

По умолчанию показ % GPU **выключен**.

- PowerShell: добавь `-ShowGpu`
- Python: добавь `--show-gpu`

## Примечания

- API Ollama доступно по `http://localhost:11435` по умолчанию (можно переопределить через `OLLAMA_PORT`).
- Загрузка GPU читается через `nvidia-smi` внутри контейнера `gpu-metrics` и показывается в процентах.
- В CLI принудительно включён **system‑prompt “строго русский”**, чтобы модель не отвечала смешанными языками (это влияет и на REPL, и на smoke).

## Как задавать вопросы из другого приложения (Ollama API)

Скрипты в `scripts/` — это только CLI‑обёртки. Любое внешнее приложение должно обращаться к HTTP API Ollama.

### Полная интеграция “извне” (RAG / AI‑ассистент в другом проекте)

Ниже — практическая схема интеграции Ollama в **любой внешний проект** (backend/desktop/бот), включая **RAG** (Retrieval‑Augmented Generation).

### 0) Что у тебя уже есть в этом репозитории

- **LLM**: доступен по HTTP (см. базовый URL ниже), используешь `POST /api/chat` или `POST /api/generate`.
- **Обязательный GPU**: контейнер `ollama` стартует только с NVIDIA GPU.

### 1) Базовая архитектура RAG (коротко)

- **Индексация (offline/периодически)**:
  - Берёшь документы (PDF/HTML/MD/код).
  - Делаешь чанки (обычно 300–800 токенов, overlap 50–150).
  - Для каждого чанка считаешь embedding (вектор) и сохраняешь в векторное хранилище (FAISS/Qdrant/pgvector/Weaviate и т.п.).
- **Ответ на вопрос (online)**:
  - Получаешь вопрос пользователя.
  - Считаешь embedding вопроса.
  - Делаешь similarity search по векторному хранилищу → топ‑K чанков.
  - Собираешь prompt/messages: *инструкция* + *контекст* (чанки) + *вопрос*.
  - Вызываешь LLM через Ollama API и получаешь ответ.

### 2) Важные практические правила (чтобы работало стабильно)

- **Не пихай “всё подряд”**: контекст ограничен `num_ctx`. Обычно лучше топ‑K=3..8 чанков.
- **Отдавай источники**: в контексте помечай чанки `source`/`title`/`url`/`path` и проси модель ссылаться на них.
- **Храни историю диалога**: RAG‑ассистенту нужны `messages` (роль+контент), а не только последний вопрос.
- **Санкционируй ответы**: если в контексте нет ответа — пусть модель честно скажет “не найдено в источниках”.

### 3) Embeddings (нужно для RAG)

Для RAG тебе нужен второй тип модели: **embedding‑модель**. В Ollama это отдельный эндпоинт:

- **POST** `/api/embeddings`

Типовой запрос:

```json
{
  "model": "nomic-embed-text",
  "prompt": "текст или вопрос"
}
```

Ответ содержит вектор (embedding). Его сохраняешь/сравниваешь в своём векторном хранилище.

Как скачать embedding‑модель (один раз):

- **POST** `/api/pull`

PowerShell:

```powershell
$body = @{ model="nomic-embed-text" } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/pull" -ContentType "application/json" -Body $body
```

Если хочешь видеть прогресс скачивания, используй стрим `POST /api/pull` так же, как в CLI (`scripts/ollama_cli.py` / `scripts/ollama-smoke.ps1`) — там это уже реализовано.

Пример (PowerShell):

```powershell
$body = @{ model="nomic-embed-text"; prompt="пример текста" } | ConvertTo-Json -Depth 5
$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/embeddings" -ContentType "application/json" -Body $body
$resp.embedding
```

### 4) LLM (чат‑режим для ассистента)

Для ассистента почти всегда удобнее **чат**:

- **POST** `/api/chat`
- Текст ответа в `message.content`

Ключевая идея RAG: ты сам добавляешь найденный контекст в `messages` перед вопросом.

Готовый system‑prompt (шаблон) для RAG‑ассистента:

```text
Ты — RAG‑ассистент.
Используй ТОЛЬКО информацию из блока "КОНТЕКСТ (RAG)".
Если ответа нет в контексте — напиши: "Не найдено в источниках".
Для каждого факта указывай ссылку на источник в формате [N] (номер чанка).
Не выдумывай, не додумывай.
Отвечай на русском языке.
```

Пример структуры сообщений:

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "stream": false,
  "options": { "num_ctx": 8192, "num_predict": 2048 },
  "messages": [
    {
      "role": "system",
      "content": "Ты ассистент по документации. Отвечай строго по предоставленным источникам. Если ответа нет в источниках — скажи об этом."
    },
    {
      "role": "system",
      "content": "КОНТЕКСТ (RAG):\n[1] source=docs/a.md\n<кусок текста>\n\n[2] source=docs/b.md\n<кусок текста>"
    },
    { "role": "user", "content": "Вопрос пользователя..." }
  ]
}
```

### 5) Минимальный RAG‑цикл (псевдокод)

1) `q_vec = embeddings(question)`
2) `chunks = vector_db.search(q_vec, top_k=5)`
3) `context = format(chunks)`
4) `answer = chat(messages=[system_rules, context, history..., user_question])`

### 6) Пример “RAG‑вызов” на Python (без внешних библиотек)

Ниже пример только “онлайн‑части” (поиск в твоей БД — это `retrieve_top_k(...)`):

```python
import json
import urllib.request

BASE = "http://localhost:11435"

def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))

def embeddings(model, text):
    r = post_json("/api/embeddings", {"model": model, "prompt": text})
    return r["embedding"]

def chat(model, messages, num_ctx=8192, num_predict=2048):
    r = post_json("/api/chat", {
        "model": model,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        "messages": messages,
    })
    return r["message"]["content"]

def retrieve_top_k(query_vec, top_k=5):
    # TODO: replace with your vector DB search
    # Return list of dicts: { "source": "...", "text": "..."}
    return []

question = "..."
qv = embeddings("nomic-embed-text", question)
chunks = retrieve_top_k(qv, top_k=5)
context = "\n\n".join([f"[{i+1}] source={c['source']}\n{c['text']}" for i, c in enumerate(chunks)])

messages = [
    {"role": "system", "content": "Ты RAG-ассистент. Отвечай только по контексту. Если ответа нет — скажи 'не найдено в источниках'."},
    {"role": "system", "content": "КОНТЕКСТ (RAG):\n" + context},
    {"role": "user", "content": question},
]

print(chat("qwen2.5:14b-instruct-q4_K_M", messages))
```

### 7) Про “строго русский” и system‑prompt

Если хочешь жёстко зафиксировать язык ответа (например, **только русский**) — добавляй это в `system`/`messages[0]` в своём внешнем приложении.

### Быстрый тест, что API живо

**GET** `/api/tags` — вернёт список локально установленных моделей.

PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:11435/api/tags"
```

Linux/macOS (`curl`):

```bash
curl -s "http://localhost:11435/api/tags"
```

### Сгенерировать ответ (без стрима)

**POST** `/api/generate`

PowerShell:

```powershell
$body = @{
  model  = "qwen2.5:14b-instruct-q4_K_M"
  prompt = "Привет! Кратко объясни квантовую механику."
  stream = $false
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/generate" -ContentType "application/json" -Body $body
$resp.response
```

Linux/macOS (`curl`, без дополнительных утилит):

```bash
curl -s "http://localhost:11435/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:14b-instruct-q4_K_M","prompt":"Привет! Кратко объясни квантовую механику.","stream":false}'
```

Ответ придёт одним JSON. Текст ответа лежит в поле `response`.

Пример тела запроса (JSON):

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "prompt": "Привет! Кратко объясни квантовую механику.",
  "stream": false
}
```

Что вернётся в ответ (важное поле):

- `response`: строка с текстом ответа

### Сгенерировать ответ (со стримом)

То же самое, но `"stream": true`.

**Важно**: ответ будет идти как **NDJSON** — много строк, где **каждая строка** это отдельный JSON‑объект. Чтобы получить полный текст — нужно **склеивать** `response` из каждой строки по порядку.

PowerShell (печатает поток сразу в консоль):

```powershell
$body = @{
  model  = "qwen2.5:14b-instruct-q4_K_M"
  prompt = "Напиши 5 пунктов: что такое квантовая механика?"
  stream = $true
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/generate" -ContentType "application/json" -Body $body |
  ForEach-Object { if ($_.response) { Write-Host -NoNewline $_.response } }
Write-Host ""
```

Linux/macOS (`curl`, стрим NDJSON без дополнительных утилит):

```bash
curl -sN "http://localhost:11435/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:14b-instruct-q4_K_M","prompt":"Напиши 5 пунктов: что такое квантовая механика?","stream":true}'
```

Ответ будет **NDJSON** (много строк). В каждой строке есть поле `response` — это кусок текста. Если просто выполнить команду выше, ты увидишь эти JSON‑строки прямо в терминале.

Windows PowerShell (стрим, надёжный вариант через .NET `HttpClient`):

```powershell
$uri = "http://localhost:11435/api/generate"
$payload = @{
  model  = "qwen2.5:14b-instruct-q4_K_M"
  prompt = "Напиши 5 пунктов: что такое квантовая механика?"
  stream = $true
} | ConvertTo-Json -Depth 10

Add-Type -AssemblyName System.Net.Http | Out-Null
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan

$req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, $uri)
$req.Content = [System.Net.Http.StringContent]::new($payload, [System.Text.Encoding]::UTF8, "application/json")

$res = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
if (-not $res.IsSuccessStatusCode) { throw "HTTP $([int]$res.StatusCode)" }

$stream = $res.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
$reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
try {
  while (-not $reader.EndOfStream) {
    $line = $reader.ReadLine()
    if (-not $line) { continue }
    $o = $line | ConvertFrom-Json
    if ($o.response) { Write-Host -NoNewline $o.response }
    if ($o.done -eq $true) { break }
  }
  Write-Host ""
} finally {
  $reader.Dispose()
  $stream.Dispose()
  $res.Dispose()
  $client.Dispose()
}
```

### Чат‑режим

**POST** `/api/chat`

PowerShell (без стрима):

```powershell
$body = @{
  model    = "qwen2.5:14b-instruct-q4_K_M"
  messages = @(
    @{ role = "system"; content = "Отвечай на русском. Кратко." }
    @{ role = "user";   content = "Составь план на день." }
  )
  stream   = $false
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/chat" -ContentType "application/json" -Body $body
$resp.message.content
```

Linux/macOS (`curl`, без дополнительных утилит):

```bash
curl -s "http://localhost:11435/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5:14b-instruct-q4_K_M",
    "messages":[
      {"role":"system","content":"Отвечай на русском. Кратко."},
      {"role":"user","content":"Составь план на день."}
    ],
    "stream":false
  }'
```
Ответ придёт одним JSON. Текст ответа лежит в `.message.content`.

Чат со стримом: `"stream": true`

**Важно**: ответ будет идти как **NDJSON**. Для вывода в консоль нужно печатать `message.content` из каждой строки.

PowerShell (стрим):

```powershell
$body = @{
  model    = "qwen2.5:14b-instruct-q4_K_M"
  messages = @(
    @{ role = "system"; content = "Отвечай на русском. Кратко." }
    @{ role = "user";   content = "Составь план на день." }
  )
  stream   = $true
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Method Post -Uri "http://localhost:11435/api/chat" -ContentType "application/json" -Body $body |
  ForEach-Object { if ($_.message -and $_.message.content) { Write-Host -NoNewline $_.message.content } }
Write-Host ""
```

Linux/macOS (`curl`, стрим NDJSON без дополнительных утилит):

```bash
curl -sN "http://localhost:11435/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5:14b-instruct-q4_K_M",
    "messages":[
      {"role":"system","content":"Отвечай на русском. Кратко."},
      {"role":"user","content":"Составь план на день."}
    ],
    "stream":true
  }'
```

Ответ будет **NDJSON** (много строк). В каждой строке кусок текста лежит в `.message.content`. Если просто выполнить команду выше, ты увидишь эти JSON‑строки прямо в терминале.

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "messages": [
    { "role": "system", "content": "Ты полезный ассистент." },
    { "role": "user", "content": "Составь план на день." }
  ],
  "stream": false
}
```

### Скачать модель (если её нет локально)

**POST** `/api/pull`

```json
{ "model": "qwen2.5:14b-instruct-q4_K_M" }
```

### Список локально установленных моделей

**GET** `/api/tags`

### Удалить модель

**POST** `/api/delete`

```json
{ "model": "qwen2.5:14b-instruct-q4_K_M" }
```

### Полезно: посмотреть параметры модели (включая контекст)

**POST** `/api/show`

```bash
curl -s "http://localhost:11435/api/show" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:14b-instruct-q4_K_M"}'
```

## Ограничения на длину ответа по API (токены)

В Ollama ограничения задаются **токенами**, а не “словами”.

### 1) Лимит генерации: `options.num_predict`

В `POST /api/generate` и `POST /api/chat` можно указать:

- `options.num_predict`: **максимум токенов**, которые модель сгенерирует в ответ.

Максимум:

- **`num_predict <= num_ctx`** (Ollama ограничивает генерацию размером контекста).
- Спец‑значения (в Ollama): `-1` = без лимита (пока не упрёшься в контекст), `-2` = “заполнить контекст”.

Пример:

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "prompt": "Напиши подробный ответ.",
  "stream": false,
  "options": { "num_predict": 2048 }
}
```

Если ответ упирается в лимит, в streaming‑режиме обычно видно `done_reason: \"length\"`.

### 2) Контекст: `options.num_ctx`

Сумма **вход (prompt/messages) + выход (ответ)** должна помещаться в контекстное окно:

- `options.num_ctx`: размер контекста в токенах.

По умолчанию Ollama может выставлять контекст, например \(n_ctx = 4096\) (это общий бюджет на вход+выход, пока не увеличишь).

Максимум:

- Практический максимум `num_ctx` ограничен **контекстным окном конкретной модели** (и доступной VRAM/памятью).
- Узнать максимум для конкретной установленной модели можно через `POST /api/show` (поле в `model_info`, например `llama.context_length`, `qwen2.context_length`, `gemma2.context_length`, `gemma3.context_length` и т.п.).

Пример запроса:

```json
{ "model": "qwen2.5:14b-instruct-q4_K_M" }
```

### Максимальный контекст у рекомендованных моделей (ориентир)

- `qwen2.5:7b-instruct-q4_K_M`: до **128K** токенов
- `qwen2.5:14b-instruct-q4_K_M`: до **128K** токенов
- `llama3.1:8b-instruct-q4_K_M`: до **128K** токенов
- `mistral:7b-instruct-v0.3-q4_K_M`: до **32K** токенов
- `qwen2.5-coder:14b-instruct-q4_K_M`: до **32K** токенов
- `gemma2:9b-instruct-q4_K_M`: до **8K** токенов
- `phi4:14b-q4_K_M`: до **16K** токенов
- `gemma3:12b-it-q4_K_M`: в описаниях часто заявляют **128K**, но в некоторых сборках/версиях Ollama может фактически быть **8K** — проверяй через `/api/show`/`ollama show` на твоей установке.

Пример:

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "prompt": "Очень длинный текст...",
  "stream": false,
  "options": { "num_ctx": 8192, "num_predict": 4096 }
}
```

### Важно

- `stream: true/false` **не меняет** максимальную длину — только формат выдачи (NDJSON vs один JSON).
- Увеличение `num_ctx` может требовать больше VRAM/памяти и снижать скорость; при слишком большом `num_ctx` модель может начать частично уходить в RAM (и становиться сильно медленнее).
