# Slide text ← source sentences (conference deck, 2026-10-17)

Each slide sentence is a translation/adaptation of a sentence from a paper in `related/`. Cite when presenting.

| Slide | Slide sentence (RU) | Source sentence | Paper |
| --- | --- | --- | --- |
| 1 | Латентная коммуникация позволяет агентам обмениваться внутренними представлениями напрямую | "a new line of work—collectively called latent communication—lets agents exchange continuous internal representations directly: embeddings, hidden states, or key–value (KV) caches" | Liu 2026, 2606.05711 |
| 2 | Каждое сообщение заставляет отправителя сгенерировать текст, а получателя — заново его прочитать | "Every message forces the sender to autoregressively decode a token sequence and the receiver to prefill that sequence." | Liu 2026 |
| 2 | Кеширование снижает цену только внутри одной модели | "Prefix caching mitigates this within a single model only." | Heo et al. 2026, 2608.03893 |
| 2 | Текст — канал с низкой пропускной способностью… сжимают в строку токенов, получатель распаковывает | "as a low-bandwidth medium, text introduces an information bottleneck. The high-dimensional internal representations must be repeatedly compressed into linear strings and then decompressed by the receiver" | Fu et al. 2025, 2510.03215 |
| 2 | Медленно — получатель ждёт; с потерями — состояние сжато в слова | "slow because the receiver must wait for the sender to decode, and lossy because much of the sender's internal state is compressed into discrete tokens" | Rossi et al. 2026, 2605.22863 |
| 3 | KV-кеш — ключи и значения, вычисленные при чтении | "All key–value pairs computed during prefill are stored in the KV-cache." | Liu 2026 |
| 3 | Переиспользовать кеш в другой модели — задача перевода представлений | "Since prefill's output is the KV cache, reusing it across models reduces to a representation problem" | Heo et al. 2026 |
| 4 | Получатель переиспользует кеш отправителя и не читает текст заново | "the receiver reuses the source's KV cache, skipping prefill" | Heo et al. 2026 |
| 4 | Кеш богаче текста и передаётся сразу целиком, без пословной генерации | "KV-Cache is a naturally richer representation than text. It also enables fully parallel communication … avoiding the slow sequential decoding" | Fu et al. 2025 |
| 4/5 | Прирост на бенчмарке сам по себе не доказывает передачу содержания | "Benchmark deltas do not by themselves establish latent-thought transmission" | Cheng et al. 2026, 2608.04893 |
| 5 | Систему не трогаем, подаём такую же корректную память, но от другого примера | "Hold the system fixed and deliver content that is equally well-formed but belongs to the wrong example." | Cheng et al. 2026 |
| 5 | Один и тот же прирост совместим и с использованием содержания, и с общим эффектом | "the same benchmark delta is compatible with content the receiver uses and with generic interface effects that merely perturb the receiver" | Cheng et al. 2026 |
| 5 | У C2C передача, привязанная к примеру, не обнаружена | "no detected example-specific transfer (C2C's released projector)" | Cheng et al. 2026 |
| 7 | Условие безубыточности объясняет, почему протокол ускоряет длинные контексты и проигрывает на коротких | "This break-even condition explains why the same protocol can accelerate long-context local inference yet regress on short prompts" | Liu 2026 |
| 7 | Перенос кеша в 2,7–25× быстрее перечитывания | "The mapper runs 2.7–25× faster than re-prefill" | Heo et al. 2026 |
| 9 | Модели читают на входе эмбеддинги слов, а скрытые состояния лежат в другой области | "Pretrained LLMs are trained to read token embeddings at the input layer, whereas decoder hidden states occupy a different region of the representation space." | Peng et al. 2026, 2608.13317 |
| 9 | Линейный перенос внутри семейства — 73–98% на 4 из 6 пар | "this linear mapper retains 73–98% of the receiver's standalone-prefill accuracy on four pairs, while two degrade sharply" | Heo et al. 2026 |
| 10 | Выравнивание между архитектурами, сжатие, стандарт оценки, гибридные протоколы | "a research agenda spanning cross-architecture alignment, secure latent channels, compression, evaluation standardisation, and hybrid text–latent protocols" | Liu 2026 |
| 3 | Эмбеддинги — вход первого блока | "Embeddings are the input to the first Transformer block." | Liu 2026 |
| 3 | Скрытое состояние — передаётся между блоками | "Hidden states are the stable, layer-wise semantic representations passed between adjacent Transformer blocks." | Liu 2026 |
| 3 | KV-кеш — ключи и значения, которые каждый слой внимания вычисляет для каждого прочитанного слова | "The collection of per-token key and value tensors computed in each self-attention layer during prefill" | Liu 2026 |
| 3 | Запрос Q не хранится — пересчитывается | "The KV cache stores only the rotated keys, while queries are recomputed at decode time." | Heo et al. 2026 |
| 3 | Чтение заполняет кеш; генерация добавляет по одному слову и обращается к сохранённым K/V | "Prefill encodes the full input to produce the first output token; decode then generates subsequent tokens iteratively using the last token and the cached key–value (KV) states." | Fu et al. 2025 |
