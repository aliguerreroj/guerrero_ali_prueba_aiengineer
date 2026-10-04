# ADR-004 — Regla de relevancia del retriever

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
ADR-003 hablaba de un umbral tras la fusión RRF. Eso no funciona: el puntaje RRF depende solo de las posiciones en los rankings, no de cuán buena es la coincidencia. Con 5 chunks, el mejor de cualquier consulta (incluida «¿quién ganó el mundial?») obtiene siempre un RRF alto, así que ningún umbral sobre RRF separa lo relevante de lo irrelevante. El agente necesita saber cuándo NO hay sustento para responder sin inventar.

## Decisión
- **RRF solo ordena.** La relevancia se decide sobre los puntajes **originales**:
  un chunk es relevante si `BM25 > umbral_bm25` **O** `coseno > umbral_semantico` (O lógico: basta una señal).
- Un BM25 `<= 0` nunca cuenta como relevante (en corpus diminutos `BM25Okapi` puede dar valores negativos).
- Si ningún chunk es relevante, `recuperar` devuelve `[]`. Solo los relevantes entran a la fusión; cada ranking (léxico, semántico) contiene únicamente los chunks relevantes por esa señal. Luego se aplica RRF (`k = 60`, constante habitual de la literatura) y se corta a `top_k`. Empates: orden original de los documentos.
- Cada resultado conserva `puntaje_bm25` y `similitud` originales para trazas y calibración.
- Fallo del embedder: se degrada a solo BM25 y se registra un aviso, en lugar de propagar la excepción.
- Valores provisionales (en `settings.yaml`, editables por entorno): `umbral_bm25 = 0.5`, `umbral_semantico = 0.3`. **Son provisionales hasta T19**, donde se calibran con el conjunto de evaluación.

### Valores observados (modelo real `paraphrase-multilingual-MiniLM-L12-v2`, 5 documentos reales)
| Consulta | Mejor coseno | Mejor BM25 |
|---|---|---|
| garantía de mi licuadora | 0.563 (doc1) | 1.634 (doc1) |
| quiero devolver un producto | 0.427 (doc4; doc2 0.386) | 1.581 (doc2) |
| envíos a Miami | 0.347 (doc3) | 2.104 (doc3) |
| cuándo me devuelven el dinero | 0.554 (doc4) | 0.0 (todos) |
| un empleado me trató mal | 0.392 (doc5) | 2.204 (doc5) |
| quién ganó el mundial (fuera de dominio) | 0.019 | 0.0 (todos) |

- `umbral_semantico = 0.3`: está por debajo del peor mejor-acierto en dominio (0.347) y muy por encima del máximo fuera de dominio (0.019). Margen: 0.047 por abajo y 0.28 por arriba; es asimétrico a propósito, porque dejar pasar un poco más de ruido lo corrige el `top_k` y el orden, mientras que perder un acierto real llevaría a escalar sin necesidad. El margen inferior es estrecho: por eso se recalibra en T19.
- `umbral_bm25 = 0.5`: por encima de los puntajes espurios más bajos observados en chunks no objetivo (0.32 y 0.34, por coincidencias de una raíz común como «garantía») y por debajo del menor acierto (1.58); fuera de dominio todos son 0. Limitación: no separa del todo, porque «un empleado me trató mal» da 1.038 a doc1 (frente a 2.204 de doc5, el acierto); ese ruido lo contiene el orden RRF y el `top_k`, no el umbral. Además, el límite superior del umbral semántico se apoya en una sola consulta fuera de dominio (muestra de una); T19 la amplía.

### Brecha léxica
«¿cuándo me devuelven el dinero?» no comparte raíz con «reembolso» (doc4): BM25 da 0 a todo. Solo la señal semántica (coseno 0.554 a doc4) la rescata, y es la razón de ser del O lógico y de la recuperación híbrida. Sin embedder (modo degradado) esa consulta devuelve `[]`.

## Alternativas consideradas
- Umbral sobre el score RRF: descartado, no mide relevancia (ver Contexto).
- Normalizar y combinar BM25 y coseno en un solo puntaje con pesos: más parámetros que calibrar y BM25 no tiene escala acotada.
- Exigir ambas señales (Y lógico): perdería paráfrasis como la del dinero.

## Consecuencias
- Positivas: «lista vacía» es una señal fiable de falta de sustento. El orquestador no escala solo por tener la lista vacía: el contexto indica al LLM que no hay documentos relevantes, el sistema acepta un `escalar` por ser el lado seguro, y la verificación de salida rechaza cifras o fuentes sin sustento (corregido el 2026-10-04: este ADR decía antes que el orquestador escalaba o pedía un dato directamente). Los puntajes originales quedan en las trazas.
- Negativas: dos umbrales que calibrar (T19); con solo 5 documentos y 6 consultas de observación el margen del umbral semántico es estrecho; el modo degradado pierde las paráfrasis.
- Respuesta a la pregunta de umbrales del entregable: los umbrales (BM25 0.5, coseno 0.3) se eligieron con los puntajes originales observados sobre los documentos reales, entre el peor acierto en dominio y el mejor puntaje fuera de dominio, y quedan marcados como provisionales hasta la calibración de T19.

## Nota de calibración (T19) — 2026-10-04: los umbrales NO cambian
Datos: `python -m tiendahogar_agent.calibracion` (embedder real `paraphrase-multilingual-MiniLM-L12-v2`, sin LLM; resultados en `evals/resultados/calibracion-2026-10-04.json` y `.md`). Muestra: 15 preguntas en dominio (golden `politica` + `hecho_critico`, último mensaje del usuario), 25 fuera de dominio (5 del golden `fuera_de_alcance` + 20 de `evals/preguntas_fuera_de_dominio.json`: clima, recetas, política, deportes, programación, matemáticas, salud, saludos, cultura, viajes, finanzas, tecnología) y 4 «límite» (recomendar una lavadora, microondas, precio de una refrigeradora, instalar una estufa), que se reportan aparte.

| Señal (puntaje máximo por pregunta) | En dominio (mín / mediana / máx) | Fuera de dominio (mín / mediana / máx) | Margen (mín dominio − máx fuera) |
|---|---|---|---|
| Coseno | 0.244 / 0.553 / 0.698 | -0.013 / 0.133 / 0.414 | **-0.170** (se solapan) |
| BM25 | 1.596 / 3.102 / 5.079 | 0.000 / 0.000 / 1.506 | +0.089 |

- Con los umbrales vigentes (BM25 > 0.5, coseno > 0.3): ninguna pregunta en dominio se queda sin chunk relevante (0 de 15), pero 4 de 25 fuera de dominio traen algún chunk (ruido): «¿Cuánto es 25 por 4?» (BM25 1.10), «Resuelve la ecuación 3x + 7 = 22» (1.24), «dolor de cabeza» (1.04) y «mejor mes para viajar a Cartagena» (BM25 1.51 y coseno 0.414). Las 4 preguntas límite también traen chunks (como cabe esperar: hablan de lavadoras, microondas, refrigeradoras y estufas).
- Margen del umbral semántico: `min(en dominio) − 0.3 = -0.056` (dos preguntas en dominio quedan por debajo, 0.244 y 0.265, y las rescata BM25 con 2.39 y 2.08) y `0.3 − max(fuera) = -0.114`. Ningún umbral de coseno separa los dos grupos con esa señal sola.
- Barrido (una señal a la vez): subir BM25 a 1.5 deja 1 de 25 fuera con chunks y 0 de 15 en dominio sin chunks; subir el coseno a 0.45 con BM25 en 1.7–2.0 deja 0 y 0. Pero el margen de esas combinaciones es mínimo: `t18-17` tiene BM25 1.596 (solo 0.09 sobre el mejor ruido, 1.506) y la consulta «quiero devolver un producto» observada arriba tiene BM25 1.581 y coseno 0.427, de modo que BM25 ≥ 1.7 y coseno 0.45 la dejarían sin chunks.
- Decisión: **se mantienen `umbral_bm25 = 0.5` y `umbral_semantico = 0.3`**. Los datos confirman que no se pierde ningún acierto en dominio, y subir los umbrales solo compraría eliminar ruido a costa de márgenes de 0.05–0.1 con una muestra de 15 preguntas en dominio. Lo asimétrico de ADR-004 se mantiene: el ruido lo contienen el `top_k`, el clasificador de intención (que atiende `fuera_de_alcance` antes del retriever), la verificación de salida y el modelo; un acierto perdido llevaría a escalar sin necesidad.
- Límites: muestra pequeña y redactada por el mismo autor; solo 5 documentos cortos (BM25 con IDF poco estable); el golden en dominio comparte vocabulario con los documentos (sesgo optimista para BM25); la calibración mide recuperación, no la calidad de la respuesta final (eso lo mide el runner de evals con el LLM real). Si cambia el corpus, el modelo de embeddings o se amplían las preguntas, repetir la calibración (el proceso está en `evals/README.md`). Los umbrales dejan de ser «provisionales»: quedan calibrados con esta muestra y con esas reservas.
