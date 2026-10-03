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
- Positivas: «lista vacía» es una señal fiable de falta de sustento: el orquestador escala a un humano o pide un dato, nunca inventa respuesta. Los puntajes originales quedan en las trazas.
- Negativas: dos umbrales que calibrar (T19); con solo 5 documentos y 6 consultas de observación el margen del umbral semántico es estrecho; el modo degradado pierde las paráfrasis.
- Respuesta a la pregunta de umbrales del entregable: los umbrales (BM25 0.5, coseno 0.3) se eligieron con los puntajes originales observados sobre los documentos reales, entre el peor acierto en dominio y el mejor puntaje fuera de dominio, y quedan marcados como provisionales hasta la calibración de T19.
