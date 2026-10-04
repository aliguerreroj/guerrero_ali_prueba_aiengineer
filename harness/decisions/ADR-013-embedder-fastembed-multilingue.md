# ADR-013 — Embedder: fastembed con MiniLM multilingüe

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
ADR-003 decidió una recuperación híbrida con un embedder `fastembed` detrás del puerto `Embedder`. Faltaba fijar el modelo concreto. Los documentos y las preguntas están en español, los clientes parafrasean, y la instalación debe seguir siendo ligera y funcionar en Docker sin red en tiempo de ejecución.

## Decisión
- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, definido en `config.py` (`MODELO_EMBEDDING_POR_DEFECTO`) y en `settings.yaml` (`embedding_model`, configurable). Vectores de 384 dimensiones.
- Ejecución con `fastembed` sobre ONNX, sin PyTorch. Adaptador `FastEmbedEmbedder` (`adaptadores/fastembed_embedder.py`): importación perezosa de `fastembed`, de modo que los tests unitarios no descargan nada (usan `FakeEmbedder` o un `TextEmbedding` inyectado).
- Multilingüe: cubre español sin un modelo aparte.
- Descarga: en la primera ejecución real (unos 0,22 GB, según el registro de la sesión de selección del modelo, `harness/progress/2026-10-02-s03.md`) y, en Docker, durante el build, con `HF_HUB_OFFLINE=1` en ejecución (ADR-017).
- La calibración del 2026-10-04 (`evals/resultados/calibracion-2026-10-04.md`) se hizo con este modelo y 5 chunks indexados. La regla de relevancia y sus umbrales están en ADR-004; no se repiten aquí.
- Si el embedder falla al indexar o al consultar, el retriever sigue solo con BM25 y deja un aviso en el log (`retriever.py`).

## Alternativas consideradas
- `sentence-transformers` con PyTorch: misma familia de modelos, pero una instalación y una imagen mucho más pesadas.
- API de embeddings remota: añade dependencia de red, costo y otro secreto, y haría el retriever no determinista en CI; el corpus es de solo 5 documentos.
- Solo BM25: sin dependencias, pero no cubre paráfrasis (ya descartado en ADR-003).

## Consecuencias
- Positivas: instalación ligera, español cubierto, ejecución local y sin red tras la primera descarga; el modelo se cambia por configuración.
- Negativas: la primera ejecución necesita red y ~0,22 GB de descarga; un modelo de paráfrasis genérico no está entrenado para este dominio, por eso la relevancia combina BM25 y coseno con umbrales calibrados (ADR-004); cambiar de modelo obliga a recalibrar los umbrales.
