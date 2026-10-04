# ADR-014 — Chunking con estrategia `auto`

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
Los 5 documentos de `data/docs/` son muy cortos: pesan 1408 bytes en total y ninguno pasa de 350 bytes (`settings.yaml` los cifra en ~200-330 caracteres). Trocearlos no aporta nada y rompería frases que el agente cita tal cual. Aun así, el sistema no debe asumir que los documentos serán siempre así de cortos.

## Decisión
`documentos.trocear` ofrece cuatro estrategias, elegidas por `Settings.chunk_strategy` (`settings.yaml`, por defecto `auto`):
- `none`: un único chunk con todo el texto.
- `fixed`: ventanas de `chunk_tamano` caracteres desplazadas `chunk_tamano - chunk_solape`.
- `recursive`: divide respetando, en este orden, párrafos (`\n\n`), líneas, frases (`. `) y palabras; el solape es la cola del chunk anterior.
- `auto`: `none` si el texto (sin espacios en los extremos) mide como máximo `chunk_umbral_corto` caracteres; si no, `recursive`.

Valores actuales: `chunk_umbral_corto: 1000`, `chunk_tamano: 500`, `chunk_solape: 50`. Con ellos los 5 documentos quedan en un chunk cada uno, es decir, 5 chunks indexados (el número que consigna `evals/resultados/calibracion-2026-10-04.md`). Cada chunk lleva en sus metadatos `titulo`, `fuente`, `estrategia`, `posicion` y `total_chunks`, y su `doc_id` se deriva del nombre de archivo (`doc1_garantia.md` -> `doc1`), que es la fuente que cita el agente. La configuración se valida (`chunk_solape` entre 0 y `chunk_tamano - 1`).

## Qué pasa si el corpus crece
Implementado: un documento largo pasa a `recursive` sin cambiar código ni configuración, y el retriever indexa todos los chunks resultantes (BM25 más embeddings en memoria).

Límites reales del código actual: el retriever embebe todos los chunks de una vez al arrancar, `InMemoryVectorStore.buscar` compara la consulta con todos los vectores (búsqueda lineal) y el retriever pide la similitud de todos los chunks en cada consulta. Es correcto y rápido para decenas o cientos de chunks; no está pensado para miles de documentos.

Trabajo futuro, NO implementado: almacén vectorial externo con índice aproximado detrás del puerto `VectorStore` (ADR-011), indexación incremental y persistente en lugar de recalcular al arrancar, calibración de `chunk_tamano` y `chunk_solape` con documentos reales largos (los valores actuales no se han medido contra ninguno), y un reranker si la recuperación pierde precisión con más contenido.

## Alternativas consideradas
- Trocear siempre con `fixed` o `recursive`: parte documentos que caben enteros y complica las citas.
- No trocear nunca (`none`): válido hoy, pero un documento largo produciría un único vector poco específico.
- Chunking semántico o por encabezados Markdown: más trabajo sin beneficio con documentos de unas pocas frases.

## Consecuencias
- Positivas: hoy no se pierde contexto; el comportamiento se adapta solo a documentos largos; las estrategias son funciones puras, fáciles de probar.
- Negativas: `recursive` con estos parámetros no está calibrado con documentos reales largos; la escalabilidad a miles de documentos requiere el trabajo futuro indicado. Los umbrales de relevancia (ADR-004) se calibraron con 1 chunk por documento y deberían recalibrarse si el chunking cambia.
