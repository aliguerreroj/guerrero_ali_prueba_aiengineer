# ADR-003 — Recuperación híbrida BM25 + fastembed

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
El corpus son 5 documentos muy cortos en español. Se necesita recuperación fiable, tests deterministas sin red y una instalación ligera para quien evalúe. Un embedder basado en PyTorch es pesado y requiere descargar modelos; solo léxico falla con paráfrasis.

## Decisión
Recuperación híbrida:
- **BM25** (`rank_bm25`) con normalización en español (minúsculas, sin tildes): determinista, sin red; base de los tests unitarios.
- **Embedder semántico** con `fastembed` (ONNX, sin PyTorch) detrás del puerto `Embedder`; un embedder falso determinista cubre los tests unitarios.
- Fusión con **Reciprocal Rank Fusion** y umbral/top_k configurables; si nada supera el umbral, la lista es vacía.
- Un test `@pytest.mark.integration` valida el embedder real y se omite por defecto.

## Alternativas consideradas
- Solo embeddings: no determinista en tests y sensible al modelo.
- sentence-transformers (PyTorch): instalación pesada.
- Solo BM25: no cubre paráfrasis.

## Consecuencias
- Tests rápidos y deterministas; el comportamiento semántico real se valida aparte.
- Dos índices que mantener y un parámetro más de calibración (umbral tras la fusión).
