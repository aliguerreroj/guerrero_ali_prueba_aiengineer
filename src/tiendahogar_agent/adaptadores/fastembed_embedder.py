"""Adaptador del puerto `Embedder` con fastembed (ONNX, sin PyTorch).

fastembed se importa de forma perezosa: los tests unitarios no lo necesitan ni
descargan modelos (se puede inyectar un `TextEmbedding` falso).
"""

from __future__ import annotations

from typing import Any

MODELO_POR_DEFECTO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedEmbedder:
    def __init__(self, modelo: str = MODELO_POR_DEFECTO, text_embedding: Any | None = None) -> None:
        self.modelo = modelo
        self._te = text_embedding

    def _motor(self) -> Any:
        if self._te is None:
            from fastembed import TextEmbedding  # importación perezosa

            self._te = TextEmbedding(model_name=self.modelo)
        return self._te

    def embed(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            return []
        return [[float(x) for x in v] for v in self._motor().embed(list(textos))]
