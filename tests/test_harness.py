"""Verifica que los documentos base existen y no fueron modificados."""
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "data" / "docs"
NOMBRES = [
    "doc1_garantia.md",
    "doc2_devoluciones.md",
    "doc3_envios.md",
    "doc4_reembolsos.md",
    "doc5_canales.md",
]


def sha256_normalizado(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_existen_los_5_documentos():
    for nombre in NOMBRES:
        assert (DOCS / nombre).is_file(), f"falta data/docs/{nombre}"


def test_checksums_coinciden():
    esperado = json.loads((RAIZ / "harness" / "docs_checksums.json").read_text(encoding="utf-8"))
    assert sorted(esperado) == sorted(NOMBRES)
    for nombre in NOMBRES:
        assert sha256_normalizado(DOCS / nombre) == esperado[nombre], nombre
