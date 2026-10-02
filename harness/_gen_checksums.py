"""Genera harness/docs_checksums.json (SHA-256 con CRLF normalizado a LF).

Uso humano puntual: python harness/_gen_checksums.py
"""
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "data" / "docs"


def sha256_normalizado(ruta: Path) -> str:
    datos = ruta.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(datos).hexdigest()


if __name__ == "__main__":
    resultado = {p.name: sha256_normalizado(p) for p in sorted(DOCS.glob("*.md"))}
    destino = RAIZ / "harness" / "docs_checksums.json"
    destino.write_text(json.dumps(resultado, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(destino)
