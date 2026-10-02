"""Verificación del entorno y del repo. Solo librería estándar.

Uso: python harness/init.py [--full]
Se detiene en el primer fallo, explica cómo arreglarlo y sale con código 1.
Nunca corrige nada automáticamente.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "data" / "docs"
CHECKSUMS = RAIZ / "harness" / "docs_checksums.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def fallo(msg: str, arreglo: str, detalle: str = "") -> None:
    print(f"❌ {msg}")
    if detalle:
        print(detalle.rstrip())
    print(f"   Cómo arreglarlo: {arreglo}")
    sys.exit(1)


def python_venv() -> Path | None:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        p = RAIZ / rel
        if p.exists():
            return p
    return None


def sha256_normalizado(ruta: Path) -> str:
    datos = ruta.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(datos).hexdigest()


def correr(py: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(py), *args],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> None:
    full = "--full" in sys.argv[1:]

    # 1. Versión de Python
    if sys.version_info < (3, 11):  # noqa: UP036
        fallo(
            f"Python {sys.version.split()[0]} < 3.11",
            "instala Python >= 3.11 y recrea el entorno: python -m venv .venv",
        )
    ok(f"Python {sys.version.split()[0]}")

    # 2. .venv y dependencias
    py = python_venv()
    if py is None:
        fallo(
            "no existe .venv",
            "python -m venv .venv ; activa el entorno ; python -m pip install -e \".[dev]\"",
        )
    for modulo in ("ruff", "pytest"):
        r = correr(py, "-m", modulo, "--version")
        if r.returncode != 0:
            fallo(
                f"{modulo} no está instalado en .venv",
                "activa el .venv y ejecuta: python -m pip install -e \".[dev]\"",
                r.stderr,
            )
    ok(f"entorno .venv y dependencias ({py.relative_to(RAIZ)})")

    # 3. Checksums de documentos
    if not CHECKSUMS.exists():
        fallo("falta harness/docs_checksums.json", "restáuralo con: git checkout -- harness/docs_checksums.json")
    esperado = json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    for nombre, sha in esperado.items():
        ruta = DOCS / nombre
        if not ruta.exists():
            fallo(f"falta data/docs/{nombre}", f"restáuralo con: git checkout -- data/docs/{nombre}")
        if sha256_normalizado(ruta) != sha:
            fallo(
                f"data/docs/{nombre} fue modificado (checksum distinto)",
                f"los documentos no se editan; restáuralo con: git checkout -- data/docs/{nombre}",
            )
    ok(f"{len(esperado)} documentos coinciden con docs_checksums.json")

    # 4. ruff
    r = correr(py, "-m", "ruff", "check", ".")
    if r.returncode != 0:
        fallo("ruff check encontró problemas", "corrige lo indicado (o: python -m ruff check . --fix) y reintenta", r.stdout + r.stderr)
    ok("ruff check")

    # 5. pytest
    r = correr(py, "-m", "pytest", "-q")
    if r.returncode != 0:
        fallo("pytest falló", "corrige los tests o el código que rompen y reintenta", r.stdout + r.stderr)
    ok("pytest")

    if full:
        print("ℹ️  evals aún no implementados")
    print("🟢 init.py en verde")


if __name__ == "__main__":
    main()
