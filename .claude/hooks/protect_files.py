"""PreToolUse (Edit|Write): bloquea la edición de archivos protegidos (exit 2)."""
import json
import sys
from pathlib import PurePosixPath

PROTEGIDOS = ("data/docs/", "harness/docs_checksums.json", "harness/context/")


if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def es_protegido(ruta: str) -> str | None:
    p = "/" + ruta.replace("\\", "/").lstrip("/")
    if PurePosixPath(p).name == ".env":
        return ".env"
    for prot in PROTEGIDOS:
        if prot.endswith("/"):
            if f"/{prot}" in p + "/" or p.endswith(f"/{prot[:-1]}"):
                return prot
        elif p.endswith(f"/{prot}"):
            return prot
    return None


def main() -> int:
    try:
        datos = json.load(sys.stdin)
    except ValueError:
        return 0
    ruta = (datos.get("tool_input") or {}).get("file_path", "")
    motivo = es_protegido(ruta) if ruta else None
    if motivo:
        print(
            f"BLOQUEADO: '{ruta}' está protegido ({motivo}). "
            "data/docs/, docs_checksums.json, harness/context/ y .env no se editan; "
            "si realmente hace falta, lo decide el humano.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
