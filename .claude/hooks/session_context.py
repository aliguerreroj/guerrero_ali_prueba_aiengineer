"""SessionStart: imprime contexto (git log, último progress, tareas pendientes)."""
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("## Últimos commits")
    r = subprocess.run(
        ["git", "log", "--oneline", "-5"], cwd=RAIZ, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    print(r.stdout.strip() or "(sin commits)")

    progreso = sorted(p for p in (RAIZ / "harness" / "progress").glob("*.md") if p.name != "README.md")
    print("\n## Último registro de progreso")
    if progreso:
        print(f"Archivo: {progreso[-1].name}\n")
        print(progreso[-1].read_text(encoding="utf-8"))
    else:
        print("(ninguno)")

    print("\n## Tareas pendientes")
    ruta = RAIZ / "harness" / "tasks.json"
    if ruta.exists():
        tareas = json.loads(ruta.read_text(encoding="utf-8")).get("tareas", [])
        for t in tareas:
            if t.get("estado") != "hecha":
                print(f"- {t['id']} [{t['estado']}] ({t['prioridad']}) {t['titulo']}")
    else:
        print("(no existe harness/tasks.json)")


if __name__ == "__main__":
    main()
