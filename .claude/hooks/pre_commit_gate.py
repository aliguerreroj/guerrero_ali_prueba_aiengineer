"""PreToolUse (Bash): si el comando es un git commit, exige harness/init.py en verde."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])


if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def es_commit(cmd: str) -> bool:
    return re.search(r"\bgit\b(\s+-\S+(\s+\S+)?)*\s+commit\b", cmd) is not None


def main() -> int:
    try:
        datos = json.load(sys.stdin)
    except ValueError:
        return 0
    cmd = (datos.get("tool_input") or {}).get("command", "")
    if not es_commit(cmd):
        return 0
    py = next(
        (str(RAIZ / r) for r in (".venv/Scripts/python.exe", ".venv/bin/python") if (RAIZ / r).exists()),
        sys.executable,
    )
    print("pre_commit_gate: ejecutando harness/init.py ...", file=sys.stderr)
    r = subprocess.run(
        [py, str(RAIZ / "harness" / "init.py")],
        cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if r.returncode != 0:
        print("COMMIT BLOQUEADO: harness/init.py falló.\n" + r.stdout + r.stderr, file=sys.stderr)
        return 2
    print("pre_commit_gate: init en verde, commit permitido.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
