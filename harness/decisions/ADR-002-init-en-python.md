# ADR-002 — `init` en Python en lugar de bash

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
El desarrollo ocurre en Windows 11 (PowerShell), pero el repo debe poder verificarse también en Mac/Linux por quien evalúe. Un `init.sh` no corre de forma nativa en Windows.

## Decisión
`harness/init.py` y los hooks se escriben en Python con solo librería estándar. Detectan el intérprete del `.venv` (`.venv\Scripts\python.exe` o `.venv/bin/python`) y calculan los checksums de documentos normalizando CRLF a LF para que den igual en cualquier sistema.

## Alternativas consideradas
- bash/`jq`: no portable a Windows.
- PowerShell: no portable a Mac/Linux.
- Makefile: requiere `make`, ausente en Windows por defecto.

## Consecuencias
- Un único script multiplataforma, sin dependencias extra.
- Python debe estar disponible para correr hooks (ya es requisito del proyecto).
