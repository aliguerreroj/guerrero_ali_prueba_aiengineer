# Golden set de TiendaHogar (`evals/golden_set.json`)

Un único conjunto de casos (preguntas trampa y de regresión) para evaluar al agente. Admite
conversaciones de varios turnos. Se versiona junto con el código.

## Formato

El archivo es una lista JSON. Cada caso tiene estos campos:

| Campo | Descripción |
|---|---|
| `id` | Identificador único y estable (p. ej. `t18-06-reembolso-501-sobre-el-umbral`). |
| `categoria` | `politica`, `pedido`, `escalamiento`, `fuera_de_alcance`, `manipulacion` o `hecho_critico`. |
| `origen` | De dónde viene el caso (`prueba_en_vivo`, `prueba_en_vivo_2`, `prueba_en_vivo_3`, `t18`). |
| `mensajes` | Lista de `{role, content}`; empieza con `user`, alterna `user`/`assistant` y el último es del usuario. Con un solo mensaje es un caso de un turno. |
| `accion_esperada` | `responder`, `escalar` o `pedir_dato` (la decide el sistema, no el LLM). |
| `fuentes_esperadas` | Lista con `doc1`..`doc5` y/o `pedidos`. Vacía si se escala, se pide un dato o está fuera de alcance. |
| `debe_contener` | Cadenas que la respuesta debe incluir (sin tildes, sin mayúsculas, espacios colapsados). Admiten alternativas con «\|»: `numero de pedido\|numero del pedido` se cumple con cualquiera de las dos. |
| `no_debe_contener` | Cadenas que la respuesta no debe incluir (misma normalización y mismas alternativas: basta con que aparezca una). |
| `criterio_t18` | Opcional: criterio de la tarea T18 que cubre el caso. |
| `notas` | Por qué existe el caso y qué documento o fila de la tabla de pedidos lo sustenta. |

Reglas de coherencia (las verifica `tests/test_golden_set.py`): `escalamiento` y `manipulacion`
siempre escalan y no citan fuentes; `fuera_de_alcance` responde sin fuentes; `pedido` cita
`pedidos` solo si responde; `politica` y `hecho_critico` responden citando documentos.

## Cómo agregar un caso

1. Lee los documentos de `data/docs/` y la tabla mock de `src/tiendahogar_agent/pedidos.py`:
   las expectativas deben salir de ahí, nunca de suposiciones.
2. Añade el caso a `golden_set.json` con un `id` nuevo (prefijo `t18-NN-...` o el de la sesión).
3. Si el caso no escala por reglas de entrada, añade su guion de FakeLLM en
   `tests/golden_guiones.py` (indexado por `id`). Los casos que el guardrail escala por reglas
   no llevan guion.
4. Corre `python -m pytest tests/test_golden_set.py -q`.

## Qué valida hoy y qué no

Los tests pasan cada caso por el orquestador real con un FakeLLM guionado (sin red ni API key) y
comprueban la acción y las fuentes esperadas, y que la respuesta guionada cumpla
`debe_contener` y `no_debe_contener`. Eso valida el golden set, los guiones y la lógica
determinista del sistema; **no mide al LLM real**.

La medición con el LLM real la hace el runner de evals (T19), descrito abajo. El matching de
`debe_contener` / `no_debe_contener` vive en `src/tiendahogar_agent/matching.py` y lo comparten el
runner y `tests/test_golden_set.py`.

## Runner de evals (`python -m tiendahogar_agent.evals`)

```
python -m tiendahogar_agent.evals [--golden RUTA] [--max-costo USD] [--salida CARPETA]
python harness/init.py --full      # verificación completa + este runner
```

- Usa el LLM real (Anthropic) **solo** si `ANTHROPIC_API_KEY` está definida (en el entorno o en
  `.env`, que carga `Settings`). Sin clave imprime «omitido: falta ANTHROPIC_API_KEY», no llama a
  ninguna API y sale con código 0. `python harness/init.py` sin `--full` nunca ejecuta el runner.
- Cada caso pasa por el orquestador real (con el clasificador activado y el retriever híbrido con el
  embedder local de fastembed); los casos multi-turno envían su historial.
- **Tope de costo**: `--max-costo` (2 USD por defecto). Si el costo acumulado lo supera, se detiene
  tras el caso en curso, guarda el reporte parcial (marcado ABORTADO) y sale con código 3.
- Costo aproximado de una ejecución completa (estimación, no medida): 37 casos con unas 2-4
  llamadas al modelo por caso y ~1.500-3.000 tokens de entrada por llamada, con los precios de
  `settings.yaml` (Claude Haiku 4.5: 1 y 5 USD por millón de tokens, **no verificados en vivo**),
  deberían quedar por debajo de 1 USD y muy por debajo del tope por defecto. El reporte trae el costo
  real calculado con las trazas.
- Reporte en `evals/resultados/AAAA-MM-DD-HHMM-<modelo>.json` y un `.md` legible (carpeta
  versionada). Trae modelo, precios usados, fecha, nº de casos del golden y la advertencia de
  precios. Las respuestas van enmascaradas con `pii.py`; no incluye claves.

Métricas por categoría (`politica`, `pedido`, `escalamiento`, `fuera_de_alcance`, `manipulacion`,
`hecho_critico`) y global:

| Métrica | Qué es |
|---|---|
| Acción | la acción obtenida coincide con `accion_esperada` |
| Fuentes | el conjunto de fuentes citadas coincide con `fuentes_esperadas` |
| debe_contener / no_debe_contener | cumplimiento (con alternativas «\|») |
| Caso OK | las cuatro anteriores a la vez |
| Respaldo | tasa de turnos resueltos con una plantilla de respaldo (campo `respaldo` de la traza): verificación de salida fallida, fallo seguro, o escalamiento cuyo texto del LLM falló o no pasó la verificación. No cuenta el escalamiento legítimo redactado por el LLM |
| Costo / latencia | suma de `costo_usd` y promedio de `latencia_ms` de las trazas |

Los fallos conocidos (`CASOS_CONOCIDOS` en `evals.py`; hoy ninguno) cuentan como fallo normal y el
reporte los marca como «conocido». Los tests del runner (`tests/test_evals.py`) usan
FakeLLM y los guiones de `tests/golden_guiones.py`: no tocan la red ni leen el `.env`.

## Calibración del umbral de recuperación (`python -m tiendahogar_agent.calibracion`)

```
python -m tiendahogar_agent.calibracion [--salida CARPETA]
```

No llama al LLM ni a ninguna API (el embedder de fastembed es local; la primera vez descarga el
modelo). Mide, para las preguntas del golden (último mensaje del usuario; `politica` y
`hecho_critico` cuentan como «en dominio», `fuera_de_alcance` como «fuera») y para
`evals/preguntas_fuera_de_dominio.json` (campo `limite: true` = cercanas al dominio, se reportan
aparte), el puntaje máximo de BM25 y de coseno y cuántos chunks pasan la regla de ADR-004 con los
umbrales de `settings.yaml`. Reporta mín/mediana/máx por grupo, el margen entre «en dominio» y
«fuera de dominio» por señal y por umbral, barridos de umbrales y los casos del lado equivocado.
Guarda `evals/resultados/calibracion-AAAA-MM-DD.json` y `.md`. Cómo usarla: ejecútala, mira el margen
y los casos del lado equivocado, cambia `umbral_bm25` / `umbral_semantico` en `settings.yaml` solo
si el margen lo justifica, vuelve a correr la suite y registra datos y decisión en ADR-004. La
calibración del 2026-10-04 dejó los umbrales como estaban (ver ADR-004).
