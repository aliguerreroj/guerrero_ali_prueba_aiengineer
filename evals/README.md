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
| `debe_contener` | Cadenas que la respuesta debe incluir (se compara sin tildes ni mayúsculas). |
| `no_debe_contener` | Cadenas que la respuesta no debe incluir (misma normalización). |
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

La medición con el LLM real (tasa de aciertos por caso y categoría) es la tarea T19.
