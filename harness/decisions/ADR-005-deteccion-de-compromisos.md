# ADR-005 — Detección de compromisos como red de seguridad

- Estado: aceptada
- Fecha: 2026-10-02

## Contexto
El agente no puede aprobar reembolsos, garantizar resultados ni hacer excepciones: lo decide un humano. La verificación de salida (T10) revisa el texto candidato del LLM con una detección de compromisos en `guardrail_compromisos.py`. Una revisión adversarial mostró que la detección léxica inicial dejaba escapar familias enteras de formulaciones («el equipo aprobó tu reembolso», «te haremos el reembolso», «te abonaremos el dinero», «tu reembolso procede»...) y bloqueaba frases inocuas («te aseguro que un humano revisará tu caso»). Intentar cubrir todas las frases posibles con regex es una carrera perdida; hace falta decidir qué papel juega esta pieza y cómo se mide.

## Decisión
- **La detección de compromisos es una red de seguridad, no la defensa principal.** Forma parte de una defensa en capas:
  1. guardrail de entrada determinista (reembolsos sobre el umbral, trato de empleados, facturación y temas legales se escalan antes de llegar al LLM);
  2. prompt de sistema que prohíbe prometer, aprobar o hacer excepciones;
  3. **ausencia de tools de aprobación**: el modelo no tiene ninguna herramienta que apruebe o ejecute un reembolso, así que cualquier «aprobación» es solo texto;
  4. verificación de salida (esta detección, más fundamentación en los chunks recuperados);
  5. evals (T19+) que miden el comportamiento extremo a extremo.
- **Límite léxico asumido.** Es una detección por expresiones regulares sobre texto normalizado (minúsculas, sin tildes). No entiende semántica: paráfrasis muy inusuales pueden escapar y una negación fuera de la lista cerrada no se reconoce.
- **Política «ante la duda, bloquear = escalar».** Un falso positivo cuesta un escalamiento innecesario a soporte; un falso negativo puede costar una promesa que la empresa no puede cumplir. Por eso, en frases ambiguas se bloquea.
- **Frontera política-impersonal / compromiso.** Se permite describir la política en tercera persona impersonal («el reembolso se aprueba tras revisar el producto», «la garantía cubre defectos de fábrica», «aceptamos devoluciones dentro de 30 días»), informar estados y negar («no puedo garantizar», «no hacemos excepciones») con una lista cerrada de negaciones pegadas al verbo. Se bloquea toda afirmación en primera persona, en futuro, de estado o con sujeto en tercera persona dirigida al caso del usuario («tu reembolso procede», «el equipo aprobó tu solicitud»), y las promesas de resultado.
- **Promesas con objeto.** «Te aseguro/prometo que...» y «seguro que...» solo bloquean si el objeto es un resultado (reembolso, aprobación, excepción, dinero, solución, «saldrá bien»). «Garantizado contra/por...» describe la cobertura del fabricante y no bloquea.
- **Criterio de aceptación medible (opción B).** Se mantiene un conjunto FIJO de frases en `tests/data/compromisos_eval.json` (lista de `{frase, esperado, familia}`), y `tests/test_compromisos_metricas.py` falla si la detección de compromisos baja del 90 % o los falsos positivos suben del 5 %. El conjunto no se afina frase por frase: se amplía con frases nuevas cuando aparezcan fallos reales.

## Métricas medidas
Conjunto fijo de 108 frases (51 compromisos, 57 legítimas): las 54 frases que reportó el revisor (39 compromisos y 15 legítimas), más frases nuevas realistas de un agente de soporte, escritas antes de cambiar el detector.

| Momento | Detección de compromisos | Falsos positivos |
|---|---|---|
| Antes de cerrar las familias | 25/51 (49.0 %) | 6/57 (10.5 %) |
| Después | 51/51 (100.0 %) | 0/57 (0.0 %) |

Advertencia de honestidad: el 100 % posterior es optimista. Las 39 frases de compromiso del revisor y las 15 legítimas se conocían al ajustar las familias (aunque se ajustaron familias y no frases sueltas), y el conjunto es pequeño. Las 12 frases nuevas de compromiso ya se detectaban antes del cambio; los 6 falsos positivos eran todos del revisor. La cifra real sobre texto no visto será menor; los evals de T19 la medirán con otro conjunto.

## Alternativas consideradas
- Cubrir exhaustivamente cada frase con regex: inviable y propenso a sobreajuste.
- Clasificador LLM para compromisos: añade latencia, coste y no determinismo en tests; puede añadirse como capa adicional, no sustituye la red léxica.
- Quitar la detección y confiar solo en el prompt: el prompt no es garantía; esta red es barata y determinista.

## Consecuencias
- Positivas: capa determinista, sin red ni API key, probada con métricas explícitas; el fallo seguro es escalar.
- Negativas: cobertura léxica incompleta; mantenimiento de listas de verbos; falsos positivos posibles en frases legítimas con verbos como «devolvemos» o «procesamos» y objeto personal.

## Limitaciones para SUBMISSION
- La detección de compromisos es léxica y no garantiza cobertura total; es una capa más, no la única.
- Paráfrasis inusuales, ironía o frases partidas en varios mensajes pueden escapar.
- Una negación fuera de la lista cerrada no se reconoce y el texto se bloquea (falla seguro).
- Las métricas (51/51 y 0/57) provienen de un conjunto fijo y pequeño de 108 frases conocido durante el ajuste; no son una estimación del desempeño en producción.
- Medición independiente del revisor sobre frases nuevas (no guardadas en el repo): detección 42/47 (89,4 %) y falsos positivos 0/51. Es la cifra más cercana al desempeño real y queda justo por debajo del 90 %. Escapes: verbos de envío o transferencia del dinero («te enviaremos el dinero», «te transferimos el valor»), estados de proceso («tu reembolso ya está en proceso», «se procesa tu reembolso») y gerundio con enclítico («devolviéndote»). Mejora futura posible.
- La protección real contra aprobaciones se apoya en que el agente no tiene tools de aprobación y en que el guardrail de entrada escala los casos prohibidos.
