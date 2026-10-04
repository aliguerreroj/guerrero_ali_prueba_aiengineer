# Calibración del umbral de relevancia — 2026-10-04

- Modelo de embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; chunks indexados: 5
- Umbrales vigentes: BM25 > 0.5, coseno > 0.3
- Preguntas: 15 en dominio (politica/hecho_critico), 25 fuera de dominio, 4 límite, 17 otras (pedido/escalamiento/manipulación; solo informativas)
- No se llamó al LLM. El margen es `min(en dominio) - max(fuera de dominio)` del puntaje máximo por pregunta.

## Distribución del puntaje máximo por pregunta

| Señal | Grupo | n | mín | mediana | máx |
|---|---|---|---|---|---|
| coseno | en_dominio | 15 | 0.244 | 0.553 | 0.698 |
| coseno | fuera_de_dominio | 25 | -0.013 | 0.133 | 0.414 |
| BM25 | en_dominio | 15 | 1.596 | 3.102 | 5.079 |
| BM25 | fuera_de_dominio | 25 | 0.000 | 0.000 | 1.506 |

## Márgenes

| Señal | Margen entre grupos | Umbral | Margen inferior (aciertos) | Margen superior (ruido) |
|---|---|---|---|---|
| coseno | -0.170 | 0.3 | -0.056 | -0.114 |
| BM25 | 0.089 | 0.5 | 1.096 | -1.006 |

Un margen negativo significa que las distribuciones se solapan (ningún umbral separa todo con esa señal sola).

## Barrido del umbral semántico (BM25 vigente)

| Umbral coseno | En dominio sin chunks | Fuera de dominio con chunks |
|---|---|---|
| 0.2 | 0 | 8 |
| 0.25 | 0 | 6 |
| 0.3 | 0 | 4 |
| 0.35 | 0 | 4 |
| 0.4 | 0 | 4 |
| 0.45 | 0 | 4 |
| 0.5 | 0 | 4 |

## Barrido del umbral BM25 (coseno vigente)

| Umbral BM25 | En dominio sin chunks | Fuera de dominio con chunks |
|---|---|---|
| 0.5 | 0 | 4 |
| 1.0 | 0 | 4 |
| 1.5 | 0 | 1 |
| 2.0 | 0 | 1 |

## Casos del lado equivocado (umbrales vigentes)

| Id | Motivo | BM25 máx | Coseno máx | Pregunta |
|---|---|---|---|---|
| t18-16-fuera-matematicas | fuera de dominio pero pasa la regla con 1 chunk(s) (ruido) | 1.102 | 0.256 | ¿Cuánto es 25 por 4? |
| fuera-12 | fuera de dominio pero pasa la regla con 1 chunk(s) (ruido) | 1.236 | 0.190 | Resuelve la ecuación 3x + 7 = 22. |
| fuera-13 | fuera de dominio pero pasa la regla con 1 chunk(s) (ruido) | 1.038 | 0.150 | Tengo dolor de cabeza desde ayer, ¿qué me tomo? |
| fuera-18 | fuera de dominio pero pasa la regla con 2 chunk(s) (ruido) | 1.506 | 0.414 | ¿Cuál es el mejor mes para viajar a Cartagena? |

## Preguntas límite (no cuentan para el margen)

| Id | BM25 máx | Coseno máx | Chunks relevantes | Pregunta |
|---|---|---|---|---|
| fuera-21 | 1.038 | 0.271 | 1 | Recomiéndame una lavadora. |
| fuera-22 | 0.000 | 0.329 | 1 | ¿Venden microondas? |
| fuera-23 | 1.038 | 0.376 | 1 | ¿Cuánto cuesta una refrigeradora? |
| fuera-24 | 1.102 | 0.283 | 3 | ¿Pueden instalarme la estufa en mi casa? |

## Detalle por pregunta

| Id | Grupo | BM25 máx | Coseno máx | Mejor doc | Chunks relevantes |
|---|---|---|---|---|---|
| vivo-01-licuadora-12-meses | en_dominio | 4.639 | 0.633 | doc1 | 4 |
| vivo-02-fuera-de-alcance-mundial | fuera_de_dominio | 0.000 | 0.023 | doc4 | 0 |
| vivo-03-queja-trato-empleado | otras | 2.204 | 0.434 | doc5 | 1 |
| vivo-04-queja-trato-vendedor | otras | 2.204 | 0.382 | doc5 | 3 |
| vivo-05-estado-pedido-existente | otras | 0.000 | 0.201 | doc4 | 0 |
| vivo-06-estado-pedido-sin-id | otras | 0.000 | 0.224 | doc3 | 0 |
| vivo-07-reembolso-sobre-umbral | otras | 2.000 | 0.350 | doc4 | 2 |
| vivo-08-devolucion-pasados-30-dias | en_dominio | 5.079 | 0.698 | doc2 | 4 |
| vivo-09-envio-otras-ciudades | en_dominio | 2.104 | 0.688 | doc3 | 3 |
| vivo-10-promesa-de-reembolso | en_dominio | 2.000 | 0.602 | doc4 | 1 |
| vivo2-01-pedido-inexistente | otras | 0.000 | 0.235 | doc5 | 0 |
| vivo2-02-seguimiento-urgente | otras | 0.000 | 0.207 | doc3 | 0 |
| vivo2-03-reportar-trato-empleado | otras | 2.204 | 0.567 | doc5 | 2 |
| vivo2-04-atendio-pesimo-vendedor | otras | 1.102 | 0.253 | doc5 | 1 |
| vivo2-05-canal-de-contacto | en_dominio | 3.306 | 0.376 | doc5 | 1 |
| vivo2-06-microondas-no-listado | en_dominio | 3.602 | 0.657 | doc1 | 2 |
| vivo2-07-envio-capital | en_dominio | 3.339 | 0.546 | doc3 | 2 |
| vivo2-08-lavadora-devolucion-defecto | en_dominio | 2.862 | 0.379 | doc2 | 3 |
| vivo3-01-lavadora-defecto-condicional | en_dominio | 2.076 | 0.265 | doc2 | 2 |
| t18-01-lavadora-defecto-multiturno | en_dominio | 2.394 | 0.244 | doc1 | 2 |
| t18-02-envio-internacional | en_dominio | 3.339 | 0.432 | doc3 | 1 |
| t18-03-producto-liquidacion | en_dominio | 3.974 | 0.553 | doc2 | 2 |
| t18-04-pedido-entregado-sin-fecha | otras | 0.000 | 0.249 | doc4 | 0 |
| t18-05-reembolso-500-en-el-umbral | en_dominio | 3.102 | 0.417 | doc4 | 2 |
| t18-06-reembolso-501-sobre-el-umbral | otras | 2.000 | 0.574 | doc4 | 2 |
| t18-07-reembolso-en-letras | otras | 2.000 | 0.479 | doc4 | 1 |
| t18-08-reembolso-formato-miles | otras | 2.338 | 0.351 | doc1 | 2 |
| t18-09-mezcla-envio-y-queja-trato | otras | 3.339 | 0.511 | doc4 | 4 |
| t18-10-inyeccion-ignorar-reglas | otras | 2.000 | 0.500 | doc4 | 2 |
| t18-11-inyeccion-rol-y-prompt | otras | 0.338 | 0.392 | doc5 | 1 |
| t18-12-reembolso-tras-lavadora | otras | 2.000 | 0.424 | doc4 | 1 |
| t18-13-fuera-clima | fuera_de_dominio | 0.000 | 0.154 | doc3 | 0 |
| t18-14-fuera-receta | fuera_de_dominio | 0.000 | 0.011 | doc1 | 0 |
| t18-15-fuera-politica | fuera_de_dominio | 0.000 | 0.094 | doc3 | 0 |
| t18-16-fuera-matematicas | fuera_de_dominio | 1.102 | 0.256 | doc4 | 1 |
| t18-17-hecho-capital-y-devolucion | en_dominio | 1.596 | 0.578 | doc3 | 3 |
| t18-18-reembolso-tras-pedido | en_dominio | 2.000 | 0.695 | doc4 | 3 |
| fuera-01 | fuera_de_dominio | 0.000 | 0.280 | doc3 | 0 |
| fuera-02 | fuera_de_dominio | 0.000 | 0.238 | doc3 | 0 |
| fuera-03 | fuera_de_dominio | 0.000 | 0.030 | doc1 | 0 |
| fuera-04 | fuera_de_dominio | 0.000 | -0.013 | doc2 | 0 |
| fuera-05 | fuera_de_dominio | 0.000 | 0.041 | doc1 | 0 |
| fuera-06 | fuera_de_dominio | 0.000 | 0.250 | doc2 | 0 |
| fuera-07 | fuera_de_dominio | 0.000 | 0.098 | doc3 | 0 |
| fuera-08 | fuera_de_dominio | 0.000 | 0.049 | doc3 | 0 |
| fuera-09 | fuera_de_dominio | 0.000 | 0.108 | doc5 | 0 |
| fuera-10 | fuera_de_dominio | 0.000 | 0.033 | doc2 | 0 |
| fuera-11 | fuera_de_dominio | 0.000 | 0.133 | doc1 | 0 |
| fuera-12 | fuera_de_dominio | 1.236 | 0.190 | doc2 | 1 |
| fuera-13 | fuera_de_dominio | 1.038 | 0.150 | doc3 | 1 |
| fuera-14 | fuera_de_dominio | 0.461 | 0.141 | doc1 | 0 |
| fuera-15 | fuera_de_dominio | 0.000 | 0.191 | doc3 | 0 |
| fuera-16 | fuera_de_dominio | 0.360 | 0.224 | doc3 | 0 |
| fuera-17 | fuera_de_dominio | 0.000 | 0.100 | doc3 | 0 |
| fuera-18 | fuera_de_dominio | 1.506 | 0.414 | doc3 | 2 |
| fuera-19 | fuera_de_dominio | 0.318 | 0.180 | doc4 | 0 |
| fuera-20 | fuera_de_dominio | 0.000 | 0.069 | doc3 | 0 |
| fuera-21 | limite | 1.038 | 0.271 | doc1 | 1 |
| fuera-22 | limite | 0.000 | 0.329 | doc1 | 1 |
| fuera-23 | limite | 1.038 | 0.376 | doc1 | 1 |
| fuera-24 | limite | 1.102 | 0.283 | doc1 | 3 |
