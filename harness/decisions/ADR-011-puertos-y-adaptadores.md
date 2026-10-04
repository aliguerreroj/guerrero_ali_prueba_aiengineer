# ADR-011 — Puertos y adaptadores, dobles de prueba y selección de proveedor

- Estado: aceptada
- Fecha: 2026-10-04

## Contexto
El agente depende de piezas que cambian según el entorno: un LLM remoto (Anthropic o Azure OpenAI), un modelo de embeddings, un almacén vectorial, una fuente de documentos, un repositorio de pedidos, un destino de eventos y un destino de trazas. Los tests deben correr sin red ni API key, y cambiar de proveedor no debe obligar a tocar la lógica del dominio (guardrails, orquestador, retriever).

## Decisión
1. **Siete puertos** como `Protocol` con `@runtime_checkable` en `src/tiendahogar_agent/puertos/__init__.py`: `LLMClient` (`completar`), `Embedder` (`embed`), `VectorStore` (`indexar`, `buscar`), `DocumentSource` (`cargar`), `OrderRepository` (`consultar_estado_pedido`), `EventBus` (`publicar`) y `TraceSink` (`registrar`). El dominio solo importa estos puertos, nunca un SDK.
2. **Adaptadores** en `src/tiendahogar_agent/adaptadores/`: `AnthropicLLM` y `AzureOpenAILLM` (traducen los errores de cada SDK a `ErrorLLM`, ver ADR-006), `FastEmbedEmbedder` (ADR-013), `InMemoryVectorStore` (`almacen_memoria.py`), `LocalEventBus` (ADR-010) y `JsonlTraceSink`. Dos puertos tienen su implementación en módulos del dominio y no en `adaptadores/`: `FileSystemDocumentSource` en `documentos.py` y `OrderRepositoryMock` en `pedidos.py`.
3. **Dobles de prueba** en `src/tiendahogar_agent/dobles.py`: `FakeLLM` (guionizado), `FakeEmbedder`, `FakeDocumentSource`, `FakeOrderRepository`, `FakeTraceSink` y `FakeEventBus`. Ningún módulo de `src` los importa de forma estática; la única referencia es un `import_module("tiendahogar_agent.dobles")` dentro de `crear_llm` (`adaptadores/fabrica_llm.py`), para el caso `llm_provider == "fake"`. Los tests sí importan `dobles` directamente.
4. **Selección por configuración**: `Settings.llm_provider` (`fake` | `anthropic` | `azure`, por defecto `fake`; valor en `settings.yaml`, claves y endpoint en `.env`). `crear_llm` devuelve el cliente; si faltan credenciales lanza `ValueError` con los nombres de los ajustes que faltan, sin valores secretos. Los adaptadores de Anthropic y Azure se importan dentro de la rama del proveedor elegido.
5. **Modo demostración de la CLI**: `cli.construir_orquestador` con `fake` no usa `FakeLLM` de `dobles` sino `LLMDemo` (`llm_demo.py`, reglas fijas, sin red) y un retriever léxico, para poder chatear sin API key. `FakeLLM` queda para tests.

## Alternativas consideradas
- Importar los SDKs directamente en el dominio: más corto, pero acopla la lógica a un proveedor y obliga a mockear SDKs en los tests.
- Una clase base abstracta (ABC) por puerto: obliga a heredar; con `Protocol` basta con cumplir la firma (tipado estructural) y los dobles no dependen de nada.
- Dobles dentro de `tests/`: el proveedor `fake` de la fábrica no podría construirlos; se mantienen en el paquete con carga dinámica para que el código de producción no los importe de forma estática.
- Un contenedor de inyección de dependencias: sobredimensionado para siete puertos; la composición es explícita en `cli.construir_orquestador`.

## Consecuencias
- Positivas: tests deterministas sin red; añadir un proveedor es escribir un adaptador y una rama en la fábrica; el dominio no cambia. En producción cada puerto se reemplazaría por un servicio gestionado (otro LLM, un almacén vectorial externo, Kafka o Event Hubs para `EventBus`, ver ADR-010) sin tocar el orquestador.
- Negativas: `dobles.py` viaja dentro del paquete instalado; la fábrica solo cubre `LLMClient` (el resto se compone a mano en la CLI); dos puertos tienen implementación fuera de `adaptadores/`, una inconsistencia menor de ubicación.
