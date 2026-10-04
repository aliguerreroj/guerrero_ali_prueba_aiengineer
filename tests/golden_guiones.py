"""Guiones de FakeLLM para el golden set (`evals/golden_set.json`), indexados por id de caso.

Cada guion es la secuencia de respuestas que daría un LLM «bien portado» para ese caso. Sirven
para validar el golden set y el orquestador de forma determinista (sin red ni API key); NO miden
al LLM real (eso es T19).

- Casos que el guardrail de entrada escala por reglas: no tienen guion; el test usa un LLM vacío
  y el aviso sale de la plantilla de respaldo por categoría (ver `PLANTILLAS_ESCALAMIENTO`).
- `clasificador=True`: el primer turno del LLM es la clasificación de intención (fuera de alcance).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from tiendahogar_agent.adaptadores.almacen_memoria import InMemoryVectorStore
from tiendahogar_agent.dobles import FakeEmbedder, FakeLLM
from tiendahogar_agent.documentos import FileSystemDocumentSource
from tiendahogar_agent.indice_lexico import IndiceLexico
from tiendahogar_agent.models import LLMResponse
from tiendahogar_agent.retriever import Retriever

_contador = itertools.count(1)


@dataclass(frozen=True)
class Guion:
    """Respuestas encoladas del FakeLLM y si el clasificador de intención está activo."""

    respuestas: tuple[LLMResponse, ...]
    clasificador: bool = False


def _tool(nombre: str, argumentos: dict) -> LLMResponse:
    return FakeLLM.llamada_tool(nombre, argumentos, id=f"gs{next(_contador)}")


def _responder(texto: str, fuentes: list[str], **extra: str) -> LLMResponse:
    return _tool("responder", {"respuesta": texto, "fuentes": fuentes, **extra})


def _simple(texto: str, fuentes: list[str]) -> Guion:
    return Guion((_responder(texto, fuentes),))


def _con_pedido(order_id: str, texto: str, fuentes: list[str], **extra: str) -> Guion:
    return Guion((
        _tool("consultar_estado_pedido", {"order_id": order_id}),
        _responder(texto, fuentes, **extra),
    ))


def _fuera_de_alcance() -> Guion:
    return Guion((_tool("clasificar_intencion", {"intencion": "fuera_de_alcance"}),), True)


GUIONES: dict[str, Guion] = {
    "vivo-01-licuadora-12-meses": _simple(
        "No exactamente: las licuadoras son electrodomésticos pequeños y tienen 6 meses de "
        "garantía desde la fecha de compra; los 12 meses aplican a los grandes, como "
        "refrigeradoras, lavadoras y estufas.", ["doc1"]),
    "vivo-02-fuera-de-alcance-mundial": _fuera_de_alcance(),
    "vivo-05-estado-pedido-existente": _con_pedido(
        "ORD-1001",
        "Tu pedido ORD-1001 (Refrigeradora) está En tránsito y su entrega estimada es en 3 días "
        "hábiles.", ["pedidos"]),
    "vivo-06-estado-pedido-sin-id": Guion((_responder(
        "Con gusto te ayudo. ¿Me compartes el número de pedido? Tiene el formato ORD-####.",
        [], accion_sugerida="pedir_dato"),)),
    "vivo-08-devolucion-pasados-30-dias": _simple(
        "Dentro de 30 días de la compra puedes devolverlo si está sin usar y en su empaque "
        "original. Pasados los 30 días solo se acepta si el producto tiene un defecto cubierto "
        "por garantía.", ["doc1", "doc2"]),
    "vivo-09-envio-otras-ciudades": _simple(
        "Los envíos a otras ciudades tardan 5-7 días hábiles (a la capital son 2-3 días hábiles). "
        "Los envíos internacionales no están disponibles.", ["doc3"]),
    "vivo-10-promesa-de-reembolso": _simple(
        "No puedo prometerte ni aprobar un reembolso. Te cuento la política: se procesa en 5-10 "
        "días hábiles después de recibir el producto devuelto, al mismo método de pago original, "
        "y los reembolsos mayores a $500 requieren la aprobación de un supervisor humano.",
        ["doc4"]),
    "vivo2-01-pedido-inexistente": _con_pedido(
        "ORD-9999",
        "No encontré un pedido con el número ORD-9999. ¿Puedes revisar el número y volver a "
        "escribirlo?", [], accion_sugerida="pedir_dato"),
    "vivo2-02-seguimiento-urgente": _simple(
        "Entiendo que lo necesitas con urgencia. Tu pedido ORD-1003 (Lavadora) está Procesando "
        "con entrega estimada de 6 días hábiles; no tengo información sobre envíos urgentes.",
        ["pedidos"]),
    "vivo2-05-canal-de-contacto": _simple(
        "Puedes escribir a soporte@tiendahogar.example para tus consultas.", ["doc5"]),
    "vivo2-06-microondas-no-listado": _simple(
        "Gracias por preguntar. El microondas no aparece listado en la política de garantía, "
        "así que prefiero no suponer un plazo para él.", ["doc1"]),
    "vivo2-07-envio-capital": _simple(
        "Los envíos a la capital tardan 2-3 días hábiles.", ["doc3"]),
    "vivo2-08-lavadora-devolucion-defecto": _simple(
        "Lamento que tu lavadora haya dejado de centrifugar. Sí puedes devolverla: pasados los 30 "
        "días de la compra, el documento de devoluciones acepta el producto si tiene un defecto "
        "cubierto por garantía, y las lavadoras tienen 12 meses de garantía desde la fecha de "
        "compra.", ["doc1", "doc2"]),
    "vivo3-01-lavadora-defecto-condicional": _simple(
        "Entiendo la molestia. Si es un defecto cubierto por garantía, puedes devolverla aunque "
        "hayan pasado los 30 días; las lavadoras tienen 12 meses de garantía.", ["doc1", "doc2"]),
    "t18-01-lavadora-defecto-multiturno": _simple(
        "Como pasaron más de 30 días, la devolución solo se acepta si el producto tiene un "
        "defecto cubierto por garantía. Las lavadoras tienen 12 meses de garantía desde la compra "
        "y esta cubre defectos de fábrica, así que sí aplica ese camino.", ["doc1", "doc2"]),
    "t18-02-envio-internacional": _simple(
        "Por ahora los envíos internacionales no están disponibles actualmente.", ["doc3"]),
    "t18-03-producto-liquidacion": _simple(
        "Los productos en oferta final («liquidación») no admiten devolución: no se aceptan "
        "devoluciones de ese tipo de productos.", ["doc2"]),
    "t18-04-pedido-entregado-sin-fecha": _con_pedido(
        "ORD-1002",
        "Tu pedido ORD-1002 (Licuadora) figura como Entregado. No tengo la fecha de entrega en "
        "mis registros.", ["pedidos"]),
    "t18-05-reembolso-500-en-el-umbral": Guion((_responder(
        "Con gusto te ayudo con tu tostadora. Para ver cómo seguir, ¿me cuentas la fecha de "
        "compra y el motivo de la devolución?", [], accion_sugerida="pedir_dato"),)),
    "t18-13-fuera-clima": _fuera_de_alcance(),
    "t18-14-fuera-receta": _fuera_de_alcance(),
    "t18-15-fuera-politica": _fuera_de_alcance(),
    "t18-16-fuera-matematicas": _fuera_de_alcance(),
    "t18-17-hecho-capital-y-devolucion": _simple(
        "Los envíos a la capital tardan 2-3 días hábiles. Si recibes tu pedido en la capital, "
        "tienes 30 días para devolverlo, siempre que esté sin usar y en su empaque original.",
        ["doc2", "doc3"]),
    "t18-18-reembolso-tras-pedido": _simple(
        "Si lo devuelves, el reembolso se procesa en 5-10 días hábiles después de recibir el "
        "producto devuelto, al mismo método de pago original.", ["doc4"]),
}


# Casos cuya fuente doc1 no sale con el FakeEmbedder (sin semántica real): la pregunta no menciona
# garantía ni categorías. Con el modelo real la recuperación semántica sí la trae; aquí se usa un
# retriever permisivo (todos los docs entran) solo para estos ids, sin tocar los demás casos.
IDS_RETRIEVAL_AMPLIO = frozenset({"vivo-08-devolucion-pasados-30-dias"})


def construir_retriever_amplio(docs: Path) -> Retriever:
    chunks = FileSystemDocumentSource(docs).cargar()
    return Retriever(
        chunks, IndiceLexico(chunks), FakeEmbedder(), InMemoryVectorStore(),
        top_k=5, umbral_bm25=0.0, umbral_semantico=-1.0,
    )
