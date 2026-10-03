"""Pruebas del formato de dominio de mensajes con tools y su validación (sin SDKs)."""

from __future__ import annotations

import pytest

from tiendahogar_agent.dobles import FakeLLM
from tiendahogar_agent.excepciones import ErrorHistorialMensajes, ErrorLLM
from tiendahogar_agent.mensajes import (
    mensaje_asistente,
    mensaje_desde_respuesta,
    mensaje_resultado_tool,
    validar_historial,
)
from tiendahogar_agent.models import LlamadaTool, LLMResponse

LL1 = LlamadaTool(id="c1", nombre="consultar_estado_pedido", argumentos={"order_id": "A1"})
LL2 = LlamadaTool(id="c2", nombre="consultar_estado_pedido", argumentos={"order_id": "Ñandú"})
USUARIO = {"role": "user", "content": "hola"}


def test_mensaje_asistente_con_llamadas():
    m = mensaje_asistente("Consulto", [LL1, LL2])
    assert m == {
        "role": "assistant",
        "content": "Consulto",
        "tool_calls": [
            {"id": "c1", "name": "consultar_estado_pedido", "arguments": {"order_id": "A1"}},
            {"id": "c2", "name": "consultar_estado_pedido", "arguments": {"order_id": "Ñandú"}},
        ],
    }


def test_mensaje_asistente_sin_llamadas_ni_clave():
    assert mensaje_asistente("hola", []) == {"role": "assistant", "content": "hola"}
    assert "tool_calls" not in mensaje_asistente(None, [])


def test_mensaje_asistente_copia_argumentos():
    m = mensaje_asistente(None, [LL1])
    m["tool_calls"][0]["arguments"]["x"] = 1
    assert "x" not in LL1.argumentos


def test_mensaje_resultado_tool():
    assert mensaje_resultado_tool("c1", '{"a": 1}') == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"a": 1}',
        "es_error": False,
    }
    assert mensaje_resultado_tool("c1", "fallo", es_error=True)["es_error"] is True


def test_mensaje_desde_respuesta():
    r = LLMResponse(texto=None, llamadas_tools=[LL1])
    assert mensaje_desde_respuesta(r) == mensaje_asistente(None, [LL1])
    assert mensaje_desde_respuesta(LLMResponse(texto="ok")) == {
        "role": "assistant",
        "content": "ok",
    }


def _historial_valido():
    return [
        USUARIO,
        mensaje_asistente(None, [LL1, LL2]),
        mensaje_resultado_tool("c2", "{}"),
        mensaje_resultado_tool("c1", "{}", es_error=True),
    ]


def test_historial_valido():
    validar_historial(_historial_valido())
    validar_historial([USUARIO])
    validar_historial([])


@pytest.mark.parametrize(
    "historial",
    [
        [USUARIO, mensaje_resultado_tool("c1", "{}")],  # sin llamada previa
        [USUARIO, mensaje_asistente(None, [LL1]), mensaje_resultado_tool("", "{}")],  # id vacío
        [USUARIO, mensaje_asistente(None, [LL1]), mensaje_resultado_tool("zz", "{}")],  # id ajeno
        [USUARIO, mensaje_asistente(None, [LL1, LL2]), mensaje_resultado_tool("c1", "{}")],
        [USUARIO, mensaje_asistente(None, [LL1]), USUARIO],  # falta el resultado
        [
            USUARIO,
            mensaje_asistente(None, [LL1]),
            mensaje_resultado_tool("c1", "{}"),
            mensaje_resultado_tool("c1", "{}"),  # repetido
        ],
        [mensaje_asistente(None, [LL1, LL1])],  # ids repetidos
        [mensaje_asistente(None, [LlamadaTool(id="", nombre="t")])],  # id de llamada vacío
        [USUARIO, mensaje_asistente(None, [])],  # asistente vacío sin tools
        [USUARIO, mensaje_asistente("  ", [])],
        [USUARIO, mensaje_asistente("", []), USUARIO],
    ],
)
def test_historial_invalido(historial):
    with pytest.raises(ErrorHistorialMensajes):
        validar_historial(historial)


def test_error_no_filtra_contenido_y_no_es_errorllm():
    secreto = "DATO-SENSIBLE-999"
    with pytest.raises(ErrorHistorialMensajes) as info:
        validar_historial([USUARIO, mensaje_resultado_tool("c1", secreto)])
    assert secreto not in str(info.value)
    assert not isinstance(info.value, ErrorLLM)


def test_fakellm_registra_y_valida():
    llm = FakeLLM([FakeLLM.texto("ok"), FakeLLM.texto("ok")])
    historial = _historial_valido()
    llm.completar(historial)
    assert llm.llamadas[0]["mensajes"] == historial
    with pytest.raises(ErrorHistorialMensajes):
        llm.completar(historial[:-1])  # falta un resultado
    assert len(llm.llamadas) == 1  # el historial inválido no se registra ni consume respuesta
    llm.completar(historial)  # la cola sigue intacta
