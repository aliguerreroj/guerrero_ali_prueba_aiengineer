"""Pruebas de los adaptadores LLM (Anthropic y Azure OpenAI) con SDKs simulados: sin red ni API key."""

from __future__ import annotations

import os
from types import SimpleNamespace as NS

import anthropic
import httpx
import openai
import pytest

from tiendahogar_agent.adaptadores.anthropic_llm import AnthropicLLM
from tiendahogar_agent.adaptadores.azure_openai_llm import AzureOpenAILLM
from tiendahogar_agent.adaptadores.fabrica_llm import crear_llm
from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeLLM
from tiendahogar_agent.excepciones import (
    ErrorHistorialMensajes,
    ErrorLLM,
    ErrorLLMAutenticacion,
    ErrorLLMConexion,
    ErrorLLMLimiteTasa,
    ErrorLLMRespuestaInvalida,
    ErrorLLMServidor,
    ErrorLLMTimeout,
)
from tiendahogar_agent.mensajes import (
    mensaje_asistente,
    mensaje_desde_respuesta,
    mensaje_resultado_tool,
)
from tiendahogar_agent.models import LlamadaTool
from tiendahogar_agent.puertos import LLMClient
from tiendahogar_agent.resiliencia import RespuestaFalloSeguro, llamar_llm_seguro

CLAVE = "clave-secreta-de-prueba-123"
MENSAJES = [
    {"role": "system", "content": "Eres un agente."},
    {"role": "user", "content": "hola"},
]
TOOL = {
    "name": "consultar_estado_pedido",
    "description": "Consulta un pedido",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch, tmp_path):
    for v in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)  # evita leer un .env real


def _settings(**kw) -> Settings:
    return Settings(**kw)


class Registro:
    """Cliente falso: guarda los parámetros y devuelve (o lanza) lo configurado."""

    def __init__(self, resultado):
        self.resultado = resultado
        self.llamadas: list[dict] = []

    def create(self, **kw):
        self.llamadas.append(kw)
        if isinstance(self.resultado, BaseException):
            raise self.resultado
        return self.resultado


def _cliente_anthropic(resultado) -> tuple[NS, Registro]:
    reg = Registro(resultado)
    return NS(messages=reg), reg


def _cliente_openai(resultado) -> tuple[NS, Registro]:
    reg = Registro(resultado)
    return NS(chat=NS(completions=reg)), reg


def _resp_anthropic(bloques, entrada=11, salida=7):
    return NS(content=bloques, usage=NS(input_tokens=entrada, output_tokens=salida))


def _resp_openai(contenido=None, tool_calls=None, entrada=11, salida=7):
    mensaje = NS(content=contenido, tool_calls=tool_calls)
    return NS(
        choices=[NS(message=mensaje)],
        usage=NS(prompt_tokens=entrada, completion_tokens=salida),
    )


def _httpx(status=None):
    req = httpx.Request("POST", "https://api.ejemplo.test/v1")
    return req, (httpx.Response(status, request=req) if status else None)


def _errores(sdk):
    """Errores reales del SDK -> excepción de dominio esperada."""
    req, _ = _httpx()

    def estado(cls, codigo):
        _, resp = _httpx(codigo)
        return cls(f"fallo {CLAVE}", response=resp, body=None)

    return [
        (sdk.APITimeoutError(request=req), ErrorLLMTimeout),
        (sdk.APIConnectionError(request=req), ErrorLLMConexion),
        (estado(sdk.RateLimitError, 429), ErrorLLMLimiteTasa),
        (estado(sdk.AuthenticationError, 401), ErrorLLMAutenticacion),
        (estado(sdk.PermissionDeniedError, 403), ErrorLLMAutenticacion),
        (estado(sdk.InternalServerError, 500), ErrorLLMServidor),
        (estado(sdk.InternalServerError, 503), ErrorLLMServidor),
        (estado(sdk.BadRequestError, 400), ErrorLLM),
        (sdk.APIError(f"fallo {CLAVE}", request=req, body=None), ErrorLLM),
    ]


# ---------- Anthropic ----------


def test_anthropic_cumple_puerto():
    cliente, _ = _cliente_anthropic(None)
    assert isinstance(AnthropicLLM(_settings(), cliente=cliente), LLMClient)


def test_anthropic_traduce_tools_system_y_parametros():
    cliente, reg = _cliente_anthropic(_resp_anthropic([NS(type="text", text="ok")]))
    llm = AnthropicLLM(_settings(llm_max_tokens=64), cliente=cliente)
    llm.completar(MENSAJES, tools=[TOOL], timeout=4.5)
    p = reg.llamadas[0]
    assert p["system"] == "Eres un agente."
    assert p["messages"] == [{"role": "user", "content": "hola"}]
    assert p["tools"] == [
        {
            "name": "consultar_estado_pedido",
            "description": "Consulta un pedido",
            "input_schema": TOOL["parameters"],
        }
    ]
    assert p["model"] == _settings().llm_model
    assert p["max_tokens"] == 64
    assert p["timeout"] == 4.5


def test_anthropic_sin_system_tools_ni_timeout():
    cliente, reg = _cliente_anthropic(_resp_anthropic([NS(type="text", text="ok")]))
    AnthropicLLM(_settings(), cliente=cliente).completar([{"role": "user", "content": "x"}])
    p = reg.llamadas[0]
    assert "system" not in p and "tools" not in p and "timeout" not in p


def test_anthropic_texto_y_tool_use_con_tokens():
    bloques = [
        NS(type="text", text="Voy a "),
        NS(type="text", text="consultar."),
        NS(type="tool_use", id="tu_1", name="consultar_estado_pedido", input={"order_id": "A1"}),
    ]
    cliente, _ = _cliente_anthropic(_resp_anthropic(bloques, entrada=30, salida=9))
    r = AnthropicLLM(_settings(), cliente=cliente).completar(MENSAJES)
    assert r.texto == "Voy a consultar."
    assert len(r.llamadas_tools) == 1
    lt = r.llamadas_tools[0]
    assert (lt.id, lt.nombre, lt.argumentos) == (
        "tu_1",
        "consultar_estado_pedido",
        {"order_id": "A1"},
    )
    assert (r.uso.entrada, r.uso.salida) == (30, 9)


def test_anthropic_pasa_reintentos_y_timeout_al_constructor(monkeypatch):
    capturado = {}

    def falso(**kw):
        capturado.update(kw)
        return NS(messages=Registro(None))

    monkeypatch.setattr(anthropic, "Anthropic", falso)
    AnthropicLLM(_settings(anthropic_api_key=CLAVE, max_reintentos_llm=4, timeout_llm_s=12))
    assert capturado["max_retries"] == 4
    assert capturado["timeout"] == 12
    assert capturado["api_key"] == CLAVE


@pytest.mark.parametrize(("error", "esperado"), _errores(anthropic))
def test_anthropic_traduce_errores(error, esperado):
    cliente, _ = _cliente_anthropic(error)
    with pytest.raises(esperado) as info:
        AnthropicLLM(_settings(), cliente=cliente).completar(MENSAJES)
    assert type(info.value) is esperado
    assert CLAVE not in str(info.value)


@pytest.mark.parametrize(
    "mala",
    [
        NS(content=None, usage=NS(input_tokens=1, output_tokens=1)),
        NS(content=[NS(type="text", text="x")], usage=None),
        NS(
            content=[NS(type="tool_use", id="1", name="t", input="no-es-dict")],
            usage=NS(input_tokens=1, output_tokens=1),
        ),
    ],
)
def test_anthropic_respuesta_mal_formada(mala):
    cliente, _ = _cliente_anthropic(mala)
    with pytest.raises(ErrorLLMRespuestaInvalida):
        AnthropicLLM(_settings(), cliente=cliente).completar(MENSAJES)


def test_anthropic_compatible_con_resiliencia():
    cliente, _ = _cliente_anthropic(_errores(anthropic)[0][0])
    llm = AnthropicLLM(_settings(), cliente=cliente)
    resultado = llamar_llm_seguro(llm, MENSAJES, _settings(), trace_id="trace-1")
    assert isinstance(resultado, RespuestaFalloSeguro)  # el ErrorLLM se captura, no se propaga


# ---------- Azure OpenAI ----------


def _settings_azure(**kw) -> Settings:
    return _settings(
        llm_provider="azure",
        azure_openai_api_key=CLAVE,
        azure_openai_endpoint="https://recurso.openai.azure.example",
        azure_openai_deployment="mi-deployment",
        **kw,
    )


def test_azure_cumple_puerto():
    cliente, _ = _cliente_openai(None)
    assert isinstance(AzureOpenAILLM(_settings_azure(), cliente=cliente), LLMClient)


def test_azure_traduce_tools_y_parametros():
    cliente, reg = _cliente_openai(_resp_openai("ok"))
    llm = AzureOpenAILLM(_settings_azure(llm_max_tokens=64), cliente=cliente)
    llm.completar(MENSAJES, tools=[TOOL], timeout=3.0)
    p = reg.llamadas[0]
    assert p["model"] == "mi-deployment"
    assert p["messages"] == MENSAJES
    assert p["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "consultar_estado_pedido",
                "description": "Consulta un pedido",
                "parameters": TOOL["parameters"],
            },
        }
    ]
    assert p["max_completion_tokens"] == 64
    assert p["timeout"] == 3.0


def test_azure_texto_tool_calls_y_tokens():
    llamadas = [
        NS(
            id="call_1",
            function=NS(name="consultar_estado_pedido", arguments='{"order_id": "A1"}'),
        )
    ]
    cliente, _ = _cliente_openai(_resp_openai("Consulto.", llamadas, entrada=20, salida=5))
    r = AzureOpenAILLM(_settings_azure(), cliente=cliente).completar(MENSAJES)
    assert r.texto == "Consulto."
    lt = r.llamadas_tools[0]
    assert (lt.id, lt.nombre, lt.argumentos) == (
        "call_1",
        "consultar_estado_pedido",
        {"order_id": "A1"},
    )
    assert (r.uso.entrada, r.uso.salida) == (20, 5)


@pytest.mark.parametrize("argumentos", ["{no es json", "[1, 2]"])
def test_azure_argumentos_invalidos(argumentos):
    llamadas = [NS(id="c", function=NS(name="t", arguments=argumentos))]
    cliente, _ = _cliente_openai(_resp_openai(None, llamadas))
    with pytest.raises(ErrorLLMRespuestaInvalida):
        AzureOpenAILLM(_settings_azure(), cliente=cliente).completar(MENSAJES)


def test_azure_respuesta_sin_choices():
    cliente, _ = _cliente_openai(NS(choices=[], usage=NS(prompt_tokens=1, completion_tokens=1)))
    with pytest.raises(ErrorLLMRespuestaInvalida):
        AzureOpenAILLM(_settings_azure(), cliente=cliente).completar(MENSAJES)


def test_azure_pasa_configuracion_al_constructor(monkeypatch):
    capturado = {}

    def falso(**kw):
        capturado.update(kw)
        return NS(chat=NS(completions=Registro(None)))

    monkeypatch.setattr(openai, "AzureOpenAI", falso)
    AzureOpenAILLM(_settings_azure(max_reintentos_llm=3, timeout_llm_s=9))
    assert capturado["max_retries"] == 3
    assert capturado["timeout"] == 9
    assert capturado["azure_endpoint"] == "https://recurso.openai.azure.example"
    assert capturado["api_key"] == CLAVE


@pytest.mark.parametrize(("error", "esperado"), _errores(openai))
def test_azure_traduce_errores(error, esperado):
    cliente, _ = _cliente_openai(error)
    with pytest.raises(esperado) as info:
        AzureOpenAILLM(_settings_azure(), cliente=cliente).completar(MENSAJES)
    assert type(info.value) is esperado
    assert CLAVE not in str(info.value)


# ---------- Fábrica ----------


def test_fabrica_fake_por_defecto():
    assert isinstance(crear_llm(_settings()), FakeLLM)


def test_fabrica_anthropic(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: NS(messages=Registro(None)))
    llm = crear_llm(_settings(llm_provider="anthropic", anthropic_api_key=CLAVE))
    assert isinstance(llm, AnthropicLLM)


def test_fabrica_azure(monkeypatch):
    monkeypatch.setattr(openai, "AzureOpenAI", lambda **kw: NS(chat=NS(completions=Registro(None))))
    assert isinstance(crear_llm(_settings_azure()), AzureOpenAILLM)


def test_fabrica_anthropic_sin_clave():
    with pytest.raises(ValueError, match="anthropic_api_key"):
        crear_llm(_settings(llm_provider="anthropic"))


def test_fabrica_azure_incompleto_no_muestra_secretos():
    s = _settings(llm_provider="azure", azure_openai_api_key=CLAVE)
    with pytest.raises(ValueError) as info:
        crear_llm(s)
    mensaje = str(info.value)
    assert "azure_openai_endpoint" in mensaje and "azure_openai_deployment" in mensaje
    assert CLAVE not in mensaje


def test_settings_acepta_provider_azure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    assert Settings().llm_provider == "azure"


# ---------- Integración (UNA llamada real mínima; se omite sin API key) ----------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requiere ANTHROPIC_API_KEY")
def test_integracion_anthropic_llamada_real_minima():
    llm = AnthropicLLM(
        _settings(anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], llm_max_tokens=16)
    )
    r = llm.completar([{"role": "user", "content": "Responde solo: ok"}])
    assert r.texto
    assert r.uso.entrada > 0


# ---------- Mensajes con tools: traducción e ida y vuelta ----------

LLAMADAS = [
    LlamadaTool(id="c1", nombre="consultar_estado_pedido", argumentos={"order_id": "Ñu-1"}),
    LlamadaTool(id="c2", nombre="consultar_estado_pedido", argumentos={"order_id": "A2"}),
]


def _historial(texto="Voy a consultar", llamadas=LLAMADAS, error=False):
    return [
        {"role": "system", "content": "Eres un agente."},
        {"role": "user", "content": "¿Dónde está mi pedido?"},
        mensaje_asistente(texto, llamadas),
        mensaje_resultado_tool("c1", '{"estado": "envío en camino"}'),
        mensaje_resultado_tool("c2", '{"error": "no existe"}', es_error=error),
    ]


def _anthropic_enviado(historial):
    cliente, reg = _cliente_anthropic(_resp_anthropic([NS(type="text", text="ok")]))
    AnthropicLLM(_settings(), cliente=cliente).completar(historial)
    return reg.llamadas[0]["messages"]


def _azure_enviado(historial):
    cliente, reg = _cliente_openai(_resp_openai("ok"))
    AzureOpenAILLM(_settings_azure(), cliente=cliente).completar(historial)
    return reg.llamadas[0]["messages"]


def test_anthropic_envia_tool_use_y_fusiona_tool_results():
    enviado = _anthropic_enviado(_historial(error=True))
    assert enviado == [
        {"role": "user", "content": "¿Dónde está mi pedido?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Voy a consultar"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "consultar_estado_pedido",
                    "input": {"order_id": "Ñu-1"},
                },
                {
                    "type": "tool_use",
                    "id": "c2",
                    "name": "consultar_estado_pedido",
                    "input": {"order_id": "A2"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": '{"estado": "envío en camino"}',
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "c2",
                    "content": '{"error": "no existe"}',
                    "is_error": True,
                },
            ],
        },
    ]


def test_anthropic_texto_none_omite_bloque_text_y_sin_error_sin_is_error():
    enviado = _anthropic_enviado(_historial(texto=None))
    assert [b["type"] for b in enviado[1]["content"]] == ["tool_use", "tool_use"]
    assert all("is_error" not in b for b in enviado[2]["content"])


def test_anthropic_sin_tools_el_historial_pasa_igual():
    historial = [
        {"role": "user", "content": "hola"},
        mensaje_asistente("Hola", []),
        {"role": "user", "content": "gracias"},
    ]
    assert _anthropic_enviado(historial) == historial


def test_azure_envia_tool_calls_y_tool_sin_es_error():
    enviado = _azure_enviado(_historial(error=True))
    assert enviado[0] == {"role": "system", "content": "Eres un agente."}
    assert enviado[2] == {
        "role": "assistant",
        "content": "Voy a consultar",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "consultar_estado_pedido",
                    "arguments": '{"order_id": "Ñu-1"}',  # ensure_ascii=False
                },
            },
            {
                "id": "c2",
                "type": "function",
                "function": {
                    "name": "consultar_estado_pedido",
                    "arguments": '{"order_id": "A2"}',
                },
            },
        ],
    }
    assert enviado[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"estado": "envío en camino"}',
    }
    assert enviado[4]["tool_call_id"] == "c2" and "es_error" not in enviado[4]


def test_azure_texto_none_y_sin_tools():
    enviado = _azure_enviado(_historial(texto=None))
    assert enviado[2]["content"] is None
    historial = [{"role": "user", "content": "hola"}, mensaje_asistente("Hola", [])]
    assert _azure_enviado(historial) == historial


@pytest.mark.parametrize("enviar", [_anthropic_enviado, _azure_enviado])
@pytest.mark.parametrize(
    "malo",
    [
        [{"role": "user", "content": "x"}, mensaje_resultado_tool("c1", "{}")],
        [{"role": "user", "content": "x"}, mensaje_asistente(None, LLAMADAS)],
        [
            {"role": "user", "content": "x"},
            mensaje_asistente(None, LLAMADAS[:1]),
            mensaje_resultado_tool("", "{}"),
        ],
    ],
)
def test_adaptadores_rechazan_historial_mal_formado(enviar, malo):
    with pytest.raises(ErrorHistorialMensajes):
        enviar(malo)


def test_historial_invalido_no_llega_al_sdk():
    cliente, reg = _cliente_anthropic(None)
    with pytest.raises(ErrorHistorialMensajes):
        AnthropicLLM(_settings(), cliente=cliente).completar([mensaje_resultado_tool("c1", "{}")])
    assert reg.llamadas == []


def _comparable(llamadas):
    return [(ll.id, ll.nombre, ll.argumentos) for ll in llamadas]


def test_ida_y_vuelta_anthropic():
    bloques = [
        NS(type="text", text="Consulto"),
        NS(type="tool_use", id="c1", name="consultar_estado_pedido", input={"order_id": "Ñu-1"}),
        NS(type="tool_use", id="c2", name="consultar_estado_pedido", input={"order_id": "A2"}),
    ]
    respuesta = AnthropicLLM._interpretar(_resp_anthropic(bloques))
    historial = [{"role": "user", "content": "hola"}, mensaje_desde_respuesta(respuesta)]
    historial += [mensaje_resultado_tool(ll.id, "{}") for ll in respuesta.llamadas_tools]
    enviado = _anthropic_enviado(historial)
    usos = [b for b in enviado[1]["content"] if b["type"] == "tool_use"]
    assert [(b["id"], b["name"], b["input"]) for b in usos] == _comparable(respuesta.llamadas_tools)
    assert [b["tool_use_id"] for b in enviado[2]["content"]] == ["c1", "c2"]
    de_vuelta = AnthropicLLM._interpretar(_resp_anthropic([NS(**b) for b in enviado[1]["content"]]))
    assert de_vuelta.texto == respuesta.texto
    assert _comparable(de_vuelta.llamadas_tools) == _comparable(respuesta.llamadas_tools)


def test_ida_y_vuelta_azure():
    tool_calls = [
        NS(id="c1", function=NS(name="consultar_estado_pedido", arguments='{"order_id": "Ñu-1"}')),
        NS(id="c2", function=NS(name="consultar_estado_pedido", arguments='{"order_id": "A2"}')),
    ]
    respuesta = AzureOpenAILLM._interpretar(_resp_openai(None, tool_calls))
    assert respuesta.texto is None
    historial = [{"role": "user", "content": "hola"}, mensaje_desde_respuesta(respuesta)]
    historial += [mensaje_resultado_tool(ll.id, "{}") for ll in respuesta.llamadas_tools]
    enviado = _azure_enviado(historial)
    reenviadas = [
        NS(id=t["id"], function=NS(name=t["function"]["name"], arguments=t["function"]["arguments"]))
        for t in enviado[1]["tool_calls"]
    ]
    de_vuelta = AzureOpenAILLM._interpretar(_resp_openai(enviado[1]["content"], reenviadas))
    assert _comparable(de_vuelta.llamadas_tools) == _comparable(respuesta.llamadas_tools)
    assert [m["tool_call_id"] for m in enviado[2:]] == ["c1", "c2"]


def test_anthropic_max_tokens_por_llamada_sobrescribe():
    cliente, reg = _cliente_anthropic(_resp_anthropic([NS(type="text", text="ok")]))
    llm = AnthropicLLM(_settings(llm_max_tokens=64), cliente=cliente)
    llm.completar(MENSAJES, max_tokens=16)
    llm.completar(MENSAJES)
    assert reg.llamadas[0]["max_tokens"] == 16
    assert reg.llamadas[1]["max_tokens"] == 64


def test_azure_max_tokens_por_llamada_sobrescribe():
    cliente, reg = _cliente_openai(_resp_openai("ok"))
    llm = AzureOpenAILLM(_settings_azure(llm_max_tokens=64), cliente=cliente)
    llm.completar(MENSAJES, max_tokens=16)
    assert reg.llamadas[0]["max_completion_tokens"] == 16


def test_anthropic_omite_bloque_text_de_solo_espacios():
    enviado = _anthropic_enviado(_historial(texto="  \n "))
    assert [b["type"] for b in enviado[1]["content"]] == ["tool_use", "tool_use"]


def test_anthropic_conserva_texto_no_vacio_antes_de_tool_use():
    enviado = _anthropic_enviado(_historial(texto=" Voy a consultar "))
    assert enviado[1]["content"][0] == {"type": "text", "text": " Voy a consultar "}


@pytest.mark.parametrize("vacio", [None, "", "   ", "\n"])
def test_adaptadores_rechazan_asistente_vacio_sin_tools(vacio):
    historial = [
        {"role": "user", "content": "hola"},
        mensaje_asistente(vacio, []),
        {"role": "user", "content": "¿sigues ahí?"},
    ]
    with pytest.raises(ErrorHistorialMensajes):
        _anthropic_enviado(historial)
    with pytest.raises(ErrorHistorialMensajes):
        _azure_enviado(historial)
