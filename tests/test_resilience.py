"""Pruebas de resiliencia (T11): fallo seguro = escalar, sin red ni sleeps."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from tiendahogar_agent.config import Settings
from tiendahogar_agent.dobles import FakeLLM
from tiendahogar_agent.excepciones import (
    ErrorLLM,
    ErrorLLMAutenticacion,
    ErrorLLMConexion,
    ErrorLLMLimiteTasa,
    ErrorLLMRespuestaInvalida,
    ErrorLLMServidor,
    ErrorLLMTimeout,
    ErrorRecuperacion,
)
from tiendahogar_agent.guardrail_compromisos import detectar_compromisos
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.guardrail_output import verificar_salida
from tiendahogar_agent.models import AgentResponse, LLMResponse
from tiendahogar_agent.resiliencia import (
    RespuestaFalloSeguro,
    llamar_llm_seguro,
    recuperar_seguro,
    respuesta_fallo_seguro,
    revisar_resultado_tool,
)

MSGS = [{"role": "user", "content": "hola"}]
PII = "correo ana@correo.com, tel 3001234567, tarjeta 4111 1111 1111 1111"


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("TIMEOUT_LLM_S", "TIMEOUT_TOOL_S", "MAX_REINTENTOS_LLM"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)


def _es_fallo_seguro(r, motivo=None):
    assert isinstance(r, RespuestaFalloSeguro)
    if motivo:
        assert r.motivo == motivo
    a = r.respuesta
    assert isinstance(a, AgentResponse)
    assert a.accion == "escalar" and a.canal == CANAL_ESCALAMIENTO
    assert a.fuentes == [] and CANAL_ESCALAMIENTO in a.respuesta
    assert not any(ch.isdigit() for ch in a.respuesta)


# ---------------------------------------------------------------- settings
def test_settings_resiliencia_por_defecto():
    s = Settings()
    assert s.timeout_llm_s > 0 and s.timeout_tool_s > 0
    assert 0 <= s.max_reintentos_llm <= 5


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("timeout_llm_s", 0), ("timeout_llm_s", -1), ("timeout_llm_s", 1000),
        ("timeout_tool_s", 0), ("timeout_tool_s", -5), ("timeout_tool_s", 1000),
        ("max_reintentos_llm", -1), ("max_reintentos_llm", 50),
    ],
)
def test_settings_rechaza_valores_invalidos(campo, valor):
    with pytest.raises(ValidationError):
        Settings(**{campo: valor})


def test_settings_acepta_limites():
    s = Settings(max_reintentos_llm=0, timeout_llm_s=1.5)
    assert s.max_reintentos_llm == 0 and s.timeout_llm_s == 1.5


# ---------------------------------------------------------------- respuesta
def test_respuesta_fallo_seguro_no_dispara_guardrails():
    r = respuesta_fallo_seguro("llm_error", trace_id="t1")
    _es_fallo_seguro(r, "llm_error")
    assert r.respuesta.trace_id == "t1"
    assert detectar_compromisos(r.respuesta.respuesta) == []
    v = verificar_salida(r.respuesta.respuesta, "escalar", [], [], None, "hola")
    assert v.ok, v.reglas_fallidas


def test_jerarquia_excepciones():
    for e in (ErrorLLMTimeout, ErrorLLMConexion, ErrorLLMLimiteTasa,
              ErrorLLMAutenticacion, ErrorLLMServidor, ErrorLLMRespuestaInvalida):
        assert issubclass(e, ErrorLLM)
    assert not issubclass(ErrorRecuperacion, ErrorLLM)


# ---------------------------------------------------------------- LLM
def test_llm_ok_pasa_timeout_de_settings():
    llm = FakeLLM([FakeLLM.texto("hola")])
    r = llamar_llm_seguro(llm, MSGS, Settings(timeout_llm_s=7), tools=[{"name": "x"}])
    assert isinstance(r, LLMResponse) and r.texto == "hola"
    assert llm.llamadas[0]["timeout"] == 7
    assert llm.llamadas[0]["tools"] == [{"name": "x"}]


def test_llm_solo_tools_no_es_vacia():
    llm = FakeLLM([FakeLLM.llamada_tool("consultar_estado_pedido", {"order_id": "ORD-1001"})])
    assert isinstance(llamar_llm_seguro(llm, MSGS, Settings()), LLMResponse)


@pytest.mark.parametrize(
    "exc",
    [ErrorLLM("x"), ErrorLLMTimeout("t"), ErrorLLMConexion("c"), ErrorLLMLimiteTasa("l"),
     ErrorLLMAutenticacion("a"), ErrorLLMServidor("s"), ErrorLLMRespuestaInvalida("r")],
)
def test_llm_error_escala(exc):
    llm = FakeLLM()
    llm.encolar_error(exc)
    r = llamar_llm_seguro(llm, MSGS, Settings(), trace_id="t")
    _es_fallo_seguro(r, "llm_error")
    assert len(llm.llamadas) == 1  # sin bucle de reintentos propio


@pytest.mark.parametrize("texto", [None, "", "   \n\t"])
def test_llm_respuesta_vacia_escala(texto):
    llm = FakeLLM([LLMResponse(texto=texto)])
    _es_fallo_seguro(llamar_llm_seguro(llm, MSGS, Settings()), "llm_respuesta_vacia")


def test_fake_llm_encolar_vacia():
    llm = FakeLLM()
    llm.encolar_vacia()
    _es_fallo_seguro(llamar_llm_seguro(llm, MSGS, Settings()), "llm_respuesta_vacia")


def test_error_inesperado_del_llm_se_propaga():
    llm = FakeLLM()
    llm.encolar_error(TypeError("bug"))
    with pytest.raises(TypeError):
        llamar_llm_seguro(llm, MSGS, Settings())


# ---------------------------------------------------------------- tool
def test_tool_error_interno_escala():
    r = revisar_resultado_tool(
        {"order_id": None, "error": "error_interno", "mensaje": "falló"}, trace_id="t"
    )
    _es_fallo_seguro(r, "tool_error_interno")


@pytest.mark.parametrize(
    "res",
    [
        {"order_id": "ORD-1001", "producto": "x", "estado": "enviado"},
        {"order_id": "ORD-9", "error": "no_encontrado", "mensaje": "m"},
        {"order_id": None, "error": "formato_invalido", "mensaje": "m"},
    ],
)
def test_tool_resultados_normales_no_escalan(res):
    assert revisar_resultado_tool(res) is None


# ---------------------------------------------------------------- retriever
class _Retr:
    def __init__(self, exc=None, valor=None):
        self.exc, self.valor = exc, valor if valor is not None else []

    def recuperar(self, consulta):
        if self.exc:
            raise self.exc
        return self.valor


@pytest.mark.parametrize(
    "exc", [ErrorRecuperacion("x"), ValueError("v"), KeyError("k"), IndexError("i")]
)
def test_retriever_fallo_escala(exc):
    _es_fallo_seguro(recuperar_seguro(_Retr(exc), "consulta", trace_id="t"), "retriever_error")


def test_retriever_ok_devuelve_lista():
    assert recuperar_seguro(_Retr(valor=[]), "q") == []


def test_retriever_error_inesperado_se_propaga():
    with pytest.raises(TypeError):
        recuperar_seguro(_Retr(TypeError("bug")), "q")


# ---------------------------------------------------------------- logging
def test_log_enmascara_pii_llm(caplog):
    llm = FakeLLM()
    llm.encolar_error(ErrorLLMServidor(f"falló con {PII}"))
    with caplog.at_level(logging.DEBUG):
        llamar_llm_seguro(llm, MSGS, Settings())
    log = caplog.text
    assert "ErrorLLMServidor" in log and "llm_error" in log
    for dato in ("ana@correo.com", "3001234567", "4111 1111 1111 1111"):
        assert dato not in log


def test_log_enmascara_pii_retriever_y_tool(caplog):
    with caplog.at_level(logging.DEBUG):
        recuperar_seguro(_Retr(ValueError(f"mal {PII}")), "q")
        revisar_resultado_tool({"error": "error_interno", "mensaje": PII})
    log = caplog.text
    assert "ValueError" in log and "retriever_error" in log and "tool_error_interno" in log
    for dato in ("ana@correo.com", "3001234567", "4111111111111111", "4111 1111 1111 1111"):
        assert dato not in log


def test_log_respuesta_vacia(caplog):
    with caplog.at_level(logging.DEBUG):
        llamar_llm_seguro(FakeLLM([LLMResponse(texto=" ")]), MSGS, Settings())
    assert "llm_respuesta_vacia" in caplog.text


# --- menores de seguridad ---------------------------------------------------

INVISIBLES = "\u200b\u200c\u200d\u2060\ufeff\u00a0\x00\x07 \u2028"


@pytest.mark.parametrize("resultado", [None, "texto", ["x"], 42])
def test_tool_resultado_no_dict_falla_cerrado(resultado, caplog):
    with caplog.at_level(logging.DEBUG):
        r = revisar_resultado_tool(resultado, trace_id="t")
    _es_fallo_seguro(r, "tool_error_interno")
    assert "tipo=" in caplog.text


def test_recuperar_seguro_registra_traceback_con_pii_enmascarada(caplog):
    def _lanzar():
        raise ValueError(f"mal {PII}")

    class _R:
        def recuperar(self, consulta):
            _lanzar()

    with caplog.at_level(logging.DEBUG):
        recuperar_seguro(_R(), "q", trace_id="t")
    log = caplog.text
    assert "Traceback" in log
    assert "_lanzar" in log
    assert "ana@correo.com" not in log
    assert "3001234567" not in log
    assert "4111 1111 1111 1111" not in log


def test_llm_texto_solo_invisibles_es_vacio():
    llm = FakeLLM([LLMResponse(texto=INVISIBLES)])
    r = llamar_llm_seguro(llm, MSGS, Settings())
    _es_fallo_seguro(r, "llm_respuesta_vacia")


def test_llamar_llm_seguro_propaga_max_tokens_opcional():
    llm = FakeLLM([FakeLLM.texto("a"), FakeLLM.texto("b")])
    llamar_llm_seguro(llm, MSGS, Settings(), max_tokens=33)
    llamar_llm_seguro(llm, MSGS, Settings())
    assert llm.llamadas[0]["max_tokens"] == 33
    assert llm.llamadas[1]["max_tokens"] is None


def test_llamar_llm_seguro_pasa_tool_choice_solo_si_se_pide():
    from tiendahogar_agent.resiliencia import llamar_llm_seguro

    llm = FakeLLM([FakeLLM.texto("a"), FakeLLM.texto("b")])
    llamar_llm_seguro(llm, [{"role": "user", "content": "x"}], Settings(), tool_choice="any")
    llamar_llm_seguro(llm, [{"role": "user", "content": "x"}], Settings())
    assert llm.llamadas[0]["tool_choice"] == "any"
    assert llm.llamadas[1]["tool_choice"] is None
