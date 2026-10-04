"""Pruebas de la CLI de chat (T16): sin red ni API key."""

from __future__ import annotations

import io
import logging
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tiendahogar_agent.cli import (
    DESPEDIDA,
    PROMPT,
    configurar_logging,
    construir_orquestador,
    formatear_respuesta,
    main,
)
from tiendahogar_agent.config import Settings
from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.models import AgentResponse
from tiendahogar_agent.orquestador import MAX_MENSAJES_HISTORIAL

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    for v in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "USAR_CLASIFICADOR_LLM", "MAX_ITERACIONES_LLM"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)  # sin .env del repo
    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(tmp_path / "logs"))  # los tests nunca escriben en el repo


class OrqFalso:
    """Orquestador falso: registra lo que recibe y responde de forma fija."""

    def __init__(self, respuestas=None):
        self.recibido = []
        self._respuestas = list(respuestas or [])
        self._n = 0

    def procesar(self, mensaje, historial=None):
        self.recibido.append((mensaje, historial))
        self._n += 1
        if self._respuestas:
            return self._respuestas.pop(0)
        return AgentResponse(respuesta=f"resp{self._n}", accion="responder", trace_id=f"t{self._n}")


class EntradaTTY(io.StringIO):
    def isatty(self):
        return True


class EntradaCtrlC:
    """Entrada que lanza KeyboardInterrupt tras las líneas dadas."""

    def __init__(self, lineas=()):
        self._lineas = list(lineas)

    def isatty(self):
        return False

    def readline(self):
        if self._lineas:
            return self._lineas.pop(0)
        raise KeyboardInterrupt


def _correr(texto, orq=None, argv=(), entrada=None):
    orq = orq or OrqFalso()
    salida, errores = io.StringIO(), io.StringIO()
    codigo = main(list(argv), entrada or io.StringIO(texto), salida, orq, errores)
    return codigo, salida.getvalue(), errores.getvalue(), orq


# ---------------------------------------------------------------- historial y formato
def test_multi_turno_acumula_historial():
    codigo, salida, _, orq = _correr("hola\nsegunda\ntercera\nsalir\n")
    assert codigo == 0
    assert [m for m, _ in orq.recibido] == ["hola", "segunda", "tercera"]
    assert orq.recibido[0][1] == []
    assert orq.recibido[1][1] == [
        {"role": "user", "content": "hola"}, {"role": "assistant", "content": "resp1"},
    ]
    assert [t["role"] for t in orq.recibido[2][1]] == ["user", "assistant", "user", "assistant"]
    assert orq.recibido[2][1][-1]["content"] == "resp2"
    assert "resp1" in salida and "resp3" in salida


def test_historial_es_copia_por_turno():
    _, _, _, orq = _correr("a\nb\n")
    assert len(orq.recibido[0][1]) == 0  # no se mutó después de entregarlo


def test_historial_se_acota():
    lineas = "".join(f"m{i}\n" for i in range(MAX_MENSAJES_HISTORIAL + 5))
    _, _, _, orq = _correr(lineas)
    ultimo = orq.recibido[-1][1]
    assert len(ultimo) <= MAX_MENSAJES_HISTORIAL and ultimo[0]["role"] == "user"


def test_formato_con_fuentes_y_canal():
    r = AgentResponse(
        respuesta="Hola", accion="escalar", fuentes=["doc4", "doc5"],
        canal=CANAL_ESCALAMIENTO, trace_id="abc-123",
    )
    texto = formatear_respuesta(r)
    assert texto.startswith("Agente: Hola")
    assert "acción: escalar" in texto and "fuentes: doc4, doc5" in texto
    assert f"canal: {CANAL_ESCALAMIENTO}" in texto and "trace_id: abc-123" in texto


def test_formato_sin_fuentes_ni_canal():
    texto = formatear_respuesta(AgentResponse(respuesta="x", accion="responder", trace_id="t"))
    assert "fuentes: ninguna" in texto and "canal" not in texto


# ---------------------------------------------------------------- salidas
@pytest.mark.parametrize("palabra", ["salir", "exit", "SALIR", "  Exit  "])
def test_salir_por_palabra(palabra):
    codigo, _, _, orq = _correr(f"hola\n{palabra}\nno debe procesarse\n")
    assert codigo == 0 and [m for m, _ in orq.recibido] == ["hola"]


def test_eof_termina_sin_error():
    codigo, _, errores, orq = _correr("hola\n")
    assert codigo == 0 and len(orq.recibido) == 1 and errores == ""


def test_ctrl_c_termina_sin_traza():
    orq = OrqFalso()
    salida, errores = io.StringIO(), io.StringIO()
    codigo = main([], EntradaCtrlC(["hola\n"]), salida, orq, errores)
    assert codigo == 0 and len(orq.recibido) == 1
    assert "Traceback" not in salida.getvalue() + errores.getvalue()


def test_ctrl_c_en_una_vez():
    class Orq:
        def procesar(self, m, h=None):
            raise KeyboardInterrupt

    assert main(["--una-vez", "hola"], io.StringIO(), io.StringIO(), Orq(), io.StringIO()) == 0


def test_interactivo_muestra_prompt_y_despedida():
    entrada = EntradaTTY("hola\nsalir\n")
    salida = io.StringIO()
    main([], entrada, salida, OrqFalso(), io.StringIO())
    assert PROMPT in salida.getvalue() and DESPEDIDA in salida.getvalue()


def test_no_interactivo_sin_prompt():
    _, salida, _, _ = _correr("hola\n")
    assert PROMPT not in salida


def test_una_vez_no_lee_entrada():
    codigo, salida, _, orq = _correr("ignorado\n", argv=["--una-vez", "¿hola?"])
    assert codigo == 0 and [m for m, _ in orq.recibido] == ["¿hola?"]
    assert "resp1" in salida


def test_linea_vacia_no_rompe_ni_entra_al_historial():
    _, _, _, orq = _correr("hola\n\n   \nfin\n")
    assert [m for m, _ in orq.recibido] == ["hola", "", "   ", "fin"]
    assert [t["content"] for t in orq.recibido[3][1] if t["role"] == "user"] == ["hola"]


# ---------------------------------------------------------------- configuración
def test_error_de_configuracion_sin_traza_ni_secretos(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    salida, errores = io.StringIO(), io.StringIO()
    codigo = main(["--una-vez", "hola"], io.StringIO(), salida, None, errores)
    assert codigo == 2
    assert "anthropic_api_key" in errores.getvalue()
    assert "Traceback" not in errores.getvalue() + salida.getvalue()
    assert salida.getvalue() == ""


def test_settings_invalido_no_filtra_valores(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "sk-ant-secreto-123")
    errores = io.StringIO()
    codigo = main([], io.StringIO(), io.StringIO(), None, errores)
    assert codigo == 2 and "secreto" not in errores.getvalue()
    assert "llm_provider" in errores.getvalue()


# ---------------------------------------------------------------- end to end (fake, sin red)
@pytest.fixture
def sin_red(monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("acceso a red prohibido en tests")

    monkeypatch.setattr(socket.socket, "connect", prohibido)
    monkeypatch.setattr(socket, "create_connection", prohibido)
    monkeypatch.setattr(socket, "getaddrinfo", prohibido)


def _e2e(*mensajes):
    orq = construir_orquestador(Settings(llm_provider="fake"))
    salida, errores = io.StringIO(), io.StringIO()
    codigo = main([], io.StringIO("\n".join(mensajes) + "\n"), salida, orq, errores)
    assert codigo == 0 and errores.getvalue() == ""
    bloques = re.split(r"(?m)^(?=Agente: )", salida.getvalue())
    return [b for b in bloques if b.strip()]


def test_e2e_politica_con_fuente_real(sin_red):
    (b,) = _e2e("¿Cuánto dura la garantía de una lavadora?")
    assert "acción: responder" in b and "fuentes: doc1" in b
    assert "12 meses" in b and "trace_id:" in b


def test_e2e_pedido_existente(sin_red):
    (b,) = _e2e("¿Cómo va mi pedido ORD-1001?")
    assert "ORD-1001" in b and "Refrigeradora" in b and "En tránsito" in b
    assert "acción: responder" in b and "fuentes: pedidos" in b


def test_e2e_pedido_inexistente(sin_red):
    (b,) = _e2e("Consulta el pedido ORD-9999")
    assert "No encontré un pedido" in b and "ORD-9999" in b
    assert "acción: pedir_dato" in b


def test_e2e_escalamiento_por_reglas(sin_red):
    (b,) = _e2e("El vendedor fue grosero conmigo")
    assert "acción: escalar" in b and f"canal: {CANAL_ESCALAMIENTO}" in b


@pytest.mark.parametrize("mensaje", ["¿Venden pizza congelada de pepperoni?", "¿Quién ganó el mundial?", "Hola"])
def test_e2e_fuera_de_alcance_responde_amable_sin_escalar(sin_red, mensaje):
    # Sin documentos relevantes ni pedido: el demo usa la plantilla amable (responder, sin fuentes).
    (b,) = _e2e(mensaje)
    assert "acción: responder" in b and "fuentes: ninguna" in b
    assert "garantía" in b and "[canal:" not in b and "problema técnico" not in b


def test_e2e_multi_turno_y_vacio(sin_red):
    bloques = _e2e("ORD-1002", "", "¿Cuánto dura la garantía de una licuadora?")
    assert len(bloques) == 3
    assert "Licuadora" in bloques[0] and "acción: pedir_dato" in bloques[1]
    assert "6 meses" in bloques[2]


def test_construir_orquestador_fake_no_usa_embeddings_reales(sin_red):
    # con el bloqueo de red, cargar un modelo real fallaría; debe ser léxico
    orq = construir_orquestador(Settings(llm_provider="fake"))
    assert type(orq._llm).__name__ == "LLMDemo"


def test_modulo_ejecutable_con_python_m(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "tiendahogar_agent.cli", "--una-vez", "ORD-1003"],
        capture_output=True, text=True, encoding="utf-8", cwd=RAIZ,
        env={"PYTHONPATH": str(RAIZ / "src"), "LLM_PROVIDER": "fake", "PATH": "",
             "TIENDAHOGAR_LOGS_DIR": str(tmp_path / "logs"),
             "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
        timeout=120, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "Lavadora" in r.stdout and "Procesando" in r.stdout
    assert "Traceback" not in r.stderr


# ---------------------------------------------------------------- logging del CLI
class OrqLogueador(OrqFalso):
    """Emite un WARNING (con traza) desde un logger del paquete durante la consulta."""

    def procesar(self, mensaje, historial=None):
        log = logging.getLogger("tiendahogar_agent.prueba_cli")
        log.warning("aviso-interno-visible-solo-en-archivo")
        try:
            raise RuntimeError("fallo-simulado")
        except RuntimeError:
            log.exception("error con traza")
        log.debug("detalle-debug")
        return super().procesar(mensaje, historial)


def _archivo_log(tmp_path):
    ruta = tmp_path / "logs" / "tiendahogar.log"
    return ruta.read_text(encoding="utf-8") if ruta.exists() else ""


def test_warning_no_sale_a_consola_sin_debug_y_va_al_archivo(tmp_path, capsys):
    codigo, salida, errores, _ = _correr("hola\n", OrqLogueador())
    assert codigo == 0
    capt = capsys.readouterr()
    for flujo in (salida, errores, capt.out, capt.err):
        assert "aviso-interno" not in flujo
        assert "Traceback" not in flujo and "fallo-simulado" not in flujo
    assert "Agente: resp1" in salida
    log = _archivo_log(tmp_path)
    assert "aviso-interno-visible-solo-en-archivo" in log
    assert "Traceback" in log and "fallo-simulado" in log
    assert "detalle-debug" not in log


def test_debug_muestra_en_stderr_y_baja_a_debug(tmp_path):
    codigo, salida, errores, _ = _correr("hola\n", OrqLogueador(), argv=["--debug"])
    assert codigo == 0
    assert "aviso-interno-visible-solo-en-archivo" in errores
    assert "detalle-debug" in errores
    assert "aviso-interno" not in salida
    assert "detalle-debug" in _archivo_log(tmp_path)


def test_handlers_no_se_acumulan_y_se_restauran(tmp_path):
    raiz = logging.getLogger()
    antes = list(raiz.handlers), raiz.level
    for _ in range(3):
        _correr("hola\n", OrqLogueador())
    assert list(raiz.handlers) == antes[0] and raiz.level == antes[1]
    # y la configuración directa es idempotente
    for _ in range(3):
        cerrar = configurar_logging(False, tmp_path / "logs", io.StringIO())
    propios = [h for h in raiz.handlers if getattr(h, "_tiendahogar_cli", False)]
    assert len(propios) == 1
    cerrar()
    assert list(raiz.handlers) == antes[0]


def test_directorio_no_escribible_no_cae(tmp_path, monkeypatch):
    bloqueo = tmp_path / "es_un_archivo"
    bloqueo.write_text("no soy un directorio", encoding="utf-8")
    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(bloqueo / "logs"))
    codigo, salida, errores, _ = _correr("hola\n", OrqLogueador())
    assert codigo == 0
    assert "Agente: resp1" in salida
    assert "aviso-interno" not in errores and "aviso-interno" not in salida


def test_directorio_no_escribible_con_debug_sigue_mostrando_en_stderr(tmp_path, monkeypatch):
    bloqueo = tmp_path / "es_un_archivo"
    bloqueo.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TIENDAHOGAR_LOGS_DIR", str(bloqueo / "logs"))
    codigo, _, errores, _ = _correr("hola\n", OrqLogueador(), argv=["--debug"])
    assert codigo == 0 and "aviso-interno" in errores


def test_error_de_configuracion_sigue_visible_sin_debug(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    salida, errores = io.StringIO(), io.StringIO()
    assert main([], io.StringIO(""), salida, None, errores) == 2
    assert "Traceback" not in errores.getvalue() and errores.getvalue().strip()
