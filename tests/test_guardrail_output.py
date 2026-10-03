"""Pruebas de la verificación de salida (T10): deterministas, sin red ni LLM."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from tiendahogar_agent.guardrail_input import CANAL_ESCALAMIENTO
from tiendahogar_agent.guardrail_output import (
    RESPUESTA_SEGURA,
    ResultadoVerificacion,
    verificar_salida,
)
from tiendahogar_agent.models import Chunk


def _chunk(texto: str, doc_id: str = "doc1") -> Chunk:
    return Chunk(texto=texto, doc_id=doc_id, metadatos={"posicion": 0})


def _ver(
    respuesta: str,
    contexto: str = "",
    *,
    accion: str = "responder",
    fuentes: list[str] | None = None,
    tool: dict | None = None,
    mensaje: str = "",
    chunks: list[Chunk] | None = None,
) -> ResultadoVerificacion:
    if chunks is None:
        chunks = [_chunk(contexto)] if contexto else []
    return verificar_salida(
        respuesta=respuesta,
        accion=accion,
        fuentes=fuentes if fuentes is not None else [],
        chunks=chunks,
        resultado_tool=tool,
        mensaje_usuario=mensaje,
    )


def _falla(res: ResultadoVerificacion, regla: str) -> bool:
    return (not res.ok) and regla in res.reglas_fallidas


# ------------------------------------------------------------------ cifras
class TestCifras:
    def test_sin_cifras_pasa(self):
        res = _ver("Puedes devolver el producto si llegó dañado.", "Devoluciones por daño.")
        assert res.ok
        assert res.reglas_fallidas == ()

    def test_cifra_en_contexto_pasa(self):
        assert _ver("Tienes 30 días para devolver.", "El plazo es de 30 días.").ok

    def test_cifra_ausente_falla(self):
        res = _ver("Tienes 45 días para devolver.", "El plazo es de 30 días.")
        assert _falla(res, "cifra_sin_sustento")
        assert any("45" in d for d in res.detalles)

    def test_chunks_vacios_con_cifras_falla(self):
        assert _falla(_ver("Tienes 30 días.", chunks=[]), "cifra_sin_sustento")

    def test_chunks_vacios_sin_cifras_pasa(self):
        assert _ver("Hola, ¿en qué te ayudo?", chunks=[]).ok

    def test_cifra_en_mensaje_usuario_pasa(self):
        assert _ver("Entiendo que pagaste 250 por el producto.", mensaje="Pagué 250 por él").ok

    def test_cifra_en_tool_pasa(self):
        tool = {"order_id": "ORD-1001", "producto": "Licuadora 600W", "estado": "enviado",
                "entrega_estimada": None}
        assert _ver("Tu licuadora de 600W va en camino.", tool=tool).ok

    def test_cifra_en_tool_numerica(self):
        assert _ver("El total es 99.5.", tool={"total": 99.5}).ok

    @pytest.mark.parametrize(
        ("resp", "ctx"),
        [
            ("Tienes doce meses.", "La garantía dura 12 meses."),
            ("Tienes 12 meses.", "La garantía dura doce meses."),
            ("Tienes treinta días.", "Plazo: 30 días."),
            ("Tienes 35 días.", "Plazo: treinta y cinco días."),
            ("Tienes treinta y cinco días.", "Plazo: 35 días."),
            ("Son quinientos pesos.", "Umbral: 500 pesos."),
            ("Son 500 pesos.", "Umbral: quinientos pesos."),
            ("Son ciento veinte pesos.", "Costo 120 pesos."),
            ("Son doscientas unidades.", "200 unidades."),
            ("Son mil pesos.", "Valor 1000 pesos."),
            ("Son dos mil pesos.", "Valor 2000 pesos."),
            ("Son 2 mil pesos.", "Valor 2000 pesos."),
            ("Son dos mil quinientos pesos.", "Valor 2500 pesos."),
            ("Son veintiún días.", "Plazo 21 días."),
            ("Son veintiuno.", "Plazo 21."),
            ("Son quince días.", "Plazo 15 días."),
            ("Son cero pesos.", "Costo 0 pesos."),
            ("Son DOCE días.", "Plazo 12 días."),
            ("Son un millón de pesos.", "Valor 1000000."),
            ("Son treinta y un días.", "Plazo 31 días."),
        ],
    )
    def test_numeros_en_letras_equivalen(self, resp, ctx):
        assert _ver(resp, ctx).ok

    @pytest.mark.parametrize(
        ("resp", "ctx"),
        [
            ("Tienes trece meses.", "La garantía dura 12 meses."),
            ("Tienes treinta y seis días.", "Plazo: 35 días."),
            ("Son quinientos uno pesos.", "Umbral: 500 pesos."),
            ("Son dos mil pesos.", "Valor 200 pesos."),
        ],
    )
    def test_numeros_en_letras_distintos_fallan(self, resp, ctx):
        assert _falla(_ver(resp, ctx), "cifra_sin_sustento")

    @pytest.mark.parametrize(
        ("resp", "ctx"),
        [
            ("Cuesta $1,200.", "Precio 1200 pesos."),
            ("Cuesta 1.200 pesos.", "Precio 1200 pesos."),
            ("Cuesta 1 200 pesos.", "Precio 1200 pesos."),
            ("Cuesta 1200 pesos.", "Precio $1,200."),
            ("Cuesta 1200 pesos.", "Precio 1.200,00 pesos."),
            ("Cuesta $1,200.50.", "Precio 1.200,50."),
            ("Descuento del 15%.", "Descuento de quince por ciento."),
            ("Descuento del quince por ciento.", "Descuento: 15 %."),
            ("Pagas 2,5 cuotas.", "Son 2.5 cuotas."),
            ("Pagas 5.0 pesos.", "Son 5 pesos."),
            ("Son 5 mil pesos.", "Valor $5,000."),
        ],
    )
    def test_formatos_equivalen(self, resp, ctx):
        assert _ver(resp, ctx).ok

    def test_formatos_distintos_fallan(self):
        assert _falla(_ver("Cuesta 1.300 pesos.", "Precio 1200 pesos."), "cifra_sin_sustento")
        assert _falla(_ver("Descuento del 25%.", "Descuento del 15%."), "cifra_sin_sustento")

    def test_rangos_no_se_aceptan_como_aproximacion(self):
        assert _falla(_ver("Llega en 7 días.", "Llega en 5-10 días."), "cifra_sin_sustento")
        assert _ver("Llega en 5 días.", "Llega en 5-10 días.").ok

    def test_rango_con_extremos_pasa(self):
        assert _ver("Llega entre 5 y 10 días.", "Llega en 5 a 10 días.").ok

    def test_decimal_no_es_su_parte_entera(self):
        assert _falla(_ver("Pagas 2 cuotas.", "Son 2.5 cuotas."), "cifra_sin_sustento")

    def test_porcentaje_con_decimal(self):
        assert _ver("Cobramos 2.5%.", "Cobro del 2.5%.").ok
        assert _falla(_ver("Cobramos 25%.", "Cobro del 2.5%."), "cifra_sin_sustento")

    # ---- uno / una / un
    def test_articulo_un_una_no_es_cifra(self):
        assert _ver("Es una buena opción, un gusto ayudarte.", "Sin cifras.").ok
        assert _ver("Te ayudo con uno de los canales.", "Sin cifras.").ok

    def test_un_dia_con_unidad_es_cifra(self):
        assert _falla(_ver("Llega en un día.", "Llega en 3 días."), "cifra_sin_sustento")
        assert _ver("Llega en un día.", "Llega en 1 día.").ok
        assert _ver("Llega en 1 día.", "Llega en un día.").ok

    def test_una_semana_no_equivale_a_7_dias(self):
        assert _falla(_ver("Llega en una semana.", "Llega en 7 días."), "cifra_sin_sustento")

    # ---- ordinales y palabras que no son cifras
    def test_ordinales_no_son_cifras(self):
        assert _ver("En segundo lugar, revisa tu correo; primero, tu factura.", "Sin cifras.").ok
        assert _ver("Es un tercer paso, no lo olvides.", "Sin cifras.").ok

    def test_no_marca_palabras_que_contienen_numeros(self):
        assert _ver("Cuéntanos tu situación, entonces vemos; la dosis y el once-mil.", "x").ok is False
        assert _ver("Cuéntanos tu situación, entonces vemos.", "Sin cifras.").ok
        assert _ver("Hay miles de opciones y cientos de tiendas.", "Sin cifras.").ok

    def test_y_conector_no_une_cifras_independientes(self):
        assert _ver("Tienes tres y cuatro opciones.", "3 opciones y 4 canales.").ok
        assert _falla(_ver("Tienes tres y cinco opciones.", "3 opciones y 4 canales."),
                      "cifra_sin_sustento")

    def test_lista_numerada_no_cuenta(self):
        resp = "Haz lo siguiente:\n1. Revisa tu correo.\n2) Escríbenos."
        assert _ver(resp, "Sin cifras.").ok

    # ---- ids y fechas
    def test_id_pedido_completo_en_tool_pasa(self):
        tool = {"order_id": "ORD-1001", "producto": "Horno", "estado": "entregado",
                "entrega_estimada": None}
        assert _ver("Tu pedido ORD-1001 ya fue entregado.", tool=tool).ok

    def test_id_pedido_en_mensaje_pasa(self):
        assert _ver("Reviso el pedido ORD-1001 ahora.", mensaje="Mi pedido es ord-1001").ok

    def test_id_pedido_espaciado_se_normaliza(self):
        assert _ver("Pedido ORD-1001.", mensaje="es el ORD - 1001").ok

    def test_id_pedido_inventado_falla(self):
        tool = {"order_id": "ORD-1001", "estado": "enviado"}
        assert _falla(_ver("Tu pedido ORD-1002 va en camino.", tool=tool), "cifra_sin_sustento")

    def test_id_no_se_parte_para_otras_cifras(self):
        tool = {"order_id": "ORD-1001", "estado": "enviado"}
        # el id legítimo no genera una «cifra 1001» extra que falle por sí sola
        res = _ver("ORD-1001 está enviado.", tool=tool)
        assert res.ok and res.detalles == ()

    def test_numero_suelto_de_id_en_contexto_se_acepta(self):
        tool = {"order_id": "ORD-1001", "estado": "enviado"}
        assert _ver("El pedido 1001 está enviado.", tool=tool).ok

    def test_id_en_respuesta_con_mensaje_numerico_no_basta(self):
        assert _falla(_ver("Pedido ORD-1002.", mensaje="mi pedido es el 1002"), "cifra_sin_sustento")

    def test_tool_con_error_no_encontrado(self):
        tool = {"order_id": "ORD-9999", "error": "no_encontrado",
                "mensaje": "No encontré un pedido con el número ORD-9999."}
        assert _ver("No encontré el pedido ORD-9999.", tool=tool).ok

    def test_fecha_iso_en_tool_pasa(self):
        tool = {"order_id": "ORD-1001", "entrega_estimada": "2026-10-10"}
        assert _ver("Llega el 2026-10-10.", tool=tool).ok

    def test_fecha_iso_inventada_falla(self):
        tool = {"order_id": "ORD-1001", "entrega_estimada": "2026-10-10"}
        assert _falla(_ver("Llega el 2026-10-11.", tool=tool), "cifra_sin_sustento")

    def test_dia_de_fecha_iso_del_contexto_se_acepta(self):
        tool = {"order_id": "ORD-1001", "entrega_estimada": "2026-10-10"}
        assert _ver("Llega el 10 de octubre.", tool=tool).ok

    def test_email_del_canal_no_cuenta(self):
        res = _ver(f"Escríbenos a {CANAL_ESCALAMIENTO}.", accion="escalar")
        assert res.ok


# ------------------------------------------------------------------ fuentes
class TestFuentes:
    def test_fuente_recuperada_pasa(self):
        res = _ver("Respuesta.", chunks=[_chunk("a", "doc1"), _chunk("b", "doc3")],
                   fuentes=["doc1", "doc3"])
        assert res.ok
        assert res.fuentes == ["doc1", "doc3"]

    def test_fuente_no_recuperada_falla(self):
        res = _ver("Respuesta.", chunks=[_chunk("a", "doc1")], fuentes=["doc1", "doc9"])
        assert _falla(res, "fuente_no_recuperada")
        assert any("doc9" in d for d in res.detalles)

    def test_fuente_con_chunks_vacios_falla(self):
        assert _falla(_ver("Respuesta.", chunks=[], fuentes=["doc1"]), "fuente_no_recuperada")

    def test_sin_fuentes_pasa(self):
        assert _ver("Respuesta.", fuentes=[]).ok

    def test_fuentes_invalidas_no_lanzan(self):
        res = verificar_salida("Hola.", "responder", None, [], None, "")  # type: ignore[arg-type]
        assert res.ok


# ------------------------------------------------------------ compromisos
class TestCompromisos:
    @pytest.mark.parametrize(
        "texto",
        [
            "Tu reembolso está aprobado.",
            "TU REEMBOLSO ESTA APROBADO",
            "Tu reembolso ya fue aprobado, tranquilo.",
            "Tu reembolso ha sido aprobado.",
            "Tu reembolso será aprobado mañana.",
            "Tu reembolso quedó autorizado.",
            "Su reembolso fue concedido.",
            "Tu devolución está aprobada.",
            "Aprobamos tu reembolso.",
            "Hemos aprobado tu reembolso.",
            "He aprobado tu reembolso.",
            "Ya aprobé tu reembolso.",
            "Te apruebo el reembolso completo.",
            "Autorizamos tu devolución.",
            "Aprobaremos el reembolso sin problema.",
            "Queda aprobado tu reembolso.",
            "Queda aprobada tu devolución.",
            "Se te aprobó el reembolso.",
            "Se aprobó tu reembolso.",
            "Aprobado tu reembolso.",
            "Ya está aprobado tu reembolso.",
            "El reembolso está aprobado.",
            "Reembolso aprobado.",
            "Listo, reembolso aprobado.",
            "Lo aprobamos.",
            "Ya lo aprobé.",
            "Excepcionalmente lo aprobamos.",
            "Estamos aprobando tu reembolso.",
            "Excepcionalmente haremos la devolución.",
            "Te devolvemos el dinero hoy mismo.",
            "Te reembolsamos el dinero mañana.",
            "Lo hemos aprobado.",
            "Hoy mismo se te autorizó la devolución.",
            "QUEDA APROBADO TU REEMBOLSO",
            "queda autorizada su devolucion",
            "Tu reintegro ya está confirmado.",
            "Te concedimos el reembolso.",
        ],
    )
    def test_aprobar_reembolso_falla(self, texto):
        assert _falla(_ver(texto), "compromiso_reembolso_aprobado")

    @pytest.mark.parametrize(
        "texto",
        [
            "El reembolso se aprueba tras revisar el producto.",
            "Un supervisor revisa y aprueba los reembolsos según la política.",
            "No puedo aprobar reembolsos; eso lo decide un supervisor.",
            "No aprobamos reembolsos sin revisar el producto.",
            "Nunca aprobamos un reembolso antes de la revisión.",
            "Para aprobar tu reembolso necesitamos revisar el producto.",
            "Los reembolsos aprobados se acreditan al mismo medio de pago.",
            "No te lo aprobamos sin revisar el producto.",
            "Un reembolso aprobado se acredita al mismo medio de pago.",
            "No podemos devolver el dinero sin revisar el producto.",
            "Si el producto es defectuoso, el supervisor aprueba la devolución.",
            "La devolución se evalúa según el estado del producto.",
        ],
    )
    def test_politica_de_reembolso_no_falla(self, texto):
        res = _ver(texto)
        assert res.ok, res.detalles

    @pytest.mark.parametrize(
        "texto",
        [
            "Te garantizo que te devolverán el dinero.",
            "Te garantizamos la devolución.",
            "TE GARANTIZO el reembolso.",
            "Te aseguro que te devolverán tu dinero.",
            "Te aseguramos que todo saldrá bien.",
            "Aseguro que recibirás tu reembolso.",
            "Te prometo que se resuelve hoy.",
            "Tu reembolso está garantizado.",
            "Es un resultado garantizado.",
            "Puedo garantizar que llegará.",
            "Estoy seguro de que te devolverán el dinero.",
            "Te lo garantizo.",
            "TE LO GARANTIZO",
            "Te lo aseguro.",
            "Tienes mi palabra.",
            "Descuida, te lo prometo.",
            "Te lo dejo garantizado.",
            "Tienes garantizado tu reembolso.",
            "Queda garantizado.",
            "Está garantizado tu reembolso.",
            "El resultado está garantizado.",
            "Te doy mi palabra.",
            "te garantizo que llega manana",
        ],
    )
    def test_garantizar_falla(self, texto):
        assert _falla(_ver(texto), "compromiso_garantia_resultado")

    @pytest.mark.parametrize(
        "texto",
        [
            "La garantía cubre defectos de fábrica.",
            "Tu producto tiene garantía de fábrica.",
            "No puedo garantizar el resultado de la revisión.",
            "No te garantizo nada hasta que lo revise un especialista.",
            "Puedes asegurar el paquete con cinta.",
            "Para asegurarte de que llegue bien, revisa la caja.",
            "El producto está garantizado por el fabricante.",
            "El producto está garantizado por 12 meses.",
            "La garantía del fabricante cubre defectos.",
            "No te lo puedo garantizar.",
            "No te lo garantizo.",
            "No podemos garantizar el resultado.",
        ],
    )
    def test_garantia_politica_no_falla(self, texto):
        res = _ver(texto, "12 meses de garantía")
        assert res.ok, res.detalles

    @pytest.mark.parametrize(
        "texto",
        [
            "Te hago una excepción.",
            "TE HAGO UNA EXCEPCION",
            "Haremos una excepción en tu caso.",
            "Podemos hacer una excepción para ti.",
            "Voy a hacer una excepción.",
            "Te concedemos una excepción.",
            "Aplicamos una excepción a tu devolución.",
            "Te damos una excepción especial.",
            "Excepcionalmente te hacemos una excepción.",
        ],
    )
    def test_excepcion_falla(self, texto):
        assert _falla(_ver(texto), "compromiso_excepcion")

    @pytest.mark.parametrize(
        "texto",
        [
            "No hacemos excepciones a la política.",
            "No puedo hacer una excepción, pero un supervisor puede revisar tu caso.",
            "Nunca hacemos excepciones con los plazos.",
            "Esta regla no tiene excepciones.",
        ],
    )
    def test_excepcion_negada_no_falla(self, texto):
        res = _ver(texto)
        assert res.ok, res.detalles


COMPROMISOS_ROUND3 = [
    # negación engañosa: solo la lista cerrada niega
    "No dudes que te lo garantizo.",
    "No te preocupes, te reembolsamos.",
    "No hay problema, te lo aprobamos.",
    "Sin duda te devolvemos el dinero.",
    "No te preocupes que te lo garantizo.",
    "No te preocupes hago una excepción.",
    # reembolso
    "Te reembolsamos el total.",
    "Te reembolsaremos.",
    "Te lo reembolsamos.",
    "Sin problema te lo reembolsamos.",
    "Lo reembolsamos.",
    "Ya te reembolsamos.",
    "Reembolsamos tu compra.",
    "Te devolveremos los 800 dólares.",
    "Te regresamos el dinero.",
    "Se te devolverá el dinero.",
    "Se te reembolsará.",
    "El dinero te será devuelto.",
    "Se te acreditará el reembolso.",
    "Tu reembolso es un hecho.",
    "Tu reembolso ya fue procesado.",
    "Tu reembolso será procesado.",
    "Procesamos tu reembolso.",
    "Aceptamos tu devolución.",
    "Tu devolución está aceptada.",
    "Te confirmo tu reembolso.",
    "Te confirmo que se aprobó.",
    "Tu caso queda aprobado.",
    "Ya está aprobado.",
    "Tu reembolso, está aprobado.",
    "Tu reembolso: aprobado.",
    "Recibirás tu reembolso sin falta.",
    "Tendrás tu dinero de vuelta.",
    "Seguro que te devuelven el dinero.",
    "Ya puedes contar con tu reembolso.",
    "Cuenta con tu reembolso.",
    "Es un hecho, te devolvemos.",
    "Dalo por hecho, te reembolsamos.",
    "Voy a devolverte el dinero.",
    "Podemos aprobarlo hoy.",
    "Vamos a aprobar tu reembolso.",
    "Ya te lo hemos aprobado.",
    "Estoy autorizando tu devolución.",
    "Todo listo, aprobado.",
    "Ya quedó listo tu reembolso.",
    "Sí, procedemos con el reembolso.",
    "Tranquilo, tu reembolso va a ser aprobado.",
    "No te preocupes, tu reembolso está aprobado.",
    "Una vez aprobado, te lo reembolsamos.",
    "No, te lo aprobamos.",
    "**Te reembolsamos** el total.",
    "TE   REEMBOLSAMOS EL TOTAL",
    "Te reem" + chr(0x200B) + "bolsamos el total y te lo garantizo.",
    # garantía / excepción
    "Garantizamos tu reembolso.",
    "Garantía asegurada.",
    "Excepcionalmente lo aceptamos.",
    "Excepcionalmente te devolvemos todo.",
    "Te saltamos la política.",
    "Por esta vez lo aceptamos.",
    "Nos saltamos el plazo por ti.",
    "De manera excepcional te lo concedemos.",
    "Te aseguro que llega mañana.",
    "Estamos seguros de que se resolverá.",
]


class TestCompromisosEstructural:
    @pytest.mark.parametrize("texto", COMPROMISOS_ROUND3)
    def test_compromiso_se_bloquea(self, texto):
        res = _ver(texto, "Plazo 800 dólares")
        assert not res.ok, texto
        assert any(r.startswith("compromiso_") for r in res.reglas_fallidas), (texto, res.reglas_fallidas)
        assert res.accion == "escalar"
        assert res.respuesta == RESPUESTA_SEGURA

    @pytest.mark.parametrize(
        "texto",
        [
            "Entiendo tu molestia, lamento el inconveniente.",
            "¿Podrías compartirme tu número de pedido?",
            "La garantía cubre defectos de fábrica durante 12 meses.",
            "Las devoluciones se aceptan dentro de 30 días desde la entrega.",
            "Aceptamos devoluciones dentro de 30 días.",
            "Aceptamos tarjetas de crédito y débito.",
            "Procesamos los pedidos en 24 horas.",
            "Un supervisor revisará tu caso y te contactará pronto.",
            "Si tu reembolso es aprobado, te avisaremos por correo.",
            "Una vez aprobado el reembolso, se acredita al mismo medio de pago.",
            "Para solicitar un reembolso necesito el número de pedido.",
            "Tu reembolso está en revisión.",
            "Los reembolsos aprobados se acreditan al mismo medio de pago.",
            "El reembolso se procesa en 5 días hábiles.",
            "No puedo aprobar reembolsos; lo revisa un supervisor.",
            "No te preocupes, te ayudo con tu consulta.",
            "Sin problema, te explico los pasos.",
            "No hay problema, dime tu número de pedido.",
            "No dudes en escribirnos si tienes otra pregunta.",
            "Te confirmo que tu pedido va en camino.",
            "Te devolvemos la llamada en 24 horas.",
            "Tu pedido ORD-1001 fue entregado.",
            "No hacemos excepciones a la política de devoluciones.",
            "No te lo puedo garantizar, pero un supervisor puede revisarlo.",
            "Puedes asegurar el paquete con cinta.",
            "Te recomiendo conservar el empaque original.",
            "Estoy aquí para ayudarte.",
            "El producto está garantizado por el fabricante por 12 meses.",
            "Tu pago fue recibido correctamente.",
            "Hemos recibido tu solicitud y la estamos revisando.",
            "Para devolver un producto debes conservar la factura.",
            "Solo un supervisor puede aprobar o rechazar un reembolso.",
            "¿Quieres que te comparta el procedimiento de devolución?",
            "Por esta política, las devoluciones requieren la factura.",
            "Aún no está aprobado; un agente lo revisará.",
            "Tu reembolso no ha sido aprobado todavía.",
            "No se aprueba ningún reembolso sin revisar el producto.",
            "No es posible aprobar reembolsos desde este chat.",
            "Puedo escalar tu caso a un humano.",
            "Con gusto te ayudo con tu devolución.",
            "Te explico cómo funciona la política de reembolsos.",
            "No podemos hacer excepciones con los plazos.",
            "Nunca nos saltamos la política.",
            "Si el producto llega dañado, el supervisor decide sobre la devolución.",
        ],
    )
    def test_frase_legitima_pasa(self, texto):
        res = _ver(texto, "30 días 12 meses 24 horas 5 días", tool={"order_id": "ORD-1001"})
        assert res.ok, (texto, res.detalles)

    def test_respuesta_segura_no_dispara_ninguna_regla(self):
        from tiendahogar_agent.guardrail_compromisos import detectar_compromisos

        assert detectar_compromisos(RESPUESTA_SEGURA) == []

    def test_negacion_no_se_extiende_a_palabras_ajenas(self):
        from tiendahogar_agent.guardrail_compromisos import detectar_compromisos

        assert detectar_compromisos("No dudes que te lo garantizo") != []
        assert detectar_compromisos("No te lo garantizo") == []

    def test_patologicos_de_veinte_mil_caracteres_son_rapidos(self):
        from tiendahogar_agent.guardrail_compromisos import detectar_compromisos

        entradas = [
            "no " * 7000,
            "te " * 7000 + "garantizo",
            "tu reembolso " * 1500,
            "tu reembolso, " + ", " * 9000 + "aprobado",
            "aprobamos " + "a" * 19900,
            "aprobamos " + "tu " * 6500,
            "se te " * 3300,
            "excepcion " * 2000,
            "a" * 20000,
            "no puedo " * 2200,
            "confirmo que " + "a " * 9000,
            "voy a " * 3300,
        ]
        t0 = time.perf_counter()
        for e in entradas:
            detectar_compromisos(e[:20000])
        assert time.perf_counter() - t0 < 5


# ---------------------------------------------------------------- escalar
class TestEscalar:
    def test_escalar_con_canal_pasa(self):
        res = _ver(f"Te paso con una persona: {CANAL_ESCALAMIENTO}.", accion="escalar")
        assert res.ok
        assert res.accion == "escalar"
        assert res.canal == CANAL_ESCALAMIENTO

    def test_canal_insensible_a_mayusculas(self):
        assert _ver("Escribe a SOPORTE@TIENDAHOGAR.EXAMPLE.", accion="escalar").ok

    def test_escalar_sin_canal_falla(self):
        res = _ver("Un humano te ayudará.", accion="escalar")
        assert _falla(res, "falta_canal_escalamiento")
        assert res.accion == "escalar"
        assert res.canal == CANAL_ESCALAMIENTO

    def test_responder_sin_canal_pasa_y_sin_canal_en_resultado(self):
        res = _ver("Claro, te explico.", accion="responder")
        assert res.ok
        assert res.canal is None

    def test_pedir_dato_pasa(self):
        res = _ver("¿Cuál es tu número de pedido?", accion="pedir_dato")
        assert res.ok
        assert res.accion == "pedir_dato"


# ---------------------------------------------------- resultado y respuesta
class TestResultado:
    def test_ok_devuelve_respuesta_original_sin_cambios(self):
        original = "  Tienes 30 días.  \n"
        res = _ver(original, "30 días", fuentes=["doc1"])
        assert res.ok
        assert res.respuesta == original
        assert res.accion == "responder"

    def test_fallo_devuelve_respuesta_segura_y_escala(self):
        res = _ver("Tienes 99 días.", "30 días")
        assert not res.ok
        assert res.respuesta == RESPUESTA_SEGURA
        assert res.accion == "escalar"
        assert res.canal == CANAL_ESCALAMIENTO
        assert res.fuentes == []
        assert res.reglas_fallidas == ("cifra_sin_sustento",)

    def test_varias_reglas_se_acumulan_en_orden_estable(self):
        res = _ver("Te garantizo 99 días.", "30 días", fuentes=["docX"])
        assert res.reglas_fallidas == (
            "cifra_sin_sustento", "fuente_no_recuperada", "compromiso_garantia_resultado"
        )
        assert len(res.detalles) >= 3

    def test_respuesta_segura_no_tiene_cifras_ni_dispara_reglas(self):
        res = _ver(RESPUESTA_SEGURA, accion="escalar", chunks=[])
        assert res.ok, res.detalles
        assert not any(c.isdigit() for c in RESPUESTA_SEGURA)
        assert CANAL_ESCALAMIENTO in RESPUESTA_SEGURA
        assert "no tengo esa información" in RESPUESTA_SEGURA.lower()

    def test_respuesta_segura_en_tuteo(self):
        assert " tu " in f" {RESPUESTA_SEGURA.lower()} " or "te " in RESPUESTA_SEGURA.lower()
        assert " usted " not in RESPUESTA_SEGURA.lower()

    def test_idempotente_sobre_su_propia_salida(self):
        primero = _ver("Tienes 99 días.", "30 días")
        segundo = verificar_salida(
            respuesta=primero.respuesta, accion=primero.accion, fuentes=primero.fuentes,
            chunks=[], resultado_tool=None, mensaje_usuario="",
        )
        assert segundo.ok
        assert segundo.respuesta == primero.respuesta

    def test_resultado_inmutable_y_estricto(self):
        res = _ver("Hola.")
        with pytest.raises(ValidationError):
            res.ok = False  # type: ignore[misc]


# ------------------------------------------------------------- robustez
class TestRobustez:
    def test_respuesta_demasiado_larga_falla_seguro_rapido(self):
        t0 = time.perf_counter()
        res = _ver("1." * 50000)
        assert time.perf_counter() - t0 < 0.3
        assert _falla(res, "respuesta_demasiado_larga")
        assert res.accion == "escalar"

    def test_contexto_enorme_es_rapido(self):
        t0 = time.perf_counter()
        _ver("Hola.", mensaje="1." * 500000)
        assert time.perf_counter() - t0 < 1

    @pytest.mark.parametrize("guion", ["‑", "–", "−", "—", "-"])
    def test_guiones_unicode_en_ids(self, guion):
        assert _ver(f"Tu pedido ORD{guion}1001 va en camino.", tool={"order_id": "ORD-1001"}).ok
        assert _falla(_ver(f"Tu pedido ORD{guion}9999 va en camino.", tool={"order_id": "ORD-1001"}),
                      "cifra_sin_sustento")

    def test_error_interno_registra_sin_pii(self, caplog, monkeypatch):
        import tiendahogar_agent.guardrail_output as mod

        def boom(*a, **k):
            raise RuntimeError("secreto@pii.com")

        monkeypatch.setattr(mod, "_verificar", boom)
        with caplog.at_level("ERROR", logger=mod.__name__):
            res = verificar_salida("Mi correo es x@y.com", "responder", [], [], None, "")
        assert not res.ok
        assert "RuntimeError" in caplog.text
        assert "pii.com" not in caplog.text and "x@y.com" not in caplog.text

    def test_entradas_raras_no_lanzan(self):
        res = verificar_salida(None, None, None, None, None, None)  # type: ignore[arg-type]
        assert isinstance(res, ResultadoVerificacion)
        assert not res.ok
        assert res.respuesta == RESPUESTA_SEGURA

    def test_respuesta_vacia_falla_seguro(self):
        res = _ver("   ")
        assert _falla(res, "respuesta_vacia")
        assert res.accion == "escalar"

    def test_tool_none_y_chunks_vacios(self):
        assert _ver("Hola.", tool=None, chunks=[], mensaje="").ok

    def test_tool_anidado_y_valores_raros(self):
        tool = {"a": [1, 2, {"b": None}], "c": True, "d": (3.5,)}
        assert _ver("Son 1, 2 y 3.5.", tool=tool).ok

    def test_tool_no_dict(self):
        assert _ver("Son 5.", tool="texto 5").ok  # type: ignore[arg-type]

    def test_accion_invalida_falla_seguro(self):
        res = _ver("Hola.", accion="inventada")
        assert _falla(res, "accion_invalida")
        assert res.accion == "escalar"

    def test_numero_enorme_no_lanza(self):
        res = _ver("Son " + "9" * 5000 + " pesos.", "Sin cifras.")
        assert _falla(res, "cifra_sin_sustento")
        assert _ver("Son " + "9" * 5000 + " pesos.", "x " + "9" * 5000).ok

    def test_texto_largo_es_rapido(self):
        largo = "Tienes treinta días y 1.200 pesos, no hay excepciones para nadie. " * 250
        ctx = "30 días 1200 pesos"
        t0 = time.perf_counter()
        res = _ver(largo, ctx)
        assert time.perf_counter() - t0 < 5
        assert res.ok

    def test_patrones_patologicos_no_cuelgan(self):
        t0 = time.perf_counter()
        _ver("te " * 20000 + "hago " + "una " * 20000, "")
        _ver("1 " * 50000 + "1.1" * 20000, "1")
        _ver("treinta y " * 20000, "")
        _ver("no " * 30000 + "te garantizo", "")
        assert time.perf_counter() - t0 < 10


def test_promesa_de_notificacion_se_bloquea():
    from tiendahogar_agent.guardrail_compromisos import R_NOTIFICACION, detectar_compromisos

    for frase in [
        "Te enviaremos un email con el número de seguimiento.",
        "Recibirás un correo de confirmación.",
        "Te llegará una notificación cuando salga.",
    ]:
        assert R_NOTIFICACION in [r for r, _ in detectar_compromisos(frase)], frase
    # negación o política + coma + promesa: la negación/política no exime la promesa
    for frase in [
        "No te lo puedo garantizar, te notificaremos por email.",
        "No hacemos excepciones, te enviaremos un correo.",
        "El reembolso se aprueba tras revisar el producto, te enviaremos un correo.",
        "Si tu reembolso es aprobado, te enviaremos un email de confirmación.",
        "No podemos aprobar eso ahora, te avisaremos por email.",
        "Si tu reembolso es aprobado, te avisaremos y te notificaremos por correo.",
    ]:
        assert R_NOTIFICACION in [r for r, _ in detectar_compromisos(frase)], frase
    legitimas = [
        "Escríbenos a soporte@tiendahogar.example y te ayudarán.",
        "Puedes contactar al equipo humano.",
        "Si tu reembolso es aprobado, te avisaremos.",
        "Te enviaré la política en este chat.",
    ]
    for frase in legitimas:
        assert not detectar_compromisos(frase), frase


def test_verificar_salida_bloquea_promesa_de_notificacion():
    from tiendahogar_agent.guardrail_compromisos import R_NOTIFICACION
    from tiendahogar_agent.guardrail_output import verificar_salida

    r = verificar_salida(
        "Te enviaremos un email con el seguimiento.", "responder", [], [], None, "hola"
    )
    assert not r.ok and R_NOTIFICACION in r.reglas_fallidas
