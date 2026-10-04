"""Pruebas del enmascarado de PII (T09). Deterministas, sin red ni LLM."""

import time

import pytest
from tiempos import tope

from tiendahogar_agent.pii import enmascarar_pii, enmascarar_pii_con_conteos

CORREOS = [
    "ana@example.com",
    "ana.perez+pedidos@mail.tienda.example.co",
    "JUAN.PEREZ@EXAMPLE.COM",
    "a_b-c@sub.dominio.example.org",
]

TELEFONOS = [
    "300 123 4567",
    "3001234567",
    "300-123-4567",
    "300.123.4567",
    "+57 300 123 4567",
    "+573001234567",
    "0057 300 123 4567",
    "(601) 234 5678",
    "601 234 5678",
    "6012345678",
    "+1 (415) 555-2671",
    "+34 612 345 678",
    "+44 20 7946 0958",
    "(300) 123-4567",
    "(300) 1234567",
    "57 300 123 4567",
    "573001234567",
    "03001234567",
    "0034 612345678",
    "300 123 4567",
    "300	123	4567",
    "３００１２３４５６７",
]

TARJETAS = [
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "5555 5555 5555 4444",
    "3782 822463 10005",
    "4111.1111.1111.1111",
    "4111  1111 1111 1111",
    "4111/1111/1111/1111",
    "4000 0000 0000 0000 006",
    "4000 0000 0000 006",
    "4000-0000-0000-0000-006",
]

INTACTOS = [
    "ORD-1001",
    "ORD-12345",
    "$1,200",
    "$1.200",
    "800 dólares",
    "USD 800",
    "1.200,50",
    "500.01",
    "2026-10-05",
    "05/10/2026",
    "en 2026",
    "código 12345",
    "referencia 123456789",
    "pedido 1001",
]


@pytest.mark.parametrize("correo", CORREOS)
def test_correos(correo):
    assert enmascarar_pii(f"Escríbeme a {correo} gracias") == "Escríbeme a [CORREO] gracias"


@pytest.mark.parametrize("tel", TELEFONOS)
def test_telefonos(tel):
    assert enmascarar_pii(f"Llámame al {tel} hoy") == "Llámame al [TELEFONO] hoy"
    assert enmascarar_pii(tel) == "[TELEFONO]"


@pytest.mark.parametrize("tarjeta", TARJETAS)
def test_tarjetas_validas(tarjeta):
    assert enmascarar_pii(f"Mi tarjeta es {tarjeta}.") == "Mi tarjeta es [TARJETA]."


def test_tarjeta_luhn_invalido_no_es_tarjeta():
    salida = enmascarar_pii("número 4111 1111 1111 1112 listo")
    assert "[TARJETA]" not in salida


def test_tarjeta_no_se_parte_en_telefonos():
    assert enmascarar_pii("5555 5555 5555 4444") == "[TARJETA]"
    assert "[TELEFONO]" not in enmascarar_pii("3782 822463 10005")


def test_telefono_no_es_tarjeta():
    assert enmascarar_pii("3001234567") == "[TELEFONO]"


@pytest.mark.parametrize("texto", INTACTOS)
def test_no_enmascara_no_pii(texto):
    assert enmascarar_pii(texto) == texto
    assert enmascarar_pii(f"Dato: {texto}.") == f"Dato: {texto}."


def test_pedido_y_monto_junto_a_pii():
    texto = "ORD-1001 por $1.200 o 800 dólares, llama al 3001234567 o escribe a a@b.co"
    esperado = "ORD-1001 por $1.200 o 800 dólares, llama al [TELEFONO] o escribe a [CORREO]"
    assert enmascarar_pii(texto) == esperado


def test_pedido_pegado_a_telefono_con_espacio():
    assert enmascarar_pii("ORD-1001 3001234567") == "ORD-1001 [TELEFONO]"
    assert enmascarar_pii("ORD-12345 +57 300 123 4567") == "ORD-12345 [TELEFONO]"


def test_varios_pii_en_una_frase():
    texto = "Soy ana@x.com, cel 300 123 4567, tarjeta 4111 1111 1111 1111 y ORD-1001"
    assert enmascarar_pii(texto) == "Soy [CORREO], cel [TELEFONO], tarjeta [TARJETA] y ORD-1001"


def test_conteos():
    texto = "a@b.co, c@d.co, 3001234567, 4111 1111 1111 1111"
    salida, conteos = enmascarar_pii_con_conteos(texto)
    assert salida == "[CORREO], [CORREO], [TELEFONO], [TARJETA]"
    assert conteos == {"correo": 2, "telefono": 1, "tarjeta": 1}


def test_vacio():
    assert enmascarar_pii("") == ""


@pytest.mark.parametrize("texto", CORREOS + TELEFONOS + TARJETAS + INTACTOS)
def test_idempotencia_por_caso(texto):
    una = enmascarar_pii(texto)
    assert enmascarar_pii(una) == una


def test_idempotencia_global():
    texto = " | ".join(CORREOS + TELEFONOS + TARJETAS + INTACTOS)
    una = enmascarar_pii(texto)
    assert enmascarar_pii(una) == una
    for marcador in ("[CORREO]", "[TELEFONO]", "[TARJETA]"):
        assert marcador in una


@pytest.mark.parametrize(
    "texto",
    ["1 " * 20000, "a@" * 20000, "a" * 40000, "1" * 40000, "a." * 20000, "+1 " * 20000],
    ids=["unos_espacio", "arroba", "letras", "digitos", "punto", "mas_uno"],
)
def test_sin_redos(texto):
    t0 = time.perf_counter()
    enmascarar_pii(texto)
    assert time.perf_counter() - t0 < tope(0.5)


@pytest.mark.parametrize("unidad", ["1 ", "a@", "a."], ids=["unos_espacio", "arroba", "punto"])
def test_sin_redos_crece_linealmente(unidad):
    """Cuadruplicar la entrada no debe multiplicar el tiempo por mucho más de 4 (cuadrático: 16)."""

    def medir(n):
        mejor = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            enmascarar_pii(unidad * n)
            mejor = min(mejor, time.perf_counter() - t0)
        return mejor

    chico, grande = medir(10000), medir(40000)
    assert grande < max(chico * 12, 0.05)


@pytest.mark.parametrize(
    "tarjeta",
    [
        "4222 2222 2222 2",  # 13 dígitos
        "4000 0000 0000 02",  # 14 dígitos
        "4000 0000 0000 0000 0",  # 17 dígitos (se ajusta abajo)
    ],
)
def test_tarjetas_ultimo_grupo_corto_si_pasan_luhn(tarjeta):
    from tiendahogar_agent.pii import _luhn

    digitos = "".join(c for c in tarjeta if c.isdigit())
    esperado = "[TARJETA]" if _luhn(digitos) else tarjeta
    assert enmascarar_pii(tarjeta) == esperado


def _completar_luhn(prefijo: str) -> str:
    from tiendahogar_agent.pii import _luhn

    return next(prefijo + str(d) for d in range(10) if _luhn(prefijo + str(d)))


@pytest.mark.parametrize("largo", [13, 14, 15, 16, 17, 18, 19])
@pytest.mark.parametrize("sep", [" ", "-"])
def test_tarjetas_largos_con_grupo_final_corto(largo, sep):
    digitos = _completar_luhn("4" + "0" * (largo - 2))
    grupos = [digitos[i : i + 4] for i in range(0, largo, 4)]
    assert enmascarar_pii(sep.join(grupos)) == "[TARJETA]"


def test_id_pegado_a_telefono_no_es_tarjeta():
    assert enmascarar_pii("ORD-1001 3001234567") == "ORD-1001 [TELEFONO]"
    assert enmascarar_pii("1001 3001234567") == "1001 [TELEFONO]"


def test_telefono_mas_otro_numero_no_es_tarjeta():
    salida = enmascarar_pii("3001234567 12345")
    assert "[TARJETA]" not in salida
    assert salida == "[TELEFONO] 12345"


def test_prefijos_sin_mas_no_dejan_residuo():
    assert enmascarar_pii("57 300 123 4567") == "[TELEFONO]"
    assert enmascarar_pii("llama 0034 612345678 ya") == "llama [TELEFONO] ya"


def test_montos_e_ids_cercanos_a_prefijos_intactos():
    for t in ["ORD-5730", "$57 300", "57 dólares", "300 pesos", "ORD-1001 y 57"]:
        assert enmascarar_pii(t) == t


def test_normalizacion_idempotente_y_conserva_texto():
    texto = "Hola ñandú á ORD-1001"
    assert enmascarar_pii(texto) == texto
    mixto = "a b ３００１２３４５６７"
    assert enmascarar_pii(enmascarar_pii(mixto)) == enmascarar_pii(mixto)


# --- el texto no PII se conserva intacto (NFKC solo para detectar) ---

NO_PII_RAROS = "½ taza… producto™ ﬁnal\u00a0listo"


def test_no_pii_con_caracteres_compatibles_queda_identico():
    assert enmascarar_pii(NO_PII_RAROS) == NO_PII_RAROS
    for c in ["½", "…", "™", "ﬁ", "\u00a0"]:
        assert enmascarar_pii(f"a{c}b") == f"a{c}b"


def test_pii_con_digitos_ancho_completo_se_enmascara():
    assert enmascarar_pii("Llámame al ３００１２３４５６７ ya") == "Llámame al [TELEFONO] ya"


def test_pii_con_nbsp_se_enmascara():
    assert enmascarar_pii("Tel 300\u00a0123\u00a04567 ok") == "Tel [TELEFONO] ok"


def test_mezcla_pii_y_texto_raro():
    t = "½ taza… ３００\u00a0１２３\u00a04567 y ana@example.com ™ ﬁn"
    assert enmascarar_pii(t) == "½ taza… [TELEFONO] y [CORREO] ™ ﬁn"


def test_idempotencia_con_caracteres_raros():
    t = "½ taza… ３００\u00a0１２３\u00a04567 ™ ﬁn ana@example.com"
    una = enmascarar_pii(t)
    assert enmascarar_pii(una) == una


@pytest.mark.parametrize("prefijo", ["Espera…", "Marca™", "ﬁnal", "…™ﬁ…™ﬁ"])
def test_expansion_nfkc_no_desalinea_indices(prefijo):
    assert enmascarar_pii(f"{prefijo} 3001234567 fin") == f"{prefijo} [TELEFONO] fin"
    assert enmascarar_pii(f"{prefijo} ana@example.com fin") == f"{prefijo} [CORREO] fin"
    assert (
        enmascarar_pii(f"{prefijo} 4111 1111 1111 1111 fin") == f"{prefijo} [TARJETA] fin"
    )
