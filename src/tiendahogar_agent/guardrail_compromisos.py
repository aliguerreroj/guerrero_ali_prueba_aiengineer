"""Detección léxica de compromisos que el agente NO puede asumir (fallo seguro).

El agente no aprueba reembolsos, no garantiza resultados ni hace excepciones: eso
lo decide un humano. Este módulo revisa el texto candidato y devuelve las reglas
violadas. Es una detección LÉXICA (no semántica): ante la duda, bloquea, porque
bloquear equivale a escalar a un humano.

Diseño estructural (en tres pasos sobre el texto normalizado: minúsculas, sin
tildes, sin caracteres invisibles, espacios colapsados):

1. NEUTRALIZAR negaciones. La negación solo cuenta si es una lista CERRADA de
   formas pegadas al verbo de compromiso: «no/nunca/jamás/tampoco» + clíticos
   (te, lo...) + modal opcional (puedo, podemos, voy a, es posible, se puede...) +
   verbo de la familia de compromiso («no te lo garantizo», «no podemos aprobar»,
   «no se aprueba», «no hacemos excepciones», «aún no está aprobado»). Cualquier
   otra palabra intermedia NO niega: «no dudes que te lo garantizo», «no te
   preocupes, te reembolsamos», «no hay problema, te lo aprobamos» siguen bloqueadas.
2. NEUTRALIZAR descripciones impersonales de política (lista cerrada): «el/los/un
   reembolso(s) se aprueba(n)/procesa(n)/acredita(n)...», «los reembolsos aprobados
   se acreditan...», y condicionales sobre el estado («si tu reembolso es aprobado»,
   «una vez aprobado el reembolso»).
3. DETECTAR sobre lo que queda, por familias de raíces verbales
   (reembols-, devolv-/devuelv-, regres-, acredit-, reintegr-, aprob-, autoriz-,
   conced-, acept-, confirm-, proces-, garantiz-, asegur-, promet-, excepcion-,
   saltar la política...) en: primera persona y futuro («te reembolsamos»), perfecto
   y gerundio («hemos aprobado», «estamos aprobando»), perifrásticas («voy a
   devolver»), pasiva con «se te»/«te será» («se te devolverá»), estados
   («tu reembolso está/queda/fue aprobado»; «es un hecho»), y promesas
   («te garantizo», «tienes mi palabra», «seguro que»).

Frontera de decisión:
- BLOQUEA: cualquier afirmación en primera persona, en futuro o de estado sobre el
  reembolso/devolución/caso DEL USUARIO, con o sin objeto y con comas/signos entre
  medias; promesas de resultado; excepciones («excepcionalmente», «por esta vez»,
  «te saltamos la política»). Los condicionales con compromiso («si cumples, te
  devolvemos...») también se bloquean.
- PERMITE: describir la política en tercera persona impersonal («el reembolso se
  aprueba tras revisar el producto», «la garantía cubre defectos de fábrica»),
  informar estados del pedido, pedir datos, escalar, y la lista cerrada de
  negaciones del paso 1.
- «te aseguro/prometo que...» y «seguro que...» solo bloquean si el objeto es un
  resultado (reembolso, aprobación, excepción, dinero, solución, «saldrá bien»...);
  «te aseguro que un humano revisará tu caso» pasa. «Garantizado contra/por...»
  describe la cobertura del fabricante y pasa.
- Métrica medida (conjunto fijo tests/data/compromisos_eval.json, ADR-005):
  tests/test_compromisos_metricas.py exige detección >= 90 % y falsos positivos <= 5 %.
- Limitaciones: es léxico; una negación fuera de la lista cerrada no se reconoce;
  paráfrasis muy inusuales pueden escapar; ciertos usos legítimos de verbos como
  «devolvemos» o «procesamos» con objeto personal se bloquean (falla seguro).
"""

from __future__ import annotations

import re

from tiendahogar_agent.texto import quitar_tildes

R_REEMBOLSO = "compromiso_reembolso_aprobado"
R_GARANTIA = "compromiso_garantia_resultado"
R_EXCEPCION = "compromiso_excepcion"

_INVISIBLES = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")
_MARCAS = re.compile(r"[*_`~]")
_ESPACIOS = re.compile(r"\s+")


def _alt(*items: str) -> str:
    return "(?:" + "|".join(items) + ")"


# ------------------------------------------------------------------ fragmentos
_CL1 = r"(?:te|se|le|les|me|nos|lo|la|los|las)"
_CLP = r"(?:te|se|le|les|lo|la|los|las)"  # clíticos de promesa (sin «me»/«nos»)
_CLS = r"(?:" + _CL1 + r"\s+){0,3}"  # clíticos opcionales
_CLS1 = r"(?:" + _CL1 + r"\s+){1,3}"  # al menos un clítico

_SUST_S = r"(?:reembolso|devolucion|reintegro)"
_SUST_A = r"(?:reembolsos?|devoluci(?:on|ones)|reintegros?)"
_CASO = r"(?:caso|solicitud|reclamo|reclamacion|peticion|excepcion|cambio)"
_DET = r"(?:tu|su|tus|sus|mi|el|la|este|ese|esta|esa|un|una)"
_SUBJ = r"\b(?:tu|su|el|la|este|ese|esta|esa|mi)\s+(?:" + _SUST_S + "|" + _CASO + r"|dinero)\b"
# objeto de compromiso. A (aprobar/autorizar/conceder, verbos de dinero): amplio.
_OBJ_A = (
    r"(?:(?:" + _DET + r"\s+)?(?:[a-z]+\s+){0,2})?"
    r"(?:" + _SUST_A + r"|dinero|plata|importe|excepcion|caso|solicitud|reclamo|reclamacion|"
    r"peticion|cambio|compra)\b"
)
# B (aceptar/confirmar/procesar): solo singular personal, para no bloquear
# «aceptamos devoluciones dentro de 30 días» ni «procesamos tu pedido».
_OBJ_B = r"(?:(?:" + _DET + r"\s+)?(?:[a-z]+\s+){0,3})?(?:" + _SUST_S + r"|dinero|excepcion)\b"

_PART_C = r"(?:aprobad|autorizad|concedid|aceptad|confirmad|procesad|acreditad|reembolsad|devuelt|reintegrad|regresad)[oa]s?"
# «garantizado contra/por/ante...» describe la cobertura del fabricante, no un compromiso
_NO_COBERTURA = r"(?!\s+(?:contra|por|ante|frente|durante|de|en)\b)"
_PART_ALL = r"(?:" + _PART_C + r"|listo|garantizad[oa]s?" + _NO_COBERTURA + r")\b"
_AUX = _alt(
    r"es", r"esta", r"estara", r"estaria", r"fue", r"ha\s+sido", r"ya\s+ha\s+sido", r"ha\s+quedado",
    r"queda", r"quedo", r"quedara", r"sera", r"va\s+a\s+ser", r"va\s+a\s+estar", r"esta\s+siendo",
)
_RELLENO = r"(?:(?:ya|ahora|oficialmente|finalmente|por\s+fin|definitivamente|totalmente|tambien)\s+)?"
_SEP = r"(?:\s|,|;|:|-)+"
_SIN_LLAMADA = r"(?!\s+(?:[a-z]+\s+){0,2}(?:llamada|llamado|mensaje|correo|respuesta|comunicacion|contacto|mail|whatsapp|telefono|pregunta|duda|consulta|inquietud|saludo)\b)"

# ---------------------------------------------------------------- neutralización
_MODAL = _alt(
    r"puedo", r"podemos", r"puedes", r"puede", r"pueden", r"podria", r"podriamos", r"voy\s+a",
    r"vamos\s+a", r"va\s+a", r"debo", r"debemos", r"tengo\s+que", r"tenemos\s+que", r"quiero",
    r"queremos", r"me\s+es\s+posible", r"nos\s+es\s+posible", r"es\s+posible", r"se\s+puede",
    r"se\s+pueden", r"estoy\s+autorizad[oa]\s+a", r"estamos\s+autorizad[oa]s\s+a",
)
_VERBO_COMPROMISO = (
    r"(?:aprob|aprueb|autoriz|conced|acept|confirm|reembols|devolv|devuelv|regres|acredit|reintegr|"
    r"proces|garantiz|asegur|promet|salt|omit|ignor|jur|exceptu|flexibiliz|dispens|tendr|recibir|"
    r"recuper|obtendr|har[ae]|hac|hag|hic|dar|dam|abon|compens|proced|estam|estar)[a-z]*"
)
_NEG_PALABRA = r"\b(?:no|nunca|jamas|tampoco)\s+"
_NEG_VERBO = re.compile(
    _NEG_PALABRA + _CLS + r"(?:" + _MODAL + r"\s+" + _CLS + r")?" + _VERBO_COMPROMISO + r"\b"
)
_NEG_EXCEPCION = re.compile(
    _NEG_PALABRA + _CLS + r"(?:" + _MODAL + r"\s+" + _CLS + r")?[a-z]+\s+"
    r"(?:(?:una|ninguna|alguna|las|esa|esta|estas|mas|otra)\s+){0,2}excepcion(?:es)?\b"
)
_NEG_ESTADO = re.compile(
    r"\bno\s+(?:esta|ha\s+sido|fue|sera|queda|quedo|estara|esta\s+siendo)\s+(?:aun\s+|todavia\s+|ya\s+)?"
    + _PART_C + r"\b"
)
_POLITICA = [
    # «el/los/un reembolso(s) se aprueba(n) ...» (impersonal, nunca «tu/su»)
    re.compile(
        r"\b(?:el|los|un|una|las|la|cada|todo)\s+" + _SUST_A + r"\s+(?:se\s+)?"
        r"(?:aprueba|aprueban|procesa|procesan|acredita|acreditan|evalua|evaluan|revisa|revisan|"
        r"autoriza|autorizan|concede|conceden|acepta|aceptan|devuelve|devuelven|reembolsa|"
        r"reembolsan|regresa|regresan|reintegra|reintegran|cubre|cubren)\b"
    ),
    # «los reembolsos aprobados se acreditan ...»
    re.compile(
        r"\b(?:el|los|un|una|las|la|cada)\s+" + _SUST_A + r"\s+(?:ya\s+)?" + _PART_C + r"\s+se\s+[a-z]+"
    ),
    # condicionales sobre el estado: «si tu reembolso es aprobado», «cuando el reembolso se apruebe»
    re.compile(
        r"\b(?:si|cuando|una\s+vez\s+que|en\s+caso\s+de\s+que|siempre\s+que|mientras|hasta\s+que|en\s+cuanto)"
        r"\s+(?:(?:tu|su|el|la|un|una|este|ese)\s+)?(?:" + _SUST_S + "|" + _CASO + r"|dinero)"
        r"(?:\s+[a-z]+){0,2}\s+(?:" + _PART_C + r"|apruebe|autorice|conceda|acepte|procese)\b"
    ),
    # «una vez aprobado el reembolso», «tras ser aprobado ...»
    re.compile(
        r"\b(?:una\s+vez\s+(?:que\s+)?|luego\s+de\s+(?:ser\s+)?|despues\s+de\s+(?:ser\s+)?|tras\s+(?:ser\s+)?)"
        r"(?:se\s+(?:apruebe|autorice|procese|conceda|acepte)|(?:sea\s+)?" + _PART_C + r")\b"
    ),
]
_NEUTRO = " | "


def _normalizar(texto: str) -> str:
    t = _INVISIBLES.sub("", quitar_tildes(texto))
    return _ESPACIOS.sub(" ", _MARCAS.sub(" ", t)).strip()


def _neutralizar(texto: str) -> str:
    for patron in (_NEG_EXCEPCION, _NEG_VERBO, _NEG_ESTADO, *_POLITICA):
        texto = patron.sub(_NEUTRO, texto)
    return texto


# -------------------------------------------------------------------- familias
_DINERO_1 = _alt(
    r"reembolsamos", r"reembolsaremos", r"reembolsare", r"reembolse", r"reembolsaron",
    r"devolvemos", r"devolveremos", r"devolvere", r"devuelvo", r"devolvimos", r"devolvi",
    r"regresamos", r"regresaremos", r"regresare", r"regrese",
    r"acreditamos", r"acreditaremos", r"acreditare", r"acredito", r"acredite",
    r"reintegramos", r"reintegraremos", r"reintegrare", r"reintegre",
    r"abonamos", r"abonaremos", r"abonare",
)
_DINERO_3 = _alt(
    r"devolvera", r"devolveran", r"devuelven", r"devolvieron", r"devolvio", r"devuelve",
    r"reembolsara", r"reembolsaran", r"reembolsan", r"reembolsaron", r"reembolsa", r"reembolso",
    r"acreditara", r"acreditaran", r"acreditan", r"acredito", r"acredita",
    r"reintegrara", r"reintegraran", r"reintegran", r"reintegra", r"reintegro",
    r"regresara", r"regresaran", r"regresan", r"regresa",
)
_DECISION_3 = _alt(
    r"aprobara", r"aprobaran", r"aprueban", r"aprobaron", r"aprobo", r"aprueba",
    r"autorizara", r"autorizaran", r"autorizan", r"autorizo", r"autoriza",
    r"concedera", r"concederan", r"conceden", r"concedio", r"concede",
)
_FORMAS_A = _alt(  # aprobar / autorizar / conceder (1.ª persona)
    r"aprobamos", r"aprobaremos", r"aprobare", r"apruebo", r"aprobe",
    r"autorizamos", r"autorizaremos", r"autorizo", r"autorice", r"autorizare",
    r"concedemos", r"concedimos", r"concedo", r"concedere", r"concedi", r"concederemos",
)
_FORMAS_B = _alt(  # aceptar / confirmar / procesar (1.ª persona)
    r"aceptamos", r"aceptaremos", r"acepto", r"aceptare", r"acepte",
    r"confirmamos", r"confirmaremos", r"confirmo", r"confirmare", r"confirme",
    r"procesamos", r"procesaremos", r"procesare", r"procese",
)
_PERF = _alt(
    r"aprobado", r"autorizado", r"concedido", r"reembolsado", r"devuelto", r"acreditado",
    r"reintegrado", r"regresado", r"aceptado", r"procesado",
)
_GER = _alt(
    r"aprobando", r"autorizando", r"concediendo", r"aceptando", r"procesando", r"reembolsando",
    r"devolviendo", r"acreditando", r"reintegrando", r"regresando", r"confirmando",
)
_MODAL_1 = r"(?:voy\s+a|vamos\s+a|puedo|podemos|quiero|queremos)"
_ENCLITICO = r"(?:te|le|lo|la|les|los|las|selo|telo|tela|sela)"
_INF_DINERO = _alt(r"reembolsar", r"devolver", r"acreditar", r"reintegrar", r"regresar")
_INF_A = _alt(r"aprobar", r"autorizar", r"conceder")
_INF_B = _alt(r"aceptar", r"confirmar", r"procesar")
_FORMAS_ESTADO_DECISION = r"(?:aprobad|autorizad|concedid)[oa]s?"
_EXC_VERBO = _alt(
    r"hago", r"hacemos", r"haremos", r"hare", r"haria", r"hariamos", r"hagamos", r"damos", r"doy",
    r"daremos", r"dare", r"concedemos", r"concedo", r"concederemos", r"ofrecemos", r"ofrezco",
    r"aplicamos", r"aplico", r"aplicaremos", r"otorgamos", r"otorgo", r"permitimos", r"autorizamos",
    r"autorizo", r"regalamos", r"regalo", r"brindamos", r"brindo",
    r"(?:podemos|puedo|voy\s+a|vamos\s+a)\s+(?:hacer|dar|conceder|aplicar|otorgar)",
)
_SALTO = _alt(
    r"saltamos", r"salto", r"saltaremos", r"saltare", r"omitimos", r"omito", r"omitiremos",
    r"ignoramos", r"ignoro", r"ignoraremos", r"pasamos\s+por\s+alto", r"obviamos", r"flexibilizamos",
    r"flexibilizo", r"exceptuamos", r"exceptuo", r"eximimos", r"eximo", r"dispensamos",
)


# objeto de resultado: lo que convierte «te aseguro/prometo que...» en compromiso
_OBJ_RES = (
    r"(?:reembols[a-z]*|devoluci[a-z]*|devolv[a-z]*|devuelv[a-z]*|dinero|plata|excepci[a-z]*|aprobaci[a-z]*|"
    r"aprob[a-z]*|aprueb[a-z]*|autoriz[a-z]*|acredit[a-z]*|reintegr[a-z]*|regres[a-z]*|solucion|"
    r"resultado|exito|aceptar[a-z]*|aceptad[a-z]*|sal(?:ga|dra)[a-z]*|resuelv[a-z]*|resolv[a-z]*|"
    r"llega[a-z]*|llegar[a-z]*|funcionar[a-z]*|arreglar[a-z]*)"
)


def _c(*patrones: str) -> re.Pattern[str]:
    return re.compile("|".join(patrones))


_REGLAS: list[tuple[str, re.Pattern[str]]] = [
    (R_REEMBOLSO, _c(
        # verbos de dinero en 1.ª persona/futuro (con o sin clítico y objeto)
        r"\b" + _CLS + _DINERO_1 + r"\b" + _SIN_LLAMADA,
        r"\b(?:te|le|se|lo|la)\s+(?:reembolso|reintegro|regreso)\b" + _SIN_LLAMADA,
        # 3.ª persona dirigida al usuario: «te devolverán», «se te reembolsará», «te será devuelto»
        r"\b(?:te|le|les)\s+(?:lo\s+|la\s+|los\s+|las\s+)?(?:" + _DINERO_3 + r"|" + _DECISION_3 + r")\b",
        r"\bse\s+(?:te|le|les|nos)\s+(?:lo\s+|la\s+)?(?:" + _DINERO_3 + r"|" + _DECISION_3 + r"|procesa\w*|acepta\w*)\b",
        r"\b(?:te|le|les)\s+(?:ya\s+)?(?:sera|seran|fue|fueron|esta|estan|queda|quedan|quedo)\s+(?:ya\s+)?"
        + _PART_C + r"\b",
        # «se aprobó», «se aprobará», «se devolverá» (sin «se te»)
        r"\bse\s+(?:aprobo|aprobara|autorizo|autorizara|concedio|concedera|devolvera|devolvio|reembolsara)\b",
        r"\bse\s+(?:acreditara|acredito|reintegrara|regresara|procesara|aprobaran)\s+(?:[a-z]+\s+){0,2}"
        r"(?:tu|su)\s+" + _SUST_A + r"\b",
        # aprobar / autorizar / conceder: clítico o con objeto
        r"\b" + _CLS1 + _FORMAS_A + r"\b",
        r"\b" + _FORMAS_A + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_A,
        # aceptar / confirmar / procesar: clítico lo/la o con objeto personal singular
        r"\b(?:lo|la)\s+" + _FORMAS_B + r"\b",
        r"\b" + _FORMAS_B + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_B,
        r"\bconfirm(?:o|amos|aremos|are)\s+que\s+(?:[a-z]+\s+){0,5}?"
        r"(?:aprobad|autorizad|concedid|reembolsad|aprobo|autorizo|concedio|aprobara)",
        # perfecto y gerundio
        r"\b" + _CLS1 + r"(?:he|hemos)\s+(?:ya\s+)?" + _PERF + r"\b",
        r"\b(?:he|hemos)\s+(?:ya\s+)?" + _PERF + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_A,
        r"\b" + _CLS + r"(?:estamos|estoy)\s+(?:ya\s+)?" + _GER + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_A,
        r"\b" + _CLS1 + r"(?:estamos|estoy)\s+(?:ya\s+)?" + _GER + r"\b",
        # perifrásticas e infinitivos con modal en 1.ª persona
        r"\b" + _MODAL_1 + r"\s+(?:" + _CL1 + r"\s+){0,2}" + _INF_DINERO + r"(?:" + _ENCLITICO + r")?\b" + _SIN_LLAMADA,
        r"\b" + _MODAL_1 + r"\s+(?:" + _CL1 + r"\s+){1,2}(?:" + _INF_A + "|" + _INF_B + r")\b",
        r"\b" + _MODAL_1 + r"\s+(?:" + _INF_A + "|" + _INF_B + r")" + _ENCLITICO + r"\b",
        r"\b" + _MODAL_1 + r"\s+" + _INF_A + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_A,
        r"\b" + _MODAL_1 + r"\s+" + _INF_B + r"\s+(?:" + _CL1 + r"\s+)?" + _OBJ_B,
        r"\b(?:haremos|hare|hago|voy\s+a\s+hacer|vamos\s+a\s+hacer)\s+(?:ya\s+)?(?:la|tu|su)\s+" + _SUST_S + r"\b",
        # estados: «tu reembolso (ya) está/queda/fue aprobado», con comas o relleno
        _SUBJ + _SEP + _RELLENO + _AUX + r"(?:\s|,|;|:|-)*" + _RELLENO + r"(?:sido\s+)?" + _PART_ALL,
        _SUBJ + _SEP + _RELLENO + r"(?:es|esta|sera|queda|quedara)\s+(?:ya\s+)?un\s+hecho\b",
        r"\b(?:esta|queda|quedo|quedara|sera|fue|ha\s+sido|estara)\s+(?:ya\s+)?" + _FORMAS_ESTADO_DECISION + r"\b",
        r"(?:^|[.!?;,\-]\s*)(?:ya\s+)?" + _SUST_S + r"\s*:?\s+(?:ya\s+)?" + _PART_C + r"\b",
        r"\b" + _SUST_S + r"\s*:\s*" + _PART_C + r"\b",
        r"\b(?:listo|todo|perfecto|hecho)\s*[,:!.\-]*\s*(?:ya\s+)?(?:aprobad|autorizad|reembolsad|concedid)[oa]s?\b",
        r"(?:^|[.!?]\s*)(?:aprobad|autorizad|concedid)[oa]s?\s*(?:[.!,]|$)",
        r"\b" + _PART_C + r"\s+(?:ya\s+)?(?:tu|su|el|la|este|ese|mi)\s+(?:" + _SUST_S + "|" + _CASO + r"|dinero)\b",
        r"\b(?:queda|quedo|esta|estara)\s+(?:ya\s+)?listo\s+(?:tu|su|el|la)\s+(?:" + _SUST_S + "|" + _CASO + r"|dinero)\b",
        r"\bproced(?:emos|o|eremos|ere)\s+con\s+(?:tu|su|el|la|este|ese)\s+(?:" + _SUST_S + "|" + _CASO + r"|dinero)\b",
        # recibirás / tendrás / cuenta con
        r"\b(?:tendras|recibiras|obtendras|recuperaras|tendra|recibira|recuperara|recibiran)\s+(?:[a-z]+\s+){0,3}"
        r"(?:" + _SUST_A + r"|dinero|plata|saldo)\b",
        r"\b(?:cuenta|cuentas|contar|cuenten)\s+con\s+(?:ello|eso)\b",
        r"\b(?:cuenta|cuentas|contar|cuenten)\s+con\s+(?:tu|su|el|la|un|una)\s+(?:[a-z]+\s+)?"
        r"(?:" + _SUST_A + r"|dinero|aprobacion|excepcion)\b",
        # familias cerradas tras revisión (T10, ADR-005)
        # 3.ª persona con objeto del usuario: «el equipo aprobó tu reembolso»
        r"\b(?:aprobo|autorizo|concedio|aprobaron|autorizaron|concedieron)\s+(?:ya\s+)?(?:tu|su)\s+(?:"
        + _SUST_S + "|" + _CASO + r")\b",
        # hacer / dar el reembolso o la devolución
        r"\b(?:haremos|hacemos|hare|hago|hice|hicimos|daremos|damos|doy|dare|di|dimos)\s+(?:ya\s+)?"
        r"(?:el|la|tu|su|un|una)\s+(?:[a-z]+\s+)?" + _SUST_S + r"\b",
        r"\bse\s+te\s+(?:hara|haran|hace|hizo)\s+(?:efectiv[oa]\s+)?(?:el|la|tu|su)\s+" + _SUST_S + r"\b",
        r"\bse\s+(?:hara|haran|hizo)\s+(?:efectiv[oa]\s+)?(?:el|la|tu|su)\s+" + _SUST_S + r"\b",
        r"\bse\s+hace\s+efectiv[oa]\s+(?:el|la|tu|su)\s+" + _SUST_S + r"\b",
        r"\bte\s+(?:lo|la)\s+(?:damos|daremos|doy|dare|hacemos|haremos|hago)\b",
        r"\b(?:te|le)\s+(?:abono|abonamos|abonaremos|compensamos|compensaremos|compenso)\b",
        r"\b(?:pagamos|pagaremos|pago|pagare)\s+(?:de\s+vuelta|de\s+nuevo)\b",
        r"\b(?:te|le)\s+(?:pagamos|pagaremos)\s+(?:el\s+)?(?:valor|dinero|total|importe)\b",
        # proceder / estar reembolsando
        r"\b(?:procederemos|procedemos|procedere|procedo|hemos\s+procedido|he\s+procedido|"
        r"vamos\s+a\s+proceder|voy\s+a\s+proceder)\s+a\s+(?:reembols|devolv|devuel|acredit|reintegr|abon|aprob|autoriz)\w*",
        r"\b(?:estaremos|estare|estamos|estoy)\s+(?:ya\s+)?(?:reembolsando|devolviendo|acreditando|reintegrando|abonando)\b"
        + _SIN_LLAMADA,
        r"\b(?:tu|su)\s+" + _SUST_S + r"\s+(?:ya\s+)?(?:procede|procedera|procedio)\b",
        r"\b(?:damos|daremos|doy|dare|dimos)\s+curso\s+a\s+(?:tu|su|el|la)\s+(?:" + _SUST_S + "|" + _CASO + r")\b",
        # el dinero lo tendrás / recibirás
        r"\b(?:dinero|reembolso|plata|saldo|valor)\s+(?:lo\s+|la\s+)?(?:tendras|recibiras|recuperaras|veras)\b",
        r"\b(?:tendras|recibiras|recuperaras)\s+(?:[a-z]+\s+){0,3}en\s+(?:tu|su)\s+(?:cuenta|tarjeta)\b",
    )),
    (R_GARANTIA, _c(
        # «te lo garantizo / te lo aseguro / te lo prometo»; «te garantizo»
        r"\b" + _CLP + r"\s+(?:" + _CLP + r"\s+)?(?:garantizo|garantizamos|garantizaremos|garantizare|garantice)\b",
        r"\b" + _CLP + r"\s+" + _CLP + r"\s+(?:aseguro|aseguramos|aseguraremos|asegurare|prometo|prometemos|"
        r"prometi|prometimos|juro|juramos|jurare)\b",
        r"\b(?:garantizo|garantizamos|garantizaremos|garantizare|garantice)\s+que\b",
        # «te aseguro/prometo (que) ...» solo es compromiso si el objeto es un resultado
        r"\b(?:aseguro|aseguramos|aseguraremos|asegurare|prometo|prometemos|prometi|prometimos|juro|juramos)\s+"
        r"(?:que\s+)?(?:[a-z]+\s+){0,6}?" + _OBJ_RES + r"\b",
        r"\b(?:garantizo|garantizamos|garantizaremos|garantice|aseguro|aseguramos|aseguraremos)\s+"
        r"(?:[a-z]+\s+){0,2}(?:" + _SUST_A + r"|resultado|exito|solucion|aprobacion|dinero)\b",
        r"\b(?:mi|nuestra)\s+palabra\b",
        r"\b(?:dejo|dejamos)\s+(?:[a-z]+\s+){0,2}garantizad[oa]s?\b",
        r"\b(?:tienes|tiene|tendras|tendra)\s+(?:ya\s+)?garantizad[oa]s?\b",
        r"\b(?:queda|quedan|quedo|quedara|sera|seran|esta|estan|estara)\s+(?:ya\s+)?garantizad[oa]s?\b" + _NO_COBERTURA,
        r"\b(?:" + _SUST_A + r"|resultado|dinero|exito|solucion|aprobacion)\s+"
        r"(?:(?:ya\s+)?(?:esta|estan|sera|seran|es|va\s+a\s+estar)\s+)?(?:ya\s+)?garantizad[oa]s?\b" + _NO_COBERTURA,
        r"\bgarantizad[oa]s?\s+(?:ya\s+)?(?:tu|su|el|la)\s+(?:" + _SUST_A + r"|resultado|exito|solucion|dinero|aprobacion)\b",
        r"\b(?:puedo|podemos|voy\s+a|vamos\s+a)\s+garantizar",
        r"\bgarantia\s+(?:ya\s+)?(?:esta\s+|queda\s+|sera\s+)?(?:asegurad|confirmad|aprobad|concedid|otorgad|garantizad)[oa]s?\b",
        r"\bsegur[oa]s?\s+(?:de\s+)?que\s+(?:[a-z]+\s+){0,6}?" + _OBJ_RES + r"\b",
        r"\bes\s+un\s+hecho\b",
        r"\bpor\s+hecho\b",
    )),
    (R_EXCEPCION, _c(
        r"\b" + _CLS + _EXC_VERBO + r"\s+(?:(?:te|le|les)\s+)?(?:[a-z]+\s+){0,2}excepcion(?:es)?\b",
        r"\bexcepcion\s+(?:para\s+ti|en\s+tu\s+caso|a\s+tu\s+favor)\b",
        r"\bexcepcion\b.{0,30}\b(?:aprobad|concedid|autorizad)[a-z]*",
        r"\bexcepcionalmente\b",
        r"\b(?:de\s+(?:manera|forma|modo)\s+excepcional|por\s+excepcion)\b",
        r"\bpor\s+(?:esta|unica)\s+vez\b",
        r"\bsolo\s+(?:por\s+)?esta\s+vez\b",
        r"\b" + _CLS + _SALTO + r"\s+(?:[a-z]+\s+){0,3}"
        r"(?:politicas?|normas?|reglas?|requisitos?|plazos?|condiciones|limites?|umbral|restricciones?)\b",
    )),
]


def detectar_compromisos(respuesta: str) -> list[tuple[str, str]]:
    """Devuelve (regla, frase) por cada regla violada (la primera coincidencia de cada una)."""
    texto = _neutralizar(_normalizar(respuesta))
    halladas: list[tuple[str, str]] = []
    for regla, patron in _REGLAS:
        m = patron.search(texto)
        if m:
            halladas.append((regla, m.group().strip(" ,;:-")))
    return halladas
