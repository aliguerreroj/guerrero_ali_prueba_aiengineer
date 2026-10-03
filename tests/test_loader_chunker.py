"""Pruebas del loader y chunker de documentos (sin red)."""

import hashlib
from pathlib import Path

import pytest

from tiendahogar_agent.config import Settings
from tiendahogar_agent.documentos import (
    ErrorDocumentos,
    FileSystemDocumentSource,
    chunk_fixed,
    chunk_none,
    chunk_recursive,
    trocear,
)
from tiendahogar_agent.puertos import DocumentSource

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


def _hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DOCS.glob("*"))}


def _largo(n=12):
    return "\n\n".join(f"Párrafo {i}. " + "palabra " * 30 for i in range(n))


def test_cumple_puerto():
    assert isinstance(FileSystemDocumentSource(DOCS), DocumentSource)


def test_cinco_docs_cinco_chunks_con_auto():
    chunks = FileSystemDocumentSource(DOCS, estrategia="auto").cargar()
    assert len(chunks) == 5
    assert [c.doc_id for c in chunks] == ["doc1", "doc2", "doc3", "doc4", "doc5"]
    assert chunks[0].metadatos["titulo"] == "Doc 1 — Política de garantía"
    assert chunks[4].metadatos["titulo"] == "Doc 5 — Canales de contacto"
    for c in chunks:
        assert c.texto.strip()
        assert c.metadatos["posicion"] == 0 and c.metadatos["total_chunks"] == 1
        assert c.metadatos["fuente"].startswith(c.doc_id + "_")


def test_desde_settings():
    s = Settings(chunk_strategy="none")
    assert len(FileSystemDocumentSource.desde_settings(DOCS, s).cargar()) == 5


def test_archivos_fuente_no_cambian():
    antes = _hashes()
    for est in ("none", "fixed", "recursive", "auto"):
        FileSystemDocumentSource(DOCS, estrategia=est, tamano=100, solape=10).cargar()
    assert _hashes() == antes


def test_none_un_chunk():
    assert chunk_none("  hola\n\nmundo  ") == ["hola\n\nmundo"]
    assert chunk_none("   ") == []


def test_fixed_tamano_y_solape():
    texto = "".join(chr(97 + i % 26) for i in range(100))
    ch = chunk_fixed(texto, 30, 10)
    assert all(len(c) <= 30 for c in ch)
    assert ch[0][-10:] == ch[1][:10]
    assert ch[0] == texto[:30] and ch[1] == texto[20:50]
    assert texto.endswith(ch[-1])


def test_fixed_sin_solape_cubre_todo():
    texto = "x" * 95
    assert "".join(chunk_fixed(texto, 30, 0)) == texto


def test_recursive_respeta_tamano_y_no_vacios():
    ch = chunk_recursive(_largo(), 300, 40)
    assert len(ch) > 3
    assert all(c and len(c) <= 300 for c in ch)
    assert ch[0].startswith("Párrafo 0")


def test_recursive_solape():
    texto = "\n".join(f"Línea número {i} del texto." for i in range(40))
    ch = chunk_recursive(texto, 200, 60)
    assert len(ch) > 2 and all(len(c) <= 200 for c in ch)
    assert all(ch[i + 1].splitlines()[0] in ch[i] for i in range(len(ch) - 1))


def test_recursive_palabra_gigante():
    ch = chunk_recursive("a" * 250, 100, 0)
    assert [len(c) for c in ch] == [100, 100, 50]


def test_texto_vacio_nunca_chunk_vacio():
    for est in ("none", "fixed", "recursive", "auto"):
        assert trocear("  \n\n ", est, 10, 2, 5) == []


def test_auto_corto_none_largo_recursive():
    corto = "texto corto"
    assert trocear(corto, "auto", 5, 1, 100) == [corto]
    assert len(trocear(_largo(), "auto", 300, 20, 100)) > 1


def test_estrategia_invalida():
    with pytest.raises(ValueError, match="Estrategia"):
        trocear("x", "magica")
    with pytest.raises(ValueError, match="Estrategia"):
        FileSystemDocumentSource(DOCS, estrategia="magica")


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        chunk_fixed("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_recursive("abc", 10, 10)


def test_documento_largo_sintetico(tmp_path):
    (tmp_path / "doc9_largo.md").write_text("# Título largo\n\n" + _largo(), encoding="utf-8")
    ch = FileSystemDocumentSource(
        tmp_path, estrategia="auto", tamano=300, solape=30, umbral_corto=200
    ).cargar()
    assert len(ch) > 3
    assert {c.doc_id for c in ch} == {"doc9"}
    assert {c.metadatos["titulo"] for c in ch} == {"Título largo"}
    assert [c.metadatos["posicion"] for c in ch] == list(range(len(ch)))
    assert {c.metadatos["total_chunks"] for c in ch} == {len(ch)}
    assert all(c.texto.strip() for c in ch)


def test_crlf_equivale_a_lf(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    cuerpo = "# Título\n\nUna línea.\nOtra línea.\n"
    (lf / "doc1_a.md").write_bytes(cuerpo.encode("utf-8"))
    (crlf / "doc1_a.md").write_bytes(cuerpo.replace("\n", "\r\n").encode("utf-8"))
    a = FileSystemDocumentSource(lf).cargar()
    b = FileSystemDocumentSource(crlf).cargar()
    assert a == b and "\r" not in b[0].texto


def test_sin_h1_deriva_titulo_del_nombre(tmp_path):
    (tmp_path / "doc2_plazos_de_pago.md").write_text("Solo texto.", encoding="utf-8")
    c = FileSystemDocumentSource(tmp_path).cargar()[0]
    assert c.doc_id == "doc2" and c.metadatos["titulo"] == "plazos de pago"


def test_errores_claros(tmp_path):
    with pytest.raises(ErrorDocumentos, match="no existe"):
        FileSystemDocumentSource(tmp_path / "nada").cargar()
    with pytest.raises(ErrorDocumentos, match="No se encontraron"):
        FileSystemDocumentSource(tmp_path).cargar()
    (tmp_path / "doc1_v.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(ErrorDocumentos, match="vacío"):
        FileSystemDocumentSource(tmp_path).cargar()


def test_bom_utf8_tolerado(tmp_path):
    (tmp_path / "doc1_prueba.md").write_bytes(b"\xef\xbb\xbf# Titulo con BOM\n\nContenido.\n")
    chunks = FileSystemDocumentSource(tmp_path).cargar()
    assert chunks[0].metadatos["titulo"] == "Titulo con BOM"
    assert all("\ufeff" not in c.texto for c in chunks)
