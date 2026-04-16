import json
from pathlib import Path

import chromadb
import pytest

from armazenamento.vetorizador import Vetorizador


PROMOCAO_COM_TODOS_CAMPOS = {
    "produto": {"nome": "Leite Piracanjuba", "quantidade": 1.0, "unidade": "L", "categoria": "laticinios"},
    "preco_promocional": 4.99,
    "preco_original": 6.49,
    "desconto_percentual": 23.1,
    "validade_inicio": "01/04/2026",
    "validade_fim": "07/04/2026",
    "mercado": "mercado_teste",
    "localizacao": "Cidade Teste",
    "periodo": "2026-04-01",
    "pagina_origem": "encarte_1_pagina_1",
}

PROMOCAO_COM_CAMPOS_NULOS = {
    "produto": {"nome": "Arroz Tio Joao", "quantidade": 5.0, "unidade": "kg", "categoria": "mercearia"},
    "preco_promocional": 22.90,
    "preco_original": None,
    "desconto_percentual": None,
    "validade_inicio": "01/04/2026",
    "validade_fim": "07/04/2026",
    "mercado": "mercado_teste",
    "localizacao": None,
    "periodo": "2026-04-01",
    "pagina_origem": "encarte_1_pagina_1",
}


def _embedding_falso(texto: str) -> list[float]:
    """Stub de embedding: vetor constante de 768 dimensões (igual ao nomic-embed-text)."""
    return [0.1] * 768


@pytest.fixture
def cliente_chromadb():
    """ChromaDB em memória para testes — não persiste em disco."""
    return chromadb.EphemeralClient()


def _criar_json(tmp_path: Path, promocoes: list[dict]) -> Path:
    arquivo = tmp_path / "promocoes_consolidadas.json"
    arquivo.write_text(json.dumps(promocoes, ensure_ascii=False))
    return arquivo


def test_criar_texto_inclui_campos_obrigatorios():
    """_criar_texto deve incluir nome do produto, preço e mercado."""
    v = Vetorizador("m", "p", _gerar_embedding=_embedding_falso, _client=chromadb.EphemeralClient())
    texto = v._criar_texto(PROMOCAO_COM_TODOS_CAMPOS)
    assert "Leite Piracanjuba" in texto
    assert "R$4.99" in texto
    assert "mercado_teste" in texto
    assert "23.1%" in texto
    assert "Cidade Teste" in texto


def test_criar_texto_omite_campos_nulos():
    """_criar_texto não deve incluir campos com valor None."""
    v = Vetorizador("m", "p", _gerar_embedding=_embedding_falso, _client=chromadb.EphemeralClient())
    texto = v._criar_texto(PROMOCAO_COM_CAMPOS_NULOS)
    assert "Desconto:" not in texto
    assert "Localização:" not in texto


def test_vetorizar_retorna_contagem(tmp_path, cliente_chromadb):
    """vetorizar() deve retornar o número de promoções inseridas."""
    caminho = _criar_json(tmp_path, [PROMOCAO_COM_TODOS_CAMPOS, PROMOCAO_COM_CAMPOS_NULOS])
    v = Vetorizador("mercado_teste", "2026-04-01", _gerar_embedding=_embedding_falso, _client=cliente_chromadb)
    contagem = v.vetorizar(caminho)
    assert contagem == 2


def test_vetorizar_idempotente(tmp_path, cliente_chromadb):
    """Chamar vetorizar() duas vezes não deve duplicar promoções na coleção."""
    caminho = _criar_json(tmp_path, [PROMOCAO_COM_TODOS_CAMPOS, PROMOCAO_COM_CAMPOS_NULOS])
    v = Vetorizador("mercado_teste", "2026-04-01", _gerar_embedding=_embedding_falso, _client=cliente_chromadb)
    v.vetorizar(caminho)
    v.vetorizar(caminho)
    colecao = cliente_chromadb.get_collection("promocoes")
    assert colecao.count() == 2


def test_vetorizar_json_vazio(tmp_path, cliente_chromadb):
    """vetorizar() com JSON vazio deve retornar 0 e não inserir nada."""
    caminho = _criar_json(tmp_path, [])
    v = Vetorizador("mercado_teste", "2026-04-01", _gerar_embedding=_embedding_falso, _client=cliente_chromadb)
    contagem = v.vetorizar(caminho)
    assert contagem == 0


def test_metadados_sem_valores_nulos(tmp_path, cliente_chromadb):
    """Nenhum metadado armazenado no ChromaDB pode conter None (não suportado)."""
    caminho = _criar_json(tmp_path, [PROMOCAO_COM_CAMPOS_NULOS])
    v = Vetorizador("mercado_teste", "2026-04-01", _gerar_embedding=_embedding_falso, _client=cliente_chromadb)
    v.vetorizar(caminho)
    colecao = cliente_chromadb.get_collection("promocoes")
    resultado = colecao.get()
    for meta in resultado["metadatas"]:
        for valor in meta.values():
            assert valor is not None, f"Metadado com valor None encontrado: {meta}"
