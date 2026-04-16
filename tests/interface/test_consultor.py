import chromadb
import pytest

from interface.consultor import Consultor


EMBEDDING_FALSO = [0.1] * 768


def _embedding_falso(texto: str) -> list[float]:
    return EMBEDDING_FALSO


def _llm_falso(sistema: str, pergunta: str) -> str:
    return f"Resposta para: {pergunta}"


@pytest.fixture
def cliente_vazio():
    """ChromaDB em memória com coleção 'promocoes' criada mas sem documentos."""
    cliente = chromadb.EphemeralClient()
    cliente.get_or_create_collection("promocoes", metadata={"hnsw:space": "cosine"})
    return cliente


@pytest.fixture
def cliente_com_dados():
    """ChromaDB em memória com 2 promoções pré-indexadas."""
    cliente = chromadb.EphemeralClient()
    colecao = cliente.get_or_create_collection("promocoes", metadata={"hnsw:space": "cosine"})
    colecao.add(
        ids=["id1", "id2"],
        documents=[
            "Produto: Arroz Tio Joao | Categoria: mercearia | Preço: R$22.9 | Mercado: mercado_teste | Período: 2026-04-01",
            "Produto: Leite Piracanjuba | Categoria: laticinios | Preço: R$4.99 | Mercado: mercado_teste | Período: 2026-04-01",
        ],
        embeddings=[EMBEDDING_FALSO, EMBEDDING_FALSO],
        metadatas=[
            {"mercado": "mercado_teste", "nome_produto": "Arroz Tio Joao", "periodo": "2026-04-01"},
            {"mercado": "mercado_teste", "nome_produto": "Leite Piracanjuba", "periodo": "2026-04-01"},
        ],
    )
    return cliente


@pytest.fixture
def consultor_sem_dados(cliente_vazio):
    """Consultor pronto com coleção vazia — para testar métodos puros."""
    return Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_falso,
        _client=cliente_vazio,
    )


def test_consultor_instancia_sem_erro(cliente_vazio):
    """Consultor deve ser instanciado sem erros com dependências injetadas."""
    consultor = Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_falso,
        _client=cliente_vazio,
    )
    assert consultor is not None
