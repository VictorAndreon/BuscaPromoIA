import chromadb
import pytest

from interface.consultor import Consultor


EMBEDDING_FALSO = [0.1] * 768


def _embedding_falso(texto: str) -> list[float]:
    return EMBEDDING_FALSO


def _llm_falso(sistema: str, pergunta: str) -> str:
    return f"Resposta para: {pergunta}"


@pytest.fixture(scope="function")
def cliente_vazio():
    """ChromaDB em memória com coleção 'promocoes' criada mas sem documentos."""
    cliente = chromadb.EphemeralClient()
    try:
        cliente.delete_collection("promocoes")
    except Exception:
        pass
    cliente.get_or_create_collection("promocoes", metadata={"hnsw:space": "cosine"})
    return cliente


@pytest.fixture(scope="function")
def cliente_com_dados():
    """ChromaDB em memória com 2 promoções pré-indexadas."""
    cliente = chromadb.EphemeralClient()
    try:
        cliente.delete_collection("promocoes")
    except Exception:
        pass
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
    """Consultor deve armazenar as dependências injetadas no construtor."""
    consultor = Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_falso,
        _client=cliente_vazio,
    )
    assert consultor._gerar_embedding is _embedding_falso
    assert consultor._chamar_llm is _llm_falso


def test_buscar_promocoes_retorna_lista(cliente_com_dados):
    """_buscar_promocoes deve retornar lista de dicts com 'documento' e 'metadados'."""
    consultor = Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_falso,
        _client=cliente_com_dados,
    )
    resultados = consultor._buscar_promocoes("qual o preço do arroz?")
    assert len(resultados) == 2
    assert "documento" in resultados[0]
    assert "metadados" in resultados[0]


def test_buscar_promocoes_colecao_vazia(consultor_sem_dados):
    """_buscar_promocoes com coleção vazia deve retornar lista vazia sem lançar erro."""
    resultados = consultor_sem_dados._buscar_promocoes("arroz")
    assert resultados == []


def test_formatar_contexto_numera_resultados(consultor_sem_dados):
    """_formatar_contexto deve numerar cada promoção no texto de contexto."""
    resultados = [
        {"documento": "Produto: Arroz | Preço: R$22.9", "metadados": {}},
        {"documento": "Produto: Leite | Preço: R$4.99", "metadados": {}},
    ]
    contexto = consultor_sem_dados._formatar_contexto(resultados)
    assert "1. Produto: Arroz" in contexto
    assert "2. Produto: Leite" in contexto


def test_formatar_contexto_lista_vazia(consultor_sem_dados):
    """_formatar_contexto com lista vazia deve informar ausência de promoções."""
    contexto = consultor_sem_dados._formatar_contexto([])
    assert "Nenhuma promoção encontrada" in contexto


def test_gerar_resposta_delega_ao_llm(consultor_sem_dados):
    """_gerar_resposta deve passar sistema formatado e a pergunta original ao _chamar_llm."""
    chamadas: list[dict] = []

    def _llm_captura(sistema: str, pergunta: str) -> str:
        chamadas.append({"sistema": sistema, "pergunta": pergunta})
        return "resposta capturada"

    consultor_sem_dados._chamar_llm = _llm_captura

    contexto = "1. Produto: Arroz | Preço: R$22.9"
    resposta = consultor_sem_dados._gerar_resposta("arroz barato?", contexto)

    assert resposta == "resposta capturada"
    assert len(chamadas) == 1
    assert chamadas[0]["pergunta"] == "arroz barato?"
    assert "1. Produto: Arroz" in chamadas[0]["sistema"]


def test_consultar_retorna_resposta_llm(cliente_com_dados):
    """consultar() deve retornar a string gerada pelo LLM."""
    consultor = Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_falso,
        _client=cliente_com_dados,
    )
    resposta = consultor.consultar("qual o menor preço de arroz?")
    assert "qual o menor preço de arroz?" in resposta


def test_consultar_colecao_vazia_passa_mensagem_ao_llm(cliente_vazio):
    """consultar() com coleção vazia deve enviar 'Nenhuma promoção encontrada' ao LLM."""
    sistemas_capturados: list[str] = []

    def _llm_captura(sistema: str, pergunta: str) -> str:
        sistemas_capturados.append(sistema)
        return "sem promoções relevantes"

    consultor = Consultor(
        _gerar_embedding=_embedding_falso,
        _chamar_llm=_llm_captura,
        _client=cliente_vazio,
    )
    consultor.consultar("arroz")
    assert "Nenhuma promoção encontrada" in sistemas_capturados[0]


def test_cli_modulo_importavel():
    """interface.cli deve ser importável e expor a função main."""
    from interface import cli
    assert callable(cli.main)
