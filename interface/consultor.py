import requests
from pathlib import Path
from typing import Callable, Optional

import chromadb


OLLAMA_URL_EMBEDDINGS = "http://localhost:11434/api/embeddings"
OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_LLM_PADRAO = "qwen2.5:7b"
NOME_COLECAO = "promocoes"
TOP_K_PADRAO = 5
TIMEOUT_EMBEDDING = 60
TIMEOUT_LLM = 120

PROMPT_SISTEMA = (
    "Você é um assistente especializado em promoções de supermercados. "
    "Com base nas promoções abaixo, responda a pergunta do usuário em português, "
    "citando mercado, preço e validade quando disponíveis. "
    "Se não houver promoções relevantes, diga isso claramente.\n\n"
    "Promoções disponíveis:\n{contexto}"
)


class ConsultorError(Exception):
    pass


class Consultor:
    def __init__(
        self,
        pasta_chromadb: Path = Path("dados") / "chromadb",
        modelo_llm: str = MODELO_LLM_PADRAO,
        top_k: int = TOP_K_PADRAO,
        _gerar_embedding: Optional[Callable[[str], list[float]]] = None,
        _chamar_llm: Optional[Callable[[str, str], str]] = None,
        _client=None,
    ) -> None:
        self._modelo_llm = modelo_llm
        self._top_k = top_k
        self._gerar_embedding = _gerar_embedding or self._embedding_ollama
        self._chamar_llm = _chamar_llm or self._llm_ollama
        cliente = _client or chromadb.PersistentClient(path=str(pasta_chromadb))
        self._colecao = cliente.get_or_create_collection(
            NOME_COLECAO,
            metadata={"hnsw:space": "cosine"},
        )

    def consultar(self, pergunta: str) -> str:
        # implementado na Task 3
        raise NotImplementedError

    def _buscar_promocoes(self, pergunta: str) -> list[dict]:
        count = self._colecao.count()
        if count == 0:
            return []
        embedding = self._gerar_embedding(pergunta)
        resultados = self._colecao.query(
            query_embeddings=[embedding],
            n_results=min(self._top_k, count),
        )
        documentos = resultados["documents"][0]
        metadados = resultados["metadatas"][0]
        return [
            {"documento": doc, "metadados": meta}
            for doc, meta in zip(documentos, metadados)
        ]

    def _formatar_contexto(self, resultados: list[dict]) -> str:
        if not resultados:
            return "Nenhuma promoção encontrada."
        linhas = [f"{i + 1}. {r['documento']}" for i, r in enumerate(resultados)]
        return "\n".join(linhas)

    def _gerar_resposta(self, pergunta: str, contexto: str) -> str:
        # implementado na Task 3
        raise NotImplementedError

    def _embedding_ollama(self, texto: str) -> list[float]:
        # implementado na Task 3
        raise NotImplementedError

    def _llm_ollama(self, sistema: str, pergunta: str) -> str:
        # implementado na Task 3
        raise NotImplementedError
