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
        resultados = self._buscar_promocoes(pergunta)
        contexto = self._formatar_contexto(resultados)
        return self._gerar_resposta(pergunta, contexto)

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
        sistema = PROMPT_SISTEMA.format(contexto=contexto)
        return self._chamar_llm(sistema, pergunta)

    def _embedding_ollama(self, texto: str) -> list[float]:
        payload = {"model": MODELO_EMBEDDING, "prompt": texto}
        try:
            resposta = requests.post(OLLAMA_URL_EMBEDDINGS, json=payload, timeout=TIMEOUT_EMBEDDING)
            resposta.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConsultorError(
                f"Não foi possível conectar ao Ollama em {OLLAMA_URL_EMBEDDINGS}. "
                "Verifique se o Ollama está rodando com 'ollama pull nomic-embed-text'."
            )
        except requests.exceptions.Timeout:
            raise ConsultorError(
                f"Timeout ao aguardar resposta do Ollama em {OLLAMA_URL_EMBEDDINGS}. "
                "Verifique se o modelo está carregado."
            )
        except requests.exceptions.HTTPError as erro:
            raise ConsultorError(f"Erro HTTP ao chamar Ollama embeddings: {erro}")
        return resposta.json()["embedding"]

    def _llm_ollama(self, sistema: str, pergunta: str) -> str:
        payload = {
            "model": self._modelo_llm,
            "system": sistema,
            "prompt": pergunta,
            "stream": False,
        }
        try:
            resposta = requests.post(OLLAMA_URL_GENERATE, json=payload, timeout=TIMEOUT_LLM)
            resposta.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConsultorError(
                f"Não foi possível conectar ao Ollama em {OLLAMA_URL_GENERATE}. "
                f"Verifique se o Ollama está rodando com 'ollama pull {self._modelo_llm}'."
            )
        except requests.exceptions.Timeout:
            raise ConsultorError(
                f"Timeout ao aguardar resposta do Ollama em {OLLAMA_URL_GENERATE}. "
                f"Verifique se o modelo '{self._modelo_llm}' está carregado."
            )
        except requests.exceptions.HTTPError as erro:
            raise ConsultorError(f"Erro HTTP ao chamar Ollama generate: {erro}")
        return resposta.json()["response"]
