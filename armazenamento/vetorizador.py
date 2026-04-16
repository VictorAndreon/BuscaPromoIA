import json
from pathlib import Path
from typing import Callable, Optional

import requests

import chromadb


OLLAMA_URL_EMBEDDINGS = "http://localhost:11434/api/embeddings"
MODELO_EMBEDDING = "nomic-embed-text"
NOME_COLECAO = "promocoes"
TIMEOUT_EMBEDDING = 60


class VetorizadorError(Exception):
    pass


class Vetorizador:
    def __init__(
        self,
        mercado: str,
        periodo: str,
        pasta_chromadb: Path = Path("dados") / "chromadb",
        _gerar_embedding: Optional[Callable[[str], list[float]]] = None,
        _client=None,
    ) -> None:
        self._mercado = mercado
        self._periodo = periodo
        self._gerar_embedding = _gerar_embedding or self._embedding_ollama
        cliente = _client or chromadb.PersistentClient(path=str(pasta_chromadb))
        self._colecao = cliente.get_or_create_collection(
            NOME_COLECAO,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[vetorizador] Inicializado | mercado={mercado} | periodo={periodo} | colecao={NOME_COLECAO}")

    def vetorizar(self, caminho_json: Path) -> int:
        dados = json.loads(caminho_json.read_text())
        if not dados:
            print("[vetorizador] JSON vazio, nada a vetorizar.")
            return 0

        ids: list[str] = []
        textos: list[str] = []
        embeddings: list[list[float]] = []
        metadados: list[dict] = []

        for i, promocao in enumerate(dados):
            texto = self._criar_texto(promocao)
            ids.append(self._criar_id(promocao, i))
            textos.append(texto)
            embeddings.append(self._gerar_embedding(texto))
            metadados.append(self._extrair_metadados(promocao))
            nome = promocao.get("produto", {}).get("nome", "?")
            print(f"[vetorizador] ({i + 1}/{len(dados)}) {nome}")

        self._colecao.upsert(
            ids=ids,
            documents=textos,
            embeddings=embeddings,
            metadatas=metadados,
        )
        print(f"[vetorizador] {len(dados)} promoção(ões) vetorizadas na coleção '{NOME_COLECAO}'")
        return len(dados)

    def _criar_texto(self, promocao: dict) -> str:
        produto = promocao.get("produto", {})
        nome = produto.get("nome", "")
        categoria = produto.get("categoria") or "outros"
        quantidade = produto.get("quantidade")
        unidade = produto.get("unidade")
        preco = promocao.get("preco_promocional", "")
        desconto = promocao.get("desconto_percentual")
        mercado = promocao.get("mercado", "")
        localizacao = promocao.get("localizacao")
        periodo = promocao.get("periodo", "")
        validade_fim = promocao.get("validade_fim")

        partes = [f"Produto: {nome}", f"Categoria: {categoria}"]
        if quantidade is not None and unidade:
            partes.append(f"Quantidade: {quantidade} {unidade}")
        quantidade_minima = promocao.get("quantidade_minima")
        partes.append(f"Preço: R${preco}")
        if quantidade_minima is not None:
            partes.append(f"Condição: leve {quantidade_minima} unidades")
        if desconto is not None:
            partes.append(f"Desconto: {desconto}%")
        partes.append(f"Mercado: {mercado}")
        if localizacao:
            partes.append(f"Localização: {localizacao}")
        partes.append(f"Período: {periodo}")
        if validade_fim:
            partes.append(f"Válido até: {validade_fim}")

        return " | ".join(partes)

    def _criar_id(self, promocao: dict, indice: int) -> str:
        pagina = promocao.get("pagina_origem", "desconhecido")
        return f"{self._mercado}_{self._periodo}_{pagina}_{indice}"

    def _extrair_metadados(self, promocao: dict) -> dict:
        produto = promocao.get("produto", {})
        campos = {
            "mercado": promocao.get("mercado", ""),
            "periodo": promocao.get("periodo", ""),
            "localizacao": promocao.get("localizacao"),
            "nome_produto": produto.get("nome", ""),
            "categoria": produto.get("categoria"),
            "preco_promocional": promocao.get("preco_promocional"),
            "preco_original": promocao.get("preco_original"),
            "desconto_percentual": promocao.get("desconto_percentual"),
            "quantidade_minima": promocao.get("quantidade_minima"),
            "pagina_origem": promocao.get("pagina_origem", ""),
            "validade_fim": promocao.get("validade_fim"),
        }
        return {k: v for k, v in campos.items() if v is not None}

    def _embedding_ollama(self, texto: str) -> list[float]:
        payload = {"model": MODELO_EMBEDDING, "prompt": texto}
        try:
            resposta = requests.post(OLLAMA_URL_EMBEDDINGS, json=payload, timeout=TIMEOUT_EMBEDDING)
            resposta.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise VetorizadorError(
                f"Não foi possível conectar ao Ollama em {OLLAMA_URL_EMBEDDINGS}. "
                "Verifique se o Ollama está rodando com 'ollama pull nomic-embed-text'."
            )
        except requests.exceptions.HTTPError as erro:
            raise VetorizadorError(f"Erro HTTP ao chamar Ollama embeddings: {erro}")
        return resposta.json()["embedding"]
