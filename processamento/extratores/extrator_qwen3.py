import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from coleta.modelos.promocao import Produto, Promocao


OLLAMA_URL = "http://localhost:11434/api/chat"
MODELO_QWEN3 = "qwen3-vl:8b"
TIMEOUT_SEGUNDOS = 300
LARGURA_MAXIMA_MODELO = 896

PROMPT_EXTRACAO = """You are analyzing a Brazilian supermarket promotional flyer image.
Extract every product promotion visible in the image.

Return ONLY a valid JSON array, no explanations, no markdown, no extra text.
Use dot as decimal separator for prices (e.g. 4.99).

Rules:
- "validade_inicio" and "validade_fim": use the promotion period shown (e.g. "validas de 23/03 a 02/04/2025" -> inicio="23/03/2025", fim="02/04/2025"). Use null if not shown.
- "preco_promocional" is required. If you cannot read the price clearly, skip the product entirely.
- "preco_original" is usually shown in small text above or crossed out near the promotional price. Look carefully for it.
- "quantidade" and "unidade": look for small text near the product name (e.g. "500g", "1kg", "2L"). Use null if not shown.
- Do not invent or guess products. Only extract what is clearly visible.

[
  {
    "nome": "product name exactly as shown",
    "categoria": "one of: laticinios, bebidas, carnes, hortigranjeiro, higiene, limpeza, mercearia, outros",
    "quantidade": null or number,
    "unidade": null or "kg" or "L" or "un" or "g" or "ml",
    "preco_promocional": number with dot decimal (required),
    "preco_original": null or number,
    "desconto_percentual": null or number,
    "validade_inicio": null or "DD/MM/YYYY",
    "validade_fim": null or "DD/MM/YYYY"
  }
]"""


class ExtratorQwen3Error(Exception):
    pass


class ExtratorQwen3:
    def __init__(self, mercado: str, periodo: str, localizacao: Optional[str] = None) -> None:
        self._mercado = mercado
        self._periodo = periodo
        self._localizacao = localizacao
        self._pasta_json = self._criar_pasta_json()
        print(f"[ExtratorQwen3] Inicializado | mercado={mercado} | periodo={periodo} | modelo={MODELO_QWEN3}")

    def extrair(self, caminho_imagem: Path) -> list[Promocao]:
        print(f"\n[extracao] Iniciando: {caminho_imagem.name}")
        imagem_base64 = self._converter_imagem_para_base64(caminho_imagem)
        texto_resposta = self._chamar_qwen3(imagem_base64)
        dados = self._parsear_resposta(texto_resposta)
        promocoes = self._converter_para_promocoes(dados, caminho_imagem.stem)
        print(f"[extracao] Concluida: {len(promocoes)} promocao(es) encontrada(s)")
        return promocoes

    def salvar(self, promocoes: list[Promocao], nome_arquivo: str) -> Path:
        caminho = self._pasta_json / f"{nome_arquivo}.json"
        dados = [self._promocao_para_dict(p) for p in promocoes]
        caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2))
        print(f"[ExtratorQwen3] {len(promocoes)} promocoes salvas em {caminho}")
        return caminho

    def _converter_imagem_para_base64(self, caminho: Path) -> str:
        imagem = Image.open(caminho)
        largura_original, altura_original = imagem.size
        print(f"[imagem] Lida: {caminho.name} ({caminho.stat().st_size / 1024:.1f} KB) | {largura_original}x{altura_original}px")

        if largura_original > LARGURA_MAXIMA_MODELO:
            proporcao = LARGURA_MAXIMA_MODELO / largura_original
            nova_altura = int(altura_original * proporcao)
            imagem = imagem.resize((LARGURA_MAXIMA_MODELO, nova_altura), Image.LANCZOS)
            print(f"[imagem] Redimensionada para {LARGURA_MAXIMA_MODELO}x{nova_altura}px para envio ao modelo")

        buffer = io.BytesIO()
        imagem.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _chamar_qwen3(self, imagem_base64: str) -> str:
        payload = {
            "model": MODELO_QWEN3,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_EXTRACAO,
                    "images": [imagem_base64],
                }
            ],
            "stream": True,
            "options": {
                "temperature": 0,
                "num_predict": 8192,
                "repeat_penalty": 1.3,
                "repeat_last_n": 64,
            },
        }
        print(f"[qwen3] Enviando imagem para o modelo '{MODELO_QWEN3}' (timeout={TIMEOUT_SEGUNDOS}s)...")
        print("[qwen3] Thinking: ", end="", flush=True)
        try:
            resposta = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEGUNDOS, stream=True)
            resposta.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ExtratorQwen3Error(
                f"Nao foi possivel conectar ao Ollama em {OLLAMA_URL}. "
                "Verifique se o Ollama esta rodando."
            )
        except requests.exceptions.Timeout:
            raise ExtratorQwen3Error(
                f"Timeout apos {TIMEOUT_SEGUNDOS}s aguardando resposta do Ollama. "
                "Tente aumentar TIMEOUT_SEGUNDOS ou usar um modelo menor."
            )
        except requests.exceptions.HTTPError as erro:
            detalhe = ""
            try:
                detalhe = erro.response.json().get("error", erro.response.text)
            except Exception:
                pass
            raise ExtratorQwen3Error(f"Erro HTTP ao chamar Ollama: {erro} | Detalhe: {detalhe}")

        texto_completo = []
        em_thinking = True
        for linha in resposta.iter_lines():
            if linha:
                chunk = json.loads(linha)
                msg = chunk.get("message", {})
                thinking = msg.get("thinking", "")
                content = msg.get("content", "")

                if thinking:
                    print(f"{thinking}", end="", flush=True)
                if content:
                    if em_thinking:
                        print(f"\n[qwen3] Gerando resposta: ", end="", flush=True)
                        em_thinking = False
                    print(content, end="", flush=True)
                    texto_completo.append(content)

                if chunk.get("done"):
                    break

        print(f"\n[qwen3] Resposta concluida ({sum(len(t) for t in texto_completo)} caracteres)")
        return "".join(texto_completo)

    def _parsear_resposta(self, texto: str) -> list[dict]:
        print("[json] Parseando resposta...")
        texto = re.sub(r"```(?:json)?", "", texto).strip()

        try:
            dados = json.loads(texto)
            if isinstance(dados, list):
                print(f"[json] Parseado com sucesso ({len(dados)} item(ns))")
                return dados
            print(f"[json] JSON valido mas tipo inesperado: {type(dados).__name__} | valor: {dados}")
        except json.JSONDecodeError:
            pass

        print("[json] Tentando extrair JSON parcial com regex...")
        match = re.search(r"\[.*\]", texto, re.DOTALL)
        if match:
            try:
                dados = json.loads(match.group())
                print(f"[json] Extraido via regex ({len(dados)} item(ns))")
                return dados
            except json.JSONDecodeError:
                pass

        print(f"[json] ERRO: nao foi possivel parsear JSON. Resposta bruta:\n{texto[:500]}")
        return []

    def _converter_para_promocoes(self, dados: list[dict], pagina_origem: str) -> list[Promocao]:
        promocoes = []
        for item in dados:
            preco = item.get("preco_promocional")
            if preco is None:
                print(f"[conversao] Item ignorado (preco_promocional nulo): {item.get('nome', '?')}")
                continue

            produto = Produto(
                nome=item.get("nome", ""),
                quantidade=item.get("quantidade"),
                unidade=item.get("unidade"),
                categoria=item.get("categoria"),
            )
            preco_promocional = float(preco)
            preco_original = item.get("preco_original")
            desconto = item.get("desconto_percentual")
            if desconto is None and preco_original is not None:
                desconto = round((1 - preco_promocional / float(preco_original)) * 100, 1)

            promocao = Promocao(
                produto=produto,
                preco_promocional=preco_promocional,
                preco_original=preco_original,
                desconto_percentual=desconto,
                validade_inicio=item.get("validade_inicio"),
                validade_fim=item.get("validade_fim"),
                mercado=self._mercado,
                localizacao=self._localizacao,
                periodo=self._periodo,
                pagina_origem=pagina_origem,
            )
            promocoes.append(promocao)
        return promocoes

    def _criar_pasta_json(self) -> Path:
        pasta = Path("dados") / self._mercado / "json" / self._periodo
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"[pasta] Criada: {pasta}")
        return pasta

    def _promocao_para_dict(self, promocao: Promocao) -> dict:
        return {
            "produto": {
                "nome": promocao.produto.nome,
                "quantidade": promocao.produto.quantidade,
                "unidade": promocao.produto.unidade,
                "categoria": promocao.produto.categoria,
            },
            "preco_promocional": promocao.preco_promocional,
            "preco_original": promocao.preco_original,
            "desconto_percentual": promocao.desconto_percentual,
            "validade_inicio": promocao.validade_inicio,
            "validade_fim": promocao.validade_fim,
            "mercado": promocao.mercado,
            "localizacao": promocao.localizacao,
            "periodo": promocao.periodo,
            "pagina_origem": promocao.pagina_origem,
        }
