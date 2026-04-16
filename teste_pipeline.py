from pathlib import Path
from datetime import date

from processamento.extratores.extrator_qwen3 import ExtratorQwen3
from processamento.consolidador_json import ConsolidadorJson
from armazenamento.vetorizador import Vetorizador

MERCADO = "stokcenter"
PERIODO = str(date.today())
LOCALIZACAO = "Petropolis"

pasta_imagens = Path("dados") / MERCADO / "imagens" / PERIODO
pasta_json = Path("dados") / MERCADO / "json" / PERIODO

imagens = sorted(pasta_imagens.glob("*.jpg"))
if not imagens:
    print(f"Nenhuma imagem encontrada em {pasta_imagens}")
    exit(1)

imagem = imagens[0]
print(f"[teste] Usando imagem: {imagem.name} ({len(imagens)} disponíveis no período)")

extrator = ExtratorQwen3(MERCADO, PERIODO, localizacao=LOCALIZACAO)

caminho_json = pasta_json / f"{imagem.stem}.json"
if caminho_json.exists():
    print(f"[teste] JSON já existe, apague {caminho_json} para reprocessar.")
else:
    promocoes = extrator.extrair(imagem)
    extrator.salvar(promocoes, imagem.stem)

consolidador = ConsolidadorJson(MERCADO, PERIODO)
caminho_consolidado = consolidador.consolidar()

vetorizador = Vetorizador(MERCADO, PERIODO)
vetorizador.vetorizar(caminho_consolidado)
