from pathlib import Path
from datetime import date

from rich.traceback import install

from coleta.scraper_base import ScraperBase
from processamento.processador_pdf import ProcessadorPDF
from processamento.extratores.extrator_qwen3 import ExtratorQwen3
from processamento.consolidador_json import ConsolidadorJson
from armazenamento.vetorizador import Vetorizador

install()

periodo = str(date.today())

coletor = ScraperBase("https://stokcenter.com.br/encarte/?id=21479&loja=Petr%C3%B3polis&unidade_id=416")

processador = ProcessadorPDF("stokcenter", periodo)
processador.processar(coletor.coletar_encartes(), "encarte")

extrator = ExtratorQwen3("stokcenter", periodo, localizacao="Petropolis")
pasta_imagens = Path("dados") / "stokcenter" / "imagens" / periodo
pasta_json = Path("dados") / "stokcenter" / "json" / periodo

for imagem in sorted(pasta_imagens.glob("*.jpg")):
    caminho_json = pasta_json / f"{imagem.stem}.json"
    if caminho_json.exists():
        print(f"[skip] {imagem.name} já processado, pulando...")
        continue
    promocoes = extrator.extrair(imagem)
    extrator.salvar(promocoes, imagem.stem)

consolidador = ConsolidadorJson("stokcenter", periodo)
caminho_consolidado = consolidador.consolidar()

vetorizador = Vetorizador("stokcenter", periodo)
vetorizador.vetorizar(caminho_consolidado)