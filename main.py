from pathlib import Path
from datetime import date

from rich.traceback import install

from coleta.scraper_base import ScraperBase
from processamento.processador_pdf import ProcessadorPDF
from processamento.extratores.extrator_qwen3 import ExtratorQwen3

install()

periodo = str(date.today())

coletor = ScraperBase("https://stokcenter.com.br/encarte/?id=21479&loja=Petr%C3%B3polis&unidade_id=416")

processador = ProcessadorPDF("stokcenter", periodo)
processador.processar(coletor.coletar_encartes(), "encarte")

extrator = ExtratorQwen3("stokcenter", periodo, localizacao="Petropolis")
pasta_imagens = Path("dados") / "stokcenter" / "imagens" / periodo
for imagem in sorted(pasta_imagens.glob("*.jpg")):
    promocoes = extrator.extrair(imagem)
    extrator.salvar(promocoes, imagem.stem)