from coleta.scraper_base import ScraperBase
from processamento.processador_pdf import ProcessadorPDF
from rich.traceback import install
from datetime import date

install()

coletor = ScraperBase("https://stokcenter.com.br/encarte/?id=21479&loja=Petr%C3%B3polis&unidade_id=416")


processador = ProcessadorPDF("stokcenter", str(date.today()))
processador.processar(coletor.coletar_encartes(), "encarte")