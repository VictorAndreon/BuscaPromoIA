import requests
from pathlib import Path
from pdf2image import convert_from_bytes
from PIL import Image

# Processa os PDFs para Imagens
class ProcessadorPDF:
    def __init__(self, mercado: str, periodo: str):
        self.mercado = mercado
        self.periodo = periodo
        self.pasta_imagens = self._criar_pasta_imagens()


    def _criar_pasta_imagens(self) -> Path:
        pasta = Path("dados") / self.mercado / "imagens" / self.periodo
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"[pasta] Criada: {pasta}")
        return pasta


    def _baixar_pdf(self, url: str) -> bytes:
        print(f"[download] Baixando PDF: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        print(f"[download] Concluido ({len(response.content) / 1024:.1f} KB)")
        return response.content


    def _converter_para_imagens(self, pdf_bytes: bytes) -> list[Image.Image]:
        print("[conversao] Convertendo paginas do PDF em imagens...")
        imagens = convert_from_bytes(pdf_bytes, dpi=150)
        print(f"[conversao] {len(imagens)} pagina(s) gerada(s)")
        return imagens


    def _salvar_imagens(self, imagens: list[Image.Image], nome: str) -> list[Path]:
        caminhos = []
        for i, imagem in enumerate(imagens):
            caminho = self.pasta_imagens / f"{nome}_pagina_{i + 1}.jpg"
            imagem.save(caminho, "JPEG")
            print(f"[salvo] {caminho}")
            caminhos.append(caminho)
        return caminhos


    def processar(self, url_pdf: list[str], nome_arquivo: str) -> list[Path]:
        print(f"\n[inicio] Processando {len(url_pdf)} PDF(s) do mercado '{self.mercado}'")
        todos_caminhos = []
        for i, url in enumerate(url_pdf):
            print(f"\n[pdf {i + 1}/{len(url_pdf)}]")
            pdf_bytes = self._baixar_pdf(url)
            imagens = self._converter_para_imagens(pdf_bytes)
            nome = f"{nome_arquivo}_{i + 1}" if len(url_pdf) > 1 else nome_arquivo
            caminhos = self._salvar_imagens(imagens, nome)
            todos_caminhos.extend(caminhos)
        print(f"\n[fim] {len(todos_caminhos)} imagem(ns) salva(s) em '{self.pasta_imagens}'")
        return todos_caminhos