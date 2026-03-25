import requests
import json
import re
from bs4 import BeautifulSoup

# Coleta os PDF's de encartes
class ScraperBase:
    def __init__(self, url:str):
        self.url = url
    
    def _buscar_pagina(self) -> BeautifulSoup:
        url = f"{self.url}"
        print(f"Acessando URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    
    def coletar_encartes(self) -> list[dict]:
        soup = self._buscar_pagina()

        encartes = []

        for item in soup.find_all("script", class_="df-shortcode-script"):
            match = re.search(r'=\s*({.*?\});', item.string, re.DOTALL)

            if match:
                print("Encontrado PDF's!")
                dados = json.loads(match.group(1))
                source = dados.get("source")
                encartes.append(source)
                print(f"[PDF]: {encartes}")

        return encartes