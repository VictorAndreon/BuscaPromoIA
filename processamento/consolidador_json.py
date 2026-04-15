import json
from pathlib import Path


NOME_ARQUIVO_CONSOLIDADO = "promocoes_consolidadas.json"


class ConsolidadorJsonError(Exception):
    pass


class ConsolidadorJson:
    def __init__(self, mercado: str, periodo: str) -> None:
        self._mercado = mercado
        self._periodo = periodo
        self._pasta_json = Path("dados") / mercado / "json" / periodo

    def consolidar(self) -> Path:
        promocoes = self._ler_paginas()
        caminho_saida = self._pasta_json / NOME_ARQUIVO_CONSOLIDADO
        caminho_saida.write_text(json.dumps(promocoes, ensure_ascii=False, indent=2))
        print(f"[consolidador] {len(promocoes)} promoção(ões) salvas em {caminho_saida}")
        return caminho_saida

    def _ler_paginas(self) -> list[dict]:
        if not self._pasta_json.exists():
            raise ConsolidadorJsonError(f"Pasta não encontrada: {self._pasta_json}")

        arquivos = sorted(
            f for f in self._pasta_json.glob("*.json")
            if f.name != NOME_ARQUIVO_CONSOLIDADO
        )
        print(f"[consolidador] {len(arquivos)} arquivo(s) de página encontrado(s)")

        todas = []
        for arquivo in arquivos:
            dados = json.loads(arquivo.read_text())
            todas.extend(dados)
            print(f"[consolidador]   {arquivo.name}: {len(dados)} item(ns)")

        return todas
