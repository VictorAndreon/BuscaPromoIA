import json
from pathlib import Path

import pytest

from processamento.consolidador_json import ConsolidadorJson


PAGINA_1 = [
    {
        "produto": {"nome": "Leite Piracanjuba", "quantidade": 1.0, "unidade": "L", "categoria": "laticinios"},
        "preco_promocional": 4.99,
        "preco_original": 6.49,
        "desconto_percentual": 23.1,
        "validade_inicio": "01/04/2026",
        "validade_fim": "07/04/2026",
        "mercado": "mercado_teste",
        "localizacao": "Cidade Teste",
        "periodo": "2026-04-01",
        "pagina_origem": "encarte_1_pagina_1",
    }
]

PAGINA_2 = [
    {
        "produto": {"nome": "Arroz Tio Joao", "quantidade": 5.0, "unidade": "kg", "categoria": "mercearia"},
        "preco_promocional": 22.90,
        "preco_original": None,
        "desconto_percentual": None,
        "validade_inicio": "01/04/2026",
        "validade_fim": "07/04/2026",
        "mercado": "mercado_teste",
        "localizacao": "Cidade Teste",
        "periodo": "2026-04-01",
        "pagina_origem": "encarte_1_pagina_2",
    }
]


def _criar_estrutura(tmp_path: Path) -> Path:
    """Cria dados/mercado_teste/json/2026-04-01/ com dois JSONs de página."""
    pasta = tmp_path / "dados" / "mercado_teste" / "json" / "2026-04-01"
    pasta.mkdir(parents=True)
    (pasta / "encarte_1_pagina_1.json").write_text(json.dumps(PAGINA_1, ensure_ascii=False))
    (pasta / "encarte_1_pagina_2.json").write_text(json.dumps(PAGINA_2, ensure_ascii=False))
    return pasta


def test_consolidar_gera_arquivo(tmp_path, monkeypatch):
    """consolidar() deve gerar o arquivo promocoes_consolidadas.json."""
    monkeypatch.chdir(tmp_path)
    _criar_estrutura(tmp_path)

    consolidador = ConsolidadorJson("mercado_teste", "2026-04-01")
    caminho = consolidador.consolidar()

    assert caminho.exists()
    assert caminho.name == "promocoes_consolidadas.json"


def test_consolidar_une_todas_paginas(tmp_path, monkeypatch):
    """consolidar() deve unir as promoções de todas as páginas em uma lista."""
    monkeypatch.chdir(tmp_path)
    _criar_estrutura(tmp_path)

    consolidador = ConsolidadorJson("mercado_teste", "2026-04-01")
    caminho = consolidador.consolidar()

    dados = json.loads(caminho.read_text())
    assert len(dados) == 2
    nomes = [d["produto"]["nome"] for d in dados]
    assert "Leite Piracanjuba" in nomes
    assert "Arroz Tio Joao" in nomes


def test_consolidar_ignora_arquivo_consolidado(tmp_path, monkeypatch):
    """Rodar consolidar() duas vezes não deve duplicar promoções."""
    monkeypatch.chdir(tmp_path)
    _criar_estrutura(tmp_path)

    consolidador = ConsolidadorJson("mercado_teste", "2026-04-01")
    consolidador.consolidar()
    caminho = consolidador.consolidar()  # segunda execução

    dados = json.loads(caminho.read_text())
    assert len(dados) == 2  # ainda 2, não 4


def test_consolidar_pasta_vazia(tmp_path, monkeypatch):
    """consolidar() com pasta sem JSONs de página deve gerar lista vazia."""
    monkeypatch.chdir(tmp_path)
    pasta = tmp_path / "dados" / "mercado_teste" / "json" / "2026-04-01"
    pasta.mkdir(parents=True)

    consolidador = ConsolidadorJson("mercado_teste", "2026-04-01")
    caminho = consolidador.consolidar()

    dados = json.loads(caminho.read_text())
    assert dados == []
