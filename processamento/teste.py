import json
from pathlib import Path

from coleta.modelos.promocao import Produto, Promocao
from processamento.extratores.extrator_qwen3 import ExtratorQwen3

MERCADO = "stokcenter"
PERIODO = "2026-03-25"
LOCALIZACAO = "Petropolis"

PASTA_IMAGENS = Path("dados") / MERCADO / "imagens" / PERIODO
PASTA_JSON = Path("dados") / MERCADO / "json" / PERIODO


def _carregar_json_existente(caminho_json: Path) -> list[Promocao]:
    dados = json.loads(caminho_json.read_text())
    promocoes = []
    for item in dados:
        produto = Produto(
            nome=item["produto"]["nome"],
            quantidade=item["produto"].get("quantidade"),
            unidade=item["produto"].get("unidade"),
            categoria=item["produto"].get("categoria"),
        )
        promocoes.append(Promocao(
            produto=produto,
            preco_promocional=item["preco_promocional"],
            preco_original=item.get("preco_original"),
            desconto_percentual=item.get("desconto_percentual"),
            validade_inicio=item.get("validade_inicio"),
            validade_fim=item.get("validade_fim"),
            mercado=item["mercado"],
            localizacao=item.get("localizacao"),
            periodo=item["periodo"],
            pagina_origem=item["pagina_origem"],
        ))
    return promocoes


def main() -> None:
    imagens = sorted(PASTA_IMAGENS.glob("*.jpg"))
    if not imagens:
        print(f"Nenhuma imagem encontrada em {PASTA_IMAGENS}")
        return

    extrator = ExtratorQwen3(mercado=MERCADO, periodo=PERIODO, localizacao=LOCALIZACAO)

    for imagem in imagens:
        caminho_json = PASTA_JSON / f"{imagem.stem}.json"

        if caminho_json.exists():
            print(f"\n[mock] {imagem.name} ja processado, carregando JSON existente...")
            promocoes = _carregar_json_existente(caminho_json)
            print(f"[mock] {len(promocoes)} promocao(es) carregada(s)")
            continue

        promocoes = extrator.extrair(imagem)
        extrator.salvar(promocoes, imagem.stem)


if __name__ == "__main__":
    main()
