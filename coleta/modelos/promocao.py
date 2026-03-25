from dataclasses import dataclass
from typing import Optional

# Arquivo de Types
@dataclass
class Produto:
    nome: str
    quantidade: Optional[float]
    unidade: Optional[str]
    categoria: Optional[str]


@dataclass
class Promocao:
    produto: Produto
    preco_promocional: float
    preco_original: Optional[float]
    desconto_percentual: Optional[float]
    validade_inicio: Optional[str]
    validade_fim: Optional[str]
    mercado: str
    localizacao: Optional[str]
    periodo: str
    pagina_origem: str
