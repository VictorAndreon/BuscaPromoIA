from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from interface.consultor import Consultor, ConsultorError


COMANDOS_SAIR = {"sair", "exit", "quit", "q"}

console = Console()


def main() -> None:
    console.print(
        Panel("[bold green]BuscaPromoIA — Consultor de Promoções[/bold green]", expand=False)
    )
    console.print("Digite sua pergunta ou [bold]'sair'[/bold] para encerrar.\n")

    consultor = Consultor()

    while True:
        pergunta = Prompt.ask("[bold cyan]Você[/bold cyan]")
        if pergunta.strip().lower() in COMANDOS_SAIR:
            console.print("[dim]Encerrando...[/dim]")
            break
        if not pergunta.strip():
            continue
        try:
            with console.status("[dim]Consultando promoções...[/dim]"):
                resposta = consultor.consultar(pergunta)
            console.print(
                Panel(resposta, title="[bold green]Resposta[/bold green]", border_style="green")
            )
        except ConsultorError as erro:
            console.print(f"[bold red]Erro:[/bold red] {erro}")


if __name__ == "__main__":
    main()
