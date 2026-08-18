import typer
from TUI.app import QiuApp
from cli.setup_command import run_setup
from config.managers import is_config,load_config,remove_config
from config.providers import PROVIDERS
from security.api_key_store import get_api_key


app = typer.Typer(
    name="qiu",
    help=(
        "QIU - AI Machine Learning "
        "Notebook Agent"
    )
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    start_qiu()


@app.command()
def setup():
    run_setup()


@app.command()
def version():

    typer.echo("QIU v0.1.0")

@app.command()
def rm_config():
    if remove_config():
        typer.echo("Remove config done!")
    else:
        typer.echo("Can not remove config :((")


def start_qiu() -> None:
    if not is_config():
        typer.echo("\nQIU is not configured.")
        typer.echo("\nRun:\n")
        typer.echo("qiu setup\n")
        raise typer.Exit(code=1)
    config = load_config()
    if config is None:
        typer.echo("Cannot load QIU config.")
        raise typer.Exit(code=1)
    provider_id = config["provider"]
    model = config["model"]
    base_url = config.get("base_url")
    api_key = get_api_key(provider_id)
    if not api_key:
        typer.echo("\nAPI key not found.")
        typer.echo("Run:")
        typer.echo("\nqiu setup\n")
        raise typer.Exit(code=1)
    if provider_id in PROVIDERS:
        provider_name = (PROVIDERS[provider_id]["label"])
    else:
        provider_name = (provider_id)
    typer.echo( "\nQIU")
    typer.echo(
        "AI Machine Learning "
        "Notebook Agent"
    )
    typer.echo("\nConfiguration")
    typer.echo(
        f"Provider : "
        f"{provider_name}"
    )
    typer.echo(
        f"Model    : "
        f"{model}"
    )
    typer.echo(
        f"Base URL : "
        f"{base_url}"
    )
    typer.echo("\nConfiguration loaded.")
    typer.echo("Starting QIU...")
    app_instance = QiuApp(config=config)
    app_instance.run()
