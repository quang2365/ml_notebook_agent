import questionary
from questionary import Choice
import typer
from config.managers import save_config
from config.providers import PROVIDERS
from security.api_key_store import save_api_key


def select_provider() -> str:
    typer.echo("\nQIU SETUP")

    choices = [
        Choice(title=f"{data['label']}", value=provider_id)
        for provider_id, data in PROVIDERS.items()
    ]

    choices.append(Choice(title="Other providers...", value="__other__"))

    selected = questionary.select(
        "Select Your Provider:",
        choices=choices,
        use_arrow_keys=True,
        use_indicator=True,
    ).ask()

    if selected is None:
        raise typer.Exit(1)
    if selected == "__other__":
        choose_other_provider = True
        return selected_other_provider()

    return selected

def select_model(provider_id: str) -> str:
    if provider_id not in PROVIDERS:
        return select_orther_model()
    provider = PROVIDERS[provider_id]
    model = provider['models']

    choices=[
        Choice(title=name,value=name)
        for name in model
    ]
    another_choice = Choice(title="anther model?",value = "__other__")
    choices.append(another_choice)
    selected_model = questionary.select(
        "Select Model:",
        choices=choices,
        use_arrow_keys=True,
        use_indicator=True,
    ).ask()
    if select_model == "__other__":
        return select_orther_model()
    if selected_model is None:
        raise typer.Exit(1)

    return selected_model
def run_setup() -> None:
    provider = select_provider()
    if provider not in PROVIDERS:
        provider_name = provider
        base_url = enter_base_url()
    else:
        provider_name = f"{PROVIDERS[provider]['label']}"
    api_key = questionary.password(
        f"Enter API Key for {provider_name}:",
        validate=lambda text: True
        if text.strip()
        else "API key cannot be empty.",
    ).ask()

    model = select_model(provider)

    if api_key is None:
        typer.echo("\nSetup cancelled.")
        raise typer.Exit(code=1)

    typer.echo("\n--- Configuration Summary ---")
    typer.echo(f"Provider : {provider_name}")
    typer.echo(f"Model    : {model}")

    confirmed = questionary.confirm(
        "Bitch, Do you wanna save this shit?",
        default=True,
    ).ask()

    if not confirmed:
        typer.echo("Setup cancelled.")
        raise typer.Exit()

    # 6. Lưu dữ liệu
    save_api_key(
        provider=provider,
        api_key=api_key.strip(),
    )

    save_config(provider,base_url,model)

    typer.echo("\n✔ QIU setup completed successfully!")

def selected_other_provider() -> str:
    other_provider = questionary.text(
        "Enter provider name:",
        validate=lambda text: True
        if text.strip()
        else "Provider name cannot be empty.",
    ).ask()
    other_provider = other_provider.strip()
    if other_provider is None:
        raise typer.Exit(1)
    return other_provider
def select_orther_model() -> str:
    model = questionary.text(
        "Enter model name:",
        validate=lambda text: True
        if text.strip()
        else "Must enter the model name"
    ).ask()

    model = model.strip()

    if not model:
        raise typer.Exit(1)
    return model
def enter_base_url() -> str:
    base_url = questionary.text(
        "Enter base url of provider:",
        validate=lambda text:True
        if text.strip()
        else "Must enter base url of provider"
    ).ask()

    base_url = base_url.strip()

    if not base_url:
        raise typer.Exit(1)
    return base_url