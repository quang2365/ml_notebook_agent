import questionary
from questionary import Choice
import typer
from model.model import create_llm
from config.managers import save_config,load_configured,append_configured
from config.providers import PROVIDERS
from security.api_key_store import save_api_key
from dotenv import load_dotenv
import os
def message(content: str) -> None:
    model = create_llm()
    result = model.invoke([{"role":"user","content":content}])
    typer.echo(result.content)
def change_config() -> None:
    typer.echo("\nProvider saved")
    configured = load_configured()
    choices = []
    for config in configured:
        provider = config['provider']
        model = config['model']
        base_url = config['base_url']
        choices.append(Choice(title=f"provider:{provider};model:{model};base_url:{base_url}",value= config))
    selected = questionary.select(
            "Select Your Provider:",
            choices=choices,
            use_arrow_keys=True,
            use_indicator=True
        ).ask()
    if selected is None:
        raise typer.Exit(1)
    save_config(
        selected["provider"],
        selected["base_url"],
        selected["model"],
    )

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
        return selected_other_provider()

    return selected

def select_model(provider_id: str) -> str:
    if provider_id not in PROVIDERS:
        return select_other_model()
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
    if selected_model == "__other__":
        return select_other_model()
    if selected_model is None:
        raise typer.Exit(1)

    return selected_model



def selected_other_provider() -> str:
    other_provider = questionary.text(
        "Enter provider name:",
        validate=lambda text: True
        if text.strip()
        else "Provider name cannot be empty.",
    ).ask()
    if other_provider is None:
        raise typer.Exit(1)
    return other_provider.strip()
def select_other_model() -> str:
    model = questionary.text(
        "Enter model name:",
        validate=lambda text: True
        if text.strip()
        else "Must enter the model name"
    ).ask()

    if not model:
        raise typer.Exit(1)
    return model.strip()
def enter_base_url() -> str:
    base_url = questionary.text(
        "Enter base url of provider:",
        validate=lambda text:True
        if text.strip()
        else "Must enter base url of provider"
    ).ask()
    if not base_url:
        raise typer.Exit(1)
    return base_url.strip()


########################################################################
def run_setup() -> None:
    load_dotenv()
    provider = select_provider()
    if provider not in PROVIDERS:
        provider_name = provider
        base_url = enter_base_url()
    else:
        provider_name = f"{PROVIDERS[provider]['label']}"
        base_url = PROVIDERS[provider]["base_url"]
    provider_api_key = provider.upper() +"_API_KEY"
    if not os.getenv(provider_api_key):
        api_key = questionary.password(
        f"Enter API Key for {provider_name}:",
        validate=lambda text: True
        if text.strip()
        else "API key cannot be empty.",
    ).ask()
    else:
        api_key = os.getenv(provider_api_key)

    if api_key is None:
        typer.echo("\nSetup cancelled.")
        raise typer.Exit(code=1)
    model = select_model(provider)

    typer.echo("\n--- Configuration Summary ---")
    typer.echo(f"Provider : {provider_name}")
    typer.echo(f"Model    : {model}")

    confirmed = questionary.confirm(
        "Save this configuration?",
        default=True,
    ).ask()

    if not confirmed:
        typer.echo("Setup cancelled.")
        raise typer.Exit()

    save_api_key(
        provider=provider,
        api_key=api_key.strip(),
    )
    save_config(provider,base_url,model)
    append_configured(provider,base_url,model)
    typer.echo("\nQIU setup completed successfully!")
