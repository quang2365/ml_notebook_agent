import keyring

SERVICE_NAME = "qiu"

def save_api_key(provider:str,api_key:str) -> None:
    keyring.set_password(SERVICE_NAME,provider,api_key)
def get_api_key(provider) -> str|None:
    return keyring.get_password(SERVICE_NAME,provider)
def delete_api_key(provider:str) -> None:
    try:
        keyring.delete_password(SERVICE_NAME,provider)
    except keyring.errors.PasswordDeleteError:
        pass