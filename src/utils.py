import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def get_connection_string() -> str:
    """Constrói a string de conexão a partir das variáveis de ambiente."""
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

def check_env_vars(provider: str):
    """Verifica se as variáveis de ambiente necessárias estão definidas."""
    db_vars = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"]
    if not all(os.getenv(var) for var in db_vars):
        missing = [var for var in db_vars if not os.getenv(var)]
        raise EnvironmentError(f"Variáveis de banco de dados ausentes: {', '.join(missing)}")

    if provider == 'google' and not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Para o provedor 'google', a GOOGLE_API_KEY é necessária.")
    elif provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Para o provedor 'openai', a OPENAI_API_KEY é necessária.")

def get_embeddings_model(provider: str, verbose: bool = False):
    """Retorna a instância do modelo de embeddings com base no provedor."""
    verbose_print = v_print(verbose)
    
    if provider == 'google':
        verbose_print("Usando o modelo de embeddings do Google (models/embedding-001).")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif provider == 'openai':
        verbose_print("Usando o modelo de embeddings da OpenAI (text-embedding-3-small).")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

def get_chat_model(provider: str, verbose: bool = False):
    """Retorna a instância do modelo de chat com base no provedor."""
    verbose_print = v_print(verbose)
    
    if provider == 'google':
        verbose_print("Usando o modelo de chat do Google (gemini-2.5-flash-lite).")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    elif provider == 'openai':
        verbose_print("Usando o modelo de chat da OpenAI (gpt-5-nano).")
        return ChatOpenAI(model="gpt-5-nano", temperature=0)
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

def v_print(verbose: bool):
    """Retorna uma função de print que só imprime se verbose for True."""
    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return print_if_verbose
