import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

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

def get_embeddings_model(provider: str):
    """Retorna a instância do modelo de embeddings com base no provedor."""
    if provider == 'google':
        print("Usando o modelo de embeddings do Google (models/embedding-001).")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif provider == 'openai':
        print("Usando o modelo de embeddings da OpenAI (text-embedding-ada-002).")
        return OpenAIEmbeddings() # Modelo padrão é 'text-embedding-ada-002'
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

def main():
    """
    Script principal para ingerir um documento PDF no banco de dados vetorial.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingere um documento PDF no banco de dados vetorial.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="document.pdf", 
        help="O caminho para o arquivo PDF a ser processado (padrão: document.pdf)."
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="documentos_pdf", 
        help="O nome da coleção no banco de dados vetorial (padrão: documentos_pdf)."
    )
    args = parser.parse_args()

    try:
        check_env_vars(args.provider)
    except EnvironmentError as e:
        print(f"Erro de configuração: {e}")
        return

    print(f"Lendo o arquivo: {args.path}...")
    try:
        loader = PyPDFLoader(args.path)
        docs = loader.load()
        print(f"Documento carregado com {len(docs)} página(s).")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o PDF: {e}")
        return

    print("Dividindo o documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(split_docs)} chunks.")

    try:
        embeddings = get_embeddings_model(args.provider)
    except ValueError as e:
        print(f"Erro: {e}")
        return

    connection_string = get_connection_string()
    collection_name = args.collection

    print(f"Salvando embeddings na coleção '{collection_name}' (Provedor: {args.provider})...")
    PGVector.from_documents(
        embedding=embeddings,
        documents=split_docs,
        collection_name=collection_name,
        connection=connection_string,
        pre_delete_collection=True,
    )

    print("\nProcesso de ingestão concluído com sucesso!")

if __name__ == '__main__':
    main()