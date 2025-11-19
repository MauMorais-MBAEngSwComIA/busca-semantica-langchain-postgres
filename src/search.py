import os
from dotenv import load_dotenv
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
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif provider == 'openai':
        return OpenAIEmbeddings()
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

class DocumentSearcher:
    """
    Uma classe para encapsular a lógica de busca de documentos em um vector store.
    """
    def __init__(self, provider: str, collection_name: str = "documentos_pdf"):
        """
        Inicializa o buscador, configurando embeddings e a conexão com o banco.
        """
        print(f"Inicializando o buscador de documentos com o provedor: {provider}...")
        check_env_vars(provider)
        
        self.embeddings = get_embeddings_model(provider)
        self.connection_string = get_connection_string()
        self.collection_name = collection_name

        try:
            self.db = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
            )
            print("Conexão com o banco de dados vetorial estabelecida com sucesso.")
        except Exception as e:
            raise ConnectionError(f"Não foi possível conectar ao banco de dados: {e}") from e

    def search_documents(self, query: str, k: int = 10):
        """
        Realiza uma busca por similaridade no banco de vetores.
        """
        print(f"Buscando por: '{query}'...")
        similar_docs = self.db.similarity_search_with_score(query, k=k)
        print(f"Encontrados {len(similar_docs)} documentos similares.")
        return similar_docs

# Carrega as variáveis de ambiente do arquivo .env no escopo global
load_dotenv()

if __name__ == '__main__':
    # Este teste assume que o 'ingest' foi executado com o provedor 'google'
    print("--- Teste da classe de busca (provedor: google) ---")
    try:
        searcher = DocumentSearcher(provider='google')
        test_query = "qualquer coisa"
        results = searcher.search_documents(test_query)
        
        if results:
            for doc, score in results:
                print("-" * 50)
                print(f"Score: {score}")
                print(f"Conteúdo: {doc.page_content[:200]}...")
                print(f"Metadados: {doc.metadata}")
                print("-" * 50)
        else:
            print("Nenhum resultado encontrado.")
            
    except (EnvironmentError, ConnectionError) as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante o teste: {e}")