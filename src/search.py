import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils import get_connection_string, check_env_vars, get_embeddings_model, get_chat_model, v_print


class DocumentSearcher:
    """
    Uma classe para encapsular a lógica de busca de documentos em um vector store.
    """
    def __init__(self, provider: str, collection_name: str = "documentos_pdf", verbose: bool = False):
        """
        Inicializa o buscador, configurando embeddings e a conexão com o banco.
        """
        self.verbose_print = v_print(verbose)
        self.verbose_print(f"Inicializando o buscador de documentos com o provedor: {provider}...")
        check_env_vars(provider)
        
        self.embeddings = get_embeddings_model(provider, verbose)
        self.connection_string = get_connection_string()
        self.collection_name = collection_name
        
        # Inicializa o LLM para geração de texto (usado em HyDE e Query2Doc)
        self.llm = get_chat_model(provider, verbose)

        try:
            self.db = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
            )
            self.verbose_print("Conexão com o banco de dados vetorial estabelecida com sucesso.")
        except Exception as e:
            raise ConnectionError(f"Não foi possível conectar ao banco de dados: {e}") from e

    def _generate_text(self, prompt_template: str, input_vars: dict) -> str:
        """Gera texto usando o LLM configurado."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(input_vars)

    def _generate_hyde_doc(self, query: str) -> str:
        """Gera um documento hipotético para a query (HyDE)."""
        prompt = """Escreva um parágrafo conciso que responda de forma clara à pergunta abaixo. 
        A resposta deve parecer um documento real, com datas, fatos ou descrições prováveis, 
        mesmo que hipotéticas. 
        Não use linguagem especulativa (não diga "talvez" ou "possivelmente").
        Pergunta: {query}
        Texto: """
        self.verbose_print("Gerando documento hipotético (HyDE)...")
        hyde_doc = self._generate_text(prompt, {"query": query})
        self.verbose_print(f"Documento Hipotético gerado:\n{hyde_doc[:200]}...")
        return hyde_doc

    def _generate_query2doc_expansion(self, query: str) -> str:
        """Expande a query com uma resposta gerada (Query2Doc)."""
        prompt = """Escreva um texto informativo e neutro, de 100 a 150 palavras, que explique o tópico abaixo. 
        Inclua definições, termos técnicos, sinônimos e contexto relacionado, mas não dê uma resposta direta 
        se for uma pergunta.
        Tópico: {query}
        Texto: """
        self.verbose_print("Gerando expansão da query (Query2Doc)...")
        answer = self._generate_text(prompt, {"query": query})
        expanded_query = f"{query} {answer}"
        self.verbose_print(f"Query expandida:\n{expanded_query[:200]}...")
        return expanded_query

    def search_documents(self, query: str, k: int = 10, strategy: str = 'default'):
        """
        Realiza uma busca por similaridade no banco de vetores.
        Strategies: 'default', 'hyde', 'query2doc', 'best'
        """
        if strategy == 'best':
            self.verbose_print("Calculando a melhor estratégia...")
            strategies = ['default', 'hyde', 'query2doc']
            best_strategy = None
            best_avg_score = float('inf') # Menor é melhor (distância)
            best_results = []
            
            # Salva o print original para usar nos logs de resumo
            original_v_print = self.verbose_print
            def no_op(*args, **kwargs): pass

            for s in strategies:
                original_v_print(f"--- Testando estratégia: {s} ---")
                
                # Silencia os logs internos da chamada recursiva
                self.verbose_print = no_op
                try:
                    results = self.search_documents(query, k, strategy=s)
                finally:
                    # Restaura o print original
                    self.verbose_print = original_v_print
                
                if not results:
                    avg_score = float('inf')
                else:
                    # Calcula a média dos scores (distância)
                    avg_score = sum(score for _, score in results) / len(results)
                
                original_v_print(f"Estratégia '{s}' - Média de Score (Distância): {avg_score:.4f}")
                
                if avg_score < best_avg_score:
                    best_avg_score = avg_score
                    best_strategy = s
                    best_results = results
            
            self.verbose_print(f"*** Estratégia Vencedora: {best_strategy} (Score: {best_avg_score:.4f}) ***")
            return best_results

        text_to_search = query
        
        if strategy == 'hyde':
            text_to_search = self._generate_hyde_doc(query)
        elif strategy == 'query2doc':
            text_to_search = self._generate_query2doc_expansion(query)
            
        self.verbose_print(f"Buscando por: '{text_to_search[:100]}...' (Strategy: {strategy})")
        
        # Nota: similarity_search_with_score aceita a string query e gera o embedding internamente
        similar_docs = self.db.similarity_search_with_score(text_to_search, k=k)
        
        self.verbose_print(f"Encontrados {len(similar_docs)} documentos similares.")
        return similar_docs

# Carrega as variáveis de ambiente do arquivo .env no escopo global
load_dotenv()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Busca documentos em um banco de dados vetorial.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="qualquer coisa", 
        help="A query para a busca (padrão: 'qualquer coisa')."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        choices=['default', 'hyde', 'query2doc', 'best'],
        help="Estratégia de busca: 'default', 'hyde', 'query2doc' ou 'best' (padrão: default)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aumenta a verbosidade para exibir logs detalhados."
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="documentos_pdf", 
        help="O nome da coleção no banco de dados vetorial (padrão: documentos_pdf)."
    )
    args = parser.parse_args()

    # Este teste assume que o 'ingest' foi executado com o provedor 'google'
    print(f"--- Teste da classe de busca (provedor: {args.provider}, coleção: {args.collection}) ---")
    try:
        searcher = DocumentSearcher(provider=args.provider, collection_name=args.collection, verbose=args.verbose)
        results = searcher.search_documents(args.query, strategy=args.strategy)
        
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