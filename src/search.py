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


    def _generate_iter_retgen_context(self, query: str, k: int) -> str:
        """
        Executa o processo ITER-RETGEN (Iterative Retrieval-Generation) com placeholders [MISSING].
        """
        self.verbose_print("Iniciando estratégia ITER-RETGEN com refinamento de lacunas...")
        print("⚠️  Atenção: A estratégia ITER-RETGEN é detalhada e pode levar alguns minutos. Por favor, aguarde...")
        
        # 1. Draft Inicial com [MISSING]
        draft_prompt = """You are an expert assistant with limited initial knowledge.
        Answer the following question, but you MUST mark MANY specific details as missing.
        Use [MISSING: ...] markers for:
        - Specific version numbers and release dates
        - Technical specifications and parameters
        - Performance metrics and benchmarks
        - Comparison data between different versions
        - Implementation details and code examples
        - Real-world use cases and case studies
        - Limitations and known issues
        - Future roadmap and upcoming features
        
        Be thorough in identifying what specific information would make the answer complete.
        Start with a basic overview but mark MANY specific details as missing.
        
        Do not generate more than 5 MISSING Markers.
        Question: {query}
        
        Answer:"""
        
        self.verbose_print("Gerando draft inicial...")
        current_draft = self._generate_text(draft_prompt, {"query": query})
        self.verbose_print(f"Draft Inicial:\n{current_draft[:200]}...")

        search_query_text = query # Fallback inicial
        
        # 2. Ciclo de Refinamento (2 iterações para não estender demais, user pediu similar ao curso)
        for i in range(2):
            self.verbose_print(f"--- Iteração {i+1}/2 ---")
            
            # Identifica lacunas e gera hipóteses/queries para busca
            query_prompt = """You received the following draft with gaps:
            {draft}
            
            For each [MISSING: ...] marker, provide information to fill that gap.
            Format each as: 'For [MISSING: topic]: provide the actual information'
            Be specific and provide real data when possible.
            Example: 'For [MISSING: version numbers]: LangChain is at version 0.1.0, LangGraph at 0.2.0'
            List information for each gap, maximum 5 items."""
            
            gap_info = self._generate_text(query_prompt, {"draft": current_draft})
            self.verbose_print(f"Informação para lacunas (Busca):\n{gap_info[:150]}...")
            
            # Recuperação (Retrieval)
            # Usa a informação gerada (hipotética) + query original para buscar documentos reais
            search_query_text = f"{query} {gap_info}"
            docs = self.db.similarity_search_with_score(search_query_text, k=k)
            retrieved_context = "\n\n".join([d.page_content for d, _ in docs])
            
            # Refinamento (Fill Gaps)
            fill_prompt = """Original question: {query}
            
            Current draft (iteration {iteration}):
            {draft}
            
            Information to help fill the gaps:
            {context}
            
            CRITICAL INSTRUCTIONS:
            1. You MUST replace AT LEAST 1-2 [MISSING: ...] markers with concrete information
            2. ACTUALLY REPLACE the text '[MISSING: xyz]' with real content - don't keep the marker
            3. Use the information above to guide what content to add
            4. Do NOT add any new [MISSING:] markers - only fill or keep existing ones
            5. If you cannot fill a gap with certainty, keep it as [MISSING: ...]
            
            Important: This is iteration {iteration}. You MUST make progress by filling gaps.
            
            Rewrite the ENTIRE answer with the [MISSING:] markers replaced:"""
            
            current_draft = self._generate_text(fill_prompt, {
                "query": query, 
                "draft": current_draft, 
                "context": retrieved_context,
                "iteration": i+1
            })
            self.verbose_print(f"Draft Refinado ({i+1}):\n{current_draft[:200]}...")

        # 3. Fase de Expansão (Expansion Phase) - v1.3.0
        # Analisa o rascunho final para ver se pode ser enriquecido
        self.verbose_print("--- Fase de Expansão ---")
        expansion_prompt = """Review this draft answer:
        {draft}
        
        Identify areas that could benefit from MORE specific information.
        Add new [MISSING: ...] markers for:
        - Technical details that were glossed over
        - Specific examples that would clarify concepts
        - Comparative data that would add context
        - Implementation specifics that developers would need
        
        Return the same text but with ADDITIONAL [MISSING: ...] markers for deeper details.
        If the answer is already comprehensive, do not add any markers."""
        
        expanded_draft = self._generate_text(expansion_prompt, {"draft": current_draft})
        new_gaps_count = expanded_draft.count("[MISSING:")
        
        if new_gaps_count > 0:
            self.verbose_print(f"Expansão identificou {new_gaps_count} novas lacunas. Iniciando iteração extra...")
            current_draft = expanded_draft
            
            # --- Iteração Extra (Expansão) ---
            query_prompt_exp = """You received the following draft with gaps:
            {draft}
            
            For each [MISSING: ...] marker, provide information to fill that gap.
            List information for each gap."""
            
            gap_info = self._generate_text(query_prompt_exp, {"draft": current_draft})
            
            search_query_text = f"{query} {gap_info}"
            docs = self.db.similarity_search_with_score(search_query_text, k=k)
            retrieved_context = "\n\n".join([d.page_content for d, _ in docs])
            
            fill_prompt_exp = """Original question: {query}
            Current draft:
            {draft}
            Information to help fill the gaps:
            {context}
            
            Rewrite the ENTIRE answer with the [MISSING:] markers replaced with detailed information."""
            
            current_draft = self._generate_text(fill_prompt_exp, {
                "query": query, 
                "draft": current_draft, 
                "context": retrieved_context
            })
            self.verbose_print(f"Draft Final Expandido:\n{current_draft[:200]}...")
            
            # Atualiza a query de busca final com o texto expandido
            search_query_text = f"{query} {current_draft}"
        else:
            self.verbose_print("Nenhuma expansão necessária. O rascunho já está completo.")
            search_query_text = f"{query} {current_draft}"

        return search_query_text

    def search_documents(self, query: str, k: int = 10, strategy: str = 'default'):
        """
        Realiza uma busca por similaridade no banco de vetores.
        Strategies: 'default', 'hyde', 'query2doc', 'best'
        """
        if strategy == 'best':
            self.verbose_print("Calculando a melhor estratégia...")
            strategies = ['default', 'hyde', 'query2doc', 'iter-retgen']
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
        elif strategy == 'iter-retgen':
            text_to_search = self._generate_iter_retgen_context(query, k)
            
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
        choices=['default', 'hyde', 'query2doc', 'iter-retgen', 'best'],
        help="Estratégia de busca: 'default', 'hyde', 'query2doc', 'iter-retgen' ou 'best' (padrão: default)."
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