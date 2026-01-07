import os
import argparse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils import check_env_vars, v_print, get_chat_model
from src.search import DocumentSearcher




def format_context(docs_with_scores, verbose_print):
    """Formata os documentos recuperados em uma string de contexto."""
    context = []
    verbose_print("\n--- Documentos Recuperados ---")
    for i, (doc, score) in enumerate(docs_with_scores):
        source_info = f"Fonte: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Página: {doc.metadata.get('page', 'N/A')}"
        context.append(f"{doc.page_content}\n({source_info})")
        verbose_print(f"Doc {i+1} (Score: {score:.4f}): {source_info}\n{doc.page_content[:100]}...")
    verbose_print("--------------------------\n")
    return "\n\n---\n\n".join(context)

def main():
    """
    Função principal para iniciar o chat CLI.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Inicia um chat com um documento PDF.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aumenta a verbosidade para exibir logs detalhados e fontes de contexto."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        choices=['default', 'hyde', 'query2doc', 'best'],
        help="Estratégia de busca: 'default', 'hyde', 'query2doc' ou 'best' (padrão: default)."
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="documentos_pdf", 
        help="O nome da coleção no banco de dados vetorial (padrão: documentos_pdf)."
    )
    args = parser.parse_args()
    verbose_print = v_print(args.verbose)

    try:
        check_env_vars(args.provider)
    except EnvironmentError as e:
        print(f"Erro de configuração: {e}")
        return

    try:
        searcher = DocumentSearcher(provider=args.provider, collection_name=args.collection, verbose=args.verbose)
        llm = get_chat_model(args.provider, args.verbose)
    except (ConnectionError, ValueError) as e:
        print(f"Erro na inicialização: {e}")
        return

    prompt_template = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    print(f"--- Chat com Documento PDF (Provedor: {args.provider}, Coleção: {args.collection}) ---")
    print("Digite sua pergunta ou 'sair' para terminar.")

    while True:
        try:
            question = input("\nPergunta: ")
            if question.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o chat. Até logo!")
                break
            if not question.strip():
                continue
            
            # Passa a estratégia escolhida para a busca
            relevant_docs = searcher.search_documents(question, strategy=args.strategy)
            
            # Se a estratégia for 'best', o usuário pediu para não mostrar os documentos recuperados no log,
            # apenas as estatísticas (que já são mostradas pelo searcher).
            # Então passamos uma função de print vazia para o format_context se strategy == 'best'.
            context_printer = verbose_print
            if args.strategy == 'best':
                def no_op(*args, **kwargs): pass
                context_printer = no_op

            context = format_context(relevant_docs, context_printer) if relevant_docs else ""
            
            if not context:
                verbose_print("Nenhum documento relevante encontrado para a consulta.")

            verbose_print("\nGerando resposta...")
            response = chain.invoke({"context": context, "question": question})
            print("\nResposta:")
            print(response)

        except Exception as e:
            print(f"\nOcorreu um erro durante o chat: {e}")

if __name__ == '__main__':
    main()
