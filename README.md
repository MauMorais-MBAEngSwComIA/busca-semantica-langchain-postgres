# Busca Semântica com LangChain e PostgreSQL

Este projeto é uma demonstração de um sistema de busca semântica e chat (RAG - Retrieval-Augmented Generation) utilizando Python e LangChain. Ele é configurado para ser flexível, permitindo o uso de modelos do **Google Gemini** ou da **OpenAI**.

## Funcionalidades

- **Provedor de LLM Flexível**: Escolha entre 'google' e 'openai' usando um simples argumento de linha de comando.
- **Ingestão de Documentos**: Um script (`src/ingest.py`) lê um arquivo PDF, o divide em partes (chunks), gera embeddings para cada parte e os armazena em um banco de dados PostgreSQL com `pgvector`.
- **Busca Semântica**: Um módulo (`src/search.py`) realiza uma busca por similaridade no banco de dados para encontrar os trechos do documento mais relevantes para uma pergunta.
- **Chat CLI**: Uma interface de linha de comando (`src/chat.py`) permite que o usuário faça perguntas. O sistema recupera o contexto relevante e usa o LLM do provedor escolhido para gerar uma resposta.

## Tech Stack

- **Linguagem**: Python 3.10+
- **Orquestração LLM**: LangChain
- **Provedores de Modelos**: Google Gemini, OpenAI
- **Banco de Dados Vetorial**: PostgreSQL 16 com `pgvector`
- **Containerização**: Docker e Docker Compose

---

## Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Configuração e Instalação

### 1. Clone o Repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd busca-semantica-langchain-postgres
```

### 2. Crie e Configure o Arquivo de Ambiente

Copie o arquivo de exemplo `.env.example` para um novo arquivo chamado `.env`.

```bash
# No Windows (prompt de comando)
copy .env.example .env

# No Linux ou macOS
cp .env.example .env
```

Abra o arquivo `.env` e preencha as chaves de API correspondentes ao provedor que você pretende usar:
- **`GOOGLE_API_KEY`**: Necessária se você for usar `--provider google`. Obtenha em [Google AI Studio](https://aistudio.google.com/app/apikey).
- **`OPENAI_API_KEY`**: Necessária se você for usar `--provider openai`. Obtenha em [OpenAI Platform](https://platform.openai.com/api-keys).

As credenciais do banco de dados já vêm pré-configuradas para o ambiente Docker.

### 3. Instale as Dependências

Crie um ambiente virtual e instale as bibliotecas Python.

```bash
# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

---

## Executando a Aplicação

### 1. Inicie o Banco de Dados

Abra um terminal na raiz do projeto e suba o container do PostgreSQL.

```bash
docker-compose up -d
```

### 2. Adicione um Documento

Coloque o arquivo PDF que você deseja processar na raiz do projeto. O nome padrão é `document.pdf`.

### 3. Execute a Ingestão de Dados

Este script processa e armazena os embeddings do seu PDF. Você deve usar o mesmo provedor na ingestão e no chat.

**Usando o provedor padrão (Google):**
```bash
python -m src.ingest
```

**Usando o provedor OpenAI:**
```bash
python -m src.ingest --provider openai
```

**Para especificar um arquivo diferente:**
```bash
python -m src.ingest --provider google --path /caminho/para/seu/arquivo.pdf
```

**Para especificar uma coleção diferente:** (Útil para organizar documentos por tema)
```bash
python -m src.ingest --collection "minha_colecao"
```

### 4. Inicie o Chat

Execute o script de chat, especificando o mesmo provedor usado na ingestão.

**Chat com Google (padrão):**
```bash
python -m src.chat
```

**Chat com OpenAI:**
```bash
python -m src.chat --provider openai
```

**Opções Adicionais do Chat:**

- `--strategy`: Escolha a estratégia de busca.
    - `default`: Busca vetorial padrão.
    - `hyde`: Gera uma resposta hipotética e busca por similaridade com ela.
    - `query2doc`: Expande a pergunta com uma resposta preliminar antes da busca.
    - `iter-retgen`: Realiza um ciclo de 2 iterações (Draft -> Busca de Lacunas -> Refinamento) para aprimorar o contexto. Identifica e preenche dados faltantes automaticamente.
    - `best`: (Recomendado) Testa as quatro estratégias (`default`, `hyde`, `query2doc`, `iter-retgen`) e usa a que tiver o melhor score de similaridade.
    
    Exemplo:
    ```bash
    python -m src.chat --strategy best
    ```

- `-v` ou `--verbose`: Exibe logs detalhados do processo. No modo `best`, mostra as estatísticas de comparação.
    ```bash
    python -m src.chat --strategy best --verbose
    ```

- `--help`: Mostra a ajuda completa com todos os parâmetros disponíveis.
    ```bash
    python -m src.chat --help
    ```

- `--collection`: Especifica a coleção de documentos no banco de dados (deve ser a mesma usada na ingestão). Padrão: `documentos_pdf`.
    ```bash
    python -m src.chat --collection "minha_colecao"
    ```

Para sair do chat, digite `sair`.