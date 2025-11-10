# test_rag.py
# (VERSÃO FINAL - Normaliza a Pergunta E usa k=10)

import sys
import os
import json # Importa json
import re # Importa regex
import unidecode # Importa unidecode
from pathlib import Path
import warnings

# --- 0. CONFIGURAÇÃO DO AMBIENTE ---
notebook_dir = os.getcwd() 
PROJECT_ROOT_PATH = os.path.abspath(notebook_dir)
SRC_PATH = os.path.join(PROJECT_ROOT_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    print(f"Adicionado ao sys.path: {SRC_PATH}")

warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')
# Remove a linha que causa o 'NameError'

# --- IMPORTA AS NOSSAS FUNÇÕES ---
try:
    from rag_pipeline.loader import load_and_process_jsons
    from rag_pipeline.vector_store import get_embedding_model, get_vector_store
    from rag_pipeline.model_setup import get_llm
    from rag_pipeline.chain import create_rag_chain
except ImportError as e:
    print(f"!!! ERRO DE IMPORTAÇÃO !!! Falha ao importar módulos da 'src/rag_pipeline'.")
    print(f"Erro: {e}")
    sys.exit(1)

# --- LÓGICA DE NORMALIZAÇÃO DE PERGUNTA ---
def carregar_dicionarios(input_dir: str) -> dict[str, dict[str, str]]:
    """Carrega os dicionários de normalização."""
    dictionaries_path = os.path.join(input_dir, 'dicionarios.json')
    try:
        with open(dictionaries_path, 'r', encoding='utf-8') as f:
            dictionaries = json.load(f)
        
        acronyms = {k.lower(): v.lower() for k, v in dictionaries.get("acronyms", {}).items()}
        standardization = {k.lower(): v.lower() for k, v in dictionaries.get("standardization_map", {}).items()}
        
        print("-> Dicionários de normalização carregados para o Testador RAG.")
        return {"acronyms": acronyms, "standardization": standardization}
    except FileNotFoundError:
        print(f"-> AVISO: 'dicionarios.json' não encontrado. Usando dicionários vazios.")
        return {"acronyms": {}, "standardization": {}}

def normalize_query(text: str, acronyms_map: dict, standardization_map: dict) -> str:
    """Aplica a mesma normalização dos dados à pergunta do usuário."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  
    text = unidecode.unidecode(text)
    
    # Ordem: Siglas primeiro, depois Padronização
    for acronym, expansion in acronyms_map.items():
        replacement_string = f"{acronym} {expansion}"
        text = re.sub(r'\b' + re.escape(acronym) + r'\b', replacement_string, text)
        
    for key, value in standardization_map.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)

    text = re.sub(r'[^a-z0-9\s.,-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# --- FIM DA LÓGICA DE NORMALIZAÇÃO ---

# --- FUNÇÃO DE LIMPEZA DE RESPOSTA ---
def clean_model_response(response: str) -> str:
    """Limpa o "eco" do prompt que o Gemma às vezes inclui na saída."""
    match = re.search(r'<start_of_turn>model\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def main():
    print("\n--- INICIANDO PIPELINE RAG COMPLETO (VERSÃO SCRIPT) ---")

    # --- FASE 0.5: CARREGAR DICIONÁRIOS ---
    DATA_INPUT_DIRECTORY = os.path.join(PROJECT_ROOT_PATH, "data", "input")
    mapas_normalizacao = carregar_dicionarios(DATA_INPUT_DIRECTORY)
    acronyms = mapas_normalizacao["acronyms"]
    standardization = mapas_normalizacao["standardization"]

    # --- FASE 1: CARREGAMENTO DOS DADOS ---
    JSONL_DIRECTORY = os.path.join(PROJECT_ROOT_PATH, "data", "output_blocos")
    all_documents, report = load_and_process_jsons(JSONL_DIRECTORY)
    if not all_documents:
        print("!!! ERRO: Nenhum documento foi carregado. Abortando.")
        return
    print(f"\n--- Fase 1 Concluída: {len(all_documents)} documentos (blocos) carregados. ---")

    # --- FASE 2: VECTOR STORE ---
    embedding_model = get_embedding_model()
    
    index_name = "faiss_index_v_smart_chunk" # Nome do novo índice
    
    vector_store = get_vector_store(
        documents=all_documents,
        embedding_model=embedding_model,
        index_path=index_name 
    )
    
    # "Alarga a rede" para k=10
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    print(f"Retriever configurado para buscar k=10 documentos.")
    
    print(f"\n--- Fase 2 Concluída: Retriever está pronto! ---")

    # --- FASE 3: CARREGAR O LLM ---
    llm = get_llm()
    if llm is None:
        print("!!! ERRO: O LLM falhou ao carregar (é 'None'). Abortando.")
        return
    print(f"\n--- Fase 3 Concluída: LLM (Gemma) está pronto! ---")

    # --- FASE 4: MONTAR O PIPELINE DE RAG ---
    rag_chain = create_rag_chain(retriever, llm)
    print(f"\n--- Fase 4 Concluída: Pipeline RAG está pronto! ---")

    # --- FASE 5: TESTE DE PERGUNTAS E RESPOSTAS ---
    print("\n\n--- INICIANDO FASE 5: TESTE DE PERGUNTAS E RESPOSTAS ---")
    perguntas_teste = [
        "Quais são os professores que podem ser orientadores do TCC?",
        "Qual é o tempo máximo permitido para que um estudante conclua o curso de Ciência da Computação?",
        "Como o aluno sabe se foi aprovado ou reprovado numa matéria?",
        "Quantas horas e que tipo de atividades contam como Atividades Complementares no curso de Ciência da Computação?",
        "Em qual momento do curso a gente pode começar a fazer o estágio de Ciência da Computação?"
    ]
    
    resultados_finais = {} 

    for i, pergunta_bruta in enumerate(perguntas_teste):
        print(f"\n--- Processando Pergunta {i+1}/{len(perguntas_teste)} ---")
        print(f"PERGUNTA (Bruta): {pergunta_bruta}")
        
        # --- NORMALIZA A PERGUNTA ---
        pergunta_normalizada = normalize_query(pergunta_bruta, acronyms, standardization)
        print(f"PERGUNTA (Normalizada): {pergunta_normalizada}")
        
        # 1. Executa o RAG com a pergunta normalizada
        resposta_bruta = rag_chain.invoke(pergunta_normalizada)
        
        # 2. Limpa a resposta
        resposta_limpa = clean_model_response(resposta_bruta)
        print(f"\nRESPOSTA (Gemma):\n{resposta_limpa}")
        
        # 3. Mostra as fontes (usando a pergunta normalizada para o debug)
        documentos_fonte = retriever.invoke(pergunta_normalizada)
        print("\nFONTES (Documentos encontrados pelo FAISS):")
        for j, doc in enumerate(documentos_fonte):
            print(f"  Fonte {j+1}: {doc.metadata.get('source_file')} (Página: {doc.metadata.get('pagina')})")
            print(f"    Trecho (Busca): {doc.page_content[:100]}...")
            print(f"    Trecho (Resposta):    {doc.metadata.get('texto_bruto_resposta', 'N/A')[:100]}...\n")
        
        resultados_finais[pergunta_bruta] = resposta_limpa
        
        print("--------------------------------------------------")

    print("\n--- Teste de Perguntas Concluído! ---")

    # --- RESUMO FINAL (Limpo) ---
    print("\n\n" + "="*60)
    print("--- RESUMO FINAL (PERGUNTAS E RESPOSTAS DIRETAS) ---")
    print("="*60 + "\n")
    
    if not resultados_finais:
        print("Nenhum resultado para exibir.")
    else:
        for i, (pergunta, resposta) in enumerate(resultados_finais.items()):
            print(f"--- Pergunta {i+1} ---")
            print(f"P: {pergunta}")
            print(f"R: {resposta}\n")
    
    print("="*60)

if __name__ == "__main__":
    main()