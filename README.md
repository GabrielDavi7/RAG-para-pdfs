# RAG para Documentos Acadêmicos (IFNMG) 📚

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** focado em documentos institucionais do IFNMG (PPCs, Matrizes Curriculares e Diretrizes de TCC). O objetivo é permitir que alunos e servidores realizem consultas em linguagem natural sobre normas acadêmicas.

## 🚀 O Projeto

O sistema processa arquivos PDF, extrai o conteúdo textual e utiliza técnicas de busca semântica para encontrar as passagens mais relevantes antes de gerar uma resposta via LLM.

### 🔬 Foco em Embeddings (Equipe 3)
Como parte do desenvolvimento, realizamos uma análise comparativa de diferentes modelos de representação vetorial (embeddings) para identificar qual oferece a melhor precisão na recuperação de informações, testamos varios modelos:
* **MiniLM** (All-MiniLM-L6-v2)
* **SBERT** (Sentence-BERT)
* **BERT base**
* **GloVe** (Baseline estatístico)

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **LLM:** gemma-2b
* **Orquestração:** LangChain / LlamaIndex
* **Vetorização:** Sentence-Transformers / PyTorch

## 📋 Pré-requisitos

Antes de começar, você vai precisar ter instalado:
* Python 3.8+
* Gerenciador de pacotes `pip`

## 🔧 Instalação e Uso

1. Clone o repositório:
   ```bash
   git clone [https://github.com/GabrielDavi7/RAG-para-pdfs.git](https://github.com/GabrielDavi7/RAG-para-pdfs.git)

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
