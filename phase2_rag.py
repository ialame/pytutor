"""Phase 2 — RAG sur la documentation Python officielle (PyTutor).

Concepts LangChain travaillés :
  1. DocumentLoader : aspirer des pages web (WebBaseLoader)
  2. TextSplitter : découper en chunks avec chevauchement
  3. Embeddings : encoder le texte en vecteurs (HuggingFace, local)
  4. VectorStore : indexer et persister sur disque (Chroma)
  5. Retriever : interface Runnable pour la recherche par similarité
  6. Chaîne RAG complète en LCEL
  7. Récupérer la réponse ET les sources (RunnableParallel)
  8. Comparaison similarity vs MMR (stretch)

ARCHITECTURE DU SCRIPT
----------------------
On sépare l'indexation (lente, faite une fois) de l'interrogation (rapide,
faite à chaque question). La fonction `build_or_load_vectorstore()` réindexe
seulement si le dossier `chroma_db/` n'existe pas. Donc :
  - 1er lancement : ~30s d'indexation (téléchargement embeddings + scraping)
  - Lancements suivants : <1s (rechargement depuis disque)

Lance ce fichier directement pour voir la démo complète.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import get_model


# ============================================================================
# 1. Sources : pages de la doc officielle Python à indexer
# ============================================================================
# Choix éditorial : on couvre les sujets qui reviennent le plus en cours.
# Tu peux ajouter/retirer librement, l'indexation est incrémentale en théorie
# (en pratique on rebuild tout pour rester simple dans ce TP).
# ============================================================================

PYTHON_DOCS_URLS = [
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/howto/functional.html",
    "https://docs.python.org/3/glossary.html",
]

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, rapide, multilingue
COLLECTION_NAME = "python_docs"


# ============================================================================
# 2. Indexation : loader -> splitter -> embeddings -> Chroma
# ============================================================================
# Si la base existe déjà sur disque, on la recharge sans rien refaire.
# Sinon on aspire les URL, on découpe, on encode, on persiste.
# ============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton de fait : un seul modèle d'embeddings réutilisé partout."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Renvoie le vectorstore prêt à l'emploi.

    - Si `chroma_db/` existe et `force_rebuild=False`, on recharge depuis disque.
    - Sinon, on aspire toutes les URL et on réindexe.
    """
    embeddings = get_embeddings()
    persist_path = Path(PERSIST_DIR)

    if persist_path.exists() and not force_rebuild:
        print(f"✓ Rechargement de l'index depuis {PERSIST_DIR}")
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    print(f"→ Indexation de {len(PYTHON_DOCS_URLS)} pages de docs.python.org...")

    # --- Chargement : WebBaseLoader gère une ou plusieurs URL ---
    # Chaque page devient UN document, avec son URL en metadata.
    loader = WebBaseLoader(PYTHON_DOCS_URLS)
    loader.requests_kwargs = {"headers": {"User-Agent": "PyTutor-Edu/0.1"}}
    raw_docs = loader.load()
    print(f"  {len(raw_docs)} pages chargées")

    # --- Découpage en chunks ---
    # chunk_size en CARACTÈRES (pas tokens). 1000 ≈ 200-250 tokens : un bon
    # compromis entre granularité (recherche précise) et contexte (chunk
    # auto-suffisant pour répondre).
    # chunk_overlap = 200 : 20% de chevauchement, on évite de couper une idée.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # L'ordre des séparateurs compte : on essaie d'abord de couper aux
        # endroits naturels, et seulement en dernier recours au milieu d'un mot.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  {len(chunks)} chunks générés")

    # --- Indexation + persistance ---
    # Chroma.from_documents calcule les embeddings et persiste en une étape.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"✓ Index créé et persisté dans {PERSIST_DIR}\n")
    return vectorstore


# ============================================================================
# 3. La chaîne RAG en LCEL
# ============================================================================
# Pattern canonique :
#   {"context": retriever | format, "question": passthrough}
#       | prompt | model | parser
#
# Lis ce code en regardant CE QU'UN INPUT TRAVERSE :
#   1. l'utilisateur passe une string : "Comment fonctionnent les générateurs ?"
#   2. le dict en entrée crée DEUX branches parallèles :
#        - branche "context" : la string -> retriever -> liste de docs -> format -> string
#        - branche "question" : la string -> passthrough -> string (inchangée)
#   3. le prompt reçoit {"context": "...", "question": "..."} et formate
#   4. le modèle répond
#   5. le parser extrait le texte
# ============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un assistant pédagogique pour des élèves de Python. "
        "Réponds à la question en t'appuyant UNIQUEMENT sur le contexte fourni.\n"
        "Si la réponse n'est pas dans le contexte, dis-le explicitement plutôt "
        "que d'inventer. Cite le ou les chunks utilisés par leur numéro entre "
        "crochets, par exemple [1] ou [2,3]."
    ),
    (
        "human",
        "Contexte :\n{context}\n\n"
        "Question : {question}\n\n"
        "Réponse pédagogique :"
    ),
])


def format_docs(docs: list[Document]) -> str:
    """Concatène les documents retrouvés en un seul bloc de contexte numéroté.

    Le numéro [N] permet au modèle de citer ses sources.
    """
    return "\n\n".join(
        f"[{i + 1}] (source: {doc.metadata.get('source', '?')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def build_rag_chain(retriever):
    """Chaîne RAG simple : input string -> output string (la réponse)."""
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | get_model()
        | StrOutputParser()
    )


def build_rag_chain_with_sources(retriever):
    """Chaîne RAG qui renvoie un dict {answer, sources}.

    Pattern intéressant : on utilise RunnableParallel pour FAIRE DEUX CHOSES
    en même temps : générer la réponse ET garder les docs retrouvés pour
    inspection. C'est le moyen idiomatique de récupérer les sources avec LCEL.
    """
    # Étape 1 : on récupère les docs UNE FOIS et on les passe en aval.
    retrieve = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    )

    # Étape 2 : à partir de {context: [Doc...], question: str}, on construit
    # la réponse en formatant les docs, puis on remet tout dans un dict final.
    answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"]),
        )
        | RAG_PROMPT
        | get_model()
        | StrOutputParser()
    )

    # `.assign(answer=...)` ajoute la clé "answer" au dict courant, sans
    # toucher aux autres clés. À la fin on a {context: [Doc...], question, answer}.
    return retrieve | RunnablePassthrough.assign(answer=answer_chain)


# ============================================================================
# 4. Démo : 3 questions, deux stratégies de retrieval
# ============================================================================

DEMO_QUESTIONS = [
    "Quelle est la différence entre une liste et un tuple ?",
    "Comment écrire un générateur en Python ?",
    "À quoi sert le mot-clé `yield` exactement ?",
]


def demo() -> None:
    # --- Indexation (ou rechargement) ---
    vectorstore = build_or_load_vectorstore()

    # --- Démo 1 : RAG simple, sortie texte ---
    print("=" * 70)
    print("DÉMO 1 — RAG simple (retriever similarity, k=4)")
    print("=" * 70)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    chain = build_rag_chain(retriever)

    question = DEMO_QUESTIONS[1]
    print(f"\nQuestion : {question}\n")
    answer = chain.invoke(question)
    print(answer)

    # --- Démo 2 : RAG avec sources ---
    print("\n" + "=" * 70)
    print("DÉMO 2 — RAG avec inspection des sources")
    print("=" * 70)
    chain_with_sources = build_rag_chain_with_sources(retriever)
    question = DEMO_QUESTIONS[2]
    print(f"\nQuestion : {question}\n")
    result = chain_with_sources.invoke(question)
    print("RÉPONSE :")
    print(result["answer"])
    print("\nSOURCES UTILISÉES :")
    for i, doc in enumerate(result["context"], 1):
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {doc.metadata.get('source', '?')}")
        print(f"      {snippet}...")

    # --- Démo 3 : comparaison similarity vs MMR ---
    print("\n" + "=" * 70)
    print("DÉMO 3 — Similarity vs MMR sur la MÊME question")
    print("=" * 70)
    question = DEMO_QUESTIONS[0]
    print(f"\nQuestion : {question}")

    sim_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        # fetch_k : on récupère 20 candidats avant de sélectionner les 3 plus diversifiés.
        # lambda_mult : 1.0 = pure similarité, 0.0 = pure diversité.
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
    )

    print("\n--- Top-3 similarity ---")
    for i, doc in enumerate(sim_retriever.invoke(question), 1):
        print(f"  [{i}] {doc.page_content[:100].replace(chr(10), ' ')}...")

    print("\n--- Top-3 MMR (diversifié) ---")
    for i, doc in enumerate(mmr_retriever.invoke(question), 1):
        print(f"  [{i}] {doc.page_content[:100].replace(chr(10), ' ')}...")

    print("\n" + "=" * 70)
    print("Observe : MMR évite les doublons sémantiques que retourne la")
    print("similarity pure. Pour une question généraliste comme 'liste vs")
    print("tuple', MMR ramène souvent des chunks plus complémentaires.")
    print("=" * 70)


if __name__ == "__main__":
    demo()


# ============================================================================
# EXERCICES (avant la Phase 3)
# ============================================================================
#
# 1. FACILE — Joue avec chunk_size (essaie 500 et 2000) et compare la qualité
#    des réponses. Force la réindexation avec build_or_load_vectorstore(force_rebuild=True).
#    À petit chunk : recherche précise mais contexte fragmenté. À gros chunk :
#    contexte riche mais retriever moins discriminant.
#
# 2. FACILE — Ajoute un mode "interactif" : une boucle while qui demande une
#    question, l'envoie à chain_with_sources, et affiche réponse + sources.
#
# 3. MOYEN — Pose une question dont la réponse N'EST PAS dans les pages
#    indexées (ex: "Comment installer un package avec pip ?"). Observe le
#    comportement du modèle. Modifie le prompt pour le rendre encore plus
#    strict sur le refus de répondre hors-contexte.
#
# 4. MOYEN — Active LangSmith dans ton .env et observe les traces. Tu verras
#    chaque branche de RunnableParallel exécutée séparément. C'est le meilleur
#    moment pour intérioriser visuellement comment LCEL fonctionne.
#
# 5. STRETCH — Remplace le retriever simple par un MultiQueryRetriever qui
#    reformule chaque question en 3 variantes via LLM avant la recherche :
#
#        from langchain.retrievers.multi_query import MultiQueryRetriever
#        multi = MultiQueryRetriever.from_llm(
#            retriever=vectorstore.as_retriever(),
#            llm=get_model(),
#        )
#
#    Compare la qualité sur des questions ambiguës ou mal formulées.
#
# 6. STRETCH — Ajoute un filtre par métadonnée. Chaque chunk a une `source`
#    (l'URL d'origine). Force le retriever à ne chercher que dans
#    'tutorial/classes.html' pour une question sur la POO :
#
#        retriever = vectorstore.as_retriever(
#            search_kwargs={
#                "k": 4,
#                "filter": {"source": "https://docs.python.org/3/tutorial/classes.html"},
#            },
#        )
# ============================================================================
