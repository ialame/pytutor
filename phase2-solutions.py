"""Solutions des exercices de la Phase 2.

Lance ce fichier pour exécuter les démos automatiquement.
L'exercice 2 (boucle interactive) et l'exercice 4 (LangSmith) sont
optionnels : décommente leur appel dans main() pour les lancer.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from config import get_model
from phase2_rag import (
    PYTHON_DOCS_URLS,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    get_embeddings,
    build_or_load_vectorstore,
    build_rag_chain,
    build_rag_chain_with_sources,
    format_docs,
    RAG_PROMPT,
)


# ============================================================================
# EXERCICE 1 — Comparer 3 chunk_size sur la même question
# ============================================================================
# Objectif : voir CONCRÈTEMENT que le chunk_size change tout. À 500 caractères,
# le retriever est précis mais le contexte est fragmenté ; à 2000, le contexte
# est riche mais on retrouve souvent du bruit (passages tangentiels).
#
# Méthode : construire 3 vectorstores dans 3 dossiers séparés, poser la même
# question, observer la longueur des chunks retrouvés et la qualité finale.
# ============================================================================

def build_vectorstore_with_chunk_size(chunk_size: int, persist_dir: str) -> Chroma:
    """Construit (ou recharge) un vectorstore avec un chunk_size donné.

    On utilise un overlap proportionnel : 20% du chunk_size.
    """
    embeddings = get_embeddings()

    if Path(persist_dir).exists():
        print(f"  ✓ Rechargement depuis {persist_dir}")
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    print(f"  → Indexation chunk_size={chunk_size} dans {persist_dir}...")
    loader = WebBaseLoader(PYTHON_DOCS_URLS)
    loader.requests_kwargs = {"headers": {"User-Agent": "PyTutor-Edu/0.1"}}
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"    → {len(chunks)} chunks générés")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def exercise_1(cleanup: bool = False) -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 1 — Impact du chunk_size sur la qualité du RAG")
    print("=" * 70)

    configs = [
        (500,  "./chroma_db_500"),
        (1000, "./chroma_db_1000"),
        (2000, "./chroma_db_2000"),
    ]

    # Question volontairement précise : on veut un détail spécifique de la doc.
    question = "Comment lève-t-on une exception personnalisée en Python ?"

    for chunk_size, persist_dir in configs:
        print(f"\n--- chunk_size = {chunk_size} ---")
        vs = build_vectorstore_with_chunk_size(chunk_size, persist_dir)
        retriever = vs.as_retriever(search_kwargs={"k": 3})

        docs = retriever.invoke(question)
        avg_len = sum(len(d.page_content) for d in docs) // len(docs)
        print(f"  Longueur moyenne des chunks retrouvés : {avg_len} caractères")

        # On affiche les 80 premiers caractères de chaque chunk pour comparer
        for i, d in enumerate(docs, 1):
            snippet = d.page_content[:80].replace("\n", " ")
            print(f"  [{i}] {snippet}...")

        # Et la réponse finale du modèle
        chain = build_rag_chain(retriever)
        answer = chain.invoke(question)
        print(f"\n  Réponse :\n  {answer[:300]}...")

    print("\n" + "-" * 70)
    print("OBSERVATIONS À RETENIR :")
    print(" - chunk=500 : chunks précis mais souvent incomplets, le modèle")
    print("   peut manquer le contexte autour du détail recherché.")
    print(" - chunk=1000 : le bon défaut pour de la doc technique.")
    print(" - chunk=2000 : beaucoup de bruit, le retriever distingue mal")
    print("   les passages pertinents des passages tangentiels.")

    if cleanup:
        for _, d in configs:
            if d != "./chroma_db_1000" and Path(d).exists():
                shutil.rmtree(d)


# ============================================================================
# EXERCICE 2 — Boucle interactive
# ============================================================================
# Objectif : transformer le RAG en mini-CLI. Idéal pour explorer ta doc
# librement et tester des questions. C'est aussi la base de ce qu'on
# transformera en UI Gradio plus tard.
# ============================================================================

def exercise_2_interactive() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 2 — Mode interactif (Ctrl+C ou ligne vide pour quitter)")
    print("=" * 70)

    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = build_rag_chain_with_sources(retriever)

    print("\nPyTutor est prêt. Pose une question sur Python.")
    while True:
        try:
            question = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ la prochaine !")
            break

        if not question:
            print("Au revoir.")
            break

        result = chain.invoke(question)
        print("\n" + "─" * 70)
        print(result["answer"])
        print("─" * 70)
        print("Sources :")
        for i, doc in enumerate(result["context"], 1):
            src = doc.metadata.get("source", "?").split("/")[-1]
            print(f"  [{i}] {src}")


# ============================================================================
# EXERCICE 3 — Prompt strict pour refuser le hors-contexte
# ============================================================================
# Objectif : voir le comportement par défaut sur une question hors-corpus,
# puis durcir le prompt pour rendre le refus systématique et propre.
#
# Le piège classique en RAG : le modèle "complète" avec ses connaissances
# générales quand le contexte ne suffit pas. C'est de l'hallucination polie,
# et c'est exactement ce qu'on veut éviter en pédagogie.
# ============================================================================

STRICT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu réponds EXCLUSIVEMENT à partir du contexte fourni ci-dessous. "
        "Règles absolues :\n"
        "1. Si la réponse n'apparaît pas littéralement dans le contexte, "
        "tu réponds : 'Cette information n'est pas dans la documentation fournie.' "
        "et tu t'arrêtes.\n"
        "2. Tu ne complètes JAMAIS avec tes connaissances générales sur Python.\n"
        "3. Tu ne fais JAMAIS d'inférence en partant d'un sujet voisin.\n"
        "4. Si tu utilises le contexte, cite les chunks par leur numéro [N].\n"
        "5. Si le contexte est ambigu ou partiel, dis explicitement ce qui "
        "manque plutôt que d'extrapoler."
    ),
    (
        "human",
        "Contexte :\n{context}\n\nQuestion : {question}",
    ),
])


def build_strict_rag_chain(retriever):
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | STRICT_RAG_PROMPT
        | get_model()
        | StrOutputParser()
    )


def exercise_3() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 3 — Prompt strict vs prompt par défaut")
    print("=" * 70)

    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain_default = build_rag_chain(retriever)
    chain_strict = build_strict_rag_chain(retriever)

    # Question HORS contexte : pip n'est pas dans les pages indexées
    # (qui couvrent tutoriel + glossaire, pas l'installation).
    out_of_context = "Comment installer le package `requests` avec pip ?"

    print(f"\nQuestion (HORS contexte) : {out_of_context}\n")

    print("--- Prompt PAR DÉFAUT ---")
    print(chain_default.invoke(out_of_context))

    print("\n--- Prompt STRICT ---")
    print(chain_strict.invoke(out_of_context))

    # Question DANS le contexte pour vérifier que le prompt strict
    # n'est pas excessivement restrictif.
    in_context = "Qu'est-ce qu'un itérable en Python ?"
    print(f"\n\nQuestion (DANS contexte, contrôle) : {in_context}\n")
    print("--- Prompt STRICT ---")
    print(chain_strict.invoke(in_context))

    print("\n" + "-" * 70)
    print("OBSERVATIONS À RETENIR :")
    print(" - Sans contrainte explicite, le modèle a tendance à 'aider' en")
    print("   complétant avec ses connaissances. C'est utile en chat général,")
    print("   c'est un BUG en RAG.")
    print(" - Le prompt strict rend le refus net et exploitable en aval")
    print("   (tu peux router vers une autre source d'info, par exemple).")


# ============================================================================
# EXERCICE 4 — Vérifier et utiliser LangSmith
# ============================================================================
# Objectif : s'assurer que LangSmith est bien configuré et exécuter une chaîne
# pour générer une trace visible dans l'UI.
#
# Cet exercice produit du code minimal mais oriente l'attention vers la
# vraie ressource pédagogique : l'UI LangSmith elle-même.
# ============================================================================

def exercise_4_langsmith() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 4 — Tracer une exécution dans LangSmith")
    print("=" * 70)

    # --- Vérification de la configuration ---
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1"}
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT", "default")

    print(f"\nLANGSMITH_TRACING : {'✓ activé' if tracing else '✗ NON activé'}")
    print(f"LANGSMITH_API_KEY : {'✓ présente' if api_key else '✗ absente'}")
    print(f"LANGSMITH_PROJECT : {project}")

    if not (tracing and api_key):
        print("\n→ Ajoute ces lignes dans ton .env (et redémarre le script) :")
        print("    LANGSMITH_TRACING=true")
        print("    LANGSMITH_API_KEY=lsv2_pt_...")
        print("    LANGSMITH_PROJECT=PyTutor2")
        print("    LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # si compte EU")
        print("\n→ Crée un compte gratuit sur https://smith.langchain.com")
        return

    # --- Exécution d'une chaîne pour générer une trace intéressante ---
    print("\nExécution d'une chaîne RAG (la trace sera envoyée à LangSmith)...")
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = build_rag_chain_with_sources(retriever)

    result = chain.invoke("Comment écrire un générateur infini en Python ?")
    print("\nRéponse :")
    print(result["answer"][:300] + "...")

    print("\n" + "-" * 70)
    print("CE QU'IL FAUT REGARDER DANS L'UI :")
    print(f"  1. Va sur https://smith.langchain.com et ouvre le projet '{project}'")
    print("  2. Tu vois ta dernière trace. Clique dessus.")
    print("  3. Le panneau de gauche montre l'arbre d'exécution :")
    print("     - le RunnableParallel 'context' + 'question' (les 2 branches")
    print("       partent en parallèle)")
    print("     - puis le format_docs (RunnableLambda)")
    print("     - puis le prompt formatté (regarde le texte exact envoyé)")
    print("     - puis l'appel au modèle (avec coût et latence)")
    print("     - puis le parser")
    print("  4. Clique sur l'appel modèle pour voir les tokens consommés")
    print("     et le prompt exact, sans rien deviner.")


# ============================================================================
# EXERCICE 5 — MultiQueryRetriever
# ============================================================================
# Objectif : améliorer le RECALL (qu'on retrouve les bonnes infos) en générant
# des reformulations de la question via un LLM avant la recherche.
#
# Effet : très utile pour les questions courtes, ambiguës, ou mal formulées.
# Coût : un appel LLM supplémentaire par question pour la reformulation.
# ============================================================================

def exercise_5() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 5 — MultiQueryRetriever vs retriever simple")
    print("=" * 70)

    vectorstore = build_or_load_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_model(),
    )

    # Question courte et ambiguë : "héritage" peut couvrir plusieurs angles
    # (syntaxe, MRO, héritage multiple, super()...). MultiQuery va générer
    # des variantes qui couvrent ces angles.
    question = "Comment marche l'héritage ?"

    print(f"\nQuestion (volontairement vague) : {question}\n")

    print("--- Retriever SIMPLE (k=3) ---")
    docs_simple = base_retriever.invoke(question)
    for i, d in enumerate(docs_simple, 1):
        snippet = d.page_content[:100].replace("\n", " ")
        print(f"  [{i}] {snippet}...")

    print("\n--- MultiQuery Retriever ---")
    # Active le logging interne pour voir les reformulations générées par le LLM
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    docs_multi = multi_retriever.invoke(question)
    print(f"\n  {len(docs_multi)} documents au total (déduplication automatique)")
    for i, d in enumerate(docs_multi, 1):
        snippet = d.page_content[:100].replace("\n", " ")
        print(f"  [{i}] {snippet}...")

    # On compare aussi les réponses finales
    print("\n--- RÉPONSE avec retriever simple ---")
    print(build_rag_chain(base_retriever).invoke(question)[:400] + "...")

    print("\n--- RÉPONSE avec MultiQuery ---")
    print(build_rag_chain(multi_retriever).invoke(question)[:400] + "...")

    print("\n" + "-" * 70)
    print("OBSERVATIONS À RETENIR :")
    print(" - Regarde les logs ci-dessus pour voir les 3 reformulations")
    print("   générées automatiquement par le LLM.")
    print(" - MultiQuery double ou triple ton coût par question (1 appel pour")
    print("   reformuler + N appels de recherche), mais améliore notablement")
    print("   le recall sur les questions courtes ou floues.")
    print(" - À ne PAS utiliser sur des questions très précises et bien")
    print("   formulées : aucun gain, juste du coût en plus.")


# ============================================================================
# EXERCICE 6 — Filtrage par métadonnée
# ============================================================================
# Objectif : limiter la recherche à une partie du corpus via les metadata.
# Chaque chunk a hérité d'une `source` = l'URL d'origine. Chroma supporte
# des filtres sur les metadata avec une syntaxe MongoDB-like.
# ============================================================================

def exercise_6() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 6 — Filtrage par métadonnée (source URL)")
    print("=" * 70)

    vectorstore = build_or_load_vectorstore()

    # --- Retriever filtré : SEULEMENT la page classes.html ---
    classes_only = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source": "https://docs.python.org/3/tutorial/classes.html"},
        },
    )

    # --- Retriever non filtré pour comparaison ---
    unfiltered = vectorstore.as_retriever(search_kwargs={"k": 4})

    question = "Comment fonctionne `super()` ?"
    print(f"\nQuestion : {question}\n")

    print("--- SANS filtre (toutes les pages indexées) ---")
    for i, d in enumerate(unfiltered.invoke(question), 1):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  [{i}] ({src}) {d.page_content[:80].replace(chr(10), ' ')}...")

    print("\n--- AVEC filtre source=classes.html ---")
    for i, d in enumerate(classes_only.invoke(question), 1):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  [{i}] ({src}) {d.page_content[:80].replace(chr(10), ' ')}...")

    # Démo de la limite du filtre : si on demande quelque chose qui n'est
    # PAS dans classes.html (ex: les exceptions), le filtre ramène du
    # contenu hors-sujet.
    off_topic = "Comment lever une exception ?"
    print(f"\n\nDémo limite — question hors-périmètre : {off_topic}\n")

    print("--- AVEC filtre source=classes.html (mauvaise idée ici) ---")
    for i, d in enumerate(classes_only.invoke(off_topic), 1):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  [{i}] ({src}) {d.page_content[:80].replace(chr(10), ' ')}...")

    print("\n" + "-" * 70)
    print("OBSERVATIONS À RETENIR :")
    print(" - Le filtre par metadata est un couteau : utile quand tu SAIS")
    print("   où chercher (ex: namespace, tag, langue, date), pénalisant")
    print("   quand tu te trompes de domaine.")
    print(" - Syntaxe Chroma : opérateurs $eq, $ne, $in, $gt, $gte, $lt,")
    print("   $lte, $and, $or. Ex: {'source': {'$in': [url1, url2]}}.")
    print(" - Pattern de prod : un mini-classifieur LLM en amont qui décide")
    print("   du filtre à appliquer (auto-routing par metadata).")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    # Exercices automatiques (pas d'input utilisateur, pas de LangSmith)
    #exercise_1()
    #exercise_3()
    exercise_5()
    # exercise_6()

    # Décommente ceci si tu as configuré LangSmith dans ton .env
    #exercise_4_langsmith()

    # Décommente ceci pour lancer le mode interactif (bloque sur input())
    #exercise_2_interactive()

    print("\n" + "=" * 70)
    print("✓ Solutions de la Phase 2 exécutées.")
    print("  → Pour la boucle interactive, décommente exercise_2_interactive()")
    print("  → Pour LangSmith, configure .env et décommente exercise_4_langsmith()")
    print("=" * 70)


if __name__ == "__main__":
    main()
