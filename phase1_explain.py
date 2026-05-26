"""Phase 1 — Le moteur d'explication structurée (PyTutor).

Concepts LangChain travaillés :
  1. ChatPromptTemplate avec system + user messages et variables
  2. Schémas Pydantic + with_structured_output (la voie moderne)
  3. LCEL : composition avec `|` et `.partial()` pour fixer des variables
  4. RunnableParallel pour exécuter plusieurs chaînes en concurrence
  5. Streaming token par token avec `.stream()`

Lance ce fichier directement (Run dans PyCharm) pour voir la démo.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from config import get_model


# ============================================================================
# 1. Schéma de sortie structurée
# ============================================================================
# On force le LLM à répondre dans ce schéma exact.
#
# Avantage énorme par rapport à parser du markdown ou du JSON à la main :
#   - Jamais de malformation (le SDK Anthropic utilise tool-calling sous le capot)
#   - Tu obtiens un objet Python typé, auto-complété par PyCharm
#   - Les `description=...` sont envoyés au modèle comme instructions
# ============================================================================

class ComprehensionQuestion(BaseModel):
    """Une question de compréhension pour vérifier l'acquisition."""

    question: str = Field(description="La question posée à l'élève.")
    expected_answer: str = Field(
        description="La réponse attendue, en une à deux phrases."
    )


class Explanation(BaseModel):
    """Réponse pédagogique complète sur un concept Python."""

    concept: str = Field(description="Le nom du concept expliqué.")
    intuition: str = Field(
        description=(
            "Explication intuitive en exactement 3 phrases. "
            "Évite le jargon, vise la clarté pour l'élève."
        )
    )
    code_example: str = Field(
        description=(
            "Un exemple de code Python minimal et commenté qui illustre "
            "le concept. Préfère des noms de variables parlants."
        )
    )
    pitfall: str = Field(
        description="Un piège classique que rencontrent les élèves sur ce concept."
    )
    comprehension_questions: list[ComprehensionQuestion] = Field(
        description="Exactement 3 questions de compréhension, de difficulté croissante."
    )


# ============================================================================
# 2. Le prompt template
# ============================================================================
# `{level}` et `{concept}` sont des variables qui seront remplacées au
# moment de l'invoke. Le prompt est lui-même un Runnable composable.
# ============================================================================

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un assistant pédagogique pour des élèves apprenant Python. "
        "Tu adaptes systématiquement ton vocabulaire et tes exemples au "
        "niveau de l'élève. Niveau actuel : {level}.\n\n"
        "Règles selon le niveau :\n"
        "  - 'débutant' : pas de jargon, exemples ultra-simples, "
        "analogies du quotidien, pas plus de 10 lignes de code.\n"
        "  - 'intermédiaire' : tu peux supposer la syntaxe de base "
        "connue, montre les usages idiomatiques (PEP 8, type hints).\n"
        "  - 'avancé' : montre les détails du modèle objet de Python, "
        "les subtilités, les cas limites, les implications de performance."
    ),
    ("human", "Explique le concept suivant : {concept}"),
])


# ============================================================================
# 3. La chaîne LCEL
# ============================================================================
# `prompt | model | (rien après, with_structured_output remplace le parser)`
#
# `with_structured_output(Explanation)` configure le modèle pour qu'il
# remplisse le schéma Pydantic. Tu peux l'imaginer comme un parser intégré
# au modèle, plus fiable qu'un PydanticOutputParser à la sortie (que tu
# verras dans des vieux tutos — il marche, mais c'est moins propre).
# ============================================================================

def explain_chain_for_level(level: str):
    """Renvoie une chaîne dont le niveau est figé.

    `.partial(level=...)` crée un nouveau prompt avec une variable fixée.
    On peut donc construire des chaînes spécialisées sans dupliquer le template.
    """
    model = get_model().with_structured_output(Explanation)
    return EXPLAIN_PROMPT.partial(level=level) | model


# ============================================================================
# 4. Génération multi-niveaux en parallèle
# ============================================================================
# RunnableParallel exécute ses sous-chaînes EN CONCURRENCE et renvoie
# un dict avec les résultats. Les deux appels au LLM partent en même temps :
# tu paies la latence d'UN seul appel au lieu de deux.
# Ouvre LangSmith après l'exécution, tu verras visuellement les deux
# traces lancées simultanément.
# ============================================================================

def build_multi_level_chain():
    """Génère les versions débutant ET intermédiaire en un seul invoke."""
    return RunnableParallel(
        beginner=explain_chain_for_level("débutant"),
        intermediate=explain_chain_for_level("intermédiaire"),
    )


# ============================================================================
# 5. Variante streaming
# ============================================================================
# `with_structured_output` ne streame pas bien (le modèle doit produire
# l'objet complet pour qu'il soit valide). Pour du streaming token par
# token, on utilise une chaîne classique avec sortie texte.
# ============================================================================

STREAM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant pédagogique Python concis et précis."),
    ("human", "Explique en 2 paragraphes : {concept}"),
])


def build_streaming_chain():
    """Chaîne texte simple, idéale pour du streaming dans un terminal/UI."""
    return STREAM_PROMPT | get_model() | StrOutputParser()


# ============================================================================
# 6. Démo
# ============================================================================

def demo() -> None:
    concept = "les générateurs Python"

    # --- Démo 1 : sortie structurée mono-niveau ---
    print("=" * 70)
    print(f"DÉMO 1 — Explication structurée : '{concept}' (intermédiaire)")
    print("=" * 70)
    chain = explain_chain_for_level("intermédiaire")
    result: Explanation = chain.invoke({"concept": concept})

    # `result` est un vrai objet Python : tu peux faire result.intuition,
    # PyCharm autocomplète, mypy est content.
    print(f"\nIntuition :\n{result.intuition}\n")
    print(f"Code :\n{result.code_example}\n")
    print(f"Piège classique :\n{result.pitfall}\n")
    print("Questions de compréhension :")
    for i, q in enumerate(result.comprehension_questions, 1):
        print(f"  {i}. {q.question}")
        print(f"     → {q.expected_answer}")

    # --- Démo 2 : multi-niveaux en parallèle ---
    print("\n" + "=" * 70)
    print(f"DÉMO 2 — RunnableParallel : 2 niveaux en concurrence")
    print("=" * 70)
    multi = build_multi_level_chain()
    results = multi.invoke({"concept": concept})

    print("\n--- Pour un DÉBUTANT ---")
    print(results["beginner"].intuition)
    print("\n--- Pour un INTERMÉDIAIRE ---")
    print(results["intermediate"].intuition)

    # --- Démo 3 : streaming ---
    print("\n" + "=" * 70)
    print("DÉMO 3 — Streaming token par token")
    print("=" * 70)
    print()
    streamer = build_streaming_chain()
    for chunk in streamer.stream({"concept": "le GIL en Python"}):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    demo()

#################        Exercices pour aller plus loin       ################
# ============================================================================
# EXERCICES (pour aller plus loin avant la Phase 2)
# ============================================================================
#
# 1. FACILE — Ajoute un niveau "avancé" à la chaîne multi-niveaux et lance
#    les 3 en parallèle. Combien de temps ça prend vs. en séquentiel ?
#    (Astuce : `import time; t = time.perf_counter(); ...`)
#
# 2. MOYEN — Utilise `.batch()` plutôt que `.invoke()` pour expliquer une
#    LISTE de concepts en parallèle. Compare avec une boucle for + invoke.
#    `chain.batch([{"concept": c} for c in ["décorateurs", "métaclasses"]])`
#
# 3. MOYEN — Ajoute un champ `analogy: str` au schéma Pydantic (analogie
#    avec un objet du quotidien) et regarde comme c'est trivial.
#
# 4. STRETCH — Crée un router avec `RunnableBranch` qui choisit
#    automatiquement entre 3 sous-chaînes selon que le concept est
#    "syntaxe" (def, lambda...), "paradigme" (POO, fonctionnel...),
#    ou "écosystème" (pip, venv, pytest...). Utilise une mini-chaîne LLM
#    de classification en amont pour décider du routage.
#    Doc : https://python.langchain.com/docs/how_to/routing/
#
# 5. ASYNC — Réécris la démo en async avec `await chain.ainvoke(...)` et
#    `asyncio.gather(...)`. Compare avec RunnableParallel.
# ============================================================================
