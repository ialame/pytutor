"""Solutions des exercices de la Phase 1.

Lance ce fichier pour exécuter les 5 démos dans l'ordre.
Chaque exercice est isolé dans sa propre fonction pour pouvoir être étudié
ou relancé séparément.
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
)

from config import get_model
# On réutilise le schéma et le prompt de la Phase 1
from phase1_explain import (
    Explanation,
    EXPLAIN_PROMPT,
    explain_chain_for_level,
)


# ============================================================================
# EXERCICE 1 — Multi-niveaux 3x : parallèle vs séquentiel
# ============================================================================
# Objectif : montrer concrètement que RunnableParallel ne paye PAS la somme
# des latences mais le MAX. Avec 3 appels qui prennent ~2s chacun, le séquentiel
# tombe à ~6s, le parallèle à ~2s.
# ============================================================================

def exercise_1() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 1 — Parallèle vs séquentiel sur 3 niveaux")
    print("=" * 70)

    concept = "les décorateurs Python"

    # --- Construction des 3 chaînes spécialisées ---
    beginner = explain_chain_for_level("débutant")
    intermediate = explain_chain_for_level("intermédiaire")
    advanced = explain_chain_for_level("avancé")

    # --- Version SÉQUENTIELLE : 3 invokes successifs ---
    t0 = time.perf_counter()
    seq_results = {
        "beginner": beginner.invoke({"concept": concept}),
        "intermediate": intermediate.invoke({"concept": concept}),
        "advanced": advanced.invoke({"concept": concept}),
    }
    seq_time = time.perf_counter() - t0
    print(f"\nSéquentiel : {seq_time:.2f}s pour 3 niveaux")

    # --- Version PARALLÈLE : un seul invoke sur un RunnableParallel ---
    parallel = RunnableParallel(
        beginner=beginner,
        intermediate=intermediate,
        advanced=advanced,
    )
    t0 = time.perf_counter()
    par_results = parallel.invoke({"concept": concept})
    par_time = time.perf_counter() - t0
    print(f"Parallèle  : {par_time:.2f}s pour 3 niveaux")
    print(f"→ Speedup  : ×{seq_time / par_time:.2f}")

    # On affiche l'intuition avancée pour vérifier que ça a bien tourné
    print(f"\nExtrait niveau avancé :\n{par_results['advanced'].intuition[:200]}...")


# ============================================================================
# EXERCICE 2 — batch() vs boucle for + invoke
# ============================================================================
# Objectif : montrer que .batch() paralléliser plusieurs INPUTS sur la MÊME
# chaîne. À ne pas confondre avec RunnableParallel qui parallélise plusieurs
# CHAÎNES sur le même input.
#
# Règle simple :
#   - plusieurs inputs, même chaîne   -> .batch(inputs)
#   - même input, plusieurs chaînes   -> RunnableParallel(a=c1, b=c2, ...)
# ============================================================================

def exercise_2() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 2 — batch() vs boucle for")
    print("=" * 70)

    concepts = ["les décorateurs", "les métaclasses", "les générateurs", "le GIL"]
    chain = explain_chain_for_level("intermédiaire")

    # --- Version BOUCLE : un invoke après l'autre ---
    t0 = time.perf_counter()
    loop_results = [chain.invoke({"concept": c}) for c in concepts]
    loop_time = time.perf_counter() - t0
    print(f"\nBoucle for : {loop_time:.2f}s pour {len(concepts)} concepts")

    # --- Version BATCH : un seul appel, parallélisé sous le capot ---
    inputs = [{"concept": c} for c in concepts]
    t0 = time.perf_counter()
    batch_results = chain.batch(inputs)
    batch_time = time.perf_counter() - t0
    print(f"batch()    : {batch_time:.2f}s pour {len(concepts)} concepts")
    print(f"→ Speedup  : ×{loop_time / batch_time:.2f}")

    # Note : tu peux contrôler le degré de parallélisme avec max_concurrency
    # pour éviter de saturer les rate limits :
    #   chain.batch(inputs, config={"max_concurrency": 5})

    print(f"\nExemple ({concepts[0]}) :\n{batch_results[0].intuition[:200]}...")


# ============================================================================
# EXERCICE 3 — Schéma Pydantic enrichi avec un champ "analogy"
# ============================================================================
# Objectif : montrer la trivialité d'ajouter un champ. La seule chose à faire,
# c'est étendre le schéma. Le modèle remplit automatiquement le nouveau champ
# grâce au mécanisme de tool-calling sous-jacent.
#
# C'est LA force de with_structured_output : tu modifies juste ta classe
# Pydantic, et le contrat avec le LLM se met à jour automatiquement.
# ============================================================================

class ExplanationWithAnalogy(BaseModel):
    """Version enrichie : on rajoute une analogie du quotidien."""

    concept: str = Field(description="Le concept expliqué.")
    intuition: str = Field(description="Explication intuitive en 3 phrases.")
    analogy: str = Field(
        description=(
            "Une analogie avec un objet ou une situation du quotidien "
            "(cuisine, sport, bureau...) qui éclaire le concept pour un débutant. "
            "Une seule phrase percutante."
        )
    )
    code_example: str = Field(description="Un exemple de code minimal et commenté.")
    pitfall: str = Field(description="Un piège classique sur ce concept.")


def exercise_3() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 3 — Schéma Pydantic enrichi avec un champ analogy")
    print("=" * 70)

    # Une seule ligne change par rapport à la Phase 1 :
    # le schéma passé à with_structured_output.
    model = get_model().with_structured_output(ExplanationWithAnalogy)
    chain = EXPLAIN_PROMPT.partial(level="débutant") | model

    for concept in ["les décorateurs", "le polymorphisme", "le contexte avec `with`"]:
        result = chain.invoke({"concept": concept})
        print(f"\n--- {concept} ---")
        print(f"Analogie : {result.analogy}")


# ============================================================================
# EXERCICE 4 — Router avec RunnableBranch et classifieur LLM
# ============================================================================
# Objectif : faire dispatcher dynamiquement vers 3 sous-chaînes selon le type
# de concept. Pattern fondamental, qu'on retrouvera partout en production.
#
# Architecture :
#
#   {"concept": "lambda"}
#         |
#         |  RunnablePassthrough.assign(category=classifier_chain)
#         v
#   {"concept": "lambda", "category": "syntaxe"}
#         |
#         |  RunnableBranch (3 conditions + un défaut)
#         v
#   réponse spécialisée syntaxe
# ============================================================================

class CategoryResult(BaseModel):
    """Catégorie d'un concept Python pour router vers la bonne sous-chaîne."""

    category: Literal["syntaxe", "paradigme", "ecosysteme"] = Field(
        description=(
            "Catégorise le concept :\n"
            " - 'syntaxe' : éléments du langage (def, lambda, comprehensions, with, yield...)\n"
            " - 'paradigme' : concepts de design (POO, fonctionnel, héritage, polymorphisme...)\n"
            " - 'ecosysteme' : outils/bibliothèques (pip, venv, pytest, numpy, requests...)"
        )
    )


def build_classifier():
    """Mini-chaîne qui catégorise un concept en 3 classes."""
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu classes des concepts Python en 3 catégories. Ne réponds que par la catégorie."),
        ("human", "Concept à catégoriser : {concept}"),
    ])
    # Important : on récupère .category (le string) plutôt que l'objet entier
    # pour pouvoir l'utiliser facilement dans les prédicats de RunnableBranch.
    return classify_prompt | get_model().with_structured_output(CategoryResult) | RunnableLambda(lambda r: r.category)


def build_specialized_chain(focus: str):
    """Construit une sous-chaîne avec un prompt système orienté `focus`."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant Python. Le concept à expliquer relève de la catégorie "
         f"'{focus}'. Adapte ton explication en conséquence :\n" +
         {
             "syntaxe":   "insiste sur la syntaxe exacte, montre un mini exemple et un anti-exemple.",
             "paradigme": "insiste sur la philosophie, le 'pourquoi', les implications de design.",
             "ecosysteme": "insiste sur l'usage pratique, l'installation, les commandes shell typiques.",
         }[focus]),
        ("human", "Explique : {concept}"),
    ])
    return prompt | get_model()


def build_router():
    """Le routeur complet : classifie puis branche."""
    classifier = build_classifier()
    syntaxe_chain    = build_specialized_chain("syntaxe")
    paradigme_chain  = build_specialized_chain("paradigme")
    ecosysteme_chain = build_specialized_chain("ecosysteme")

    # RunnableBranch prend N couples (prédicat, chaîne) + une chaîne par défaut.
    # Le prédicat reçoit le dict courant et renvoie True/False.
    branch = RunnableBranch(
        (lambda x: x["category"] == "syntaxe",    syntaxe_chain),
        (lambda x: x["category"] == "paradigme",  paradigme_chain),
        ecosysteme_chain,  # défaut : si rien ne matche
    )

    # On enrichit le dict d'entrée avec la catégorie, puis on branche.
    # Note : .assign() conserve la clé "concept" pour que les sous-chaînes
    # puissent toujours lire {concept} dans leur template.
    return RunnablePassthrough.assign(category=classifier) | branch


def exercise_4() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 4 — Router avec RunnableBranch")
    print("=" * 70)

    router = build_router()

    test_concepts = [
        "les list comprehensions",   # syntaxe attendue
        "le polymorphisme",           # paradigme attendu
        "pytest",                     # écosystème attendu
        "lambda",                     # syntaxe
    ]

    # On veut aussi voir la catégorie choisie. On reconstruit le pipeline en
    # gardant la catégorie dans la sortie pour la démo (en prod on s'en passe).
    classifier = build_classifier()
    debug_chain = RunnablePassthrough.assign(
        category=classifier,
        answer=router,
    )

    for concept in test_concepts:
        result = debug_chain.invoke({"concept": concept})
        category = result["category"]
        # AIMessage -> on prend les 200 premiers caractères du contenu textuel
        snippet = result["answer"].content[:200].replace("\n", " ")
        print(f"\n[{category:>10}] {concept}")
        print(f"             → {snippet}...")


# ============================================================================
# EXERCICE 5 — Async natif avec asyncio.gather
# ============================================================================
# Objectif : montrer la voie async pure et comparer avec RunnableParallel.
#
# Conclusion attendue : les deux donnent des temps quasi identiques parce
# que RunnableParallel utilise asyncio sous le capot. La différence est
# stylistique :
#   - asyncio.gather : tu écris le contrôle d'exécution explicitement
#   - RunnableParallel : tu décris la composition, l'exécution est implicite
#
# En pratique, on utilise RunnableParallel dans les chaînes (composable),
# et asyncio.gather dans le code applicatif (boucle de chat, batch jobs...).
# ============================================================================

async def exercise_5() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 5 — Async avec asyncio.gather vs RunnableParallel")
    print("=" * 70)

    concept = "les générateurs Python"
    beginner = explain_chain_for_level("débutant")
    intermediate = explain_chain_for_level("intermédiaire")
    advanced = explain_chain_for_level("avancé")

    # --- Version asyncio.gather pure ---
    # On utilise .ainvoke (la variante async) sur chaque chaîne, et gather
    # attend toutes les coroutines en concurrence.
    t0 = time.perf_counter()
    beg_r, int_r, adv_r = await asyncio.gather(
        beginner.ainvoke({"concept": concept}),
        intermediate.ainvoke({"concept": concept}),
        advanced.ainvoke({"concept": concept}),
    )
    gather_time = time.perf_counter() - t0
    print(f"\nasyncio.gather    : {gather_time:.2f}s")

    # --- Version RunnableParallel avec .ainvoke (async aussi) ---
    parallel = RunnableParallel(beginner=beginner, intermediate=intermediate, advanced=advanced)
    t0 = time.perf_counter()
    results = await parallel.ainvoke({"concept": concept})
    parallel_time = time.perf_counter() - t0
    print(f"RunnableParallel  : {parallel_time:.2f}s")

    # --- Version synchrone pour la référence ---
    t0 = time.perf_counter()
    parallel.invoke({"concept": concept})
    sync_time = time.perf_counter() - t0
    print(f"Sync (RunnablePar): {sync_time:.2f}s")

    print(f"\nÉchantillon ({concept}, débutant) :")
    print(beg_r.intuition[:200] + "...")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    # Les 4 premiers exercices sont synchrones
    # exercise_1()
    # exercise_2()
    # exercise_3()
    exercise_4()
    # Le 5e est async, on l'enrobe dans asyncio.run
    #asyncio.run(exercise_5())

    print("\n" + "=" * 70)
    print("✓ Tous les exercices de la Phase 1 sont résolus.")
    print("=" * 70)


if __name__ == "__main__":
    main()
