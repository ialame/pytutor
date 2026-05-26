"""Solutions des exercices de la Phase 3.

Chaque exercice est isolé dans sa fonction et peut être lancé séparément.
L'exercice 2 (LangSmith) nécessite que .env soit configuré.
"""
from __future__ import annotations

import os
import sys

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_model
from phase3_agent import (
    TOOLS,
    SYSTEM_PROMPT,
    build_agent,
    print_trace,
    search_python_docs,
    generate_exercise,
    check_pep8,
)


# ============================================================================
# EXERCICE 1 — Ajouter un 5e tool current_python_version()
# ============================================================================
# Objectif : montrer la trivialité d'ajouter un tool. La seule chose à faire,
# c'est d'écrire une fonction @tool et de l'ajouter à la liste passée à
# create_agent. Le modèle découvre automatiquement le nouvel outil grâce à son
# nom et sa docstring.
# ============================================================================

@tool
def current_python_version() -> str:
    """Renvoie la version Python du processus qui exécute PyTutor.

    À utiliser quand l'élève pose des questions sur la compatibilité, les
    features récentes (match/case en 3.10, ExceptionGroup en 3.11, etc.),
    ou simplement quand il demande quelle version est utilisée.
    """
    return f"Python {sys.version.split()[0]} ({sys.platform})"


def exercise_1() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 1 — Ajout d'un tool current_python_version()")
    print("=" * 70)

    # On rajoute simplement le nouveau tool à la liste existante.
    agent = create_agent(
        model=get_model(),
        tools=TOOLS + [current_python_version],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Quelle version de Python utilises-tu pour exécuter mon code ? "
            "Est-ce que je peux utiliser le pattern matching avec `match` / `case` ?"
        ))]
    })
    print_trace(result["messages"])


# ============================================================================
# EXERCICE 2 — LangSmith : observer une trace d'agent
# ============================================================================
# Objectif : générer une trace dans LangSmith pour observer visuellement la
# boucle ReAct. C'est de loin la meilleure pédagogie pour comprendre comment
# un agent "pense".
# ============================================================================

def exercise_2_langsmith() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 2 — Tracer un agent dans LangSmith")
    print("=" * 70)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"true", "1"}
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    print(f"\nLANGCHAIN_TRACING_V2 : {'✓' if tracing else '✗'}")
    print(f"LANGCHAIN_API_KEY    : {'✓' if api_key else '✗'}")
    print(f"LANGCHAIN_PROJECT    : {project}")

    if not (tracing and api_key):
        print("\n→ Configure ces variables dans ton .env puis relance.")
        return

    # On lance une question MULTI-TOOL pour avoir une trace riche
    print("\nExécution d'un agent multi-tool (la trace part vers LangSmith)...")
    agent = build_agent()
    agent.invoke({
        "messages": [HumanMessage(content=(
            "Cherche dans la doc ce qu'est un dictionnaire en Python, "
            "puis génère un exercice débutant sur ce sujet, "
            "et vérifie que le code de référence est PEP 8 compliant."
        ))]
    })
    print("✓ Trace envoyée.")

    print("\n" + "-" * 70)
    print("CE QU'IL FAUT REGARDER DANS L'UI LANGSMITH :")
    print(f"  https://smith.langchain.com → projet '{project}'")
    print()
    print("  1. Tu vois une trace 'LangGraph' (l'agent EST un graphe LangGraph)")
    print("  2. Clique : tu vois l'arbre d'exécution.")
    print("     - Noeud 'agent' (model call) : Claude réfléchit, décide d'appeler")
    print("       search_python_docs avec ses arguments. Tu peux voir le prompt")
    print("       EXACT envoyé : system + history + schémas des tools.")
    print("     - Noeud 'tools' : exécution du tool, sortie complète.")
    print("     - Re-noeud 'agent' : Claude reçoit le résultat, décide d'appeler")
    print("       generate_exercise.")
    print("     - ... etc, jusqu'à la décision finale 'pas de tool, réponse'.")
    print("  3. Chaque appel modèle montre les tokens consommés et la latence.")
    print("  4. Insight clé : tu vois CE QUE LE MODÈLE A VU à chaque tour. Il")
    print("     n'a pas de 'mémoire magique' — c'est juste l'historique qui")
    print("     grossit et qu'on lui repasse à chaque fois.")


# ============================================================================
# EXERCICE 3 — Tool avec paramètre timeout configurable
# ============================================================================
# Objectif : montrer qu'un paramètre avec valeur par défaut est exposé au
# modèle, qui peut décider de le surcharger selon la situation.
#
# Subtilité importante : la valeur par défaut n'est PAS un secret côté agent.
# Elle apparaît dans le schéma du tool envoyé au modèle, et le modèle décide
# lui-même quand l'override en se basant sur le contexte de la question.
# ============================================================================

@tool
def run_python_code_v2(code: str, timeout_seconds: int = 5) -> str:
    """Exécute du code Python et renvoie stdout + stderr.

    Args:
        code: Le code Python à exécuter, complet et autonome.
        timeout_seconds: Délai max avant interruption. Défaut : 5 secondes.
            Augmente cette valeur SEULEMENT si le code requiert manifestement
            un calcul long (entraînement, traitement de gros datasets, etc.).
            Maximum raisonnable : 60 secondes.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    timeout_seconds = max(1, min(timeout_seconds, 60))  # clamp défensif

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        parts.append(f"Exit code: {result.returncode} (timeout utilisé: {timeout_seconds}s)")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return (
            f"ERREUR: timeout après {timeout_seconds} secondes. "
            "Si tu pensais que le code devait être long, augmente timeout_seconds. "
            "Sinon, il y a sans doute une boucle infinie ou une récursion non terminale."
        )
    finally:
        Path(path).unlink(missing_ok=True)


def exercise_3() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 3 — Tool avec paramètre timeout_seconds configurable")
    print("=" * 70)

    # On remplace l'ancien run_python_code par v2 dans la liste de tools
    tools_v2 = [t for t in TOOLS if t.name != "run_python_code"] + [run_python_code_v2]

    agent = create_agent(
        model=get_model(),
        tools=tools_v2,
        system_prompt=SYSTEM_PROMPT,
    )

    # Question 1 : code rapide, l'agent ne devrait PAS surcharger le timeout
    print("\n--- Cas 1 : code rapide (timeout par défaut attendu) ---")
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Exécute ce code pour me dire la sortie :\n"
            "```python\nprint(sum(range(100)))\n```"
        ))]
    })
    print_trace(result["messages"])

    # Question 2 : code volontairement long, l'agent devrait surcharger
    print("\n\n--- Cas 2 : calcul long, l'agent doit augmenter le timeout ---")
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "J'ai écrit un code qui calcule la somme des nombres premiers "
            "jusqu'à 5 millions. Ça met une bonne dizaine de secondes à tourner "
            "sur ma machine. Tu peux l'exécuter pour vérifier le résultat ?\n\n"
            "```python\n"
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, int(n**0.5)+1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n"
            "\n"
            "total = sum(n for n in range(2, 5_000_000) if is_prime(n))\n"
            "print(total)\n"
            "```"
        ))]
    })
    print_trace(result["messages"])

    print("\n" + "-" * 70)
    print("OBSERVATIONS :")
    print(" - Au cas 1, l'agent appelle le tool SANS spécifier timeout_seconds")
    print("   (il garde la valeur par défaut 5s).")
    print(" - Au cas 2, l'agent voit dans ta question que le code prend ~10s,")
    print("   et passe explicitement timeout_seconds=30 (ou similaire).")
    print(" - C'est UNE preuve que la docstring conditionne le comportement :")
    print("   la phrase 'Augmente cette valeur SEULEMENT si...' est lue et")
    print("   appliquée par le modèle.")


# ============================================================================
# EXERCICE 4 — Tool qui appelle un sous-LLM (evaluate_student_answer)
# ============================================================================
# Objectif : pattern de composition "tool dans lequel se cache un LLM".
# Ici on évalue la réponse d'un élève via une mini-chaîne LCEL avec sortie
# structurée Pydantic. Vu de l'agent, c'est un tool comme un autre. Vu de
# l'intérieur, c'est un appel LLM avec son propre prompt et schéma.
# ============================================================================

class AnswerEvaluation(BaseModel):
    """Évaluation pédagogique d'une réponse d'élève."""

    is_correct: bool = Field(
        description="True si la réponse est essentiellement correcte, "
                    "False si elle contient une erreur substantielle."
    )
    score: int = Field(
        description="Note de 0 à 10. 0 = totalement faux, 10 = parfait. "
                    "Tiens compte de la précision technique ET de la clarté."
    )
    feedback: str = Field(
        description="Feedback pédagogique en 2-3 phrases pour l'élève. "
                    "Souligne ce qui est juste, puis ce qu'il faut corriger ou préciser."
    )
    missing_concepts: list[str] = Field(
        description="Concepts importants que la réponse aurait dû mentionner mais omet. "
                    "Liste vide si la réponse est complète."
    )


# La mini-chaîne LCEL utilisée à l'intérieur du tool. On la construit une seule
# fois en module-level pour éviter de la recréer à chaque appel.
_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un évaluateur pédagogique de réponses Python. Sois exigeant "
     "techniquement mais bienveillant dans la formulation."),
    ("human",
     "Question posée à l'élève :\n{question}\n\n"
     "Réponse de l'élève :\n{answer}\n\n"
     "Évalue cette réponse."),
])

# Note : on utilise temperature=0.0 ici parce qu'une éval doit être reproductible.
_evaluator_chain = (
    _eval_prompt
    | get_model().with_structured_output(AnswerEvaluation)
)


@tool
def evaluate_student_answer(question: str, answer: str) -> str:
    """Évalue la réponse d'un élève à une question technique sur Python.

    À utiliser quand un élève a écrit une réponse et que tu veux la noter
    objectivement plutôt que de te contenter d'un "oui" ou "non" qualitatif.
    Renvoie un score, un feedback, et la liste des concepts manquants.

    Args:
        question: La question posée à l'élève.
        answer: La réponse complète de l'élève à évaluer.
    """
    # Sous-LLM : on appelle une mini-chaîne LCEL avec sortie structurée.
    result = _evaluator_chain.invoke({"question": question, "answer": answer})

    # On reformate l'objet Pydantic en string pour l'agent.
    # (Les tools renvoient toujours du texte, c'est leur protocole.)
    missing = ", ".join(result.missing_concepts) if result.missing_concepts else "(aucun)"
    return (
        f"Correct : {'oui' if result.is_correct else 'non'}\n"
        f"Note : {result.score}/10\n"
        f"Concepts manquants : {missing}\n"
        f"Feedback : {result.feedback}"
    )


def exercise_4() -> None:
    print("\n" + "=" * 70)
    print("EXERCICE 4 — Tool 'evaluate_student_answer' (sous-LLM caché)")
    print("=" * 70)

    tools_with_eval = TOOLS + [evaluate_student_answer]
    agent = create_agent(
        model=get_model(),
        tools=tools_with_eval,
        system_prompt=SYSTEM_PROMPT,
    )

    # Scénario réaliste : l'enseignant donne la question ET la réponse
    # de l'élève à l'agent, en demandant une évaluation objective.
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Un élève a répondu à une question d'examen. Évalue-la pour moi "
            "et formule un commentaire que je pourrais lui envoyer.\n\n"
            "Question : À quoi sert le mot-clé `yield` en Python et en quoi "
            "diffère-t-il de `return` ?\n\n"
            "Réponse de l'élève : « `yield` permet de retourner plusieurs "
            "valeurs depuis une fonction, contrairement à `return` qui n'en "
            "retourne qu'une seule. »"
        ))]
    })
    print_trace(result["messages"])

    print("\n" + "-" * 70)
    print("ARCHITECTURE EN COUCHES À VISUALISER :")
    print(" 1. AGENT PRINCIPAL (Claude) : décide d'appeler evaluate_student_answer.")
    print(" 2. TOOL evaluate_student_answer : Python exécute la fonction.")
    print(" 3. SOUS-LLM (Claude à nouveau) : appelé en interne par la mini-chaîne.")
    print("    Reçoit son propre prompt, renvoie un objet Pydantic structuré.")
    print(" 4. Le tool reformate l'objet en string et le rend à l'agent.")
    print(" 5. L'agent intègre le résultat et formule sa réponse à l'utilisateur.")
    print()
    print(" → 2 appels LLM dans ce tour, pour 1 invocation d'agent.")
    print(" → Si tu actives LangSmith, tu verras les 2 traces emboîtées.")


# ============================================================================
# EXERCICE 5 — Streaming avec agent.stream()
# ============================================================================
# Objectif : passer de .invoke() à .stream() pour avoir une UX réactive.
# create_agent expose plusieurs modes de stream :
#   - "updates"  : on reçoit chaque MISE À JOUR du graphe (node terminé)
#                  → idéal pour afficher "l'agent a appelé tel tool"
#   - "messages" : on reçoit chaque TOKEN du modèle au fur et à mesure
#                  → idéal pour l'effet machine à écrire
#   - "values"   : on reçoit l'état COMPLET à chaque étape (verbeux)
# ============================================================================

def exercise_5_updates() -> None:
    """Streaming au niveau des ÉTAPES (mode 'updates')."""
    print("\n" + "=" * 70)
    print("EXERCICE 5a — Streaming par étapes (stream_mode='updates')")
    print("=" * 70)

    agent = build_agent()
    print("\nQuestion envoyée. Affichage des étapes au fil de l'eau :\n")

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=(
            "Cherche dans la doc ce qu'est un itérable, "
            "puis génère un exercice débutant sur ce sujet."
        ))]},
        stream_mode="updates",
    ):
        # chunk est un dict {nom_du_noeud: {clés_modifiées: valeurs}}.
        # Avec create_agent, on a deux noeuds principaux : 'agent' et 'tools'.
        for node_name, node_update in chunk.items():
            messages = node_update.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [{node_name}] → appel {tc['name']}({list(tc['args'].keys())})")
                elif hasattr(msg, "name") and msg.name:
                    # ToolMessage : a un .name pointant vers le tool qui a renvoyé
                    preview = (msg.content or "")[:80].replace("\n", " ")
                    print(f"  [{node_name}] ← retour de {msg.name}: {preview}...")
                elif msg.content:
                    preview = msg.content[:120].replace("\n", " ")
                    print(f"  [{node_name}] 💬 {preview}...")


def exercise_5_tokens() -> None:
    """Streaming token par token (mode 'messages')."""
    print("\n" + "=" * 70)
    print("EXERCICE 5b — Streaming token par token (stream_mode='messages')")
    print("=" * 70)

    agent = build_agent()
    print("\nRéponse qui s'affiche au fur et à mesure (effet machine à écrire) :\n")

    # Mode 'messages' : on reçoit des tuples (chunk_message, metadata).
    # Le chunk est un AIMessageChunk (ou ToolMessage selon le noeud).
    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=(
            "Explique-moi en deux phrases ce qu'est une compréhension de liste."
        ))]},
        stream_mode="messages",
    ):
        # On ne stream que le texte produit par le noeud 'agent' (pas les tools)
        if metadata.get("langgraph_node") == "agent" and chunk.content:
            # chunk.content peut être un string ou une liste de blocs selon la version.
            text = chunk.content if isinstance(chunk.content, str) else "".join(
                b.get("text", "") for b in chunk.content if isinstance(b, dict)
            )
            print(text, end="", flush=True)
    print("\n")


def exercise_5() -> None:
    exercise_5_updates()
    exercise_5_tokens()

    print("-" * 70)
    print("QUAND UTILISER QUEL MODE :")
    print(" - 'updates'  : UI qui montre 'l'agent réfléchit / l'agent utilise X'.")
    print("                Idéal pour les workflows long-running où l'utilisateur")
    print("                attendrait sinon dans le vide.")
    print(" - 'messages' : effet machine à écrire, comme ChatGPT et Claude.ai.")
    print("                Indispensable pour une UX de chat moderne.")
    print(" - On combine souvent les deux : on stream les tokens du texte final,")
    print("   et on affiche des badges 'utilise tool X' pour les étapes intermédiaires.")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    exercise_1()
    exercise_3()
    exercise_4()
    exercise_5()

    # Décommente si LangSmith est configuré dans ton .env :
    # exercise_2_langsmith()

    print("\n" + "=" * 70)
    print("✓ Solutions de la Phase 3 exécutées.")
    print("  → Pour LangSmith, configure .env et décommente exercise_2_langsmith()")
    print("=" * 70)


if __name__ == "__main__":
    main()
