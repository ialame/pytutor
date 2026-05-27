"""Phase 3 — L'agent PyTutor (LangChain 1.x).

Concepts travaillés :
  1. Le décorateur @tool : transformer une fonction Python en outil pour le LLM
  2. La docstring comme contrat avec le modèle (le modèle LIT la docstring)
  3. create_agent : la voie officielle LangChain 1.x pour les agents ReAct
  4. La boucle Réfléchir -> Agir -> Observer en pratique
  5. Multi-tool : enchaîner plusieurs tools pour répondre à une question complexe
  6. Mémoire conversationnelle : maintenir l'historique entre tours

ARCHITECTURE
------------
On définit 4 tools qui s'appuient sur les phases précédentes :
  - search_python_docs : utilise le RAG de la Phase 2
  - run_python_code   : exécute du code Python en subprocess
  - generate_exercise : utilise la chaîne d'explication de la Phase 1
  - check_pep8        : vérifie le style avec pycodestyle

Puis on les passe à `create_agent`, qui construit la boucle ReAct.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from config import get_model
from phase1_explain import explain_chain_for_level
from phase2_rag import build_or_load_vectorstore


# ============================================================================
# 1. Les tools
# ============================================================================
# IMPORTANT : la docstring de chaque tool est envoyée AU MODÈLE comme description.
# Le modèle décide quand appeler quel tool en lisant cette docstring. Donc une
# docstring vague = agent qui prend de mauvaises décisions. Sois précis sur :
#   - À QUOI sert le tool
#   - QUAND l'utiliser (et quand ne pas l'utiliser)
#   - Ce que renvoient les arguments
# ============================================================================

# Cache lazy : on charge le retriever une seule fois, à la première utilisation
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        vs = build_or_load_vectorstore()
        _retriever = vs.as_retriever(search_kwargs={"k": 3})
    return _retriever


@tool
def search_python_docs(query: str) -> str:
    """Recherche dans la documentation Python officielle.

    À utiliser pour toute question FACTUELLE sur Python : syntaxe d'une
    construction, comportement standard d'une fonction built-in, vocabulaire
    technique. PAS pour générer du code ou évaluer du code de l'élève.

    Args:
        query: La question ou les mots-clés à rechercher dans la documentation.
    """
    docs = _get_retriever().invoke(query)
    if not docs:
        return "Aucun résultat dans la doc Python pour cette recherche."
    return "\n\n---\n\n".join(
        f"Source: {d.metadata.get('source', '?')}\n{d.page_content}"
        for d in docs
    )


@tool
def run_python_code(code: str) -> str:
    """Exécute du code Python dans un sous-processus et renvoie stdout + stderr.

    À utiliser quand :
      - l'élève montre un code qui ne marche pas (pour voir la vraie erreur)
      - tu veux vérifier qu'un exemple que tu vas donner fonctionne réellement
      - tu veux mesurer la sortie d'un bout de code pour pouvoir l'expliquer

    Limites : timeout 5 secondes, environnement Python standard, pas de pip.

    Args:
        code: Le code Python à exécuter, complet et autonome.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return (
            "ERREUR: timeout après 5 secondes. Le code boucle probablement "
            "à l'infini (boucle while, génération récursive sans condition d'arrêt, etc.)"
        )
    finally:
        Path(path).unlink(missing_ok=True)


@tool
def generate_exercise(topic: str, difficulty: str = "intermédiaire") -> str:
    """Génère un exercice Python sur un concept avec ses questions de compréhension.

    À utiliser quand l'élève demande explicitement de la pratique, des
    exercices, ou pour vérifier sa compréhension d'un concept.

    Args:
        topic: Le concept Python à pratiquer (ex: "décorateurs", "list comprehensions").
        difficulty: Le niveau visé. Valeurs valides : "débutant", "intermédiaire", "avancé".
    """
    if difficulty not in {"débutant", "intermédiaire", "avancé"}:
        difficulty = "intermédiaire"

    chain = explain_chain_for_level(difficulty)
    explanation = chain.invoke({"concept": topic})

    questions = "\n".join(
        f"  Q{i}. {q.question}\n     → Réponse attendue : {q.expected_answer}"
        for i, q in enumerate(explanation.comprehension_questions, 1)
    )
    return (
        f"Exercice sur '{topic}' (niveau {difficulty}) :\n\n"
        f"Code de référence :\n{explanation.code_example}\n\n"
        f"Questions :\n{questions}\n\n"
        f"Piège classique : {explanation.pitfall}"
    )


@tool
def check_pep8(code: str) -> str:
    """Vérifie la conformité PEP 8 d'un bout de code Python.

    À utiliser quand l'élève partage du code et veut savoir s'il est
    propre, ou quand tu donnes un exemple que tu veux valider stylistiquement.

    Args:
        code: Le code Python à analyser.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pycodestyle", "--max-line-length=100", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if not result.stdout.strip():
            return "✓ Aucune violation PEP 8 détectée."
        # On masque le chemin du fichier temporaire pour ne pas embrouiller l'agent
        cleaned = result.stdout.replace(path, "<code>")
        return f"Problèmes PEP 8 détectés :\n{cleaned}"
    except FileNotFoundError:
        return (
            "Le module pycodestyle n'est pas installé. "
            "Réponds à l'élève en lui suggérant `pip install pycodestyle`."
        )
    finally:
        Path(path).unlink(missing_ok=True)


# ============================================================================
# 2. Construction de l'agent
# ============================================================================
# create_agent prend :
#   - model        : soit une instance, soit une string "provider:model"
#   - tools        : la liste de Python callables décorés @tool
#   - system_prompt: instructions générales (remplace l'ancien paramètre `prompt`)
# Il renvoie un graphe LangGraph compilé, prêt à être invoke/stream.
# ============================================================================

TOOLS = [search_python_docs, run_python_code, generate_exercise, check_pep8]

SYSTEM_PROMPT = """Tu es PyTutor, un assistant pédagogique pour des élèves apprenant Python.

Tu as accès à 4 outils :
  - search_python_docs : pour récupérer du contenu de la doc Python officielle
  - run_python_code    : pour exécuter du code et observer ce qui se passe
  - generate_exercise  : pour générer un exercice et ses questions
  - check_pep8         : pour vérifier le style PEP 8

Règles de comportement :
1. Pour toute question factuelle sur Python (syntaxe, builtin, vocabulaire),
   utilise search_python_docs AVANT de répondre. Cite la source dans ta réponse.
2. Quand l'élève montre un code qui ne marche pas, exécute-le toi-même avec
   run_python_code AVANT d'expliquer pourquoi, pour voir la vraie erreur.
3. Quand tu donnes un exemple de code dans ta réponse, exécute-le d'abord pour
   être sûr qu'il marche.
4. Sois pédagogique : explique POURQUOI, pas seulement COMMENT.
5. Adapte ton vocabulaire au niveau de l'élève (déduis-le de ses questions).
6. Réponds en français.
"""


def build_agent():
    """Construit l'agent PyTutor avec ses 4 tools."""
    return create_agent(
        model=get_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# ============================================================================
# 3. Affichage lisible des traces
# ============================================================================
# L'agent renvoie un dict avec une clé "messages" contenant TOUT l'historique :
# system message, human message, ai messages (avec leurs tool_calls éventuels),
# tool messages (les résultats), réponse finale. Cette fonction formate ça
# proprement pour qu'on voie le déroulé de la boucle ReAct.
# ============================================================================

def print_trace(messages, from_index: int = 0) -> None:
    """Affiche le déroulé de la boucle ReAct depuis un index donné."""
    for msg in messages[from_index:]:
        if isinstance(msg, SystemMessage):
            continue  # on n'affiche pas le system prompt
        elif isinstance(msg, HumanMessage):
            print(f"\n👤 USER:\n  {msg.content}")
        elif isinstance(msg, AIMessage):
            # Un AIMessage peut contenir du texte ET/OU des appels de tools
            if msg.content:
                print(f"\n🤖 AI:\n  {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\n🤖 AI décide d'appeler {len(msg.tool_calls)} tool(s) :")
                for tc in msg.tool_calls:
                    args_preview = str(tc["args"])[:100]
                    print(f"   → {tc['name']}({args_preview}{'...' if len(str(tc['args'])) > 100 else ''})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:300].replace("\n", "\n     ")
            ellipsis = "..." if len(msg.content) > 300 else ""
            print(f"\n🔧 TOOL[{msg.name}]:\n     {preview}{ellipsis}")


# ============================================================================
# 4. Démos
# ============================================================================

def demo_factual_question() -> None:
    """Question factuelle : devrait déclencher search_python_docs."""
    print("\n" + "=" * 70)
    print("DÉMO 1 — Question factuelle")
    print("=" * 70)
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content="À quoi sert exactement le mot-clé `yield` ?")]
    })
    print_trace(result["messages"])


def demo_buggy_code() -> None:
    """Code qui boucle à l'infini : devrait déclencher run_python_code."""
    print("\n" + "=" * 70)
    print("DÉMO 2 — Debugging d'un code qui boucle")
    print("=" * 70)

    buggy = """\
def double_each(numbers):
    for n in numbers:
        numbers.append(n * 2)
    return numbers

print(double_each([1, 2, 3]))
"""
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Mon code ne se termine jamais. Tu peux m'aider ?\n\n"
            f"```python\n{buggy}```"
        ))]
    })
    print_trace(result["messages"])


def demo_multi_tool() -> None:
    """Question complexe : devrait chaîner plusieurs tools."""
    print("\n" + "=" * 70)
    print("DÉMO 3 — Multi-tool (explication + exercice)")
    print("=" * 70)
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Explique-moi les list comprehensions en t'appuyant sur la doc, "
            "puis génère-moi un exercice de niveau intermédiaire pour pratiquer."
        ))]
    })
    print_trace(result["messages"])


def chat_loop() -> None:
    """Boucle de chat interactive avec MÉMOIRE conversationnelle.

    Note importante sur la mémoire : create_agent ne stocke RIEN entre les
    invocations. Pour avoir une conversation continue, on doit accumuler les
    messages côté client et les renvoyer à chaque tour. C'est exactement ce
    qu'on fait ci-dessous.
    """
    print("\n" + "=" * 70)
    print("MODE CHAT — Ctrl+C ou ligne vide pour quitter")
    print("=" * 70)

    agent = build_agent()
    history: list = []  # accumulera HumanMessage / AIMessage / ToolMessage

    while True:
        try:
            user_input = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt.")
            return

        if not user_input:
            print("Au revoir.")
            return

        # On ajoute le nouveau message au contexte complet
        history.append(HumanMessage(content=user_input))
        index_before = len(history) - 1

        # L'agent renvoie l'historique complet (input + tous les nouveaux messages)
        result = agent.invoke({"messages": history})
        history = result["messages"]

        # On n'affiche que les messages produits ce tour-ci
        print_trace(history, from_index=index_before)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo_factual_question()
    demo_buggy_code()
    demo_multi_tool()
    # Décommente pour passer en mode chat interactif :
    chat_loop()


# ============================================================================
# EXERCICES
# ============================================================================
#
# 1. FACILE — Ajoute un 5e tool `current_python_version()` qui renvoie
#    sys.version. Pose une question genre "quelle version de Python tu
#    utilises ?" et regarde l'agent l'appeler.
#
# 2. MOYEN — Active LangSmith (.env) et observe une trace complète d'agent.
#    Tu verras visuellement la boucle Réfléchir/Agir/Observer, et pour
#    chaque appel modèle tu verras la liste de tools envoyée + la décision
#    prise. C'est de loin la meilleure ressource pédagogique pour
#    comprendre comment un agent "pense".
#
# 3. MOYEN — Modifie run_python_code pour autoriser un timeout configurable
#    (5s par défaut, mais paramétrable). Attention : les paramètres avec
#    valeur par défaut sont aussi exposés au modèle, qui peut décider de
#    les surcharger. Vérifie en posant une question qui pourrait nécessiter
#    plus de temps.
#
# 4. STRETCH — Ajoute un tool `evaluate_student_answer(question, answer)`
#    qui prend une question et la réponse de l'élève, et retourne une
#    évaluation (correct/incorrect + explication). Utilise une mini-chaîne
#    LCEL avec sortie structurée Pydantic à l'intérieur du tool. C'est le
#    pattern "tool qui appelle un autre LLM en interne".
#
# 5. STRETCH — Passe en mode streaming avec agent.stream() au lieu de
#    .invoke(). Affiche les tokens au fur et à mesure et les appels de
#    tools en temps réel. C'est l'UX qu'on attend d'un vrai chatbot.
# ============================================================================
