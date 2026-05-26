"""
Test étendu : on demande au LLM de fournir AUSSI le code de tracé.

Trois choses sont déléguées au LLM :
  1. la résolution du problème de structure (positions au repos / déformées)
  2. le facteur d'amplification du déplacement pour la visualisation
  3. le code Python de tracé (matplotlib)

Aucun calcul ET aucun code de visualisation n'est écrit côté Python.
Seul l'orchestrateur (LangChain + exec) est encore présent.

ATTENTION SÉCURITÉ : ce script exécute du code généré par le LLM via exec().
Acceptable dans un cadre pédagogique contrôlé. Ne JAMAIS faire ça
en production sans un environnement isolé (subprocess, sandbox, docker).
"""

import os
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# ============================================================================
# Schéma : on ajoute un champ texte pour le code Python de tracé
# ============================================================================
class Node(BaseModel):
    label: str = Field(description="Identifiant du nœud")
    x_m: float = Field(description="Coordonnée x en mètres")
    y_m: float = Field(description="Coordonnée y en mètres")


class Bar(BaseModel):
    node_start: str
    node_end: str


class TrussSolutionWithCode(BaseModel):
    """Résolution complète d'un treillis + code matplotlib de tracé."""

    nodes_rest: list[Node] = Field(description="Positions des nœuds au repos")
    nodes_displaced: list[Node] = Field(
        description="Positions des nœuds après chargement (déplacements RÉELS, "
                    "sans amplification). Mêmes labels et ordre que nodes_rest."
    )
    bars: list[Bar] = Field(description="Connectivité des barres")
    displacement_scale: float = Field(
        default=1.0,
        description="Facteur d'amplification à appliquer aux déplacements pour la lisibilité."
    )
    plot_code: str = Field(
        description=(
            "Code Python complet d'une fonction plot_truss(geom). "
            "L'argument geom expose les attributs nodes_rest (list[Node]), "
            "nodes_displaced (list[Node]), bars (list[Bar]), displacement_scale (float). "
            "Chaque Node a .label, .x_m, .y_m. Chaque Bar a .node_start et .node_end. "
            "La fonction doit tracer en bleu plein la configuration au repos, en rouge "
            "pointillé la configuration déformée AMPLIFIÉE par displacement_scale, "
            "annoter les nœuds, conserver le ratio d'aspect, et afficher une légende. "
            "Imports nécessaires inclus dans le code. Ne pas inclure de markdown ni de "
            "```python```, juste du code Python brut."
        )
    )
    commentaire: str = Field(default="")


# ============================================================================
# Prompt : on précise juste le rôle, pas la méthode
# ============================================================================
SYSTEM = """Tu es un ingénieur structures. Tu réponds en fournissant les données
nécessaires à la visualisation graphique du résultat ET le code Python de tracé
correspondant."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{problem}"),
])

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
chain = prompt | llm.with_structured_output(TrussSolutionWithCode)


# ============================================================================
# Énoncé : identique au test précédent
# ============================================================================
PROBLEM = """
Soit un treillis plan composé de trois barres articulées concourant au nœud A,
situé à l'origine (0, 0).

Les trois autres extrémités des barres sont des appuis fixes :
  - B1 = (0, 1) m
  - B2 = (1, 1) m
  - B3 = (-1, 1) m

Toutes les barres ont les mêmes caractéristiques :
  - module d'Young E = 200 GPa
  - section transversale A = 100 mm²

Une charge ponctuelle F = (0, -10000) N est appliquée au nœud A
(force verticale de 10 kN dirigée vers le bas).

Fournis les données du treillis au repos et déformé, ainsi que le code Python
qui les trace sur un même graphique.
"""


# ============================================================================
# Exécution
# ============================================================================
if __name__ == "__main__":
    print("Interrogation du LLM (données + code matplotlib)...\n")
    result = chain.invoke({"problem": PROBLEM})

    print("=" * 70)
    print("DONNÉES RENVOYÉES PAR LE LLM")
    print("=" * 70)
    print(f"\nFacteur d'amplification : {result.displacement_scale}\n")
    print("Nœuds au repos :")
    for n in result.nodes_rest:
        print(f"  {n.label:>4} : ({n.x_m:+.6f}, {n.y_m:+.6f}) m")
    print("\nNœuds déformés (déplacements réels) :")
    for n in result.nodes_displaced:
        print(f"  {n.label:>4} : ({n.x_m:+.6f}, {n.y_m:+.6f}) m")
    if result.commentaire:
        print(f"\nCommentaire :\n{result.commentaire}")

    print()
    print("=" * 70)
    print("CODE PYTHON GÉNÉRÉ PAR LE LLM")
    print("=" * 70)
    print(result.plot_code)
    print("=" * 70)

    # Exécution du code généré (cadre pédagogique uniquement)
    print("\nExécution du code généré...\n")
    local_ns = {}
    exec(result.plot_code, local_ns)

    if "plot_truss" not in local_ns:
        raise RuntimeError("Le LLM n'a pas défini de fonction plot_truss().")

    local_ns["plot_truss"](result)