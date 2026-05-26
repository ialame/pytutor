"""
Test des limites d'un LLM sur un treillis général à N barres.

Format des appuis (convention par DDL) :
  supports = {indice_noeud: [ex, ey], ...}
  où ex = 0 signifie déplacement bloqué selon x, ex = 1 signifie libre selon x
  (idem pour ey selon y). Permet notamment :
    [0, 0]  appui fixe (rotule)
    [1, 0]  appui glissant horizontal (rouleau sur sol)
    [0, 1]  appui glissant vertical
    [1, 1]  équivalent à "pas d'appui"
  Les nœuds non listés dans supports sont entièrement libres.

Entrée :
  - nodes      : liste de positions [[x, y], ...] indexées 0..N-1
  - bars       : liste de connectivités [[i, j], ...] par indices
  - supports   : dict {i: [ex, ey]} (voir ci-dessus)
  - loads      : dict {i: [Fx, Fy]} en N
  - section_A  : section commune en m²
  - material   : matériau (le LLM connaît E pour les matériaux courants)

Sortie du LLM :
  - displacements_m : [[ux, uy], ...] pour chaque nœud, en m
  - displacement_scale : facteur d'amplification pour visualisation
  - plot_code : fonction plot_truss(nodes, bars, displacements_m, supports, scale)
  - commentaire : libre
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
# Schéma de sortie
# ============================================================================
class GeneralTrussSolution(BaseModel):
    """Solution d'un treillis plan à N barres + code de tracé."""

    displacements_m: list[list[float]] = Field(
        description=(
            "Déplacements [ux, uy] de chaque nœud, en mètres. Une entrée par nœud "
            "dans l'ordre des nœuds en entrée. Convention : pour un nœud i listé "
            "dans supports avec [ex, ey], ex=0 impose ux=0 et ey=0 impose uy=0. "
            "Quand ex=1 (resp. ey=1), le DDL correspondant est libre. Les nœuds "
            "absents de supports sont entièrement libres."
        )
    )
    displacement_scale: float = Field(
        default=1.0,
        description="Facteur d'amplification recommandé pour visualiser la déformée."
    )
    plot_code: str = Field(
        description=(
            "Code Python d'une fonction plot_truss(nodes, bars, displacements_m, "
            "supports, scale). Arguments : "
            "nodes (list[[float, float]]), "
            "bars (list[[int, int]]), "
            "displacements_m (list[[float, float]]), "
            "supports (dict[int, [int, int]] avec [ex, ey], 0=bloqué, 1=libre), "
            "scale (float). "
            "La fonction trace les barres au repos en bleu plein, les barres "
            "déformées (positions au repos + scale*déplacement) en rouge pointillé "
            "sur le même graphique. Pour chaque nœud d'appui, choisis un marqueur "
            "qui reflète le type : un triangle pour [0,0] (rotule fixe), un cercle "
            "barré ou losange pour les appuis partiels [1,0] ou [0,1]. Les nœuds "
            "libres en cercle plein. Annote chaque nœud par son indice. Conserve le "
            "ratio d'aspect, ajoute une légende. Inclus tous les imports. Pas de "
            "markdown, juste du code Python brut directement exécutable."
        )
    )
    commentaire: str = Field(default="")


# ============================================================================
# Mise en forme texte du problème
# ============================================================================
def format_problem(nodes, bars, supports, loads, section_A_m2, material):
    nodes_str = "\n".join(
        f"  {i:>2} : ({n[0]:+.3f}, {n[1]:+.3f}) m" for i, n in enumerate(nodes)
    )
    bars_str = "\n".join(
        f"  barre {k} : nœuds {b[0]} -- {b[1]}" for k, b in enumerate(bars)
    )

    if supports:
        def describe(s):
            ex, ey = s
            return (("libre" if ex else "bloqué") + " en x, "
                    + ("libre" if ey else "bloqué") + " en y")
        sup_str = "\n".join(
            f"  nœud {i} : [ex={s[0]}, ey={s[1]}]  ({describe(s)})"
            for i, s in supports.items()
        )
    else:
        sup_str = "  (aucun appui)"

    if loads:
        loads_str = "\n".join(
            f"  nœud {i} : F = ({fx:+.1f}, {fy:+.1f}) N"
            for i, (fx, fy) in loads.items()
        )
    else:
        loads_str = "  (aucune charge)"

    return f"""Treillis plan à barres articulées.

Nœuds (positions en mètres) :
{nodes_str}

Barres (connectivité par indices de nœuds) :
{bars_str}

Appuis (convention par DDL : 0 = bloqué, 1 = libre) :
{sup_str}

Charges appliquées :
{loads_str}

Matériau : {material}
Section identique pour toutes les barres : A = {section_A_m2*1e6:.1f} mm²
   = {section_A_m2:.3e} m²

Calcule les déplacements (ux, uy) de TOUS les nœuds, en mètres, en respectant
strictement les contraintes des appuis (ux = 0 si ex = 0, uy = 0 si ey = 0).
Fournis également le code Python de tracé."""


# ============================================================================
# Chaîne LangChain
# ============================================================================
SYSTEM = """Tu es un ingénieur structures. Tu résous des problèmes de treillis plans
articulés à N barres avec des conditions d'appui spécifiées DDL par DDL (déplacement
nul ou libre selon x et selon y indépendamment). Tu fournis les déplacements de tous
les nœuds en respectant strictement les contraintes, ainsi que le code Python
permettant de tracer le treillis au repos et déformé sur un même graphique."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{problem}"),
])

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
chain = prompt | llm.with_structured_output(GeneralTrussSolution)


# ============================================================================
# Exemple : 3 barres avec la nouvelle convention d'appuis
# ============================================================================

# --- Configuration 1 : 3 barres, appuis [0,0] partout (équivalent à avant) ---
nodes = [
    [ 0.0,  0.0],   # 0 : nœud fixe
    [ 0.0,  1.0],   # 1 :
    [ 1.0,  0.0],   # 2 : nœud fixe
    [ 1.0, 1.0],
    [1.0,  2.0]    # 4 :
]
bars = [
    [0, 1],
    [0, 3],
    [1, 3],
    [2, 3],
    [1, 4],
    [3, 4],
]
supports = {
    0: [0, 0],
    2: [0, 0]
}
loads = {4: [10000.0/2, -10000.0/2]}
section_A_m2 = 1.0e-4
material = "acier"

# --- Configuration 2 : poutre Pratt 5 barres, 1 rotule + 1 rouleau ---
# Décommenter pour passer au cas plus exigeant.
#
# nodes = [
#     [0.0, 0.0],   # 0 : appui gauche
#     [2.0, 0.0],   # 1 : nœud libre milieu bas
#     [4.0, 0.0],   # 2 : appui droit
#     [2.0, 1.5],   # 3 : nœud libre sommet
# ]
# bars = [
#     [0, 1], [1, 2], [0, 3], [3, 2], [1, 3],
# ]
# supports = {
#     0: [0, 0],   # rotule fixe
#     2: [1, 0],   # rouleau : libre en x, bloqué en y
# }
# loads = {3: [0.0, -50000.0]}
# section_A_m2 = 5.0e-4
# material = "acier"


# ============================================================================
# Exécution
# ============================================================================
if __name__ == "__main__":
    problem_text = format_problem(nodes, bars, supports, loads, section_A_m2, material)
    print("=" * 70)
    print("PROBLÈME ENVOYÉ AU LLM")
    print("=" * 70)
    print(problem_text)
    print()

    print("Interrogation du LLM...\n")
    result = chain.invoke({"problem": problem_text})

    print("=" * 70)
    print("DÉPLACEMENTS RENVOYÉS PAR LE LLM")
    print("=" * 70)
    for i, (ux, uy) in enumerate(result.displacements_m):
        if i in supports:
            ex, ey = supports[i]
            tag = f"  [appui ex={ex}, ey={ey}]"
        else:
            tag = "  (libre)"
        print(f"  Nœud {i:>2} : ux = {ux*1000:+10.4f} mm, uy = {uy*1000:+10.4f} mm{tag}")

    print(f"\nFacteur d'amplification : {result.displacement_scale}")
    if result.commentaire:
        print(f"\nCommentaire du LLM :\n{result.commentaire}")

    print()
    print("=" * 70)
    print("CODE DE TRACÉ GÉNÉRÉ PAR LE LLM")
    print("=" * 70)
    print(result.plot_code)
    print("=" * 70)

    print("\nExécution du code généré...\n")
    namespace = {}
    exec(result.plot_code, namespace)

    if "plot_truss" not in namespace:
        raise RuntimeError("Le LLM n'a pas défini de fonction plot_truss(...).")

    import math

    # Diagonale du domaine
    xs, ys = [n[0] for n in nodes], [n[1] for n in nodes]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    # Plus grand déplacement réel
    max_d = max(math.hypot(ux, uy) for ux, uy in result.displacements_m) or 1e-12

    # Cible : la déformée représente ~10 % de la diagonale du domaine
    target_visual = 0.10 * diag
    safe_scale = max(result.displacement_scale, target_visual / max_d)

    print(f"Scale LLM = {result.displacement_scale:g} | Scale corrigé = {safe_scale:g}")

    namespace["plot_truss"](
        nodes, bars, result.displacements_m, supports, safe_scale
    )