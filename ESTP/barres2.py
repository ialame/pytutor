"""
Test des limites d'un LLM en calcul de structure.

Protocole expérimental :
- On ne donne AUCUNE indication de méthode (pas de mention de matrice de rigidité,
  pas d'étapes imposées, pas de formules).
- On décrit seulement la géométrie, les constantes physiques et la charge.
- On demande la donnée nécessaire au tracé : positions au repos et positions déformées.
- Python ne fait que dessiner. Aucun calcul physique côté code.

Ce qu'on observe :
- Le LLM trouve-t-il seul une méthode de résolution ?
- Applique-t-il un facteur d'amplification (sinon les déplacements sont invisibles
  sur un dessin à l'échelle réelle, de l'ordre du dixième de millimètre sur 1 m) ?
- Préserve-t-il la symétrie ?
- Le sens du déplacement est-il cohérent avec le sens de la charge ?
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
# Schéma de sortie : juste ce qu'il faut pour dessiner
# ============================================================================
class Node(BaseModel):
    label: str = Field(description="Identifiant du nœud (par exemple 'A', 'B1', etc.)")
    x_m: float = Field(description="Coordonnée x en mètres")
    y_m: float = Field(description="Coordonnée y en mètres")


class Bar(BaseModel):
    node_start: str = Field(description="Label du nœud de départ")
    node_end: str = Field(description="Label du nœud d'arrivée")


class TrussGeometry(BaseModel):
    """Géométrie d'un treillis au repos ET après chargement."""

    nodes_rest: list[Node] = Field(
        description="Positions des nœuds au repos (avant chargement)"
    )
    nodes_displaced: list[Node] = Field(
        description="Positions des nœuds après application de la charge. "
                    "Mêmes labels que nodes_rest, dans le même ordre."
    )
    bars: list[Bar] = Field(
        description="Connectivité des barres, par paires de labels de nœuds."
    )
    displacement_scale: float = Field(
        default=1.0,
        description="Facteur d'amplification appliqué aux déplacements pour la "
                    "lisibilité du tracé. 1.0 signifie échelle réelle."
    )
    commentaire: str = Field(
        default="",
        description="Notes éventuelles sur la résolution ou le choix d'échelle."
    )


# ============================================================================
# Prompt MINIMAL : aucune méthode imposée, aucune étape suggérée
# ============================================================================
SYSTEM = """Tu es un ingénieur structures. Tu réponds à des problèmes de mécanique
des structures en fournissant les données nécessaires à la visualisation graphique
du résultat."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{problem}"),
])

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
chain = prompt | llm.with_structured_output(TrussGeometry)


# ============================================================================
# Énoncé : géométrie + constantes + charge, aucune indication de méthode
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

Fournis les données nécessaires au tracé du treillis au repos et déformé,
sur un même graphique.
"""


# ============================================================================
# Exécution + tracé
# ============================================================================
def plot_truss(geom: TrussGeometry):
    """Trace en appliquant explicitement l'amplification déclarée par le LLM."""
    rest = {n.label: (n.x_m, n.y_m) for n in geom.nodes_rest}

    # On reconstruit les positions déformées en amplifiant
    # le déplacement (position_déformée - position_repos) par le facteur.
    s = geom.displacement_scale
    disp = {}
    for n_rest, n_dep in zip(geom.nodes_rest, geom.nodes_displaced):
        dx = n_dep.x_m - n_rest.x_m
        dy = n_dep.y_m - n_rest.y_m
        disp[n_rest.label] = (n_rest.x_m + s * dx, n_rest.y_m + s * dy)

    fig, ax = plt.subplots(figsize=(8, 8))

    for bar in geom.bars:
        x0r, y0r = rest[bar.node_start]; x1r, y1r = rest[bar.node_end]
        ax.plot([x0r, x1r], [y0r, y1r], 'b-', lw=2, alpha=0.7)
        x0d, y0d = disp[bar.node_start]; x1d, y1d = disp[bar.node_end]
        ax.plot([x0d, x1d], [y0d, y1d], 'r--', lw=2, alpha=0.9)

    for label, (x, y) in rest.items():
        ax.plot(x, y, 'bo', ms=8)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8),
                    color='blue', fontsize=11)
    for label, (x, y) in disp.items():
        ax.plot(x, y, 'rs', ms=8)

    ax.plot([], [], 'b-', label='Au repos')
    ax.plot([], [], 'r--', label=f'Déformé (×{s:g})')
    ax.legend(loc='upper right'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Treillis 3 barres — réponse du LLM')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    print("Interrogation du LLM (sans indication de méthode)...\n")
    result = chain.invoke({"problem": PROBLEM})

    print("=" * 70)
    print("DONNÉES RENVOYÉES PAR LE LLM")
    print("=" * 70)
    print(f"\nFacteur d'amplification : {result.displacement_scale}\n")
    print("Nœuds au repos :")
    for n in result.nodes_rest:
        print(f"  {n.label:>4} : ({n.x_m:+.6f}, {n.y_m:+.6f}) m")
    print("\nNœuds déformés :")
    for n in result.nodes_displaced:
        print(f"  {n.label:>4} : ({n.x_m:+.6f}, {n.y_m:+.6f}) m")
    print("\nBarres :")
    for b in result.bars:
        print(f"  {b.node_start} -- {b.node_end}")
    if result.commentaire:
        print(f"\nCommentaire du LLM :\n{result.commentaire}")

    plot_truss(result)