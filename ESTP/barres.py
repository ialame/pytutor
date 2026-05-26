"""
Exercice — Pilotage d'un LLM pour le calcul des déplacements d'un treillis 3 barres.

Particularité : aucun code Python ne fait le moindre calcul physique.
Tout est délégué au LLM via une chaîne LangChain unique.
Cela permet d'observer expérimentalement la fiabilité (ou non) d'un LLM
sur un calcul d'ingénieur élémentaire.

Référence analytique (à comparer après exécution) :
    u = 0.0000 mm
    v = 0.2929 mm  (≈ PL / (EA·(1 + 1/√2)))
    K11 = 1.4142e+07 N/m
    K22 = 3.4142e+07 N/m
    K12 = 0
"""

import os

from dotenv import load_dotenv
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # ou via la config PyCharm

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# ============================================================================
# 1. Schéma de la sortie attendue
# ============================================================================
class TrussSolution(BaseModel):
    """Solution d'un treillis plan : déplacement du nœud libre + matrice de rigidité."""

    u_mm: float = Field(description="Déplacement horizontal du nœud libre en mm (positif vers la droite)")
    v_mm: float = Field(description="Déplacement vertical du nœud libre en mm (positif vers le bas)")

    K11_N_per_m: float = Field(description="Coefficient K[0,0] de la matrice de rigidité au nœud libre, en N/m")
    K12_N_per_m: float = Field(description="Coefficient K[0,1] = K[1,0], en N/m")
    K22_N_per_m: float = Field(description="Coefficient K[1,1] de la matrice de rigidité au nœud libre, en N/m")

    raisonnement: str = Field(
        description="Démarche détaillée : longueurs et angles de chaque barre, "
                    "raideurs k_i, assemblage de K, résolution. 8 lignes maximum."
    )


# ============================================================================
# 2. Prompt système
# ============================================================================
SYSTEM_PROMPT = """Tu es un ingénieur structures. Tu résous des problèmes de treillis
plans articulés par la méthode des déplacements (matrice de rigidité).

Pour une barre i reliant le nœud libre A à un support B_i :
- longueur : L_i = ||B_i - A||
- angle : θ_i = atan2(B_iy - A_y, B_ix - A_x), mesuré depuis l'axe horizontal
- raideur axiale : k_i = E_i * A_i / L_i
- contribution à la matrice de rigidité au nœud libre :
      k_i * [[cos²θ_i,        cosθ_i sinθ_i],
             [cosθ_i sinθ_i,  sin²θ_i      ]]

Démarche stricte :
1. Pour chaque barre, calcule L_i, θ_i, k_i (en N/m).
2. Assemble la matrice K (2x2) au nœud libre en sommant les contributions.
3. Résous K * [u; v] = F où F est la force appliquée au nœud libre.
4. Convertis u et v en millimètres.

Conventions :
- u positif vers la droite, v positif vers le bas (donc une charge descendante
  donne une composante F_y positive dans cette convention).
- Tous les calculs en unités SI à l'intérieur, conversion finale en mm.

Sois rigoureux numériquement. Détaille tes calculs intermédiaires dans le champ
'raisonnement' pour qu'on puisse les vérifier."""


# ============================================================================
# 3. Chaîne LangChain (la SEULE pièce active du script)
# ============================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{problem}"),
])

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=2048)

chain = prompt | llm.with_structured_output(TrussSolution)


# ============================================================================
# 4. Le problème (texte brut)
# ============================================================================
PROBLEM = """
Treillis plan à 3 barres articulées se rejoignant au nœud apex A situé à l'origine (0, 0).

Coordonnées des supports (fixes) :
  - B1 = (0,    +1.0) m   -- barre 1 verticale, longueur 1 m
  - B2 = (+1.0, +1.0) m   -- barre 2 diagonale, à 45° de l'horizontale
  - B3 = (-1.0, +1.0) m   -- barre 3 diagonale, à 135° de l'horizontale

Toutes les barres ont les mêmes caractéristiques :
  - module d'Young E = 200 GPa = 2.0e11 Pa
  - section A = 100 mm² = 1.0e-4 m²

Charge appliquée au nœud A :
  - F = (0, +10000) N  (10 kN, purement verticale vers le bas dans la convention
    où v est positif vers le bas).

Calculer le déplacement de A.
"""


# ============================================================================
# 5. Exécution et affichage
# ============================================================================
if __name__ == "__main__":
    print("Interrogation du LLM via LangChain...\n")
    result = chain.invoke({"problem": PROBLEM})

    print("=" * 70)
    print("RÉPONSE DU LLM")
    print("=" * 70)
    print(f"u = {result.u_mm:+.4f} mm")
    print(f"v = {result.v_mm:+.4f} mm")
    print()
    print(f"K11 = {result.K11_N_per_m:.4e} N/m")
    print(f"K12 = {result.K12_N_per_m:.4e} N/m")
    print(f"K22 = {result.K22_N_per_m:.4e} N/m")
    print()
    print("Raisonnement du LLM :")
    print(result.raisonnement)

    print()
    print("=" * 70)
    print("RÉFÉRENCE ANALYTIQUE (calcul à la main)")
    print("=" * 70)
    print("u = +0.0000 mm")
    print("v = +0.2929 mm   (PL / (EA · (1 + 1/√2)))")
    print()
    print("K11 = 1.4142e+07 N/m   (EA/(L√2))")
    print("K12 = 0.0000e+00 N/m   (par symétrie)")
    print("K22 = 3.4142e+07 N/m   (EA/L · (1 + 1/√2))")