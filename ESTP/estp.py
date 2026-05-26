import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
# === 1. Physique : formule analytique de l'amplitude en régime permanent ===
def steady_state_amplitude(m, k, c, A, omega):
    """X = (A/ωₙ²) / √((1-r²)² + (2ζr)²)"""
    omega_n = np.sqrt(k / m)
    zeta    = c / (2 * np.sqrt(m * k))
    r       = omega / omega_n
    return (A / omega_n**2) / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)

# === 2. LangChain : structure de sortie attendue (Pydantic) ===
class SeismicParameters(BaseModel):
    """Paramètres physiques d'un oscillateur 1 DDL sous séisme sinusoïdal."""
    m: float     = Field(description="Masse en kg")
    k: float     = Field(description="Raideur en N/m")
    zeta: float  = Field(description="Taux d'amortissement (entre 0 et 1)")
    A: float     = Field(description="Amplitude de l'accélération en m/s²")
    omega: float = Field(description="Pulsation en rad/s")

# === 3. Une seule chaîne : prompt paramétré → LLM → sortie structurée ===
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu extrais les paramètres physiques de la requête utilisateur en unités SI. "
     "Convertis systématiquement : tonnes → kg, Hz → rad/s (×2π), pourcentage → fraction. "
     "Si une information essentielle manque, lève une erreur explicite plutôt que d'inventer."),
    ("human", "{query}")
])

chain = prompt | llm.with_structured_output(SeismicParameters)

# === 4. Pipeline : 1 ligne LLM, 3 lignes Python ===
def answer(query: str) -> dict:
    p = chain.invoke({"query": query})                       # ← seul appel LLM
    c = 2 * p.zeta * np.sqrt(p.m * p.k)                      # Python pur
    X = steady_state_amplitude(p.m, p.k, c, p.A, p.omega)    # Python pur
    return {"parametres": p.model_dump(), "deplacement_mm": X * 1000}


result = answer(
    "Bâtiment 50 tonnes, raideur 2e7 N/m, amortissement 5 %, séisme 3 Hz à 1 m/s²"
)
print(f"Déplacement max : {result['deplacement_mm']:.2f} mm")
print(f"Paramètres extraits : {result['parametres']}")