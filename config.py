"""Configuration centrale du projet PyTutor.

Toutes les phases importent depuis ce module pour rester cohérent.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Charge automatiquement les variables du .env dès l'import
load_dotenv()

# --- Choix du modèle ---
# Haiku = rapide et économique, parfait pour itérer toute la journée.
# Sonnet = plus intelligent, à activer quand tu veux comparer la qualité.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

DEFAULT_MODEL = HAIKU


def get_model(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatAnthropic:
    """Renvoie une instance configurée de ChatAnthropic.

    Lève une erreur claire si la clé API n'est pas trouvée.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY introuvable. "
            "As-tu créé un fichier .env à partir de .env ?"
        )
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
