"""test_hello.py — valide l'installation et donne un avant-goût LCEL.

Lance avec :  python test_hello.py
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_model

# 1. Définir le gabarit de prompt
prompt = ChatPromptTemplate.from_template(
    "Raconte-moi une blague courte sur le thème : {sujet}"
)
# 2. Initialiser le modèle (nécessite la clé API)
model = get_model()
# 3. Créer le parser de sortie
parser = StrOutputParser()
# 4. Assembler la chaîne (LCEL)
chain = prompt | model | parser
# 5. Exécuter la chaîne
reponse = chain.invoke({"sujet": "Cours de mathématiques"})
print(reponse)
