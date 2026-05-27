Intuition :
Un générateur est une fonction qui produit une séquence de valeurs une à la fois, au lieu de tout calculer et retourner d'un coup. Imagine une usine qui fabrique des objets à la demande plutôt que de les stocker tous en entrepôt : elle économise de l'espace et de l'énergie. En Python, les générateurs utilisent le mot-clé `yield` pour "pausable" et reprendre l'exécution, ce qui les rend très efficaces pour traiter de grandes quantités de données.

Code :

# Fonction classique : calcule et retourne TOUT d'un coup
def nombres_classique(n: int) -> list:
    resultat = []
    for i in range(n):
        resultat.append(i ** 2)
    return resultat

# Générateur : produit les valeurs UNE À LA FOIS
def nombres_generateur(n: int):
    for i in range(n):
        yield i ** 2  # Pause ici, attend qu'on demande la prochaine valeur

# Utilisation
print("Classique :", nombres_classique(5))  # [0, 1, 4, 9, 16]

print("Générateur :")
gen = nombres_generateur(5)
print(next(gen))  # 0 (première valeur)
print(next(gen))  # 1 (deuxième valeur)

# Ou avec une boucle (plus courant)
for valeur in nombres_generateur(5):
    print(valeur)  # Affiche 0, 1, 4, 9, 16 une à une


Piège classique :
Oublier que les générateurs ne peuvent être parcourus qu'une seule fois. Une fois épuisés, ils ne produisent plus de valeurs. Si tu essaies de les réutiliser, tu dois en créer une nouvelle instance. Aussi, confondre `yield` avec `return` : `yield` pause et reprend, tandis que `return` termine définitivement la fonction.

Questions de compréhension :
  1. Quelle est la différence principale entre une fonction classique qui retourne une liste et un générateur ?
     → Une fonction classique calcule et retourne toutes les valeurs à la fois en mémoire, tandis qu'un générateur produit les valeurs une à la fois à la demande, ce qui économise la mémoire.
  2. Comment appelle-t-on le mot-clé qui permet à une fonction de devenir un générateur ?
     → Le mot-clé `yield`. Quand une fonction contient `yield`, elle devient un générateur au lieu d'une fonction classique.
  3. Pourquoi les générateurs sont-ils particulièrement utiles pour traiter des fichiers très volumineux ou des flux de données infinis ?
     → Parce qu'ils ne chargent pas toutes les données en mémoire à la fois, mais les traitent progressivement. Cela permet de traiter des données plus grandes que la RAM disponible et de réagir immédiatement aux premières valeurs sans attendre le calcul complet.
