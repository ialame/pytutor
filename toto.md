======================================================================
EXERCICE 5 — MultiQueryRetriever vs retriever simple
======================================================================
✓ Rechargement de l'index depuis ./chroma_db

Question (volontairement vague) : Comment marche l'héritage ?

--- Retriever SIMPLE (k=3) ---
  [1] 9.5. Inheritance¶ Of course, a language feature would not be worthy of the name “class” without supp...
  [2] race condition¶A condition of a program where the behavior depends on the relative timing or orderin...
  [3] |     f()   |     ~^^   |   File "<stdin>", line 3, in f   |     raise ExceptionGroup('there were pr...

--- MultiQuery Retriever ---

  9 documents au total (déduplication automatique)
  [1] Note that comparing objects of different types with < or > is legal provided that the objects have a...
  [2] Previous topic Enum HOWTO   Next topic Logging HOWTO    This page  Report a bug Improve this page  S...
  [3] 4.10. Intermezzo: Coding Style      Previous topic 3. An Informal Introduction to Python   Next topi...
  [4] cyclic isolate¶A subgroup of one or more objects that reference each other in a reference cycle, but...
  [5] race condition¶A condition of a program where the behavior depends on the relative timing or orderin...
  [6] 9.5. Inheritance¶ Of course, a language feature would not be worthy of the name “class” without supp...
  [7] Use isinstance() to check an instance’s type: isinstance(obj, int) will be True only if obj.__class_...
  [8] Having seen the mechanics behind the iterator protocol, it is easy to add iterator behavior to your ...
  [9] The cumulative effect of these changes is to turn generators from one-way producers of information i...

--- RÉPONSE avec retriever simple ---
# Comment marche l'héritage en Python ?

L'héritage permet à une classe (appelée **classe dérivée**) d'hériter des propriétés et méthodes d'une autre classe (appelée **classe de base**). [1]

## Syntaxe de base

Pour créer une classe qui hérite d'une autre, voici la syntaxe : [1]

```python
class ClasseDérivée(ClasseDeBase):
    <instruction-1>
    .
    .
    .
    <instruction-N>
```

## Points ...

--- RÉPONSE avec MultiQuery ---
# L'héritage en Python

Basé sur le contexte fourni [6], voici comment fonctionne l'héritage :

## Syntaxe de base

Pour créer une classe qui hérite d'une autre, on utilise cette syntaxe :

```python
class ClasseDérivée(ClasseDeBase):
    <instruction-1>
    ...
    <instruction-N>
```

La `ClasseDeBase` doit être définie dans un espace de noms accessible depuis la classe dérivée.

## Exemple avec...

----------------------------------------------------------------------
OBSERVATIONS À RETENIR :
 - Regarde les logs ci-dessus pour voir les 3 reformulations
   générées automatiquement par le LLM.
 - MultiQuery double ou triple ton coût par question (1 appel pour
   reformuler + N appels de recherche), mais améliore notablement
   le recall sur les questions courtes ou floues.
 - À ne PAS utiliser sur des questions très précises et bien
   formulées : aucun gain, juste du coût en plus.

======================================================================
✓ Solutions de la Phase 2 exécutées.
  → Pour la boucle interactive, décommente exercise_2_interactive()
  → Pour LangSmith, configure .env et décommente exercise_4_langsmith()
======================================================================