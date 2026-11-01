# R0

Simulateur de propagation d'épidémie, pensé comme un outil d'aide à la décision
pour les premières semaines d'une nouvelle épidémie. Code ouvert, licence
Apache 2.0.

Ce dépôt contient la **phase 1** de la feuille de route : le socle du modèle.

- **Modèle déclaratif** : les compartiments, leurs durées, leurs étages d'Erlang,
  leur contagiosité et les transitions entre eux (avec des probabilités qui
  peuvent dépendre de l'âge) se décrivent dans un fichier YAML, sans toucher au
  code. Voir [`configs/`](configs/).
- **Structure par âge** avec une matrice de contacts (rendue réciproque).
- **Deux moteurs sur la même définition** : déterministe (équations
  différentielles, Runge-Kutta 4) et stochastique (chaîne binomiale, population
  entière et conservée exactement). Tous deux en JAX.
- **R0, β et Rt** calculés par la matrice de nouvelle génération ; taux de
  croissance initial et temps de doublement par linéarisation.
- **Tests analytiques** : conservation de la population, taille finale
  `z = 1 − exp(−R0·z)`, seuil d'immunité collective `1 − 1/R0`, épidémie
  sous-critique, taux de croissance, probabilité d'extinction `1/R0`,
  concordance des deux moteurs.

## Installation

```bash
uv venv -p 3.12
uv pip install -e ".[dev]"
```

## Utilisation

En ligne de commande :

```bash
r0 simulate configs/sepiahrd_france.yaml
r0 simulate configs/sepiahrd_france.yaml --engine stochastic --replicates 50 --out traj.csv
```

En Python :

```python
import r0
from r0.model import growth_rate

model = r0.load("configs/sepiahrd_france.yaml")
res = r0.simulate_deterministic(model, days=365)
res.compartment("H")      # lits occupés par jour
res.inflow("H")           # admissions par jour
res.rt()                  # Rt par jour
growth_rate(model)        # taux de croissance initial (par jour)

sto = r0.simulate_stochastic(model, days=365, replicates=100, seed=1)
sto.attack_rate()         # part de la population infectée, par réplication
```

## Modèles fournis

| Fichier | Contenu |
| --- | --- |
| `configs/seir.yaml` | SEIR de référence, population homogène |
| `configs/sepiahrd_france.yaml` | SEPIAHRD par âge (5 classes), durées d'Erlang |

Les populations, contacts et probabilités par âge de `sepiahrd_france.yaml` sont
des **ordres de grandeur illustratifs**, pas des données sourcées. Ils seront
remplacés par des données publiques (INSEE, matrices de contacts publiées) lors
de la phase géographique.

## Organisation du code

```
src/r0/
  model/     configuration (YAML + Pydantic), structure compilée, R0 et Rt
  engines/   moteur déterministe, moteur stochastique, résultats
  cli.py     ligne de commande
tests/       tests analytiques et de cohérence
```

## Tests

```bash
pytest -q
```
