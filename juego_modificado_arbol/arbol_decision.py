"""
Árbol de decisión (clasificación) escrito desde cero con NumPy.

Autor: Diego Ignacio Moreno Zapien

Idea (algoritmo CART):
  1. En cada nodo se prueban todas las preguntas del tipo  x[j] <= umbral.
  2. Se elige la pregunta que más reduce la impureza de Gini.
  3. Los datos se separan en dos grupos (izquierda: sí, derecha: no) y se repite
     en cada grupo hasta que el nodo es puro, muy pequeño o se llega a la
     profundidad máxima.
  4. Cada hoja guarda la proporción (ponderada) de cada clase: esa es la
     probabilidad que devuelve predict_proba().

Gini de un nodo:  G = 1 - sum_k p_k^2   (0 = nodo puro, 0.5 = mitad y mitad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Nodo:
    # Datos de una hoja
    proba: np.ndarray            # probabilidad de cada clase en este nodo
    n: int                       # cuántas muestras llegaron aquí
    # Datos de un nodo interno (None si es hoja)
    caracteristica: Optional[int] = None
    umbral: Optional[float] = None
    izq: Optional["Nodo"] = None  # x[caracteristica] <= umbral
    der: Optional["Nodo"] = None  # x[caracteristica] >  umbral

    @property
    def es_hoja(self) -> bool:
        return self.caracteristica is None


def gini(pesos_por_clase: np.ndarray) -> float:
    """Impureza de Gini a partir del peso total de cada clase."""
    pesos_por_clase = np.asarray(pesos_por_clase, dtype=float)
    total = pesos_por_clase.sum()
    if total <= 0:
        return 0.0
    p = pesos_por_clase / total
    return float(1.0 - np.sum(p * p))


class ArbolDecision:
    def __init__(self, profundidad_max: int = 5, min_muestras_hoja: int = 5,
                 balancear_clases: bool = True) -> None:
        self.profundidad_max = profundidad_max
        self.min_muestras_hoja = min_muestras_hoja
        self.balancear_clases = balancear_clases
        self.raiz: Optional[Nodo] = None
        self.clases: np.ndarray = np.array([])

    # ------------------------------------------------------------ entrenamiento
    def entrenar(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "ArbolDecision":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.clases, y_idx = np.unique(y, return_inverse=True)

        # Peso de cada muestra. Con balanceo, cada clase "pesa" lo mismo en total
        # (igual que class_weight="balanced"): peso = n / (num_clases * n_clase).
        if self.balancear_clases:
            conteos = np.bincount(y_idx, minlength=len(self.clases))
            peso_clase = len(y) / (len(self.clases) * conteos)
            w = peso_clase[y_idx]
        else:
            w = np.ones(len(y))

        self.raiz = self._construir(X, y_idx, w, profundidad=0)
        return self

    def _pesos_clase(self, y_idx: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.bincount(y_idx, weights=w, minlength=len(self.clases))

    def _construir(self, X: np.ndarray, y_idx: np.ndarray, w: np.ndarray, profundidad: int) -> Nodo:
        pc = self._pesos_clase(y_idx, w)
        nodo = Nodo(proba=pc / pc.sum(), n=len(y_idx))

        # Condiciones de paro: nodo puro, profundidad o muy pocas muestras.
        if (gini(pc) == 0.0 or profundidad >= self.profundidad_max
                or len(y_idx) < 2 * self.min_muestras_hoja):
            return nodo

        mejor = self._mejor_division(X, y_idx, w)
        if mejor is None:
            return nodo

        j, umbral = mejor
        mascara = X[:, j] <= umbral
        nodo.caracteristica, nodo.umbral = j, umbral
        nodo.izq = self._construir(X[mascara], y_idx[mascara], w[mascara], profundidad + 1)
        nodo.der = self._construir(X[~mascara], y_idx[~mascara], w[~mascara], profundidad + 1)
        return nodo

    def _mejor_division(self, X: np.ndarray, y_idx: np.ndarray, w: np.ndarray) -> Optional[Tuple[int, float]]:
        """Busca (característica, umbral) que minimiza el Gini ponderado de los hijos."""
        n, m = X.shape
        k = len(self.clases)
        total = self._pesos_clase(y_idx, w)
        mejor_impureza = gini(total)  # solo aceptamos divisiones que mejoren
        mejor = None
        hoja = self.min_muestras_hoja

        for j in range(m):
            orden = np.argsort(X[:, j], kind="mergesort")
            xs, ys, ws = X[orden, j], y_idx[orden], w[orden]
            # Peso acumulado de cada clase a la izquierda para cada posible corte.
            una_caliente = np.zeros((n, k))
            una_caliente[np.arange(n), ys] = ws
            izq = np.cumsum(una_caliente, axis=0)       # fila i = primeros i+1 datos
            der = total - izq
            for i in range(hoja - 1, n - hoja):
                if xs[i] == xs[i + 1]:
                    continue  # no se puede cortar entre dos valores iguales
                wi, wd = izq[i].sum(), der[i].sum()
                imp = (wi * gini(izq[i]) + wd * gini(der[i])) / (wi + wd)
                if imp < mejor_impureza - 1e-12:
                    mejor_impureza = imp
                    mejor = (j, float((xs[i] + xs[i + 1]) / 2))  # umbral a la mitad
        return mejor

    # ------------------------------------------------------------- predicción
    def _hoja(self, x: np.ndarray) -> Nodo:
        nodo = self.raiz
        while not nodo.es_hoja:  # se recorre UNA sola ruta, de la raíz a una hoja
            nodo = nodo.izq if x[nodo.caracteristica] <= nodo.umbral else nodo.der
        return nodo

    def predecir_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._hoja(x).proba for x in X])

    def predecir(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        return self.clases[np.argmax(self.predecir_proba(X), axis=1)]

    def exactitud(self, X, y) -> float:
        return float(np.mean(self.predecir(X) == np.asarray(y)))

    # ------------------------------------------------------- interpretabilidad
    def reglas(self, nombres: Sequence[str]) -> str:
        """Devuelve el árbol como texto (una pregunta por línea)."""
        lineas: List[str] = []

        def visitar(nodo: Nodo, sangria: str) -> None:
            if nodo.es_hoja:
                clase = self.clases[int(np.argmax(nodo.proba))]
                lineas.append(f"{sangria}-> clase {clase}  (p={nodo.proba.max():.2f}, n={nodo.n})")
                return
            nombre = nombres[nodo.caracteristica]
            lineas.append(f"{sangria}si {nombre} <= {nodo.umbral:.1f}:")
            visitar(nodo.izq, sangria + "    ")
            lineas.append(f"{sangria}si no ({nombre} > {nodo.umbral:.1f}):")
            visitar(nodo.der, sangria + "    ")

        visitar(self.raiz, "")
        return "\n".join(lineas)

    def profundidad(self) -> int:
        def prof(n: Nodo) -> int:
            return 0 if n.es_hoja else 1 + max(prof(n.izq), prof(n.der))
        return prof(self.raiz)

    def num_hojas(self) -> int:
        def hojas(n: Nodo) -> int:
            return 1 if n.es_hoja else hojas(n.izq) + hojas(n.der)
        return hojas(self.raiz)


def dividir_entrenamiento_prueba(X, y, prueba: float = 0.2, semilla: int = 42):
    """Separación estratificada: cada clase conserva su proporción en ambos conjuntos."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(semilla)
    idx_prueba = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        idx_prueba.extend(idx[: max(1, int(round(len(idx) * prueba)))])
    mascara = np.zeros(len(y), dtype=bool)
    mascara[idx_prueba] = True
    return X[~mascara], X[mascara], y[~mascara], y[mascara]
