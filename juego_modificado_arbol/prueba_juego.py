"""
Prueba automática (sin ventana ni teclado) del juego con árbol de decisión.

Autor: Diego Ignacio Moreno Zapien

1. Comprueba el árbol con ejemplos pequeños cuyo resultado se conoce.
2. Simula a un "jugador" que salta cuando la bala está cerca (con algo de
   error humano) y guarda los datos igual que el modo MANUAL del juego.
3. Entrena el árbol con esos datos y muestra sus reglas.
4. Deja que el árbol juegue solo y cuenta balas esquivadas y golpes.

Solo necesita numpy:  python -I prueba_juego.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbol_decision import ArbolDecision, dividir_entrenamiento_prueba, gini  # noqa: E402
from mundo import BALA_W, NOMBRES_CARACTERISTICAS, Muestra, Mundo  # noqa: E402


def comprobar_arbol() -> None:
    print("1) Comprobaciones básicas del árbol")
    assert gini([5, 0]) == 0.0
    assert abs(gini([5, 5]) - 0.5) < 1e-12
    # Una sola pregunta basta: clase 1 si x0 > 3
    X = [[i, 0] for i in range(10)]
    y = [int(i > 3) for i in range(10)]
    a = ArbolDecision(profundidad_max=3, min_muestras_hoja=1).entrenar(X, y)
    assert a.exactitud(X, y) == 1.0 and a.raiz.umbral == 3.5, a.reglas(["x0", "x1"])
    # XOR: dos niveles de preguntas bastarían, pero la PRIMERA pregunta
    # (x0 <= 0.5 o x1 <= 0.5) no reduce nada el Gini (cada lado queda 50/50).
    # Como el algoritmo es voraz y solo acepta cortes que mejoran, se queda en
    # una hoja: exactitud 0.5. Es la desventaja "optimización voraz".
    X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 5
    y = [0, 1, 1, 0] * 5
    a = ArbolDecision(profundidad_max=2, min_muestras_hoja=1).entrenar(X, y)
    print("   XOR (profundidad 2) -> exactitud", a.exactitud(X, y), "hojas", a.num_hojas(),
          "(esperado: 0.5 y 1, por ser voraz)")
    print("   OK: gini, umbral a la mitad (3.5)\n")


def jugador_simulado(mundo: Mundo, rng: random.Random, umbral_frames: list) -> bool:
    """Salta cuando la bala está a ~10 frames de distancia (8.5 a 12, al azar por bala)."""
    return mundo.en_suelo and 0 < mundo.distancia <= mundo.rapidez * umbral_frames[0]


def generar_datos(num_balas: int, semilla: int):
    rng = random.Random(semilla)
    mundo = Mundo(semilla)
    datos = []
    umbral = [rng.uniform(8.5, 12)]
    while mundo.esquivadas + mundo.golpes < num_balas:
        saltar = jugador_simulado(mundo, rng, umbral)
        if mundo.en_suelo and mundo.distancia > -BALA_W:   # igual que el modo MANUAL
            r, d = mundo.caracteristicas()
            datos.append(Muestra(r, d, int(saltar)))
        if mundo.paso(saltar) is not None:
            umbral[0] = rng.uniform(8.5, 12)   # el "humano" no es exacto
    return datos, mundo


def jugar_auto(arbol, num_balas: int, semilla: int) -> Mundo:
    mundo = Mundo(semilla)
    idx1 = list(arbol.clases).index(1)
    while mundo.esquivadas + mundo.golpes < num_balas:
        saltar = mundo.en_suelo and arbol.predecir_proba([mundo.caracteristicas()])[0][idx1] >= 0.5
        mundo.paso(saltar)
    return mundo


def main() -> None:
    comprobar_arbol()

    print("2) Datos del jugador simulado (modo MANUAL)")
    datos, m = generar_datos(num_balas=80, semilla=1)
    saltos = sum(d.salto for d in datos)
    print(f"   balas: {m.esquivadas + m.golpes}, esquivadas: {m.esquivadas}, golpes: {m.golpes}")
    print(f"   muestras: {len(datos)}  (saltar=1: {saltos}, saltar=0: {len(datos) - saltos})\n")

    print("3) Entrenamiento del árbol")
    X = [[d.rapidez_bala, d.distancia] for d in datos]
    y = [d.salto for d in datos]
    X_ent, X_pru, y_ent, y_pru = dividir_entrenamiento_prueba(X, y, prueba=0.2)
    arbol = ArbolDecision(profundidad_max=5, min_muestras_hoja=5, balancear_clases=True).entrenar(X_ent, y_ent)
    print(f"   hojas: {arbol.num_hojas()}, profundidad: {arbol.profundidad()}")
    print(f"   exactitud entrenamiento: {arbol.exactitud(X_ent, y_ent):.3f}")
    print(f"   exactitud prueba:        {arbol.exactitud(X_pru, y_pru):.3f}")
    print("   reglas (clase 1 = saltar):")
    for linea in arbol.reglas(NOMBRES_CARACTERISTICAS).splitlines():
        print("     " + linea)
    print()

    print("4) El árbol juega solo (modo AUTO), 200 balas nuevas")
    m = jugar_auto(arbol, num_balas=200, semilla=99)
    print(f"   esquivadas: {m.esquivadas}, golpes: {m.golpes}")
    sin_saltar = Mundo(99)
    while sin_saltar.esquivadas + sin_saltar.golpes < 200:
        sin_saltar.paso(False)
    print(f"   (referencia: si nunca salta -> esquivadas {sin_saltar.esquivadas}, golpes {sin_saltar.golpes})")


if __name__ == "__main__":
    main()
