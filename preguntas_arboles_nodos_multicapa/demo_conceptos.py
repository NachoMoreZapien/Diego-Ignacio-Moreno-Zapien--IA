"""
Script de apoyo para el cuestionario de árboles de decisión y redes multicapa.
Autor: Diego Ignacio Moreno Zapien

Solo usa la biblioteca estándar de Python. Los datos son pequeños y de ejemplo,
pensados únicamente para ilustrar los conceptos de las preguntas 2, 4, 5 y 9.
"""

import math
import sys

# En Windows, si la salida se redirige a un archivo o tubería, Python usa cp1252
# y los acentos salen mal; forzamos UTF-8 para que se vean igual en todos lados.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pregunta 2: elementos de un árbol (raíz, nodo interno, rama, hoja)
# ---------------------------------------------------------------------------
# Cada nodo es un diccionario. Los nodos con "clase" son hojas; los demás
# tienen una condición (variable <= umbral) y dos ramas: "si" y "no".
ARBOL_EJEMPLO = {
    "variable": "asistencia", "umbral": 70,          # nodo raíz
    "si": {
        "variable": "tareas", "umbral": 50,          # nodo interno
        "si": {"clase": "RIESGO"},                    # hoja
        "no": {"clase": "SIN RIESGO"},                # hoja
    },
    "no": {"clase": "SIN RIESGO"},                    # hoja
}


def recorrer_arbol(nodo, dato, nivel=0):
    """Baja por el árbol imprimiendo cada pregunta y la rama tomada."""
    sangria = "   " * nivel
    if "clase" in nodo:
        print(f"{sangria}hoja -> {nodo['clase']}")
        return nodo["clase"]
    valor = dato[nodo["variable"]]
    respuesta = "si" if valor <= nodo["umbral"] else "no"
    tipo = "raíz" if nivel == 0 else "nodo interno"
    print(f"{sangria}[{tipo}] ¿{nodo['variable']} <= {nodo['umbral']}? "
          f"(valor {valor}) -> rama '{respuesta}'")
    return recorrer_arbol(nodo[respuesta], dato, nivel + 1)


# ---------------------------------------------------------------------------
# Pregunta 4: pesos y sesgo de una neurona
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def neurona(entradas, pesos, sesgo):
    """Suma ponderada + sesgo, y después la activación sigmoide."""
    z = sum(x * w for x, w in zip(entradas, pesos)) + sesgo
    return z, sigmoide(z)


# ---------------------------------------------------------------------------
# Pregunta 5: cómo aprende cada modelo
# ---------------------------------------------------------------------------
# Datos de ejemplo: (asistencia en %, tareas entregadas en %) -> 1 = riesgo
DATOS = [
    ((95, 90), 0), ((88, 75), 0), ((80, 95), 0), ((72, 60), 0),
    ((65, 40), 1), ((60, 55), 1), ((50, 30), 1), ((68, 20), 1),
    ((90, 40), 0), ((55, 80), 1),
]


def gini(etiquetas):
    """Impureza de Gini: 0 = grupo puro, 0.5 = mitad y mitad (dos clases)."""
    if not etiquetas:
        return 0.0
    p = sum(etiquetas) / len(etiquetas)
    return 1.0 - p ** 2 - (1 - p) ** 2


def mejor_corte(datos):
    """El árbol aprende probando variables y umbrales: se queda con el que
    deja los dos grupos más puros (menor Gini ponderado)."""
    mejor = None
    for var in range(2):
        for umbral in sorted({x[var] for x, _ in datos}):
            izq = [y for x, y in datos if x[var] <= umbral]
            der = [y for x, y in datos if x[var] > umbral]
            if not izq or not der:
                continue
            g = (len(izq) * gini(izq) + len(der) * gini(der)) / len(datos)
            if mejor is None or g < mejor[0]:
                mejor = (g, var, umbral)
    return mejor


def entrenar_neurona(datos, epocas=300, tasa=0.5):
    """La neurona aprende ajustando pesos y sesgo con descenso de gradiente
    (pérdida de entropía cruzada). Las entradas se escalan a [0, 1]."""
    w = [0.0, 0.0]
    b = 0.0
    historial = []
    for epoca in range(1, epocas + 1):
        perdida = 0.0
        for (x1, x2), y in datos:
            x = (x1 / 100, x2 / 100)
            _, p = neurona(x, w, b)
            perdida += -(y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12))
            error = p - y                     # derivada de la pérdida respecto a z
            w[0] -= tasa * error * x[0]
            w[1] -= tasa * error * x[1]
            b -= tasa * error
        if epoca in (1, 10, 50, 100, epocas):
            historial.append((epoca, perdida / len(datos), list(w), b))
    return w, b, historial


# ---------------------------------------------------------------------------
# Pregunta 9: comparar dos modelos con datos de prueba
# ---------------------------------------------------------------------------
def metricas(reales, predichos, positiva=1):
    vp = sum(1 for r, p in zip(reales, predichos) if r == positiva and p == positiva)
    fp = sum(1 for r, p in zip(reales, predichos) if r != positiva and p == positiva)
    fn = sum(1 for r, p in zip(reales, predichos) if r == positiva and p != positiva)
    vn = sum(1 for r, p in zip(reales, predichos) if r != positiva and p != positiva)
    exactitud = (vp + vn) / len(reales)
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    return {"VP": vp, "FP": fp, "FN": fn, "VN": vn,
            "exactitud": exactitud, "precision": precision, "recall": recall}


def main():
    print("=" * 64)
    print("Pregunta 2: recorrido de un árbol de decisión")
    print("=" * 64)
    alumno = {"asistencia": 62, "tareas": 45}
    print(f"Alumno de ejemplo: {alumno}")
    recorrer_arbol(ARBOL_EJEMPLO, alumno)

    print()
    print("=" * 64)
    print("Pregunta 4: pesos y sesgo")
    print("=" * 64)
    entradas = (0.6, 0.4)
    pesos = (2.0, -1.0)
    for sesgo in (-1.0, 0.0, 1.0):
        z, a = neurona(entradas, pesos, sesgo)
        print(f"entradas={entradas} pesos={pesos} sesgo={sesgo:+.1f} "
              f"-> z={z:+.2f}  salida={a:.3f}  ({'se activa' if a >= 0.5 else 'no se activa'})")
    _, a0 = neurona((0.0, 0.0), pesos, 1.0)
    print(f"Con entradas en cero y sesgo +1.0 la salida es {a0:.3f}: el sesgo "
          "permite activarse sin entradas.")

    print()
    print("=" * 64)
    print("Pregunta 5: cómo aprende cada modelo")
    print("=" * 64)
    g, var, umbral = mejor_corte(DATOS)
    nombres = ("asistencia", "tareas")
    print(f"Gini del conjunto completo: {gini([y for _, y in DATOS]):.3f}")
    print(f"Árbol -> mejor primer corte: {nombres[var]} <= {umbral} "
          f"(Gini ponderado después del corte: {g:.3f})")
    w, b, hist = entrenar_neurona(DATOS)
    print("Neurona -> ajuste de pesos por gradiente:")
    for epoca, perdida, wh, bh in hist:
        print(f"  época {epoca:3d}: pérdida={perdida:.3f}  "
              f"w=({wh[0]:+.2f}, {wh[1]:+.2f})  b={bh:+.2f}")
    aciertos = sum(1 for (x1, x2), y in DATOS
                   if (neurona((x1 / 100, x2 / 100), w, b)[1] >= 0.5) == bool(y))
    print(f"Aciertos de la neurona en los datos de entrenamiento: {aciertos}/{len(DATOS)}")

    print()
    print("=" * 64)
    print("Pregunta 9: comparar dos modelos con datos de prueba")
    print("=" * 64)
    # 1 = llegó tarde, 0 = llegó a tiempo. Predicciones inventadas para el ejemplo.
    reales = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    arbol = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    red = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    for nombre, pred in (("Árbol", arbol), ("Red", red)):
        m = metricas(reales, pred)
        print(f"{nombre:5s}: VP={m['VP']} FP={m['FP']} FN={m['FN']} VN={m['VN']}  "
              f"exactitud={m['exactitud']:.2f}  precisión={m['precision']:.2f}  "
              f"recall(tarde)={m['recall']:.2f}")
    print("Lectura: la exactitud se parece, pero el recall de 'tarde' muestra qué "
          "modelo detecta mejor los retrasos reales.")


if __name__ == "__main__":
    main()
