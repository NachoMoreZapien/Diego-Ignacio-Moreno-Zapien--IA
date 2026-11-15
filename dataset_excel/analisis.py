"""
Analisis exploratorio de un dataset sin encabezados ni etiquetas (dataset.csv).

Autor: Diego Ignacio Moreno Zapien
Materia: Inteligencia Artificial

Objetivo de la practica: encontrar relaciones entre las columnas y descubrir que dataset es.
En clase se hizo en Excel; este script repite el analisis en Python y ademas genera
'resumen_segmentos.csv' para abrirlo en Excel.

Ejecutar:  python analisis.py
Solo usa la biblioteca estandar.
"""

import csv
import math
import statistics as st
from pathlib import Path

CARPETA = Path(__file__).parent
COLUMNAS = ["A", "B", "C", "D"]


def cargar(ruta=CARPETA / "dataset.csv"):
    with open(ruta, newline="") as f:
        return [[float(v) for v in fila] for fila in csv.reader(f) if fila]


def columna(datos, j):
    return [fila[j] for fila in datos]


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den


def resumen(valores):
    return {
        "min": min(valores), "max": max(valores),
        "media": st.mean(valores), "desv": st.stdev(valores),
    }


# ---------------------------------------------------------------------------
# K-means simple: agrupa sin saber las etiquetas
# ---------------------------------------------------------------------------
def distancia2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def kmeans(datos, k=3, iteraciones=100):
    # Inicializacion determinista: el punto con C minimo, el de C maximo y el de C mediano.
    orden = sorted(range(len(datos)), key=lambda i: datos[i][2])
    centros = [datos[orden[0]], datos[orden[len(orden) // 2]], datos[orden[-1]]][:k]
    asignacion = [0] * len(datos)
    for _ in range(iteraciones):
        nueva = [min(range(k), key=lambda c: distancia2(p, centros[c])) for p in datos]
        if nueva == asignacion:
            break
        asignacion = nueva
        for c in range(k):
            miembros = [p for p, a in zip(datos, asignacion) if a == c]
            if miembros:
                centros[c] = [st.mean(col) for col in zip(*miembros)]
    return asignacion, centros


def regla_simple(fila):
    """Regla encontrada al observar las columnas C y D (largo y ancho del petalo)."""
    _, _, c, d = fila
    if c < 2.5:
        return 0
    return 1 if d < 1.75 else 2


def main():
    datos = cargar()
    print(f"Filas: {len(datos)}  Columnas: {len(datos[0])}\n")

    print("== Estadisticas globales ==")
    for j, nombre in enumerate(COLUMNAS):
        r = resumen(columna(datos, j))
        print(f"{nombre}: min={r['min']:.1f} max={r['max']:.1f} media={r['media']:.3f} desv={r['desv']:.3f}")

    print("\n== Segmentos de 50 filas ==")
    segmentos = [datos[0:50], datos[50:100], datos[100:150]]
    filas_csv = []
    for s, seg in enumerate(segmentos, 1):
        medias = [st.mean(columna(seg, j)) for j in range(4)]
        desv = [st.stdev(columna(seg, j)) for j in range(4)]
        print(f"Segmento {s}: medias " + "  ".join(f"{n}={m:.2f}" for n, m in zip(COLUMNAS, medias)))
        filas_csv.append([f"Segmento {s}", "media"] + [round(m, 3) for m in medias])
        filas_csv.append([f"Segmento {s}", "desviacion"] + [round(d, 3) for d in desv])

    print("\n== Correlacion de Pearson ==")
    print("     " + "     ".join(COLUMNAS))
    for i in range(4):
        fila = [pearson(columna(datos, i), columna(datos, j)) for j in range(4)]
        print(COLUMNAS[i] + "  " + " ".join(f"{v:+.2f}" for v in fila))

    print("\n== Agrupamiento sin etiquetas (k-means, k=3) ==")
    asignacion, centros = kmeans(datos)
    for s in range(3):
        grupo = asignacion[s * 50:(s + 1) * 50]
        conteo = {g: grupo.count(g) for g in sorted(set(grupo))}
        print(f"Segmento {s + 1} -> grupos k-means {conteo}")
    # Pureza: cuantas filas caen en el grupo mayoritario de su segmento
    aciertos = sum(max(asignacion[s * 50:(s + 1) * 50].count(g) for g in range(3)) for s in range(3))
    print(f"Coincidencia k-means vs. segmentos: {aciertos}/150 = {aciertos / 150:.1%}")

    print("\n== Regla simple con C y D ==")
    reales = [i // 50 for i in range(150)]
    pred = [regla_simple(f) for f in datos]
    correctos = sum(p == r for p, r in zip(pred, reales))
    print("Si C < 2.5 -> segmento 1; si no, D < 1.75 -> segmento 2; si no -> segmento 3")
    print(f"Aciertos: {correctos}/150 = {correctos / 150:.1%}")

    salida = CARPETA / "resumen_segmentos.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Segmento", "Medida"] + COLUMNAS)
        w.writerows(filas_csv)
    print(f"\nResumen guardado en {salida.name} (se puede abrir en Excel).")


if __name__ == "__main__":
    main()
