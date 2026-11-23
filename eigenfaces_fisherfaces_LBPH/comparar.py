"""Compara Eigenfaces, Fisherfaces y LBPH con el mismo dataset de rostros.

Autor: Diego Ignacio Moreno Zapien

El dataset es una carpeta con una subcarpeta por persona (la que arma
haarcascade_dataset/capturar_rostros.py):

    dataset/
        persona1/   rostro_0001.png, rostro_0002.png, ...
        persona2/   ...

Para que la comparación sea justa, los tres métodos usan exactamente las mismas
imágenes: de cada persona se separa un porcentaje para prueba (no se usa al
entrenar) y con esas imágenes se mide el acierto.

Uso:
    python comparar.py                          # usa ../haarcascade_dataset/dataset
    python comparar.py --dataset ruta/dataset --prueba 0.25 --guardar
"""

import argparse
import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np

CARPETA = Path(__file__).resolve().parent
DATASET_POR_DEFECTO = CARPETA.parent / "haarcascade_dataset" / "dataset"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
TAM = 100  # Eigenfaces y Fisherfaces exigen que todas las imágenes midan lo mismo


def cargar_dataset(carpeta, tam=TAM):
    """Lee las imágenes en gris, las lleva a tam x tam y les pone etiqueta.

    Devuelve (imagenes, etiquetas, nombres). La etiqueta de cada persona es su
    posición en la lista de nombres, que va en orden alfabético.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {carpeta}")

    nombres = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    imagenes, etiquetas = [], []
    for etiqueta, nombre in enumerate(nombres):
        for archivo in sorted((carpeta / nombre).iterdir()):
            if archivo.suffix.lower() not in EXTENSIONES:
                continue
            img = cv.imread(str(archivo), cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (tam, tam):
                img = cv.resize(img, (tam, tam), interpolation=cv.INTER_AREA)
            imagenes.append(img)
            etiquetas.append(etiqueta)
    return imagenes, np.array(etiquetas, dtype=np.int32), nombres


def separar(imagenes, etiquetas, proporcion_prueba=0.2, semilla=7):
    """Separa entrenamiento y prueba por persona (estratificado).

    Cada persona deja al menos una imagen para prueba y al menos una para
    entrenamiento. La semilla fija hace que la separación sea siempre igual.
    """
    rng = random.Random(semilla)
    entrena, prueba = [], []
    for etiqueta in sorted(set(etiquetas.tolist())):
        indices = [i for i, e in enumerate(etiquetas) if e == etiqueta]
        rng.shuffle(indices)
        n_prueba = max(1, round(len(indices) * proporcion_prueba))
        if len(indices) - n_prueba < 1:
            raise ValueError(f"La persona {etiqueta} necesita al menos 2 imágenes")
        prueba += indices[:n_prueba]
        entrena += indices[n_prueba:]

    def tomar(idx):
        return [imagenes[i] for i in idx], etiquetas[idx]

    return tomar(entrena), tomar(prueba)


def crear_reconocedores():
    """Los tres reconocedores de cv.face (necesita opencv-contrib-python)."""
    if not hasattr(cv, "face"):
        raise RuntimeError("Falta el módulo cv.face: pip install opencv-contrib-python")
    return {
        "Eigenfaces": cv.face.EigenFaceRecognizer_create(),
        "Fisherfaces": cv.face.FisherFaceRecognizer_create(),
        "LBPH": cv.face.LBPHFaceRecognizer_create(),
    }


def evaluar(nombre, reconocedor, entrena, prueba):
    """Entrena, predice el conjunto de prueba y regresa las métricas."""
    x_ent, y_ent = entrena
    x_pru, y_pru = prueba

    t0 = time.perf_counter()
    reconocedor.train(x_ent, y_ent)
    t_entrena = time.perf_counter() - t0

    aciertos, distancias = 0, []
    t0 = time.perf_counter()
    for img, real in zip(x_pru, y_pru):
        predicha, distancia = reconocedor.predict(img)
        aciertos += int(predicha == real)
        distancias.append(distancia)
    t_predice = (time.perf_counter() - t0) / len(x_pru)

    return {
        "metodo": nombre,
        "acierto": aciertos / len(x_pru),
        "aciertos": aciertos,
        "total": len(x_pru),
        "t_entrena_ms": t_entrena * 1000,
        "t_predice_ms": t_predice * 1000,
        "distancia_media": float(np.mean(distancias)),
    }


ARCHIVOS_MODELO = {
    "Eigenfaces": "modelo_eigenfaces.xml",
    "Fisherfaces": "modelo_fisherfaces.xml",
    "LBPH": "modelo_lbph.xml",
}


def comparar(carpeta, proporcion_prueba=0.2, semilla=7, guardar=False, salida=CARPETA):
    imagenes, etiquetas, nombres = cargar_dataset(carpeta)
    print("Personas:", nombres)
    for i, nombre in enumerate(nombres):
        print(f"  {nombre} (etiqueta {i}): {int((etiquetas == i).sum())} imágenes")
    if len(nombres) < 2:
        raise ValueError("Se necesitan al menos 2 personas: Fisherfaces separa clases y "
                         "con una sola no hay nada que comparar.")

    entrena, prueba = separar(imagenes, etiquetas, proporcion_prueba, semilla)
    print(f"Entrenamiento: {len(entrena[0])} imágenes | Prueba: {len(prueba[0])} imágenes\n")

    resultados = []
    for nombre, reconocedor in crear_reconocedores().items():
        r = evaluar(nombre, reconocedor, entrena, prueba)
        resultados.append(r)
        if guardar:
            ruta = Path(salida) / ARCHIVOS_MODELO[nombre]
            reconocedor.write(str(ruta))
            r["tam_modelo_kb"] = ruta.stat().st_size / 1024

    encabezado = f"{'Método':<12} {'Acierto':>14} {'Entrenar':>10} {'Predecir':>10} {'Dist. media':>12}"
    if guardar:
        encabezado += f" {'Modelo':>10}"
    print(encabezado)
    print("-" * len(encabezado))
    for r in resultados:
        linea = (f"{r['metodo']:<12} {r['aciertos']:>3}/{r['total']:<3} {r['acierto']:>6.1%} "
                 f"{r['t_entrena_ms']:>8.1f}ms {r['t_predice_ms']:>8.2f}ms {r['distancia_media']:>12.1f}")
        if guardar:
            linea += f" {r['tam_modelo_kb']:>8.0f}KB"
        print(linea)
    print("\nOjo: la distancia de cada método está en su propia escala; solo sirve para "
          "comparar predicciones del mismo método, no entre métodos.")
    return resultados


def main():
    p = argparse.ArgumentParser(description="Compara Eigenfaces, Fisherfaces y LBPH.")
    p.add_argument("--dataset", default=str(DATASET_POR_DEFECTO), help="carpeta con una subcarpeta por persona")
    p.add_argument("--prueba", type=float, default=0.2, help="proporción de imágenes para prueba")
    p.add_argument("--semilla", type=int, default=7)
    p.add_argument("--guardar", action="store_true", help="guarda los tres modelos en XML")
    a = p.parse_args()
    try:
        comparar(a.dataset, a.prueba, a.semilla, a.guardar)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()
