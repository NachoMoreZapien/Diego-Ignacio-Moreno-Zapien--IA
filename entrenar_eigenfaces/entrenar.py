"""Entrena un reconocedor Eigenfaces con un dataset de rostros y lo guarda en XML.

Autor: Diego Ignacio Moreno Zapien

El dataset tiene una subcarpeta por persona (la que arma
haarcascade_dataset/capturar_rostros.py). El nombre de cada persona se guarda
dentro del mismo XML con setLabelInfo, así el reconocedor puede mostrar el
nombre y no solo un número.

Uso:
    python entrenar.py                                # ../haarcascade_dataset/dataset
    python entrenar.py --dataset ruta --componentes 30 --ver
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

CARPETA = Path(__file__).resolve().parent
DATASET_POR_DEFECTO = CARPETA.parent / "haarcascade_dataset" / "dataset"
MODELO_POR_DEFECTO = CARPETA / "modelo_eigenfaces.xml"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
TAM = 100  # Eigenfaces necesita que todas las imágenes midan lo mismo


def cargar_dataset(carpeta, tam=TAM):
    """Lee las imágenes en gris a tam x tam. Devuelve (imagenes, etiquetas, nombres)."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {carpeta}")

    nombres = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    imagenes, etiquetas = [], []
    for etiqueta, nombre in enumerate(nombres):
        cuantas = 0
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
            cuantas += 1
        print(f"  {nombre} (etiqueta {etiqueta}): {cuantas} imágenes")

    if len(imagenes) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes para entrenar Eigenfaces.")
    return imagenes, np.array(etiquetas, dtype=np.int32), nombres


def entrenar(imagenes, etiquetas, nombres, componentes=0):
    """Crea y entrena el reconocedor.

    componentes = cuántas eigenfaces conservar (0 = todas las posibles, que son
    como máximo el número de imágenes). Menos componentes = modelo más chico y
    menos sensible a detalles, pero también puede perder información.
    """
    if not hasattr(cv, "face"):
        raise RuntimeError("Falta el módulo cv.face: pip install opencv-contrib-python")
    reconocedor = cv.face.EigenFaceRecognizer_create(num_components=componentes)
    reconocedor.train(imagenes, etiquetas)
    for etiqueta, nombre in enumerate(nombres):
        reconocedor.setLabelInfo(etiqueta, nombre)
    return reconocedor


def a_imagen(vector, tam=TAM):
    """Lleva un vector de 10000 valores a una imagen 100x100 visible (0-255)."""
    img = vector.reshape(tam, tam)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def guardar_eigenfaces(reconocedor, destino, cuantas=8):
    """Guarda la cara promedio y las primeras eigenfaces en un solo PNG."""
    media = a_imagen(reconocedor.getMean())
    vectores = reconocedor.getEigenVectors()
    caras = [a_imagen(vectores[:, i]) for i in range(min(cuantas, vectores.shape[1]))]
    cv.imwrite(str(destino), np.hstack([media] + caras))
    return len(caras)


def main():
    p = argparse.ArgumentParser(description="Entrena Eigenfaces y guarda el modelo.")
    p.add_argument("--dataset", default=str(DATASET_POR_DEFECTO))
    p.add_argument("--modelo", default=str(MODELO_POR_DEFECTO), help="archivo XML de salida")
    p.add_argument("--componentes", type=int, default=0, help="eigenfaces a conservar (0 = todas)")
    p.add_argument("--ver", action="store_true", help="guarda eigenfaces.png con la cara promedio y las eigenfaces")
    a = p.parse_args()

    try:
        print("Leyendo dataset:", a.dataset)
        imagenes, etiquetas, nombres = cargar_dataset(a.dataset)
        reconocedor = entrenar(imagenes, etiquetas, nombres, a.componentes)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")

    reconocedor.write(a.modelo)
    print(f"Entrenado con {len(imagenes)} imágenes de {len(nombres)} persona(s), "
          f"{reconocedor.getEigenVectors().shape[1]} eigenfaces.")
    print("Modelo escrito:", a.modelo)
    if len(nombres) == 1:
        print("Aviso: con una sola persona el modelo siempre responde esa persona; "
              "usa el umbral de reconocer.py para detectar desconocidos.")
    if a.ver:
        destino = Path(a.modelo).with_name("eigenfaces.png")
        n = guardar_eigenfaces(reconocedor, destino)
        print(f"Cara promedio + {n} eigenfaces guardadas en: {destino}")


if __name__ == "__main__":
    main()
