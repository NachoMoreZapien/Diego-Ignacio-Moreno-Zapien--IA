"""Pruebas de entrenar.py y reconocer.py sin cámara ni fotos reales.

Autor: Diego Ignacio Moreno Zapien

Usa un dataset sintético (figuras con ruido, no son caras) en una carpeta
temporal para comprobar el entrenamiento, el guardado con nombres, la recarga
del XML, el umbral de "desconocido" y la imagen de eigenfaces.
"""

import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

import entrenar
import reconocer

TAM = entrenar.TAM


def patron(indice, rng):
    img = np.full((TAM, TAM), 90, np.uint8)
    if indice == 0:
        cv.circle(img, (30, 35), 12, 220, -1)
        cv.circle(img, (70, 35), 12, 220, -1)
    else:
        cv.rectangle(img, (20, 60), (80, 75), 230, -1)
    img = np.roll(img, tuple(rng.integers(-3, 4, size=2)), axis=(0, 1))
    return np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)


def comprobar(condicion, mensaje):
    print(("[OK]   " if condicion else "[FALLA] ") + mensaje)
    if not condicion:
        raise SystemExit(1)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, nombre in enumerate(["ana", "beto"]):
            (tmp / "dataset" / nombre).mkdir(parents=True)
            for j in range(12):
                cv.imwrite(str(tmp / "dataset" / nombre / f"{j}.png"), patron(i, rng))
        cv.imwrite(str(tmp / "dataset" / "ana" / "grande.png"), cv.resize(patron(0, rng), (150, 150)))

        imagenes, etiquetas, nombres = entrenar.cargar_dataset(tmp / "dataset")
        comprobar(nombres == ["ana", "beto"] and len(imagenes) == 25, "carga 25 imágenes de 2 personas")
        comprobar(all(img.shape == (TAM, TAM) for img in imagenes), "redimensiona a 100x100")

        modelo = entrenar.entrenar(imagenes, etiquetas, nombres, componentes=10)
        comprobar(modelo.getEigenVectors().shape == (TAM * TAM, 10), "conserva 10 eigenfaces")

        ruta = tmp / "modelo.xml"
        modelo.write(str(ruta))
        cargado = reconocer.cargar_modelo(ruta)
        comprobar(cargado.getLabelInfo(0) == "ana" and cargado.getLabelInfo(1) == "beto",
                  "los nombres se guardan dentro del XML")

        nuevas = [(patron(0, rng), "ana"), (patron(1, rng), "beto")]
        aciertos = sum(reconocer.identificar(cargado, img, umbral=1e9)[0] == real for img, real in nuevas * 5)
        comprobar(aciertos == 10, f"reconoce imágenes nuevas ({aciertos}/10)")

        nombre, _ = reconocer.identificar(cargado, nuevas[0][0], umbral=0)
        comprobar(nombre == "desconocido", "con umbral 0 todo es desconocido")
        nombre, _ = reconocer.identificar(cargado, cv.resize(nuevas[1][0], (200, 200)), umbral=1e9)
        comprobar(nombre == "beto", "acepta una cara de otro tamaño y la redimensiona")

        recorte = reconocer.recortar(np.zeros((300, 400), np.uint8), (350, 250, 80, 80))
        comprobar(recorte.shape == (TAM, TAM), "el recorte en una esquina queda de 100x100")

        n = entrenar.guardar_eigenfaces(cargado, tmp / "eigenfaces.png", cuantas=8)
        img = cv.imread(str(tmp / "eigenfaces.png"), cv.IMREAD_GRAYSCALE)
        comprobar(n == 8 and img.shape == (TAM, TAM * 9), "eigenfaces.png = promedio + 8 eigenfaces")

        try:
            reconocer.cargar_modelo(tmp / "no_existe.xml")
            comprobar(False, "debe avisar si falta el modelo")
        except FileNotFoundError:
            comprobar(True, "avisa si todavía no se ha entrenado")

    print("\nTodas las pruebas pasaron.")


if __name__ == "__main__":
    main()
