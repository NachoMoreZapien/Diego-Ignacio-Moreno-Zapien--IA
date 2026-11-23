"""
Pruebas sin cámara de capturar_rostros.py.

Autor: Diego Ignacio Moreno Zapien

No necesitan webcam ni fotos reales: revisan que el XML cargue, que las
funciones de recorte/numeración hagan lo que dicen y que el ciclo de captura
guarde archivos en la carpeta correcta. Todo se escribe en una carpeta temporal
que se borra sola al terminar.

    python prueba_captura.py
"""

import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

import capturar_rostros as cr


def prueba_clasificador():
    clasificador, ruta = cr.cargar_clasificador()
    assert not clasificador.empty()
    print(f"[OK] clasificador cargado desde {ruta.name}")
    return clasificador


def prueba_sin_rostros(clasificador):
    # Una imagen lisa y una con ruido no deben tener caras.
    lisa = np.full((240, 320), 128, np.uint8)
    ruido = np.random.default_rng(0).integers(0, 256, (240, 320), dtype=np.uint8)
    for nombre, img in (("lisa", lisa), ("ruido", ruido)):
        rostros = cr.detectar_rostros(clasificador, cr.preparar_gris(img))
        assert rostros == [], f"falso positivo en imagen {nombre}: {rostros}"
    print("[OK] 0 detecciones en imagen lisa y en ruido")


def prueba_recorte():
    gris = np.arange(200 * 300, dtype=np.uint32).reshape(200, 300).astype(np.uint8)
    cara = cr.recortar_rostro(gris, (50, 40, 80, 80), tam=100)
    assert cara.shape == (100, 100)
    # Rostro pegado a la esquina: el margen no debe salirse de la imagen.
    esquina = cr.recortar_rostro(gris, (0, 0, 60, 60), tam=64)
    assert esquina.shape == (64, 64)
    print("[OK] recorte siempre de tam x tam, también en la esquina")


def prueba_mas_grande():
    assert cr.rostro_mas_grande([]) is None
    r = cr.rostro_mas_grande([(0, 0, 10, 10), (5, 5, 40, 30), (1, 1, 20, 20)])
    assert r == (5, 5, 40, 30)
    print("[OK] se elige el rostro de mayor área")


def prueba_numeracion(tmp):
    carpeta = tmp / "num"
    carpeta.mkdir()
    assert cr.siguiente_indice(carpeta) == 1
    for n in (1, 2, 7):
        (carpeta / f"rostro_{n:04d}.png").write_bytes(b"")
    assert cr.siguiente_indice(carpeta) == 8
    print("[OK] la numeración continúa (no sobrescribe capturas anteriores)")


def prueba_ciclo(tmp):
    # Carpeta con 5 imágenes de prueba (sin caras reales).
    fotos = tmp / "fotos"
    fotos.mkdir()
    for i in range(5):
        img = np.full((240, 320, 3), 40 * i, np.uint8)
        cv.imwrite(str(fotos / f"f{i}.png"), img)

    # 1) Con el detector real: no hay caras, no se guarda nada.
    n = cr.capturar("sin_caras", tmp / "dataset", maximo=10, fuente=str(fotos),
                    cada=1, mostrar=False)
    assert n == 0

    # 2) Simulando el detector (siempre "ve" una cara en el centro) para
    #    comprobar el guardado, el tamaño y el tope --max.
    original = cr.detectar_rostros
    cr.detectar_rostros = lambda clf, gris, tam_minimo=60: [(110, 70, 100, 100)]
    try:
        n = cr.capturar("simulado", tmp / "dataset", maximo=3, tam=100,
                        fuente=str(fotos), cada=1, mostrar=False)
    finally:
        cr.detectar_rostros = original
    archivos = sorted((tmp / "dataset" / "simulado").glob("*.png"))
    assert n == 3 and len(archivos) == 3, (n, archivos)
    img = cv.imread(str(archivos[0]), cv.IMREAD_UNCHANGED)
    assert img.shape == (100, 100), img.shape  # gris, 100x100
    print(f"[OK] ciclo de captura: {[a.name for a in archivos]} de 100x100 en gris; --max respetado")


def main():
    print("OpenCV", cv.__version__)
    clasificador = prueba_clasificador()
    prueba_sin_rostros(clasificador)
    prueba_recorte()
    prueba_mas_grande()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        prueba_numeracion(tmp)
        prueba_ciclo(tmp)
    print("Todas las pruebas pasaron.")


if __name__ == "__main__":
    main()
