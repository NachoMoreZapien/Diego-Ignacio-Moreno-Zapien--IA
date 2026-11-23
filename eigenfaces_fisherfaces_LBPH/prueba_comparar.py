"""Prueba de comparar.py sin necesitar fotos reales.

Autor: Diego Ignacio Moreno Zapien

Genera un dataset sintético en una carpeta temporal: 3 "personas", cada una con
un patrón de figuras propio (círculos/rectángulos en posiciones fijas) más ruido
y pequeños desplazamientos. No son caras, solo sirven para comprobar que la
carga, la separación, el entrenamiento y la predicción funcionan.
"""

import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

import comparar

TAM = comparar.TAM


def patron_persona(indice, rng):
    """Imagen base distinta para cada persona."""
    img = np.full((TAM, TAM), 90, np.uint8)
    if indice == 0:
        cv.circle(img, (30, 35), 12, 220, -1)
        cv.circle(img, (70, 35), 12, 220, -1)
    elif indice == 1:
        cv.rectangle(img, (20, 60), (80, 75), 230, -1)
    else:
        cv.line(img, (10, 10), (90, 90), 240, 8)
        cv.line(img, (90, 10), (10, 90), 30, 8)
    dx, dy = rng.integers(-3, 4, size=2)
    img = np.roll(img, (dy, dx), axis=(0, 1))
    ruido = rng.normal(0, 12, img.shape)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)


def crear_dataset(carpeta, por_persona=15, semilla=0):
    rng = np.random.default_rng(semilla)
    for i, nombre in enumerate(["ana", "beto", "carla"]):
        destino = Path(carpeta) / nombre
        destino.mkdir(parents=True)
        for j in range(por_persona):
            cv.imwrite(str(destino / f"rostro_{j:04d}.png"), patron_persona(i, rng))
    # una imagen de otro tamaño y un archivo que no es imagen: deben manejarse bien
    cv.imwrite(str(Path(carpeta) / "ana" / "grande.png"), cv.resize(patron_persona(0, rng), (160, 160)))
    (Path(carpeta) / "beto" / "notas.txt").write_text("no soy imagen")
    (Path(carpeta) / "suelto.png").write_bytes(b"")  # archivo suelto en la raíz: se ignora


def comprobar(condicion, mensaje):
    print(("[OK]   " if condicion else "[FALLA] ") + mensaje)
    if not condicion:
        raise SystemExit(1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        crear_dataset(tmp)

        imagenes, etiquetas, nombres = comparar.cargar_dataset(tmp)
        comprobar(nombres == ["ana", "beto", "carla"], "solo se toman subcarpetas, en orden alfabético")
        comprobar(len(imagenes) == 46, f"se leen 46 imágenes y se ignora el .txt ({len(imagenes)})")
        comprobar(all(img.shape == (TAM, TAM) for img in imagenes), "todas quedan de 100x100")

        (x_ent, y_ent), (x_pru, y_pru) = comparar.separar(imagenes, etiquetas, 0.2, semilla=7)
        comprobar(len(x_ent) + len(x_pru) == 46, "entrenamiento + prueba = total")
        comprobar(set(y_pru.tolist()) == {0, 1, 2}, "todas las personas tienen imágenes de prueba")
        otra = comparar.separar(imagenes, etiquetas, 0.2, semilla=7)
        comprobar(np.array_equal(otra[1][1], y_pru), "con la misma semilla la separación se repite")

        print()
        resultados = comparar.comparar(tmp, guardar=True, salida=tmp)
        print()
        # Con 3 personas, adivinar al azar da 33 %. No se exige 100 %: los datos son
        # sintéticos y LBPH, que mira texturas locales, se confunde más con el ruido.
        for r in resultados:
            comprobar(r["acierto"] > 1 / 3, f"{r['metodo']} acierta más que el azar ({r['acierto']:.0%})")
        comprobar(all((Path(tmp) / f).is_file() for f in comparar.ARCHIVOS_MODELO.values()),
                  "se escriben los tres XML")

        # Un modelo guardado se vuelve a cargar y predice lo mismo que el original
        original = cv.face.LBPHFaceRecognizer_create()
        original.train(x_ent, y_ent)
        ruta = Path(tmp) / "recarga.xml"
        original.write(str(ruta))
        recargado = cv.face.LBPHFaceRecognizer_create()
        recargado.read(str(ruta))
        iguales = all(original.predict(img)[0] == recargado.predict(img)[0] for img in x_pru)
        comprobar(iguales, "un XML recargado predice lo mismo que el modelo original")

        try:
            solo = Path(tmp) / "uno"
            (solo / "persona1").mkdir(parents=True)
            for j in range(3):
                cv.imwrite(str(solo / "persona1" / f"{j}.png"), patron_persona(0, np.random.default_rng(j)))
            comparar.comparar(solo)
            comprobar(False, "con una sola persona debe avisar")
        except ValueError:
            comprobar(True, "con una sola persona avisa que se necesitan 2")

    print("\nTodas las pruebas pasaron.")


if __name__ == "__main__":
    main()
