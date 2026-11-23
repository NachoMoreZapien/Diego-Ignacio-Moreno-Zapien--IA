"""
Captura de rostros con Haar Cascade para armar un dataset.

Autor: Diego Ignacio Moreno Zapien
Materia: Inteligencia Artificial

Idea general
------------
1. Se lee una imagen (de la cámara, de un video o de una carpeta de fotos).
2. Se pasa a escala de grises y se ecualiza el histograma (el detector de Haar
   trabaja con intensidades, no con color, y la ecualización ayuda con poca luz).
3. El clasificador en cascada (haarcascade_frontalface_alt.xml) busca rostros
   de frente en varias escalas (detectMultiScale).
4. De cada cuadro se guarda SOLO el rostro más grande (para no meter al dataset
   caras de otras personas que pasen por detrás).
5. El recorte se guarda en gris y redimensionado a un tamaño fijo (100x100 por
   defecto). Así todas las imágenes miden lo mismo, que es lo que piden
   Eigenfaces y Fisherfaces al entrenar.

Uso
---
    python capturar_rostros.py --persona diego               # cámara 0
    python capturar_rostros.py --persona diego --max 200     # más fotos
    python capturar_rostros.py --persona diego --fuente fotos/   # carpeta de imágenes
    python capturar_rostros.py --persona diego --fuente video.mp4
    python capturar_rostros.py --persona diego --sin-ventana  # sin imshow

Las imágenes quedan en dataset/<persona>/rostro_0001.png, rostro_0002.png, ...
Con la cámara: ESC termina antes de llegar a --max.
"""

import argparse
from pathlib import Path

import cv2 as cv

CARPETA = Path(__file__).resolve().parent
CASCADA_LOCAL = CARPETA / "haarcascade_frontalface_alt.xml"
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_clasificador(ruta=None):
    """Carga el XML del detector.

    Primero usa el XML que viene en esta carpeta. Solo si no existe intenta el
    que trae instalado OpenCV (cv.data.haarcascades), porque algunas versiones
    de opencv-python ya no incluyen los XML.
    """
    candidatos = []
    if ruta:
        candidatos.append(Path(ruta))
    candidatos.append(CASCADA_LOCAL)
    datos_cv = getattr(cv, "data", None)
    if datos_cv is not None and getattr(datos_cv, "haarcascades", None):
        candidatos.append(Path(datos_cv.haarcascades) / "haarcascade_frontalface_alt.xml")

    for candidato in candidatos:
        if candidato.is_file():
            clasificador = cv.CascadeClassifier(str(candidato))
            if not clasificador.empty():
                return clasificador, candidato
    raise RuntimeError("No se encontró/cargó haarcascade_frontalface_alt.xml "
                       f"(se probó: {[str(c) for c in candidatos]})")


def preparar_gris(cuadro):
    """BGR -> gris + ecualización de histograma."""
    if cuadro.ndim == 3:
        gris = cv.cvtColor(cuadro, cv.COLOR_BGR2GRAY)
    else:
        gris = cuadro
    return cv.equalizeHist(gris)


def detectar_rostros(clasificador, gris, tam_minimo=60):
    """Devuelve una lista de rectángulos (x, y, w, h).

    scaleFactor=1.2: en cada pasada la imagen se reduce un 20 % para encontrar
    caras de distintos tamaños. minNeighbors=5: una detección debe repetirse en
    5 ventanas vecinas para aceptarse (menos falsos positivos).
    """
    rostros = clasificador.detectMultiScale(
        gris, scaleFactor=1.2, minNeighbors=5, minSize=(tam_minimo, tam_minimo)
    )
    return [tuple(int(v) for v in r) for r in rostros]


def rostro_mas_grande(rostros):
    """El rectángulo de mayor área, o None si no hay ninguno."""
    if not rostros:
        return None
    return max(rostros, key=lambda r: r[2] * r[3])


def recortar_rostro(gris, rect, tam=100, margen=0.10):
    """Recorta el rostro con un pequeño margen y lo lleva a tam x tam.

    El margen (10 % por lado) evita cortar la barbilla o la frente; se limita a
    los bordes de la imagen para no salirse.
    """
    x, y, w, h = rect
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = gris.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    recorte = gris[y0:y1, x0:x1]
    return cv.resize(recorte, (tam, tam), interpolation=cv.INTER_AREA)


def siguiente_indice(carpeta):
    """Para no sobrescribir capturas anteriores: sigue la numeración existente."""
    numeros = []
    for archivo in carpeta.glob("rostro_*.png"):
        try:
            numeros.append(int(archivo.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(numeros, default=0) + 1


def fuentes_de_cuadros(fuente, camara):
    """Generador de cuadros BGR desde cámara, video o carpeta/archivo de imagen."""
    if fuente is None:
        cap = cv.VideoCapture(camara)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camara}")
        try:
            while True:
                ok, cuadro = cap.read()
                if not ok:
                    break
                yield cuadro
        finally:
            cap.release()
        return

    ruta = Path(fuente)
    if ruta.is_dir():
        for archivo in sorted(ruta.iterdir()):
            if archivo.suffix.lower() in EXTENSIONES_IMAGEN:
                cuadro = cv.imread(str(archivo))
                if cuadro is not None:
                    yield cuadro
    elif ruta.suffix.lower() in EXTENSIONES_IMAGEN:
        cuadro = cv.imread(str(ruta))
        if cuadro is None:
            raise RuntimeError(f"No se pudo leer la imagen {ruta}")
        yield cuadro
    else:  # se intenta como video
        cap = cv.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video {ruta}")
        try:
            while True:
                ok, cuadro = cap.read()
                if not ok:
                    break
                yield cuadro
        finally:
            cap.release()


def capturar(persona, salida, maximo=100, tam=100, cada=3, fuente=None,
             camara=0, mostrar=True, cascada=None):
    """Recorre los cuadros y guarda rostros. Devuelve cuántos guardó.

    cada=3 -> solo se guarda un rostro cada 3 cuadros, para que las fotos no
    sean casi idénticas (da tiempo a mover un poco la cabeza).
    En carpetas/imágenes sueltas conviene cada=1.
    """
    clasificador, ruta_xml = cargar_clasificador(cascada)
    print(f"Clasificador: {ruta_xml.name}")

    carpeta = Path(salida) / persona
    carpeta.mkdir(parents=True, exist_ok=True)
    indice = siguiente_indice(carpeta)
    guardados = 0

    for n, cuadro in enumerate(fuentes_de_cuadros(fuente, camara)):
        gris = preparar_gris(cuadro)
        rostros = detectar_rostros(clasificador, gris)
        principal = rostro_mas_grande(rostros)

        if principal is not None and n % cada == 0:
            cara = recortar_rostro(gris, principal, tam)
            cv.imwrite(str(carpeta / f"rostro_{indice:04d}.png"), cara)
            indice += 1
            guardados += 1

        if mostrar:
            vista = cuadro.copy()
            for (x, y, w, h) in rostros:
                color = (0, 255, 0) if (x, y, w, h) == principal else (0, 0, 255)
                cv.rectangle(vista, (x, y), (x + w, y + h), color, 2)
            cv.putText(vista, f"{persona}: {guardados}/{maximo}", (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.imshow("Captura de rostros (ESC = salir)", vista)
            if cv.waitKey(1) & 0xFF == 27:
                break

        if guardados >= maximo:
            break

    if mostrar:
        cv.destroyAllWindows()
    print(f"Rostros guardados en {carpeta}: {guardados}")
    return guardados


def main():
    p = argparse.ArgumentParser(description="Arma un dataset de rostros con Haar Cascade.")
    p.add_argument("--persona", required=True, help="nombre de la subcarpeta (una por persona)")
    p.add_argument("--salida", default=str(CARPETA / "dataset"), help="carpeta raíz del dataset")
    p.add_argument("--max", type=int, default=100, help="cuántos rostros guardar")
    p.add_argument("--tam", type=int, default=100, help="lado del recorte cuadrado en píxeles")
    p.add_argument("--cada", type=int, default=None,
                   help="guardar 1 de cada N cuadros (por defecto 3 en cámara/video, 1 en imágenes)")
    p.add_argument("--fuente", default=None, help="imagen, carpeta de imágenes o video (si no, cámara)")
    p.add_argument("--camara", type=int, default=0, help="índice de la cámara")
    p.add_argument("--cascada", default=None, help="ruta a otro XML de Haar")
    p.add_argument("--sin-ventana", action="store_true", help="no mostrar ventanas")
    a = p.parse_args()

    cada = a.cada
    if cada is None:
        es_imagenes = a.fuente is not None and (
            Path(a.fuente).is_dir() or Path(a.fuente).suffix.lower() in EXTENSIONES_IMAGEN)
        cada = 1 if es_imagenes else 3

    capturar(a.persona, a.salida, a.max, a.tam, max(1, cada), a.fuente,
             a.camara, not a.sin_ventana, a.cascada)


if __name__ == "__main__":
    main()
