"""Reconoce rostros con el modelo Eigenfaces entrenado por entrenar.py.

Autor: Diego Ignacio Moreno Zapien

Con la cámara: detecta la cara con Haar Cascade, la recorta igual que en la
captura del dataset (gris, 100x100) y le pregunta al modelo quién es.
Si la distancia pasa del umbral, se muestra "desconocido".

Uso:
    python reconocer.py                       # cámara 0, ESC para salir
    python reconocer.py --umbral 1800
    python reconocer.py --fuente fotos/       # imágenes ya recortadas (sin cámara)
"""

import argparse
from pathlib import Path

import cv2 as cv

CARPETA = Path(__file__).resolve().parent
MODELO_POR_DEFECTO = CARPETA / "modelo_eigenfaces.xml"
CASCADA = CARPETA.parent / "haarcascade_dataset" / "haarcascade_frontalface_alt.xml"
TAM = 100
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}


def cargar_modelo(ruta):
    if not Path(ruta).is_file():
        raise FileNotFoundError(f"No existe {ruta}. Primero ejecuta: python entrenar.py")
    reconocedor = cv.face.EigenFaceRecognizer_create()
    reconocedor.read(str(ruta))
    return reconocedor


def identificar(reconocedor, cara_gris, umbral):
    """Regresa (nombre, distancia). La cara debe venir en gris."""
    if cara_gris.shape != (TAM, TAM):
        cara_gris = cv.resize(cara_gris, (TAM, TAM), interpolation=cv.INTER_AREA)
    etiqueta, distancia = reconocedor.predict(cara_gris)
    if distancia > umbral:
        return "desconocido", distancia
    return reconocedor.getLabelInfo(etiqueta) or f"persona {etiqueta}", distancia


def recortar(gris, rect, margen=0.10):
    """Mismo recorte que capturar_rostros.py: 10 % de margen y 100x100."""
    x, y, w, h = rect
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = gris.shape
    recorte = gris[max(0, y - dy):min(alto, y + h + dy), max(0, x - dx):min(ancho, x + w + dx)]
    return cv.resize(recorte, (TAM, TAM), interpolation=cv.INTER_AREA)


def desde_imagenes(reconocedor, carpeta, umbral):
    """Clasifica imágenes de rostros ya recortados (útil para probar sin cámara)."""
    archivos = [p for p in sorted(Path(carpeta).rglob("*")) if p.suffix.lower() in EXTENSIONES]
    for archivo in archivos:
        img = cv.imread(str(archivo), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        nombre, distancia = identificar(reconocedor, img, umbral)
        print(f"{archivo.name:<20} -> {nombre:<12} (distancia {distancia:.0f})")


def desde_camara(reconocedor, umbral, camara=0):
    detector = cv.CascadeClassifier(str(CASCADA))
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar el detector: {CASCADA}")
    cap = cv.VideoCapture(camara)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camara}")
    try:
        while True:
            ok, cuadro = cap.read()
            if not ok:
                break
            gris = cv.equalizeHist(cv.cvtColor(cuadro, cv.COLOR_BGR2GRAY))
            for rect in detector.detectMultiScale(gris, 1.2, 5, minSize=(60, 60)):
                nombre, distancia = identificar(reconocedor, recortar(gris, rect), umbral)
                x, y, w, h = rect
                color = (0, 0, 255) if nombre == "desconocido" else (0, 200, 0)
                cv.rectangle(cuadro, (x, y), (x + w, y + h), color, 2)
                cv.putText(cuadro, f"{nombre} {distancia:.0f}", (x, y - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv.imshow("Eigenfaces (ESC para salir)", cuadro)
            if cv.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Reconoce rostros con Eigenfaces.")
    p.add_argument("--modelo", default=str(MODELO_POR_DEFECTO))
    # 1600 sale de medir con el dataset real: caras de la persona no vistas al entrenar
    # dieron 458-1343 y el ruido aleatorio ~2100. Recalíbralo si cambia el dataset.
    p.add_argument("--umbral", type=float, default=1600,
                   help="distancia máxima para aceptar a alguien (ajústala con tus pruebas)")
    p.add_argument("--fuente", default=None, help="carpeta de rostros recortados en vez de la cámara")
    p.add_argument("--camara", type=int, default=0)
    a = p.parse_args()

    try:
        reconocedor = cargar_modelo(a.modelo)
        if a.fuente:
            desde_imagenes(reconocedor, a.fuente, a.umbral)
        else:
            desde_camara(reconocedor, a.umbral, a.camara)
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()
