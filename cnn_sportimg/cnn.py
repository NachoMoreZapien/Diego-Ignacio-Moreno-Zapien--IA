"""
CNN: clasificacion de imagenes deportivas (10 deportes).

Autor: Diego Ignacio Moreno Zapien
Materia: Inteligencia Artificial

Uso:
    python cnn.py revisar  --datos sportimages/          # solo cuenta imagenes (no necesita TensorFlow)
    python cnn.py entrenar --datos sportimages/ --epocas 20
    python cnn.py predecir foto1.jpg foto2.jpg            # usa el modelo guardado
    python cnn.py prueba                                  # prueba con imagenes sinteticas

El dataset es una carpeta con una subcarpeta por deporte:
    sportimages/americano/*.jpg, sportimages/basket/*.jpg, ...
El nombre de cada subcarpeta es la etiqueta de la clase.

La parte de datos y metricas solo usa NumPy y Pillow; TensorFlow/Keras
se importa unicamente cuando se construye, entrena o carga la red.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

# Las imagenes del dataset miden 28 x 21 pixeles (ancho x alto) y son a color.
ALTO, ANCHO, CANALES = 21, 28, 3
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO_POR_DEFECTO = os.path.join(AQUI, "sport_model.keras")
CLASES_POR_DEFECTO = os.path.join(AQUI, "clases.json")


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

def leer_imagen(ruta):
    """Abre una imagen, la pasa a RGB y la deja de 28x21. Regresa uint8 (21, 28, 3)."""
    with Image.open(ruta) as img:
        img = img.convert("RGB")  # las imagenes en gris o con alfa quedan con 3 canales
        if img.size != (ANCHO, ALTO):
            img = img.resize((ANCHO, ALTO), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def cargar_dataset(carpeta, mostrar=True):
    """Lee todas las imagenes. Cada subcarpeta es una clase (orden alfabetico).

    Regresa X (n, 21, 28, 3) uint8, y (n,) int, y la lista de nombres de clase.
    La etiqueta se toma del nombre de la carpeta de cada imagen, asi no
    importa en que orden las recorra el sistema operativo.
    """
    clases = sorted(d for d in os.listdir(carpeta)
                    if os.path.isdir(os.path.join(carpeta, d)))
    if not clases:
        raise ValueError(f"'{carpeta}' no tiene subcarpetas (una por deporte).")

    imagenes, etiquetas, omitidas = [], [], 0
    for indice, clase in enumerate(clases):
        ruta_clase = os.path.join(carpeta, clase)
        archivos = sorted(f for f in os.listdir(ruta_clase)
                          if f.lower().endswith(EXTENSIONES))
        for nombre in archivos:
            try:
                imagenes.append(leer_imagen(os.path.join(ruta_clase, nombre)))
                etiquetas.append(indice)
            except OSError:
                omitidas += 1  # archivo danado o que no es imagen
        if mostrar:
            print(f"  {indice}: {clase:<12} {etiquetas.count(indice):>6} imagenes")

    if omitidas and mostrar:
        print(f"  ({omitidas} archivos no se pudieron leer y se omitieron)")
    X = np.stack(imagenes) if imagenes else np.empty((0, ALTO, ANCHO, CANALES), np.uint8)
    return X, np.array(etiquetas, dtype=np.int64), clases


def dividir_estratificado(X, y, proporcion, semilla=0):
    """Separa una parte (proporcion) de cada clase. Asi entrenamiento y prueba
    quedan con la misma mezcla de deportes aunque una clase tenga mas fotos."""
    rng = np.random.default_rng(semilla)
    idx_a, idx_b = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        corte = int(round(len(idx) * proporcion))
        idx_b.extend(idx[:corte])
        idx_a.extend(idx[corte:])
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    return X[idx_a], X[idx_b], y[idx_a], y[idx_b]


def dividir_por_orden(X, y, proporcion):
    """Separa el ULTIMO tramo (proporcion) de cada clase, sin revolver.

    Las imagenes del dataset son cuadros seguidos de videos (basket_0001,
    basket_0002, ...) y cargar_dataset las lee en orden de nombre. Si se
    revuelven al azar, la prueba queda llena de cuadros casi identicos a los
    de entrenamiento y la exactitud sale inflada. Cortando por orden, la red
    se evalua con el final de cada video, que nunca vio.
    """
    idx_a, idx_b = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)  # ya vienen en orden de archivo
        corte = len(idx) - int(round(len(idx) * proporcion))
        idx_a.extend(idx[:corte])
        idx_b.extend(idx[corte:])
    rng = np.random.default_rng(0)
    idx_a = np.array(idx_a)
    rng.shuffle(idx_a)  # revolver solo el entrenamiento ayuda a que aprenda parejo
    idx_b = np.array(idx_b)
    return X[idx_a], X[idx_b], y[idx_a], y[idx_b]


def normalizar(X):
    """Pixeles de 0..255 a 0..1 (la red aprende mejor con valores chicos)."""
    return X.astype("float32") / 255.0


def one_hot(y, n_clases):
    """Etiqueta 3 -> [0,0,0,1,0,...]. Es el formato que pide la salida softmax."""
    m = np.zeros((len(y), n_clases), dtype="float32")
    m[np.arange(len(y)), y] = 1.0
    return m


# ---------------------------------------------------------------------------
# 2. Metricas (sin scikit-learn)
# ---------------------------------------------------------------------------

def matriz_confusion(y_real, y_pred, n_clases):
    """Fila = clase real, columna = clase predicha."""
    m = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(m, (y_real, y_pred), 1)
    return m


def reporte(y_real, y_pred, clases):
    """Precision, recall y F1 por clase, como texto."""
    m = matriz_confusion(y_real, y_pred, len(clases))
    vp = np.diag(m).astype(float)
    predichos = m.sum(axis=0)
    reales = m.sum(axis=1)
    precision = np.divide(vp, predichos, out=np.zeros_like(vp), where=predichos > 0)
    recall = np.divide(vp, reales, out=np.zeros_like(vp), where=reales > 0)
    suma = precision + recall
    f1 = np.divide(2 * precision * recall, suma, out=np.zeros_like(vp), where=suma > 0)

    lineas = [f"{'clase':<12}{'precision':>10}{'recall':>8}{'f1':>8}{'n':>7}"]
    for i, c in enumerate(clases):
        lineas.append(f"{c:<12}{precision[i]:>10.2f}{recall[i]:>8.2f}{f1[i]:>8.2f}{reales[i]:>7}")
    exactitud = vp.sum() / max(m.sum(), 1)
    lineas.append(f"{'exactitud':<12}{exactitud:>34.4f}{m.sum():>7}")
    return "\n".join(lineas), m


# ---------------------------------------------------------------------------
# 3. Red convolucional (Keras)
# ---------------------------------------------------------------------------

def importar_keras():
    try:
        import keras  # Keras 3 (viene con TensorFlow 2.16 o superior)
        return keras
    except ImportError:
        sys.exit("Falta TensorFlow/Keras. Instala con:  pip install tensorflow")


def construir_modelo(n_clases):
    """Dos bloques convolucionales y una capa densa.

    Bloque = Conv2D (detecta bordes, colores, texturas) -> LeakyReLU ->
    MaxPooling (reduce el tamano a la mitad) -> Dropout (apaga neuronas al
    azar para que no memorice). Al final, Dense + softmax da una probabilidad
    por deporte.
    """
    keras = importar_keras()
    L = keras.layers
    modelo = keras.Sequential([
        keras.Input(shape=(ALTO, ANCHO, CANALES)),
        # Bloque 1: 32 filtros de 3x3. 21x28 -> 11x14
        L.Conv2D(32, (3, 3), padding="same"),
        L.LeakyReLU(negative_slope=0.1),
        L.MaxPooling2D((2, 2), padding="same"),
        L.Dropout(0.25),
        # Bloque 2: 64 filtros, combina lo que encontro el bloque 1. 11x14 -> 6x7
        L.Conv2D(64, (3, 3), padding="same"),
        L.LeakyReLU(negative_slope=0.1),
        L.MaxPooling2D((2, 2), padding="same"),
        L.Dropout(0.25),
        # Clasificador
        L.Flatten(),
        L.Dense(64),
        L.LeakyReLU(negative_slope=0.1),
        L.Dropout(0.5),
        L.Dense(n_clases, activation="softmax"),
    ])
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def guardar_graficas(historial, ruta):
    """Exactitud y perdida por epoca (entrenamiento vs validacion)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no esta instalado: no se guardan graficas)")
        return
    h = historial.history
    epocas = range(1, len(h["accuracy"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(epocas, h["accuracy"], "o-", label="entrenamiento")
    a1.plot(epocas, h["val_accuracy"], "s-", label="validacion")
    a1.set_title("Exactitud"); a1.set_xlabel("epoca"); a1.legend()
    a2.plot(epocas, h["loss"], "o-", label="entrenamiento")
    a2.plot(epocas, h["val_loss"], "s-", label="validacion")
    a2.set_title("Perdida"); a2.set_xlabel("epoca"); a2.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)
    print(f"Graficas guardadas en {ruta}")


def entrenar(carpeta, epocas=20, lote=64, modelo_ruta=MODELO_POR_DEFECTO,
             clases_ruta=CLASES_POR_DEFECTO, graficas=True, semilla=0, division="azar"):
    print(f"Leyendo imagenes de {carpeta}")
    X, y, clases = cargar_dataset(carpeta)
    print(f"Total: {len(X)} imagenes, {len(clases)} clases, forma {X.shape[1:]}")

    # 80 % entrenamiento / 20 % prueba; del 80 %, otro 20 % para validacion.
    # "azar": revuelve (como en clase). "video": corta por orden de cuadro.
    if division == "video":
        X_ent, X_prueba, y_ent, y_prueba = dividir_por_orden(X, y, 0.2)
        X_ent, X_val, y_ent, y_val = dividir_estratificado(X_ent, y_ent, 0.2, semilla + 1)
    else:
        X_ent, X_prueba, y_ent, y_prueba = dividir_estratificado(X, y, 0.2, semilla)
        X_ent, X_val, y_ent, y_val = dividir_estratificado(X_ent, y_ent, 0.2, semilla + 1)
    print(f"Division '{division}': entrenamiento {len(X_ent)}, validacion {len(X_val)}, "
          f"prueba {len(X_prueba)}")

    n = len(clases)
    modelo = construir_modelo(n)
    modelo.summary()
    historial = modelo.fit(
        normalizar(X_ent), one_hot(y_ent, n),
        validation_data=(normalizar(X_val), one_hot(y_val, n)),
        epochs=epocas, batch_size=lote, verbose=2,
    )

    perdida, exactitud = modelo.evaluate(normalizar(X_prueba), one_hot(y_prueba, n), verbose=0)
    print(f"\nPrueba -> perdida {perdida:.4f}, exactitud {exactitud:.4f}")
    y_pred = modelo.predict(normalizar(X_prueba), verbose=0).argmax(axis=1)
    texto, m = reporte(y_prueba, y_pred, clases)
    print(texto)
    print("Matriz de confusion (fila = real, columna = predicha):")
    print(m)

    modelo.save(modelo_ruta)
    with open(clases_ruta, "w", encoding="utf-8") as f:
        json.dump(clases, f, ensure_ascii=False, indent=2)
    print(f"Modelo guardado en {modelo_ruta} y clases en {clases_ruta}")
    if graficas:
        guardar_graficas(historial, os.path.join(os.path.dirname(modelo_ruta), "graficas.png"))
    return modelo, clases, exactitud


def predecir(rutas, modelo_ruta=MODELO_POR_DEFECTO, clases_ruta=CLASES_POR_DEFECTO):
    keras = importar_keras()
    modelo = keras.models.load_model(modelo_ruta)
    with open(clases_ruta, encoding="utf-8") as f:
        clases = json.load(f)
    X = normalizar(np.stack([leer_imagen(r) for r in rutas]))
    probs = modelo.predict(X, verbose=0)
    for ruta, p in zip(rutas, probs):
        i = int(p.argmax())
        print(f"{os.path.basename(ruta)}: {clases[i]} ({p[i]:.0%})")
    return probs


# ---------------------------------------------------------------------------
# 4. Prueba con datos sinteticos (no necesita el dataset real)
# ---------------------------------------------------------------------------

def crear_dataset_sintetico(carpeta, por_clase=40, semilla=0):
    """Tres 'deportes' falsos que se distinguen por el color dominante,
    con ruido. Sirve para comprobar que todo el flujo funciona."""
    rng = np.random.default_rng(semilla)
    colores = {"rojo": (200, 40, 40), "verde": (40, 180, 60), "azul": (40, 60, 200)}
    for nombre, color in colores.items():
        os.makedirs(os.path.join(carpeta, nombre), exist_ok=True)
        for k in range(por_clase):
            base = np.array(color, dtype=float) + rng.normal(0, 35, (ALTO, ANCHO, CANALES))
            img = np.clip(base, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(carpeta, nombre, f"img_{k:03d}.png"))
    # Una imagen de otro tamano y en gris, para probar que se ajusta sola
    Image.fromarray(np.full((50, 70), 128, np.uint8)).save(
        os.path.join(carpeta, "azul", "gris_grande.jpg"))
    # Un archivo que no es imagen, para probar que se omite
    with open(os.path.join(carpeta, "rojo", "roto.jpg"), "w") as f:
        f.write("esto no es una imagen")


def prueba():
    ok = True

    def revisar(condicion, mensaje):
        nonlocal ok
        print(("OK   " if condicion else "FALLA") + " " + mensaje)
        ok = ok and condicion

    with tempfile.TemporaryDirectory() as tmp:
        datos = os.path.join(tmp, "datos")
        crear_dataset_sintetico(datos)
        X, y, clases = cargar_dataset(datos, mostrar=False)
        revisar(clases == ["azul", "rojo", "verde"], f"clases en orden alfabetico: {clases}")
        revisar(X.shape == (121, ALTO, ANCHO, CANALES), f"forma de X: {X.shape}")
        revisar(X.dtype == np.uint8, "pixeles uint8")
        revisar(np.bincount(y).tolist() == [41, 40, 40], f"imagenes por clase: {np.bincount(y).tolist()}")

        Xa, Xb, ya, yb = dividir_estratificado(X, y, 0.2)
        revisar(len(Xa) + len(Xb) == len(X), f"division sin perder imagenes: {len(Xa)} + {len(Xb)}")
        revisar(np.bincount(yb).tolist() == [8, 8, 8], f"prueba estratificada: {np.bincount(yb).tolist()}")

        # Division por video: la prueba debe ser el final de cada clase, sin revolver
        Xa, Xb, ya, yb = dividir_por_orden(np.arange(len(y)), y, 0.2)
        finales = all(Xb[yb == c].tolist() == np.flatnonzero(y == c)[-8:].tolist() for c in range(3))
        revisar(len(Xa) + len(Xb) == len(y) and finales, "division por video: prueba = ultimo 20 % de cada clase")

        N = normalizar(X)
        revisar(N.min() >= 0 and N.max() <= 1, "normalizacion en [0, 1]")
        revisar(one_hot(np.array([2, 0]), 3).tolist() == [[0, 0, 1], [1, 0, 0]], "one-hot")

        texto, m = reporte(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0]), ["a", "b", "c"])
        revisar(m.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]], "matriz de confusion")
        revisar("0.6667" in texto, "exactitud 4/6 en el reporte")

        try:
            import keras  # noqa: F401
        except ImportError:
            print("AVISO TensorFlow/Keras no esta instalado: se omite el entrenamiento de prueba.")
        else:
            modelo_ruta = os.path.join(tmp, "m.keras")
            _, _, exactitud = entrenar(datos, epocas=8, lote=16, modelo_ruta=modelo_ruta,
                                       clases_ruta=os.path.join(tmp, "c.json"), graficas=False)
            revisar(exactitud > 0.8, f"la red separa los colores sinteticos (exactitud {exactitud:.2f})")
            probs = predecir([os.path.join(datos, "rojo", "img_000.png")], modelo_ruta,
                             os.path.join(tmp, "c.json"))
            revisar(abs(float(probs.sum()) - 1) < 1e-4, "softmax suma 1")

    print("\nTODO OK" if ok else "\nHAY FALLAS")
    return ok


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CNN para clasificar imagenes deportivas")
    sub = p.add_subparsers(dest="accion", required=True)

    r = sub.add_parser("revisar", help="contar imagenes por clase (sin TensorFlow)")
    r.add_argument("--datos", required=True)

    e = sub.add_parser("entrenar", help="entrenar y guardar la red")
    e.add_argument("--datos", required=True)
    e.add_argument("--epocas", type=int, default=20)
    e.add_argument("--lote", type=int, default=64)
    e.add_argument("--division", choices=["azar", "video"], default="azar",
                   help="azar = revolver como en clase; video = probar con el final de cada video")

    pr = sub.add_parser("predecir", help="clasificar imagenes con el modelo guardado")
    pr.add_argument("imagenes", nargs="+")

    sub.add_parser("prueba", help="prueba con imagenes sinteticas")

    a = p.parse_args()
    if a.accion == "revisar":
        X, y, clases = cargar_dataset(a.datos)
        print(f"Total: {len(X)} imagenes en {len(clases)} clases; forma {X.shape[1:]}")
    elif a.accion == "entrenar":
        entrenar(a.datos, a.epocas, a.lote, division=a.division)
    elif a.accion == "predecir":
        predecir(a.imagenes)
    else:
        sys.exit(0 if prueba() else 1)


if __name__ == "__main__":
    main()
