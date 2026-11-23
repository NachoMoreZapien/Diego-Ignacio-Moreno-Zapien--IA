# Dataset de rostros con Haar Cascade

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Usar el detector de rostros **Haar Cascade** de OpenCV con la cámara web para capturar
imágenes de la cara de cada persona y guardarlas en carpetas, formando el dataset que después
se usa para entrenar los reconocedores Eigenfaces, Fisherfaces y LBPH.

> Nota: el objetivo es la captura de rostros con la webcam para guardarlos como imágenes, que
> después se usan en las actividades siguientes del curso (entrenar Eigenfaces con carpetas de
> caras por persona).

## Archivos

| Archivo | Qué hace |
|---|---|
| [`capturar_rostros.py`](capturar_rostros.py) | Detecta rostros (cámara, video o carpeta de fotos) y guarda los recortes en `dataset/<persona>/`. |
| [`prueba_captura.py`](prueba_captura.py) | Pruebas sin cámara: carga del XML, recorte, numeración y ciclo de guardado. |
| `haarcascade_frontalface_alt.xml` | Modelo del detector (ver Referencias). |
| `.gitignore` | Evita subir `dataset/` (son fotos de caras de personas reales). |

## Cómo se usa

```
pip install opencv-python numpy
python capturar_rostros.py --persona diego          # cámara 0, guarda 100 rostros
python capturar_rostros.py --persona diego --max 200 --tam 100
python capturar_rostros.py --persona otra --fuente fotos/    # desde una carpeta de imágenes
python prueba_captura.py                             # pruebas sin cámara
```

Con la cámara se abre una ventana: rectángulo **verde** = rostro que se está guardando,
**rojo** = otros rostros detectados que se ignoran. **ESC** termina antes de llegar a `--max`.
Si se vuelve a ejecutar con la misma persona, la numeración continúa (no se pisan las fotos).

## Cómo funciona

1. **Gris + ecualización.** El detector de Haar compara sumas de intensidad entre regiones
   claras y oscuras (ojos más oscuros que las mejillas, puente de la nariz más claro, etc.),
   así que no necesita color. `equalizeHist` reparte mejor el contraste cuando hay poca luz.
2. **Cascada de clasificadores.** El XML contiene muchas etapas; cada ventana de la imagen
   tiene que pasar todas para considerarse cara. Las primeras etapas son muy simples y
   descartan rápido casi todo el fondo, por eso funciona en tiempo real.
3. **`detectMultiScale`.** Recorre la imagen en varias escalas (`scaleFactor=1.2`) y solo acepta
   una detección si se repite en varias ventanas vecinas (`minNeighbors=5`); `minSize` descarta
   detecciones demasiado pequeñas, que suelen ser falsos positivos.
4. **Un solo rostro por cuadro.** Se guarda el rectángulo de mayor área, para que una persona
   que pase por detrás no se cuele en la carpeta de otra.
5. **Recorte normalizado.** Se agrega un margen del 10 % y se redimensiona a `tam x tam` en
   gris. Eigenfaces y Fisherfaces exigen que todas las imágenes de entrenamiento tengan el
   mismo tamaño; si se guardan recortes de tamaño variable hay que redimensionarlos después.
6. **Una de cada N.** Con cámara se guarda un rostro cada 3 cuadros (`--cada`), para que las
   fotos no salgan casi idénticas y dé tiempo a mover un poco la cabeza o cambiar de gesto.

### Decisiones de diseño

- Cargar el XML solo desde `cv.data.haarcascades` no funciona con la versión de OpenCV
  instalada en esta computadora (5.0.0): esa carpeta ya no trae los XML, así que el clasificador
  queda vacío. Aquí se usa primero el XML de esta carpeta y solo como respaldo el de OpenCV.
- En lugar de guardar todas las caras de cada cuadro, aquí se guarda la más grande, en gris, de tamaño fijo y con un tope (`--max`).
- Se agregaron la entrada desde video o carpeta de imágenes y el modo `--sin-ventana`, que
  permiten probar el programa sin cámara.

## Pruebas

`python prueba_captura.py` (OpenCV 5.0.0, Python 3.12) imprime `[OK]` en: carga del
clasificador, 0 falsos positivos en una imagen lisa y en una de ruido, recorte siempre de
`tam x tam` (también en una esquina), elección del rostro más grande, numeración que continúa
y ciclo de guardado (con el detector simulado: 3 archivos de 100x100 en gris, respetando
`--max`). La prueba con la **cámara real** no se pudo hacer de forma automática.

## Referencias

- `haarcascade_frontalface_alt.xml`: copiado sin cambios; es el
  detector de OpenCV creado por **Rainer Lienhart**, distribuido bajo la *Intel License
  Agreement For Open Source Computer Vision Library* (Copyright (C) 2000, Intel Corporation),
  cuyo texto se conserva completo dentro del propio archivo.
- Documentación de OpenCV: *Cascade Classifier* (`cv::CascadeClassifier::detectMultiScale`).
- P. Viola y M. Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, 2001.
