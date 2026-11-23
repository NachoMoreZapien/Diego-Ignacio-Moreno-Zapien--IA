# Entrenamiento de Eigenfaces

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Entrenar un reconocedor facial **Eigenfaces** con el dataset de rostros (una carpeta por persona),
guardar el modelo y usarlo para reconocer caras con la cámara.

## Archivos

| Archivo | Qué hace |
|---|---|
| [`entrenar.py`](entrenar.py) | Lee el dataset, entrena Eigenfaces y guarda `modelo_eigenfaces.xml` con los nombres de las personas dentro. |
| [`reconocer.py`](reconocer.py) | Abre la cámara, detecta la cara con Haar Cascade y dice quién es (o "desconocido"). |
| [`prueba_eigenfaces.py`](prueba_eigenfaces.py) | Pruebas sin cámara con un dataset sintético. |

El modelo y `eigenfaces.png` no se suben (`.gitignore`), porque se generan a partir de fotos de caras reales.

## Cómo se usa

```
pip install opencv-contrib-python numpy
python ../haarcascade_dataset/capturar_rostros.py --persona persona1 # 1. armar el dataset
python entrenar.py --ver                                             # 2. entrenar
python reconocer.py                                                  # 3. reconocer (ESC sale)
python reconocer.py --fuente carpeta_con_caras/                      #    o sin cámara
python prueba_eigenfaces.py
```

## Cómo funciona Eigenfaces

1. **Cada cara es un vector.** Una imagen de 100×100 en gris se "aplana" en 10 000 números.
   Por eso todas las imágenes tienen que medir lo mismo.
2. **Cara promedio.** Se promedian todas las caras del dataset y se le resta ese promedio a cada una.
3. **PCA.** Se buscan las direcciones en las que más varían las caras. Cada dirección es otro vector
   de 10 000 números que, visto como imagen, parece un "fantasma" de cara: eso es una **eigenface**.
   Las primeras explican la mayor parte de la variación (sobre todo luz y contorno).
4. **Proyección.** Cada cara de entrenamiento se resume con unos cuantos números: cuánto tiene de cada
   eigenface. Pasa de 10 000 valores a 20 o menos.
5. **Reconocer.** La cara nueva se proyecta igual y se busca la cara de entrenamiento más cercana.
   `predict()` regresa su etiqueta y la **distancia**. Si la distancia pasa del umbral, se responde "desconocido".

Con `--ver` se guarda `eigenfaces.png`: primero la cara promedio y después las primeras 8 eigenfaces.

## Detalles de esta versión

- **El nombre de cada persona va dentro del XML** (`setLabelInfo`). Así `reconocer.py` muestra el nombre de la carpeta
  y no "0", sin depender de un archivo aparte ni del orden de las carpetas.
- **Todas las imágenes se redimensionan a 100×100** al cargarlas, aunque alguna venga de otro tamaño.
- **El recorte de la cámara es igual al de la captura** (gris, 10 % de margen, 100×100). Si el recorte
  cambia, cambian las distancias y el modelo falla.
- **`--componentes`** permite quedarse con menos eigenfaces (modelo más chico).

## Resultados con el dataset real

Dataset: una persona, 20 imágenes de 100×100.

- `python entrenar.py --ver` entrena con 20 imágenes y genera 20 eigenfaces.
- **Calibración del umbral:** entrené con 15 fotos y medí la distancia de las otras 5, que el modelo no había visto.

| Imagen | Distancia |
|---|---|
| Fotos de la persona no vistas al entrenar | 458 – 1343 |
| Las mismas fotos, al 50 % de brillo | ~1700 |
| Ruido aleatorio | ~2100 |
| Imagen gris lisa | 2112 |

Por eso el umbral por defecto es **1600**. Se puede cambiar con `--umbral`.

**Limitaciones que se ven en los números:**
- **La luz importa mucho.** La misma cara con menos luz ya se acerca al umbral. Conviene capturar el dataset
  con la misma luz con la que se va a reconocer, o con varias luces distintas.
- **Con una sola persona**, el modelo siempre responde esa persona y solo el umbral separa a los demás.
  Para probar de verdad con desconocidos hay que capturar al menos otra persona y volver a entrenar.
- La prueba con la cámara en vivo no se pudo automatizar. Hay que hacerla con `python reconocer.py`.

## Pruebas

`python prueba_eigenfaces.py` (OpenCV 5.0.0, Python 3.12) da `[OK]` en todo:

- Carga de 2 personas.
- Redimensionado a 100×100.
- 10 componentes.
- Nombres guardados en el XML.
- 10/10 imágenes nuevas reconocidas.
- Umbral de desconocido.
- Recorte en la esquina.
- `eigenfaces.png`.
- Aviso si falta el modelo.

Los datos de la prueba son sintéticos: figuras con ruido, no caras.

## Referencias

- Turk, M. y Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience.
- Documentación de OpenCV, módulo `face` (opencv-contrib): `EigenFaceRecognizer`.
- El detector `haarcascade_frontalface_alt.xml` se usa desde [`haarcascade_dataset`](../haarcascade_dataset/README.md) (OpenCV, licencia de Intel).
