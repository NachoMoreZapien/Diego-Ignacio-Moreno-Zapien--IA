# CNN: clasificación de imágenes deportivas

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Entrenar una **red neuronal convolucional (CNN)** que reconozca a qué deporte pertenece una foto,
con un dataset de 10 deportes (americano, basket, beisball, boxeo, ciclismo, f1, futbol, golf,
natacion, tenis). Las imágenes son pequeñas: **21 × 28 píxeles a color**.

## Archivos

| Archivo | Qué hace |
|---|---|
| [`cnn.py`](cnn.py) | Programa completo: lee el dataset, entrena, evalúa, guarda el modelo y clasifica fotos nuevas. Incluye una prueba con imágenes sintéticas. |
| [`CNN.ipynb`](CNN.ipynb) | El mismo proceso en un cuaderno, paso por paso, con gráficas y ejemplos de errores. |

El dataset y el modelo entrenado (`sport_model.keras`, `clases.json`, `graficas.png`) no se suben (`.gitignore`): pesan mucho y se generan al entrenar.

## Cómo se usa

```
pip install tensorflow numpy pillow matplotlib
python cnn.py revisar  --datos sportimages/      # cuenta las imágenes (no necesita TensorFlow)
python cnn.py entrenar --datos sportimages/ --epocas 20                  # división al azar (como en clase)
python cnn.py entrenar --datos sportimages/ --epocas 20 --division video # prueba con el final de cada video
python cnn.py predecir foto1.jpg foto2.jpg
python cnn.py prueba                             # comprueba el flujo con imágenes inventadas
```

`sportimages/` debe tener una subcarpeta por deporte; el nombre de la subcarpeta es la etiqueta.

## Cómo funciona

1. **Carga.** Cada imagen se abre con Pillow, se pasa a RGB (si viene en gris o con transparencia) y se
   deja de 28 × 21. La etiqueta sale del nombre de la carpeta, así no depende del orden en que el sistema
   operativo entregue los archivos. Los archivos dañados se saltan y se avisa cuántos fueron.
2. **División.** 80 % entrenamiento y 20 % prueba; del 80 % se aparta otro 20 % para validación.
   La división es **estratificada**: cada parte tiene la misma proporción de cada deporte.
   Con `--division video` la prueba es el **último 20 % de cada deporte** sin revolver (ver Resultados).
3. **Preprocesamiento.** Los píxeles pasan de 0–255 a 0–1 y las etiquetas a *one-hot*
   (`3 → [0,0,0,1,0,0,0,0,0,0]`), que es lo que compara la salida softmax.
4. **Red.**

   | Capa | Salida | Para qué |
   |---|---|---|
   | Conv2D 32 filtros 3×3 + LeakyReLU | 21×28×32 | Detecta bordes, colores y texturas simples |
   | MaxPooling 2×2 + Dropout 0.25 | 11×14×32 | Reduce tamaño y evita memorizar |
   | Conv2D 64 filtros 3×3 + LeakyReLU | 11×14×64 | Combina lo anterior en formas más grandes (pasto, cancha, casco) |
   | MaxPooling 2×2 + Dropout 0.25 | 6×7×64 | |
   | Flatten → Dense 64 + LeakyReLU + Dropout 0.5 | 64 | Decide con todo lo que encontraron los filtros |
   | Dense 10 softmax | 10 | Probabilidad de cada deporte |

   **LeakyReLU** deja pasar un poco de los valores negativos (pendiente 0.1) para que ninguna neurona
   se quede "muerta". El optimizador es **Adam** (tasa 0.001) con pérdida *categorical crossentropy*.
5. **Evaluación.** Exactitud en el set de prueba, precisión/recall/F1 por deporte y matriz de confusión
   (calculadas con NumPy, sin scikit-learn). El cuaderno además muestra las imágenes mal clasificadas
   para ver con qué deportes se confunde la red.
6. **Predicción.** Una foto de cualquier tamaño se reduce a 28 × 21 y se le pide a la red el deporte
   más probable.

## Decisiones de esta versión

- **Dos bloques convolucionales en lugar de uno.** Con imágenes tan chicas, un solo bloque solo ve
  manchas de color; el segundo combina esas manchas en formas, sin agregar muchos parámetros
  (6×7×64 = 2 688 entradas a la capa densa en lugar de 11×14×32 = 4 928).
- **Adam en lugar de SGD con *decay*.** El argumento `decay` de los optimizadores ya no existe en Keras 3;
  Adam ajusta la tasa por sí solo y converge en menos épocas.
- **Formato `.keras`** para guardar el modelo (el `.h5` es el formato antiguo).
- **Lectura y métricas sin TensorFlow.** Así `revisar` y `prueba` funcionan aunque TensorFlow no esté instalado.

## Resultados

Dataset *sportimages*: 77 128 imágenes de 21 × 28 a color, 10 deportes. Entrenamiento de 20 épocas en CPU
(Python 3.12, TensorFlow 2.21, Keras 3.15), unos 8 segundos por época.

### Primero: división al azar (como en clase)

| | Exactitud |
|---|---|
| Entrenamiento (época 20) | 99.3 % |
| Validación (época 20) | 99.7 % |
| **Prueba** (15 426 imágenes) | **99.7 %** |

Demasiado bueno. Revisando el dataset, las imágenes son **cuadros seguidos de videos**
(`basket_0001.jpg`, `basket_0002.jpg`, ...) y hay **1 982 archivos repetidos exactos**. Al revolver,
casi cada imagen de prueba tiene un cuadro vecino casi idéntico en el entrenamiento, así que la red
no está generalizando: está reconociendo cuadros que prácticamente ya vio.

### Después: división por video (`--division video`)

La prueba es el último 20 % de cada deporte, que la red nunca vio:

| | Exactitud |
|---|---|
| Entrenamiento (época 20) | 99.4 % |
| Validación (época 20) | 99.8 % |
| **Prueba** (15 426 imágenes) | **91.4 %** |

La validación sigue alta porque sale al azar de la parte de entrenamiento (tiene el mismo problema);
la cifra que vale es la de prueba.

| Deporte | Precisión | Recall | F1 |
|---|---|---|---|
| americano | 0.75 | 0.99 | 0.85 |
| basket | 0.81 | 0.85 | 0.83 |
| beisball | 0.99 | 0.85 | 0.92 |
| boxeo | 0.98 | 0.72 | 0.83 |
| ciclismo | 1.00 | 1.00 | 1.00 |
| f1 | 0.80 | 0.97 | 0.88 |
| futbol | 0.98 | 1.00 | 0.99 |
| golf | 0.98 | 0.93 | 0.96 |
| natacion | 1.00 | 0.74 | 0.85 |
| tenis | 1.00 | 1.00 | 1.00 |

Errores más comunes (de la matriz de confusión):
- **boxeo → americano** (248) y **beisball → americano** (125): muchas cosas terminan en "americano",
  que tiene la mayor cantidad de imágenes y escenas con mucha gente.
- **natacion → basket** (199): las albercas y las canchas techadas comparten colores y luz artificial.
- **basket → f1** (140).
- **ciclismo** y **tenis** salen perfectos: tienen colores de fondo muy característicos (carretera, cancha).

El resultado cambia un poco en cada entrenamiento porque la red empieza con pesos al azar: al ejecutar
el cuaderno [`CNN.ipynb`](CNN.ipynb) con la misma división dio **90.7 %** (1 434 errores de 15 426),
y en su última celda clasifica bien golf y f1 pero confunde natación con basket, justo el error más común.

**Conclusión:** la red sí aprende a distinguir deportes (91.4 % con imágenes nunca vistas), pero la
exactitud cercana al 100 % que da la división al azar es engañosa por los cuadros repetidos.

### Pruebas del programa

`python cnn.py prueba`: 13 comprobaciones `OK` (carga, división estratificada y por video, normalización,
one-hot, matriz de confusión, entrenamiento con imágenes sintéticas con exactitud 1.00 y softmax).

## Créditos

- Dataset de imágenes deportivas (21 × 28, 10 clases): material de clase, basado en el publicado con el
  tutorial *Clasificación de imágenes en Python* de Aprende Machine Learning (Juan Ignacio Bagnato).
  Se descarga como `sportimages.zip` (63 MB) del repositorio
  [jbagnato/machine-learning](https://github.com/jbagnato/machine-learning) y se descomprime junto a `cnn.py`.
