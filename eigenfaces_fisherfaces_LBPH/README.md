# Comparación de Eigenfaces, Fisherfaces y LBPH

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Entrenar los tres reconocedores faciales de OpenCV (Eigenfaces, Fisherfaces y LBPH) con un dataset
de rostros organizado en una carpeta por persona, y compararlos.

En lugar de tres scripts que solo entrenan y guardan el modelo, aquí hay **un solo programa** que
entrena los tres con **las mismas imágenes** y los evalúa con imágenes que no vieron al entrenar.
Así la comparación da números: acierto, tiempo de entrenamiento, tiempo por predicción y tamaño del modelo.

## Archivos

| Archivo | Qué hace |
|---|---|
| [`comparar.py`](comparar.py) | Carga el dataset, separa entrenamiento/prueba, entrena y evalúa los tres métodos. |
| [`prueba_comparar.py`](prueba_comparar.py) | Pruebas con un dataset sintético, sin necesitar fotos reales. |

## Cómo se usa

```
pip install opencv-contrib-python numpy
python comparar.py                              # usa ../haarcascade_dataset/dataset
python comparar.py --dataset otra/carpeta --prueba 0.25
python comparar.py --guardar                    # además guarda los tres modelos XML
python prueba_comparar.py
```

El dataset se arma con [`haarcascade_dataset`](../haarcascade_dataset/README.md), una subcarpeta por persona:

```
python ../haarcascade_dataset/capturar_rostros.py --persona persona1
python ../haarcascade_dataset/capturar_rostros.py --persona persona2
```

> Se necesitan **al menos 2 personas**. Fisherfaces busca lo que separa a una clase de otra;
> con una sola no hay nada que separar, y tampoco tiene sentido medir el acierto.

## Cómo funciona cada método

| | Eigenfaces | Fisherfaces | LBPH |
|---|---|---|---|
| **Idea** | PCA: busca las direcciones en las que más varían **todas** las caras | LDA: busca las direcciones que mejor **separan a las personas** | Describe la **textura local** de cada zona de la cara |
| **Cómo compara** | Proyecta la cara en esas "eigenfaces" y busca la más cercana | Igual, pero en el espacio de Fisher (máx. personas − 1 dimensiones) | Compara histogramas de patrones binarios por región |
| **Luz** | Muy sensible: la iluminación es justo lo que más varía | Menos sensible: aprende a ignorar lo que no distingue personas | Bastante robusto: cada patrón solo compara un píxel con sus vecinos |
| **Tamaño fijo** | Sí, todas las imágenes iguales | Sí | No es necesario |
| **Agregar personas** | Hay que reentrenar todo | Hay que reentrenar todo | Se puede usar `update()` sin reentrenar |

**LBP (Local Binary Pattern):** para cada píxel se mira a sus 8 vecinos. Cada vecino más claro que el centro
da un 1, y cada uno más oscuro da un 0. Así se forma un número de 8 bits que describe la textura de ese punto.
La imagen se divide en una cuadrícula de 8×8 y se saca el histograma de esos números en cada celda.

**Distancia:** `predict()` regresa la etiqueta y una distancia. Mientras más baja, más se parece.
Cada método la mide en su propia escala, así que no se puede comparar la distancia de Eigenfaces con la de LBPH.
Sí sirve para poner un umbral de "desconocido" dentro de un mismo método.

## Cómo se hace la comparación

1. Todas las imágenes se leen en gris y se llevan a 100×100.
2. Se toma el 20 % de las imágenes de **cada persona** para prueba (con semilla fija, así siempre se separan igual).
3. Los tres métodos se entrenan con el mismo 80 % y predicen el mismo 20 %.

## Resultados

**Con el dataset real:** por ahora solo hay una persona capturada (20 imágenes),
así que el programa avisa que necesita al menos dos. Falta capturar otra persona para correr la comparación real.

**Con el dataset sintético de la prueba** (3 "personas" hechas de figuras con ruido, 46 imágenes, 9 de prueba),
salida real de `prueba_comparar.py`:

| Método | Acierto | Entrenar | Predecir | Modelo |
|---|---|---|---|---|
| Eigenfaces | 9/9 (100 %) | 48.6 ms | 0.61 ms | 9271 KB |
| Fisherfaces | 9/9 (100 %) | 27.8 ms | 0.05 ms | 706 KB |
| LBPH | 7/9 (77.8 %) | 57.5 ms | 2.12 ms | 3740 KB |

Los tiempos cambian un poco en cada ejecución. Estos datos no son caras, así que solo sirven para ver el comportamiento:

- **LBPH falló 2.** Las imágenes sintéticas son zonas planas con ruido, y LBPH lee ese ruido como textura.
  En caras reales, que sí tienen textura, normalmente es el más robusto a cambios de luz.
- **Eigenfaces guarda el modelo más grande.** Guarda una eigenface de 100×100 por cada componente.
- **Fisherfaces es el más chico y rápido.** Con 3 personas solo usa 2 dimensiones (personas − 1).

## Referencias

- Turk, M. y Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience.
- Belhumeur, P., Hespanha, J. y Kriegman, D. (1997). *Eigenfaces vs. Fisherfaces*. IEEE TPAMI.
- Ahonen, T., Hadid, A. y Pietikäinen, M. (2006). *Face Description with Local Binary Patterns*. IEEE TPAMI.
- Documentación de OpenCV, módulo `face` (opencv-contrib).
