# Análisis de un dataset desconocido

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Se entregó un archivo [`dataset.csv`](dataset.csv) con **150 filas y 4 columnas numéricas**, sin encabezados ni etiquetas.
Hay que encontrar relaciones entre los datos y deducir qué dataset es.
En clase se trabajó en Excel. Aquí el análisis está hecho con [`analisis.py`](analisis.py), que además genera
[`resumen_segmentos.csv`](resumen_segmentos.csv) para abrirlo en Excel.

```
python analisis.py
```

## 1. Primera observación: hay 3 bloques de 50 filas

Al ordenar o graficar las columnas, se nota un cambio claro en las filas 51 y 101. Los promedios por bloque:

| Segmento | A | B | C | D |
|---|---|---|---|---|
| 1 (filas 1–50) | 5.01 | **3.43** | **1.46** | **0.25** |
| 2 (filas 51–100) | 5.94 | 2.77 | 4.26 | 1.33 |
| 3 (filas 101–150) | 6.59 | 2.97 | 5.55 | 2.03 |

- **C y D separan perfectamente el segmento 1:** sus valores son mucho más chicos y casi no varían
  (desviación de 0.17 y 0.10).
- **B es la excepción:** el segmento 1 tiene la B más **grande**, no la más chica.
  O sea, no todas las columnas crecen juntas.
- Los segmentos 2 y 3 se parecen más entre sí y se traslapan un poco.

## 2. Relaciones entre columnas (correlación de Pearson)

|   | A | B | C | D |
|---|---|---|---|---|
| A | 1.00 | −0.12 | 0.87 | 0.82 |
| B | −0.12 | 1.00 | −0.43 | −0.37 |
| C | 0.87 | −0.43 | 1.00 | **0.96** |
| D | 0.82 | −0.37 | **0.96** | 1.00 |

- **C y D casi siempre suben juntas (0.96).** Parecen medir dos dimensiones del mismo objeto.
- **A también se relaciona fuerte con C y D.**
- **B va casi por su lado**, y hasta tiene una relación ligeramente negativa con las demás.

## 3. ¿Se pueden encontrar los grupos sin saber dónde empiezan?

Para probarlo se programó un **k-means** con k = 3, que agrupa los puntos sin usar el número de fila:

| Segmento | Resultado de k-means |
|---|---|
| 1 | Los 50 en el mismo grupo |
| 2 | 48 en un grupo, 2 en otro |
| 3 | 36 en un grupo, 14 en otro |

Coincide con los segmentos en **134 de 150 filas (89.3 %)**.
Esto confirma que el segmento 1 está muy separado y que los segmentos 2 y 3 se traslapan.

También funciona una **regla simple con solo C y D**:

```
si C < 2.5              -> segmento 1
si no, si D < 1.75      -> segmento 2
si no                   -> segmento 3
```

Con esa regla se aciertan **144 de 150 filas (96 %)**. Es básicamente un árbol de decisión de dos niveles hecho a mano.

## 4. Conclusión: es el dataset Iris

Todas las pistas apuntan al **dataset Iris** (Fisher, 1936):

- 150 flores, 50 de cada especie: *Iris setosa*, *Iris versicolor* e *Iris virginica*.
- **Columnas:**
  - A = largo del sépalo
  - B = ancho del sépalo
  - C = largo del pétalo
  - D = ancho del pétalo

  Todas están en cm.
- Los promedios del segmento 1 (5.006, 3.428, 1.462, 0.246) son exactamente los publicados para *setosa*.
- Es conocido justo por lo que encontramos: *setosa* se separa fácil y *versicolor* y *virginica* se traslapan.

## Cómo hacerlo en Excel

1. Abre `dataset.csv` y agrega una fila de encabezados: A, B, C y D.
2. **Promedio por segmento:** `=PROMEDIO(A2:A51)`, `=PROMEDIO(A52:A101)` y `=PROMEDIO(A102:A151)`.
3. **Desviación:** `=DESVEST.M(A2:A51)`.
4. **Correlación:** `=COEF.DE.CORREL(C2:C151; D2:D151)`.
5. Inserta un **gráfico de dispersión** de C contra D. Se ven claramente las tres nubes de puntos.

## Referencias

- El dataset y el enunciado vienen de la práctica de la materia. El análisis y el código de esta carpeta son propios.
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics, 7(2).
- UCI Machine Learning Repository: Iris dataset.
