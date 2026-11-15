# Ejercicios de lógica — Introducción a la IA

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

Son cinco problemas clásicos que se resuelven como **problemas de búsqueda**. Todo está en
[`ejercicios.py`](ejercicios.py) y solo usa la biblioteca estándar de Python.

```
python ejercicios.py
```

## 1. N reinas: backtracking

Se coloca una reina por fila. Al probar una columna, se descarta si ya hay otra reina en la misma
columna o en la misma diagonal. Para las diagonales se usan dos conjuntos: en una diagonal `\`
el valor `fila − col` es constante, y en una diagonal `/` lo constante es `fila + col`.
Si ninguna columna sirve, se regresa a la fila anterior.

**Resultado:** con 4 reinas hay **2 soluciones**, que son reflejo una de la otra:

```
. R . .      . . R .
. . . R      R . . .
R . . .      . . . R
. . R .      . R . .
```

El script también cuenta las soluciones de n = 1 a 8: `[1, 0, 0, 2, 10, 4, 40, 92]`.

## 2. Josephus: posición segura con k = 2

Quien tiene el cuchillo elimina a la siguiente persona viva y pasa el cuchillo.
Se resolvió de dos formas:

- **Simulación** con una cola circular (`deque`).
- **Fórmula cerrada:** si `n = 2^m + L` con `0 ≤ L < 2^m`, sobrevive la posición `2L + 1`,
  contando desde quien empieza con el cuchillo.

Como el inicio es aleatorio, la función recibe la posición inicial y devuelve el lugar seguro absoluto.
El script comprueba que la fórmula y la simulación coinciden para n = 1…199 y cualquier inicio.

**Ejemplos con inicio en 1:**

| n | Sobrevive |
|---|---|
| 5 | 3 |
| 7 | 7 |
| 10 | 5 |
| 41 | 19 |

## 3. Monjes y caníbales: búsqueda en anchura (BFS)

**Estado:** `(monjes en A, caníbales en A, ¿bote en A?)`.

**Reglas del problema:**
- Al bote suben 1 o 2 personas.
- En ninguna orilla, ni en el bote, puede haber más caníbales que monjes cuando hay al menos un monje.

La búsqueda en anchura garantiza la solución con **menos viajes: 11**.

| # | Isla A | Isla B | Sentido |
|---|---|---|---|
| 0 | mmmccc | – | |
| 1 | mmmc | cc | → |
| 2 | mmmcc | c | ← |
| 3 | mmm | ccc | → |
| 4 | mmmc | cc | ← |
| 5 | mc | mmcc | → |
| 6 | mmcc | mc | ← |
| 7 | cc | mmmc | → |
| 8 | ccc | mmm | ← |
| 9 | c | mmmcc | → |
| 10 | cc | mmmc | ← |
| 11 | – | mmmccc | → |

## 4. Laberinto: búsqueda en profundidad (DFS) recursiva

En cada celda libre se marca la celda como visitada y se intenta avanzar a la derecha, abajo, izquierda y arriba.
Si se llega a la salida (`1`), el camino se conserva. Si todas las direcciones fallan, se hace *backtracking*:
la celda se quita del camino.

**Detalle de esta versión:** aquí no se sobreescribe el laberinto con `0`.
Se usa un conjunto de visitados y además se devuelve el **camino** encontrado, no solo `true` o `false`.

```
*#***#..     # = pared, * = camino, S = salida
*#*#***S
***###.#
```

> La búsqueda en profundidad encuentra *un* camino, no necesariamente el más corto.
> Para el más corto habría que usar búsqueda en anchura o A*.

## 5. Conteo de islas: flood fill

Se recorre la matriz fila por fila. Cada vez que aparece un `1` no visitado, se cuenta una isla nueva
y se recorren todas sus celdas conectadas por arista con una **pila explícita**. Con la pila, una isla grande
no puede provocar un desbordamiento de recursión. El mapa original no se modifica.

**Resultado** con el mapa de ejemplo: **5 islas**, de tamaños `[4, 3, 1, 1, 1]`.

**Complejidad:** O(filas × columnas), porque cada celda se visita una sola vez.

## Referencias

- Los enunciados vienen de los ejercicios de introducción de la materia.
- El laberinto y el mapa de islas de ejemplo son los de los ejercicios de clase. El código de esta carpeta es una implementación propia en Python.
- Russell & Norvig, *Inteligencia Artificial: Un enfoque moderno*, capítulos 3 (búsqueda) y 6 (satisfacción de restricciones).
