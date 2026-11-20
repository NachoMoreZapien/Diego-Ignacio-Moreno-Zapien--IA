# Dataset: pelotas rebotantes

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Diseñar el dataset para entrenar un modelo que controle a un muñeco dentro de un escenario
cerrado, donde debe esquivar pelotas que se mueven y rebotan en las paredes. No se pide programar
el juego, sino decidir **qué datos se guardan, cómo se etiquetan, cuántos hacen falta y qué pasa
si el dataset está mal hecho**.

---

## 1. Planteamiento del problema

Un muñeco aparece en un escenario rectangular junto con varias pelotas en movimiento. En cada
instante el muñeco puede moverse o quedarse quieto. Queremos un dataset que le permita a un
modelo (árbol de decisión o red multicapa) aprender, a partir del estado del juego, **si el muñeco
está a punto de ser golpeado** y, con eso, qué movimiento conviene hacer.

## 2. Suposiciones del escenario

Para que el problema sea manejable se fijan estas reglas:

| Elemento | Suposición |
|---|---|
| Escenario | Rectángulo de tamaño fijo (por ejemplo 800 × 600 px) con cuatro paredes. |
| Muñeco | Empieza en el centro. Puede moverse en 8 direcciones (↑ ↓ ← → y las 4 diagonales) o quedarse quieto, siempre a la misma velocidad. |
| Pelotas | Entre 1 y 3; la cantidad se elige al iniciar la partida y no cambia. |
| Inicio de las pelotas | Posición y dirección iniciales aleatorias (nunca encima del muñeco). |
| Rebotes | Al chocar con una pared, la pelota rebota y su velocidad cambia (cada pelota por separado). |
| Información disponible | En todo momento se conocen la posición, dirección y velocidad de cada pelota y cuántas hay. |

## 3. Variable objetivo

**`golpeado`** (booleana): `1` si el muñeco es tocado por alguna pelota, `0` si no.

Etiquetar solo el frame exacto del choque daría muy pocos `1`. Por eso se propone etiquetar el
frame como `1` si el muñeco es golpeado **dentro de los siguientes N frames** (por ejemplo,
N = 15 ≈ 0.25 s). Así el modelo aprende a reconocer el peligro *antes* de que ocurra.

Opcionalmente se guarda también la columna **`accion`** (el movimiento que hizo el jugador en ese
frame), útil si después se quiere entrenar al modelo para decidir el movimiento directamente.

## 4. Variables de entrada (una fila por frame)

| Columna | Tipo | Descripción |
|---|---|---|
| `frame` | entero | Número de frame (1 frame cada 16 ms ≈ 60 FPS). |
| `muneco_x`, `muneco_y` | entero (px) | Posición del muñeco. |
| `num_pelotas` | entero (1–3) | Pelotas en la partida. |
| `pelota{i}_x`, `pelota{i}_y` | entero (px) | Posición de la pelota *i* (i = 1, 2, 3). |
| `pelota{i}_vx`, `pelota{i}_vy` | real (px/frame) | Velocidad con dirección de la pelota *i*. Con estos dos valores ya quedan la dirección y la rapidez. |
| `pelota{i}_dist` | real (px) | Distancia de la pelota *i* al muñeco: √((x_p − x_m)² + (y_p − y_m)²). |
| `muneco_dist_pared` | real (px) | Distancia del muñeco a la pared más cercana (ayuda a no quedar acorralado). |
| `accion` | categoría | Movimiento elegido (opcional, ver arriba). |
| `golpeado` | 0 / 1 | Variable objetivo. |

Si en una partida hay menos de 3 pelotas, las columnas de las pelotas que no existen se llenan
con un valor fijo (por ejemplo, distancia muy grande y velocidad 0) para que todas las filas
tengan el mismo número de columnas.

Ejemplo de dos filas (valores ilustrativos):

| frame | muneco_x | muneco_y | num_pelotas | pelota1_x | pelota1_y | pelota1_vx | pelota1_vy | pelota1_dist | … | golpeado |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | 400 | 300 | 2 | 520 | 260 | −4.0 | 1.5 | 126.5 | … | 0 |
| 121 | 405 | 300 | 2 | 516 | 261.5 | −4.0 | 1.5 | 117.5 | … | 1 |

## 5. Tamaño mínimo razonable

Para cubrir al menos una vez cada combinación básica se cuentan:

- cantidad de pelotas: **3** casos (1, 2 o 3);
- acciones del muñeco: **9** (8 direcciones + quedarse quieto);
- dirección de llegada de la pelota respecto al muñeco: **8** (dividida en 8 sectores).

**3 × 9 × 8 = 216 escenarios distintos.** Ese número es solo el piso para que ninguna combinación
falte; en la práctica cada escenario debe repetirse varias veces con posiciones y velocidades
distintas (por ejemplo, unas 20 repeticiones → ~4 300 situaciones). Como cada partida produce
60 filas por segundo, unos pocos minutos de juego bastan para juntar decenas de miles de filas.

También hay que vigilar el **balance de clases**: los golpes son raros, así que conviene revisar
que haya suficientes filas con `golpeado = 1` (o darles más peso al entrenar).

## 6. Riesgos si el dataset está mal hecho

- **Decisiones equivocadas:** si casi no hay ejemplos de golpes, el modelo aprende a decir
  siempre "no me van a golpear" y nunca esquiva.
- **Moverse hacia la pelota:** si faltan casos de ciertas direcciones de llegada, el modelo no
  sabe reaccionar a ellas y puede caminar justo hacia la pelota.
- **Quedar atrapado en una esquina:** si no se registra la distancia a las paredes, el modelo
  huye de las pelotas sin notar que se está acorralando.
- **Etiquetas mal sincronizadas:** si `golpeado` se marca en un frame distinto al real, el modelo
  aprende relaciones falsas.
- **Datos de un solo jugador o un solo estilo:** el modelo copia también sus malos hábitos.

## 7. Granularidad

- **Tiempo:** una fila por frame, cada 16 ms (≈ 60 FPS). Es suficiente porque una pelota rápida
  avanza solo unos pocos píxeles por frame.
- **Espacio:** posiciones y distancias en píxeles; velocidades en píxeles por frame.
- Si el dataset crece demasiado, se puede guardar uno de cada 2 o 3 frames sin perder mucha
  información.

---

## Referencias

- Este documento es un diseño propio: cambia el conteo de acciones del muñeco a 9
  (8 direcciones + quieto), por lo que el tamaño mínimo queda en 216 en lugar de 168, y agrega la
  etiqueta con ventana de N frames, la distancia a la pared y el manejo del balance de clases.
- Apuntes de clase de Inteligencia Artificial (diseño de datasets, 05/oct/2026).
