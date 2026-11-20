# Juego modificado con árbol de decisión

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Tomar el juego de clase (un muñeco que salta para esquivar una bala; en el modo automático lo
controlaba una red neuronal multicapa, MLP) y modificarlo para que el modo automático lo controle
un **árbol de decisión**.

> Nota: el juego base de clase usaba una red `MLPClassifier`; esta actividad la cambia por un
> árbol de decisión.

## Archivos

| Archivo | Qué hace |
|---|---|
| [`juego_arbol.py`](juego_arbol.py) | El juego con ventana (pygame): menú, modo manual, entrenamiento y modo automático. |
| [`mundo.py`](mundo.py) | La física del juego sin gráficos (salto, bala, choques). La usan el juego y la prueba. |
| [`arbol_decision.py`](arbol_decision.py) | Árbol de decisión (CART con Gini) programado desde cero con NumPy. |
| [`prueba_juego.py`](prueba_juego.py) | Prueba automática sin ventana: genera datos, entrena el árbol y lo deja jugar solo. |
| `assets/` | Imágenes del juego (ver Referencias). |
| `reglas_arbol.txt` | Ejemplo de reglas que se guardan al pulsar **R** (salió de una prueba con datos simulados). |

## Cómo se usa

```
pip install pygame numpy
python juego_arbol.py      # el juego
python prueba_juego.py     # la prueba sin ventana (solo numpy)
```

En el menú: **M** jugar en manual (ESPACIO = saltar), **T** entrenar, **A** automático,
**R** ver las reglas del árbol, **C** / **L** guardar / cargar `datos_arbol.csv`, **Q** salir.
Durante la partida, **ESC** regresa al menú.

## Cómo funciona

1. **Datos.** En modo manual, en cada frame en que el muñeco está en el suelo (es decir, cuando
   *puede* decidir) se guarda una fila `(rapidez_bala, distancia, salto)`, donde `salto = 1`
   solo en el frame en que se pulsó ESPACIO.
2. **Entrenamiento.** Con 80 % de las filas se construye el árbol y con el otro 20 % se mide la
   exactitud. El árbol elige, en cada nodo, la pregunta `característica <= umbral` que más reduce
   la impureza de Gini (`G = 1 − Σ p²`), con profundidad máxima 5 y al menos 5 datos por hoja.
3. **Modo automático.** En cada frame se recorre el árbol de la raíz a una hoja; si la hoja dice
   que la probabilidad de saltar es ≥ 0.5, el muñeco salta.

### Decisiones propias (diferencias con el juego de clase)

- **El árbol está programado a mano** (`arbol_decision.py`) en lugar de usar scikit-learn, para
  ver cómo se elige cada corte. Se puede imprimir como reglas legibles (tecla **R**), que es la
  gran ventaja de un árbol frente a una MLP.
- **Distancia con signo** (`bala.x − jugador.x`). Con valor absoluto, una bala que ya pasó y una
  que viene a la misma distancia se ven iguales para el modelo.
- **Etiqueta = el frame del salto** y solo se registran frames en el suelo, que es exactamente
  cuando el modelo decide en modo automático. Así lo que se aprende coincide con lo que se usa.
- **Clases balanceadas.** Hay un salto por cada ~90 frames de "no saltar"; sin balancear, el
  árbol aprendería a no saltar nunca. Cada clase recibe el mismo peso total.
- No se necesita escalar los datos (un árbol solo compara con umbrales), a diferencia de la MLP.

## Resultados de `prueba_juego.py` (salida real)

- Datos de un jugador simulado (80 balas): 7422 filas, 80 saltos.
- Árbol: 10 hojas, profundidad 5; exactitud 0.977 en entrenamiento y 0.977 en prueba.
- **Modo automático, 200 balas nuevas: 200 esquivadas, 0 golpes** (sin saltar: 0 y 200).
- Caso XOR: el árbol voraz se queda en una hoja (exactitud 0.5), porque ninguna primera
  pregunta reduce el Gini. Ejemplo de la desventaja "optimización voraz".

Ojo: la exactitud (~0.98) dice poco con clases tan desbalanceadas; lo que importa es cuántas
balas esquiva el árbol jugando.

## Referencias

- El código de esta carpeta es propio.
- Las imágenes de `assets/` (`mono_frame_1..4.png`, `purple_ball.png`, `ufo.png`, `fondo.jpg`) no
  son de mi autoría: son las del juego base usado en clase, sin cambios.
