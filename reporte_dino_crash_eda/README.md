# Dino run: análisis exploratorio de datos (EDA conceptual)

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Antes de recolectar datos del juego del dinosaurio (Dino run), definir **qué dataset haría falta**
para tres preguntas distintas. Para cada una: variable objetivo, variables de entrada,
granularidad (qué representa cada fila), tamaño mínimo razonable y riesgos si el dataset está mal definido.

Además, [`simular_dataset.py`](simular_dataset.py) genera datos **simulados** con esas columnas para
comprobar que las definiciones funcionan (no son datos del juego real).

```
python simular_dataset.py                 # imprime el resumen
python simular_dataset.py --csv salida    # además guarda frames.csv y partidas.csv
```

## Contexto del juego

- El dinosaurio corre solo. **Espacio** = saltar (sube y baja); **flecha abajo** = agacharse (se hace más bajo).
- Obstáculos: cactus chico, cactus grande y pájaros a distintas alturas. Algunos pájaros vuelan tan alto
  que no hace falta hacer nada.
- Los obstáculos no se mueven por sí mismos: se acercan a la velocidad del escenario, que **aumenta**
  conforme avanza la partida.
- El juego corre a ~60 cuadros por segundo: un frame cada ~16 ms.

## Escenario 1: ¿morirá en el siguiente frame?

| Elemento | Definición |
|---|---|
| Objetivo (Y) | `muere_siguiente_frame`: binaria (1 = choca en el frame siguiente) |
| Entradas (X) | `distancia_obstaculo`, `obstaculo_tipo`, `obstaculo_altura`, `velocidad`, `dino_y` (altura del salto), `agachado` |
| Granularidad | **Una fila por frame** (16 ms) |
| Tamaño mínimo | Varias decenas de partidas, no una sola: cada partida aporta **un solo positivo** |
| Riesgos | Desbalance extremo (casi todo es 0); etiquetar mal el frame (marcar el frame del choque en lugar del anterior); no guardar la velocidad, que cambia cuánto se acerca el obstáculo en un frame |

Lo importante es la relación **distancia / velocidad**: cuántos frames faltan para que el obstáculo llegue.

## Escenario 2: ¿cuántos puntos hará la partida al morir?

| Elemento | Definición |
|---|---|
| Objetivo (Y) | `puntos_finales`: numérica |
| Entradas (X) | Solo lo que se sabe **antes** de que termine: p. ej. obstáculos superados en los primeros 10 s, tipos vistos, reacciones del jugador |
| Granularidad | **Una fila por partida** (resumen) |
| Tamaño mínimo | Decenas de partidas como mínimo; idealmente de jugadores distintos |
| Riesgos | **Fuga de información**: usar duración total, frame final o score final como entrada «adivina» el resultado porque el puntaje crece con el tiempo; partidas cortadas (cerrar el juego) que no terminan en muerte |

## Escenario 3: ¿qué tipo de obstáculo viene después?

| Elemento | Definición |
|---|---|
| Objetivo (Y) | `siguiente_obstaculo`: categórica (cactus chico, cactus grande, pájaro medio, pájaro alto) |
| Entradas (X) | `obstaculo_tipo` actual, `velocidad`, `score`/tiempo de partida, tipos anteriores |
| Granularidad | Una fila por frame, o mejor **una fila por obstáculo** (los frames repiten la misma información) |
| Tamaño mínimo | Cientos de obstáculos para ver la frecuencia de cada tipo |
| Riesgos | Si el juego elige el obstáculo al azar, **no hay patrón que aprender**; algunos tipos solo aparecen a cierta velocidad, así que con partidas cortas faltarían clases |

## Qué mostró la simulación (salida real, 30 partidas, semilla 42)

- 60 422 filas por frame; solo **29 positivos (0.048 %)** en el escenario 1 → confirma el desbalance.
  Una partida llegó al tope de 15 000 frames sin morir y por eso no aporta positivo.
- Escenario 2: correlación **duración vs puntos = 0.996** → usar la duración sería trampa.
  Con algo que sí se conoce temprano (obstáculos superados a los 10 s) la correlación baja a **0.352**,
  que es lo que realmente se podría predecir.
- Escenario 3: frames con obstáculo cercano: cactus chico 30.8 %, cactus grande 23.4 %, pájaro medio 15.1 %,
  pájaro alto 10.3 %, sin obstáculo 20.5 %. Muertes: 12 con cactus grande, 9 con cactus chico, 8 con
  pájaro medio y **0 con pájaro alto** (no hace falta esquivarlo).
- El score a los 10 s resultó **constante** en todas las partidas: depende solo del tiempo, así que no
  sirve como variable de entrada.

### Limitaciones del simulador

- Física, tamaños y probabilidades son inventados para el ejemplo; no son los del juego original.
- El «jugador» es una regla simple con descuidos aleatorios.
- En el escenario 3, dos obstáculos seguidos del mismo tipo se cuentan como uno al calcular `siguiente_obstaculo`.

## Referencias

- Los tres escenarios son los planteados en clase. La redacción y el simulador de esta carpeta
  son propios.
