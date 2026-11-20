# Diferencias entre redes neuronales multicapa y árboles de decisión

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Explicar las diferencias, ventajas y desventajas de las **redes neuronales multicapa (MLP)** y los
**árboles de decisión**: definir cada modelo, dar sus ventajas y desventajas técnicas, señalar la
diferencia principal y compararlos en una tabla según cómo procesan y aprenden.

---

## 1. Red neuronal multicapa (MLP)

Una MLP es un modelo formado por capas de neuronas artificiales: una capa de entrada, una o más
capas ocultas y una capa de salida. Cada neurona calcula una suma ponderada de sus entradas, le
suma un sesgo y aplica una función de activación no lineal (ReLU, sigmoide, tanh). Los pesos y
sesgos se ajustan durante el entrenamiento con **retropropagación** y **descenso de gradiente**
para reducir una función de pérdida.

### Ventajas

- **Aproxima funciones no lineales:** con suficientes neuronas puede representar relaciones muy
  complejas entre las variables (teorema de aproximación universal).
- **Aprende sus propias representaciones:** las capas ocultas combinan las entradas y crean
  características intermedias sin que haya que diseñarlas a mano.
- **Salidas probabilísticas:** con sigmoide o softmax entrega una probabilidad por clase, útil para
  fijar umbrales de decisión.
- **Optimización por gradiente:** se entrena por lotes y puede seguir aprendiendo con datos nuevos
  sin empezar de cero.
- **Escala con los datos:** mejora a medida que hay más ejemplos y aprovecha GPU.

### Desventajas

- **Costo computacional:** entrenar requiere muchas épocas y operaciones con matrices.
- **Sensible a la escala de los datos:** hay que normalizar o estandarizar las entradas.
- **Muchos hiperparámetros:** número de capas y neuronas, tasa de aprendizaje, activación,
  épocas, regularización.
- **Sobreajuste:** con pocos datos memoriza el conjunto de entrenamiento; necesita técnicas como
  dropout o parada temprana.
- **Optimización no convexa:** el gradiente puede quedarse en mínimos locales y el resultado
  depende de la inicialización.
- **Baja interpretabilidad:** es una "caja negra"; es difícil explicar por qué dio una respuesta.

## 2. Árbol de decisión

Un árbol de decisión clasifica (o predice) haciendo preguntas sobre las características, una tras
otra. Cada nodo interno aplica una condición del tipo `x_i <= t`, cada rama es una respuesta y cada
hoja da el resultado. Se construye dividiendo los datos de forma **recursiva**, eligiendo en cada
nodo la condición que más reduce la impureza (índice de Gini o entropía).

### Ventajas

- **Interpretable:** sus reglas se leen como `si ... entonces ...` y se pueden explicar a cualquier
  persona.
- **No necesita normalizar:** solo compara valores con umbrales, así que la escala no importa.
- **Fronteras no lineales:** al combinar varios cortes separa clases que no se dividen con una
  sola recta.
- **Selección implícita de variables:** las variables que no ayudan simplemente no aparecen en el
  árbol.
- **Entrenamiento rápido e inferencia eficiente:** predecir es recorrer unas cuantas condiciones.
- **Poco preprocesamiento:** acepta variables numéricas y categóricas.

### Desventajas

- **Sobreajuste:** si crece sin límite termina con una hoja por ejemplo.
- **Alta varianza:** pequeños cambios en los datos pueden producir un árbol muy distinto.
- **Construcción voraz:** elige el mejor corte en cada paso sin mirar hacia adelante, por lo que no
  garantiza el mejor árbol global (por ejemplo, le cuesta el XOR).
- **Fronteras en escalones:** los cortes son paralelos a los ejes, así que aproxima mal fronteras
  diagonales o suaves.
- **Profundidad difícil de ajustar:** poca profundidad subajusta, mucha sobreajusta; hay que podar
  o limitar `max_depth` / `min_samples_leaf`.

## 3. Diferencia principal

La diferencia de fondo es **qué aprende cada modelo y cómo lo hace**:

- La **MLP** es una composición de funciones
  `f(x) = f_L(f_{L-1}( ... f_1(x)))`, donde cada capa es `f_k(h) = σ(W_k·h + b_k)`. Su estructura
  se fija antes de entrenar y lo que aprende son los **valores continuos** de `W` y `b`, ajustados
  por gradiente para minimizar el error.
- El **árbol** es una **partición recursiva** del espacio de entrada en regiones rectangulares
  mediante condiciones `x_i <= t`. Lo que aprende es la **estructura misma** (qué variable, qué
  umbral y en qué orden), sin derivadas, con una búsqueda voraz.

En pocas palabras: la red aprende *cuánto pesa cada conexión*; el árbol aprende *qué preguntas
hacer*.

## 4. Tabla comparativa

| Aspecto | Red neuronal multicapa | Árbol de decisión |
|---|---|---|
| Entrada de datos | Vector numérico, normalizado; las categorías se codifican (one-hot). | Variables numéricas o categóricas sin escalar. |
| Procesamiento | Todas las entradas pasan por todas las capas: sumas ponderadas + activaciones. | El dato sigue un solo camino de la raíz a una hoja, evaluando una condición por nodo. |
| Predicción | Valor continuo / probabilidad en la capa de salida. | Clase (o valor) guardado en la hoja alcanzada. |
| Forma de avanzar | Hacia adelante, capa por capa (*forward pass*). | De arriba hacia abajo, rama por rama. |
| Forma de proceso | Paralelo (operaciones de matrices). | Secuencial (una pregunta a la vez). |
| Qué aprende | Pesos y sesgos. | Variables, umbrales y forma del árbol. |
| Forma de aprendizaje | Minimiza una pérdida con descenso de gradiente. | Divisiones que reducen la impureza (Gini / entropía). |
| Repetición del entrenamiento | Muchas épocas sobre los mismos datos. | Una sola construcción recursiva. |
| Corrección del error | Retropropagación: el error se reparte hacia atrás y se corrigen los pesos. | No corrige hacia atrás: crea nuevos nodos o se poda después. |
| Uso de características | Usa todas las entradas a la vez y las combina. | Usa solo las variables que aparecen en el camino del dato. |
| Interpretabilidad | Baja (caja negra). | Alta (reglas legibles). |
| Resultado final | Una función continua con muchos parámetros. | Un conjunto de reglas `si ... entonces ...`. |

## 5. Conclusión

Ninguno es mejor en todos los casos. El árbol conviene cuando hay pocos datos, variables
mezcladas y se necesita explicar la decisión. La MLP conviene cuando hay muchos datos y relaciones
complejas, aunque a costa de más cómputo y menos claridad. En la práctica 6
([juego modificado con árbol de decisión](../juego_modificado_arbol/README.md)) se ve esto
directamente: el árbol no necesitó normalizar y sus reglas se pueden imprimir, mientras que la MLP
original solo daba un resultado sin explicación.

## Referencias

- Se consultó como referencia el trabajo de **Sergio Cardona García** (repositorio
  `sergio-cardona-04/Inteligencia_artificial`, carpeta `arbol_red_diferencias`, notas a mano del
  05 al 07 de octubre de 2026). Este documento es una redacción propia que sigue la misma
  estructura de la actividad (definiciones, ventajas, desventajas, diferencia principal y tabla).
- Apuntes de clase de Inteligencia Artificial (05/oct/2026).
