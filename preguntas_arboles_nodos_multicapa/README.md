# Preguntas sobre árboles de decisión y redes multicapa

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Cuestionario de dos partes sobre árboles de decisión y redes neuronales multicapa (MLP):

- **Parte I (preguntas 1 a 5):** conceptos y definiciones.
- **Parte II (preguntas 6 a 9):** análisis de casos (fraude bancario, riesgo de reprobar, triaje
  hospitalario y entregas a tiempo), eligiendo y justificando el modelo.

Las respuestas de abajo están redactadas con mis propias palabras. Como apoyo agregué
[`demo_conceptos.py`](demo_conceptos.py), un script corto (solo Python estándar) que muestra con
números pequeños las ideas de las preguntas 2, 4, 5 y 9.

---

## Parte I. Conceptos y definiciones

### Pregunta 1. ¿Qué es un árbol de decisión y cuál es su objetivo en clasificación?

Es un modelo que clasifica un dato haciéndole una serie de preguntas sobre sus características,
una tras otra, como un diagrama de flujo. Cada pregunta es una condición sencilla
(por ejemplo, `asistencia <= 70 %`) y, según la respuesta, el dato baja por un camino u otro hasta
llegar a una clase.

Su objetivo en un problema de clasificación es **partir el conjunto de datos en grupos cada vez
más "puros"**, es decir, grupos donde casi todos los ejemplos sean de la misma clase. Para elegir
cada pregunta usa una medida de impureza (índice de Gini o entropía/ganancia de información) y se
queda con la condición que más la reduce.

### Pregunta 2. Elementos de un árbol de decisión

| Elemento | Qué es |
|---|---|
| **Nodo raíz** | La primera pregunta del árbol. Recibe todos los datos de entrenamiento y aplica la condición que mejor los separa. |
| **Nodo interno** | Cualquier pregunta intermedia. Recibe solo el subconjunto de datos que llegó por su camino y lo vuelve a dividir. |
| **Rama** | La conexión entre un nodo y el siguiente; representa una respuesta posible a la condición (por ejemplo "sí" / "no"). |
| **Hoja** | Nodo final, ya sin preguntas. Guarda la clase que se asigna (normalmente la clase mayoritaria de los ejemplos que llegaron ahí). |

Ejemplo pequeño (el script lo imprime):

```
[raíz] ¿asistencia <= 70?
 ├─ sí → [nodo interno] ¿tareas <= 50?
 │        ├─ sí → hoja: RIESGO
 │        └─ no → hoja: SIN RIESGO
 └─ no → hoja: SIN RIESGO
```

### Pregunta 3. ¿Qué es una red neuronal multicapa y qué hace cada capa?

Una MLP es un conjunto de neuronas artificiales organizadas en capas, donde cada neurona de una
capa está conectada con todas las de la capa siguiente. Cada neurona calcula una suma ponderada
de sus entradas, le suma un sesgo y aplica una función de activación (ReLU, sigmoide, etc.).
Gracias a esas activaciones no lineales la red puede aprender relaciones que no son líneas rectas.

- **Capa de entrada:** no hace cálculos; solo recibe las características del dato
  (una neurona por variable) y las pasa a la siguiente capa.
- **Capas ocultas:** combinan y transforman las entradas para crear representaciones
  intermedias (patrones) que ayudan a resolver el problema. Puede haber una o varias.
- **Capa de salida:** produce la respuesta final; en clasificación suele dar una probabilidad
  por clase (sigmoide para dos clases, softmax para varias).

### Pregunta 4. ¿Qué representan los pesos y los sesgos?

- **Pesos (w):** indican cuánto influye cada entrada en una neurona. Un peso grande hace que esa
  entrada cuente mucho; un peso cercano a cero la vuelve casi irrelevante; un peso negativo hace
  que la entrada "empuje" en sentido contrario.
- **Sesgo (b):** es un valor que se suma después de la suma ponderada. Desplaza el punto en el
  que la neurona se activa, así la neurona puede activarse (o no) aunque todas las entradas
  valgan cero. Es como el término independiente de una recta `y = m·x + b`.

La salida de una neurona es `a = f(w1·x1 + w2·x2 + ... + b)`. En el script se ve cómo al cambiar
solo el sesgo cambia la decisión con las mismas entradas.

### Pregunta 5. Diferencia principal en cómo aprenden

- El **árbol de decisión** aprende **construyendo su estructura**: en cada nodo prueba variables y
  umbrales, se queda con la división que más reduce la impureza y repite de forma recursiva
  (método voraz). Lo aprendido son reglas `si ... entonces ...`. No usa derivadas.
- La **MLP** tiene una **estructura fija** (capas y neuronas elegidas antes) y aprende
  **ajustando números**: calcula una predicción, mide el error con una función de pérdida y usa
  retropropagación (backpropagation) y descenso de gradiente para corregir poco a poco los pesos
  y sesgos. Lo hace durante muchas pasadas (épocas) sobre los datos.

En resumen: el árbol aprende *qué preguntas hacer*; la red aprende *cuánto pesa cada conexión*.

---

## Parte II. Análisis y aplicación

### Pregunta 6. Detección de compras fraudulentas (banco)

Datos: monto, hora, ciudad, tipo de establecimiento, compras en el día e historial del cliente.

**Red neuronal multicapa**

| Ventajas | Desventajas |
|---|---|
| Detecta combinaciones sutiles entre variables (por ejemplo, monto normal pero ciudad y hora raras para ese cliente). | Es una "caja negra": cuesta explicar por qué marcó una compra como fraude. |
| Funciona bien con muchos datos, que un banco sí tiene (millones de transacciones). | Necesita mucho preprocesamiento: escalar números y codificar ciudad y tipo de establecimiento. |
| Puede reentrenarse seguido para adaptarse a nuevas formas de fraude. | El fraude es muy poco frecuente (clases desbalanceadas); sin cuidado la red aprende a decir siempre "no fraude". |
| Da una probabilidad, útil para poner un umbral de alerta. | Entrenarla y ajustar hiperparámetros cuesta más tiempo y cómputo. |

**Árbol de decisión**

| Ventajas | Desventajas |
|---|---|
| Sus reglas se pueden leer y explicar al cliente o a un auditor ("monto > X y ciudad distinta a la habitual y hora de madrugada"). | Un solo árbol tiende a sobreajustarse y puede cambiar mucho con pocos datos nuevos. |
| Acepta variables numéricas y categóricas casi sin preparar y no necesita normalizar. | Sus fronteras son "en escalones" (cortes paralelos a los ejes), así que capta peor relaciones finas entre variables. |
| Es muy rápido al predecir, importante porque la compra se autoriza en milisegundos. | Con fraudes nuevos o muy disfrazados suele tener menor precisión que la red. |

**¿Cuál usaría y por qué?** Usaría la **red neuronal como detector principal**, porque el banco
tiene muchísimos datos y el fraude se esconde en combinaciones de variables que la red capta
mejor. Pero no la usaría sola: la acompañaría de un árbol (o de reglas extraídas de él) para
explicar las alertas, ya que el banco debe poder justificar por qué bloquea una tarjeta, y
manejaría el desbalance de clases (ponderar la clase fraude o sobremuestrear). Si el banco tuviera
pocos datos o la explicación fuera obligatoria en cada caso, elegiría el árbol (o un conjunto de
árboles como Random Forest).

### Pregunta 7. Estudiantes en riesgo de reprobar (misma precisión)

Si ambos modelos tienen casi la misma precisión, tomaría en cuenta:

1. **Interpretabilidad:** el objetivo es *ayudar* al alumno. Un árbol dice "riesgo porque la
   asistencia es menor a 70 % y entregó menos de la mitad de las tareas", y eso le indica al
   profesor qué hacer. La red solo da "riesgo: 0.82".
2. **Cantidad de datos:** una escuela tiene pocos registros (cientos, no millones). La red
   necesita más datos para no sobreajustarse; el árbol funciona bien con pocos.
3. **Costo y mantenimiento:** el árbol se entrena en segundos, se puede revisar a mano y
   cualquier persona del área puede actualizarlo.
4. **Tipo de variables:** son pocas y fáciles de entender; no hay patrones complejos que
   justifiquen una red.
5. **Equidad:** con reglas visibles es más fácil detectar si el modelo trata injustamente a un
   grupo de alumnos.
6. **Otras métricas además de la precisión:** importa sobre todo el *recall* de la clase
   "riesgo" (no dejar pasar alumnos que sí van a reprobar). Revisaría esa métrica en los dos.

**Elección:** el **árbol de decisión**. Con el mismo desempeño, gana porque es explicable,
barato y sus reglas se convierten directamente en acciones (tutorías, recordatorios de tareas).

### Pregunta 8. Atención prioritaria en un hospital

**No, la mayor precisión no basta por sí sola.** En un hospital la decisión afecta la salud de
personas, así que también importa poder explicarla y revisarla.

Consecuencias de elegir la red sin más análisis:

- **Falta de explicación:** el médico no sabe por qué el sistema mandó a un paciente al final de
  la fila; no puede detectar si fue un error ni justificarlo ante el paciente o la familia.
- **Responsabilidad legal y ética:** si un paciente grave no fue priorizado, hay que poder
  explicar la decisión; con una caja negra eso es muy difícil.
- **Errores ocultos y sesgos:** la red pudo aprender atajos de los datos (por ejemplo, depender
  de la edad de más) y nadie lo notaría hasta que haya un daño.
- **Desconfianza del personal:** si los médicos no entienden el sistema, lo ignoran o, peor, lo
  siguen sin cuestionarlo.
- **Qué error se comete:** no todos los errores pesan igual. Un falso negativo (no priorizar a un
  paciente grave) es mucho peor que un falso positivo. Hay que comparar los modelos por *recall*
  de la clase "prioritario", no solo por precisión general.

Lo que haría: usar la red solo si su ventaja es clara en los falsos negativos, siempre como
**apoyo** al médico (la decisión final la toma una persona), agregar herramientas de explicación
(importancia de variables, SHAP o LIME) y validar el sistema con casos reales antes de usarlo.
Si la diferencia de precisión es pequeña, preferiría el árbol por ser explicable.

### Pregunta 9. Pedido: árbol dice "a tiempo", red dice "probablemente tarde"

Una sola predicción no basta para saber cuál modelo es mejor; hay que mirar cómo se comportan en
general y qué tan seguros están en este caso. Analizaría:

1. **Desempeño histórico con datos de prueba** (pedidos que el modelo no vio al entrenar):
   exactitud, precisión, recall y F1 de cada modelo, y su matriz de confusión. Sobre todo el recall
   de la clase "tarde", porque avisar de un retraso real es lo más útil para la empresa.
2. **Confianza de cada predicción:** la probabilidad que da la red (¿0.51 o 0.95?) y la pureza de
   la hoja del árbol (¿en esa hoja el 55 % llegó a tiempo o el 98 %?). "Probablemente" indica que la
   red no está muy segura.
3. **Casos parecidos:** cómo les fue a ambos modelos con pedidos similares en distancia, tráfico,
   clima y hora.
4. **Datos actuales:** si el tráfico o el clima de hoy son raros (lluvia fuerte, un evento), puede
   ser que el árbol no tenga una rama para esa situación y la red la capte mejor, o al revés.
5. **Explicación del árbol:** seguir el camino de reglas para ver si tiene sentido (por ejemplo,
   si decidió solo por la distancia e ignoró el tráfico).
6. **Resultado real:** cuando el pedido llegue, registrar quién acertó y llevar la cuenta; con
   el tiempo eso indica qué modelo confiar más (o usar los dos juntos).

---

## Script de apoyo: `demo_conceptos.py`

```
python demo_conceptos.py
```

No necesita librerías externas. Muestra:

- un árbol pequeño escrito a mano (pregunta 2) y por qué camino baja un alumno de ejemplo;
- una neurona con pesos y sesgo (pregunta 4), y cómo cambia su salida si se modifica el sesgo;
- cómo un árbol elige un corte con Gini y cómo una neurona ajusta sus pesos por gradiente
  durante varias épocas (pregunta 5);
- la comparación de dos modelos con matriz de confusión, precisión y recall sobre datos de
  prueba inventados para el ejemplo (pregunta 9).

Los datos del script son **pequeños y de ejemplo**, solo para ilustrar los conceptos; no son
datos reales de ningún banco, escuela, hospital ni empresa.

## Referencias

- Se consultó como referencia el trabajo de **Sergio Cardona García** (repositorio
  `sergio-cardona-04/Inteligencia_artificial`, carpeta `preguntas_arboles_nodos_multicapa`), de
  donde se tomaron los enunciados de las preguntas. Las respuestas aquí son redacción propia.
- Apuntes de clase de Inteligencia Artificial sobre árboles de decisión (Gini, entropía) y redes
  multicapa (pesos, sesgos, retropropagación).
