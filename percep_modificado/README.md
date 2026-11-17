# Perceptrón modificado

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

## Actividad

Partir del programa de perceptrón en C visto en clase (una neurona de dos entradas con sigmoide,
entrenada sobre una tabla de verdad) y modificarlo.

> Nota: el enunciado escrito de la actividad no está disponible; el código de clase entrena la
> neurona con la tabla de **XOR**. Esta versión se centra en eso: qué compuertas puede aprender
> una sola neurona, por qué XOR no, y qué modificación lo permite.

## Archivo

- [`perceptron.c`](perceptron.c): programa propio, comentado.

```
gcc -Wall -O2 -o perceptron perceptron.c -lm
./perceptron
```

## Cómo funciona

1. Cada neurona calcula `y = sigmoide(w0·x0 + w1·x1 + b)`, con `sigmoide(s) = 1 / (1 + e^(-s))`.
2. Se entrena con la **regla delta** durante 20 000 épocas (tasa 0.5, semilla fija 7):
   - `error = objetivo − y`
   - `delta = error · y · (1 − y)` (la derivada de la sigmoide)
   - `w_i += tasa · delta · x_i`, `b += tasa · delta`
3. Se prueba con AND, OR, NAND y XOR. Una salida ≥ 0.5 se toma como 1.
4. **Modificación:** se agrega una tercera entrada `x2 = x0·x1` y se vuelve a entrenar XOR.

## Resultados (salida real del programa)

| Compuerta | Aciertos | Salidas para (0,0) (0,1) (1,0) (1,1) |
|---|---|---|
| AND | 4/4 | 0.0000, 0.0164, 0.0164, 0.9806 |
| OR | 4/4 | 0.0163, 0.9897, 0.9897, 1.0000 |
| NAND | 4/4 | 1.0000, 0.9836, 0.9836, 0.0194 |
| XOR (2 entradas) | 2/4 | 0.5161, 0.5000, 0.4839, 0.4678 |
| XOR + entrada `x0·x1` | 4/4 | 0.0221, 0.9820, 0.9820, 0.0146 |

### Por qué XOR falla con una sola neurona

Una neurona dibuja **una sola recta** `w0·x0 + w1·x1 + b = 0` y pone los 1 de un lado y los 0 del otro.
En XOR los 1 están en (0,1) y (1,0) y los 0 en (0,0) y (1,1), en esquinas opuestas: ninguna recta los separa.
Por eso el entrenamiento se queda con todas las salidas cerca de 0.5 (error cuadrático ≈ 0.27).

### Por qué funciona la entrada extra

Con `x2 = x0·x1` la neurona puede «castigar» el caso (1,1). Los pesos aprendidos fueron
`w0 = w1 ≈ 7.8`, `w2 ≈ −16.0`, `b ≈ −3.8`:

- (0,0): −3.8 → 0
- (0,1) y (1,0): 7.8 − 3.8 = 4.0 → 1
- (1,1): 7.8 + 7.8 − 16.0 − 3.8 = −4.2 → 0

En tres dimensiones los cuatro puntos sí se pueden separar con un plano. La otra salida clásica
es usar una red multicapa (una capa oculta), que hace esa combinación por sí sola.

## Cambios respecto al programa de clase

| Programa de clase | Esta versión | Motivo |
|---|---|---|
| `sigmoide(s) = 1/(1 + (-1*s))`, o sea `1/(1 − s)` | `1/(1 + exp(-s))` | La fórmula original no es una sigmoide: en `s = 1` divide entre cero y para `s > 1` da negativos. Falta la exponencial. |
| Ajuste `K·Error·x` | `tasa · Error · y(1−y) · x` | Se incluye la derivada de la sigmoide (regla delta completa). |
| 3 000 000 épocas imprimiendo todo | 20 000 épocas, solo el resultado final | La salida era enorme; 20 000 basta para converger. |
| Pesos fijos escritos a mano en `InitNt` | Se usan los pesos recién entrenados | Evita copiar números a mano. |
| Solo una tabla | AND, OR, NAND, XOR y XOR modificado | Permite comparar. |

## Referencias

- Programa base del perceptrón visto en clase de Inteligencia Artificial. El código de esta
  carpeta es propio.
