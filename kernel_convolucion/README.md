# CNN: kernel de convolución

**Autor:** Diego Ignacio Moreno Zapien
**Materia:** Inteligencia Artificial

Ejercicio de convolución a mano, como la que hace una capa convolucional de una CNN.

## Datos

- **Imagen:** 5×5 en escala de grises (valores de 0 a 255).
- **Kernel:** 3×3 con todos los valores en 1.
- **Bordes:** se rellenan con ceros (*zero padding*), así la salida también es de 5×5.
- **Saturación:** si el resultado pasa de 255 se deja en 255; si es menor que 0 se deja en 0.

```
Imagen de entrada          Kernel
 10  40  25  80   5        1 1 1
 60  15  90  20  35        1 1 1
 30  70  12  45 100        1 1 1
  8  55  65  18  50
 22  11  95  28   9
```

## Procedimiento

Para cada píxel se centra el kernel encima, se multiplica cada valor por el número del kernel
que le toca y se suman los 9 productos. Como el kernel es de unos, el resultado es simplemente
la **suma del píxel y sus 8 vecinos**. Los vecinos que caen fuera de la imagen valen 0.

**Esquina (fila 1, columna 1):** solo tiene 3 vecinos dentro de la imagen.

```
 0  0  0
 0 10 40     10 + 40 + 60 + 15 = 125
 0 60 15
```

**Centro (fila 3, columna 3):**

```
15 90 20
70 12 45     15 + 90 + 20 + 70 + 12 + 45 + 55 + 65 + 18 = 390  ->  255 (se satura)
55 65 18
```

## Resultado

Sumas sin saturar:

```
125 240 270 255 140
225 352 397 412 285
238 405 390 435 268
196 368 399 422 250
 96 256 272 265 105
```

Imagen de salida (con saturación a 255):

```
125 240 255 255 140
225 255 255 255 255
238 255 255 255 255
196 255 255 255 250
 96 255 255 255 105
```

## Observación

Un kernel lleno de unos suma 9 píxeles, así que casi todo el centro se satura en 255
y la imagen se ve "quemada". Solo las orillas, que suman menos vecinos, quedan por debajo.
Por eso normalmente se usa el kernel de **promedio** (cada valor en 1/9): suaviza la imagen
sin salirse del rango 0–255.
