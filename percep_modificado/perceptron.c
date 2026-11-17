/*
 * Perceptrón modificado
 * Autor: Diego Ignacio Moreno Zapien
 * Materia: Inteligencia Artificial
 *
 * Una neurona (perceptrón) con activación sigmoide entrenada con la regla
 * delta sobre tablas de verdad de dos entradas.
 *
 * Qué hace el programa:
 *   1. Entrena la neurona para AND, OR, NAND y XOR con las mismas condiciones.
 *   2. Muestra que AND, OR y NAND se aprenden, pero XOR no: XOR no es
 *      linealmente separable y una sola neurona solo traza una recta.
 *   3. Modificación: agrega una tercera entrada x0*x1. Con esa entrada extra
 *      XOR sí se vuelve separable y la misma neurona lo aprende.
 *
 * Compilar y ejecutar:
 *   gcc -Wall -O2 -o perceptron perceptron.c -lm
 *   ./perceptron
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPOCAS      20000   /* pasadas completas por la tabla */
#define TASA        0.5     /* tasa de aprendizaje */
#define SEMILLA     7       /* semilla fija: resultados repetibles */
#define MAX_ENTRADAS 3

/* Una neurona: pesos para cada entrada más un sesgo (bias). */
typedef struct {
    int n;                       /* número de entradas que usa */
    double w[MAX_ENTRADAS];
    double b;
} Neurona;

/* Sigmoide correcta: 1 / (1 + e^(-s)). Su salida siempre queda entre 0 y 1. */
static double sigmoide(double s)
{
    return 1.0 / (1.0 + exp(-s));
}

/* Pesos iniciales pequeños y aleatorios en [-0.5, 0.5]. */
static void iniciar(Neurona *nr, int n)
{
    nr->n = n;
    for (int i = 0; i < n; i++)
        nr->w[i] = (double)rand() / RAND_MAX - 0.5;
    nr->b = (double)rand() / RAND_MAX - 0.5;
}

/* Salida de la neurona: sigmoide(w·x + b). */
static double salida(const Neurona *nr, const double *x)
{
    double net = nr->b;
    for (int i = 0; i < nr->n; i++)
        net += nr->w[i] * x[i];
    return sigmoide(net);
}

/*
 * Un paso de la regla delta para un ejemplo.
 *   error = objetivo - y
 *   delta = error * y * (1 - y)      (y*(1-y) es la derivada de la sigmoide)
 *   w_i  += TASA * delta * x_i
 *   b    += TASA * delta
 * Devuelve el error cuadrático del ejemplo (antes de ajustar).
 */
static double entrenar_ejemplo(Neurona *nr, const double *x, double objetivo)
{
    double y = salida(nr, x);
    double error = objetivo - y;
    double delta = error * y * (1.0 - y);

    for (int i = 0; i < nr->n; i++)
        nr->w[i] += TASA * delta * x[i];
    nr->b += TASA * delta;

    return error * error;
}

/* Arma las entradas de un renglón: (x0, x1) y, si se pide, x0*x1. */
static void armar_entrada(double *x, int a, int b, int con_producto)
{
    x[0] = a;
    x[1] = b;
    if (con_producto)
        x[2] = a * b;
}

/*
 * Entrena una compuerta lógica.
 *   tabla[k] = salida esperada para (a, b) = (k>>1, k&1), k = 0..3
 *   con_producto = 1 agrega la entrada x0*x1.
 * Imprime pesos finales, salidas y cuántos renglones acierta (umbral 0.5).
 */
static int probar_compuerta(const char *nombre, const int tabla[4], int con_producto)
{
    Neurona nr;
    double x[MAX_ENTRADAS];
    double ecm = 0.0;

    iniciar(&nr, con_producto ? 3 : 2);

    for (int e = 0; e < EPOCAS; e++) {
        ecm = 0.0;
        for (int k = 0; k < 4; k++) {
            armar_entrada(x, k >> 1, k & 1, con_producto);
            ecm += entrenar_ejemplo(&nr, x, tabla[k]);
        }
        ecm /= 4.0;
    }

    printf("== %s%s ==\n", nombre, con_producto ? " (con entrada extra x0*x1)" : "");
    printf("Pesos: w0=%.3f  w1=%.3f", nr.w[0], nr.w[1]);
    if (con_producto)
        printf("  w2=%.3f", nr.w[2]);
    printf("  bias=%.3f\n", nr.b);

    int aciertos = 0;
    for (int k = 0; k < 4; k++) {
        int a = k >> 1, b = k & 1;
        armar_entrada(x, a, b, con_producto);
        double y = salida(&nr, x);
        int pred = y >= 0.5;
        aciertos += (pred == tabla[k]);
        printf("  %d,%d -> %.4f  (esperado %d) %s\n",
               a, b, y, tabla[k], pred == tabla[k] ? "ok" : "FALLA");
    }
    printf("Error cuadratico medio final: %.6f | aciertos %d/4\n\n", ecm, aciertos);
    return aciertos;
}

int main(void)
{
    /* Salidas esperadas en el orden (0,0), (0,1), (1,0), (1,1). */
    const int AND[4]  = {0, 0, 0, 1};
    const int OR[4]   = {0, 1, 1, 1};
    const int NAND[4] = {1, 1, 1, 0};
    const int XOR[4]  = {0, 1, 1, 0};

    srand(SEMILLA);
    printf("Perceptron con sigmoide | epocas=%d tasa=%.2f semilla=%d\n\n",
           EPOCAS, TASA, SEMILLA);

    int total = 0;
    total += probar_compuerta("AND", AND, 0) == 4;
    total += probar_compuerta("OR", OR, 0) == 4;
    total += probar_compuerta("NAND", NAND, 0) == 4;
    int xor_simple = probar_compuerta("XOR", XOR, 0);
    int xor_mod = probar_compuerta("XOR", XOR, 1);

    printf("Resumen:\n");
    printf("  Compuertas lineales aprendidas (AND, OR, NAND): %d/3\n", total);
    printf("  XOR con 2 entradas: %d/4 aciertos (no es linealmente separable)\n", xor_simple);
    printf("  XOR con entrada x0*x1: %d/4 aciertos\n", xor_mod);
    return 0;
}
