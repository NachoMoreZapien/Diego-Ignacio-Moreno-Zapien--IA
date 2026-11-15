"""
Ejercicios de introduccion a la Inteligencia Artificial.

Autor: Diego Ignacio Moreno Zapien
Materia: Inteligencia Artificial

Contiene cinco problemas clasicos resueltos como problemas de busqueda:
    1. N reinas            -> backtracking
    2. Josephus (k = 2)    -> simulacion + formula cerrada
    3. Monjes y canibales  -> busqueda en anchura (BFS) sobre estados
    4. Laberinto           -> busqueda en profundidad (DFS) recursiva
    5. Conteo de islas     -> recorrido con DFS iterativo (flood fill)

Ejecutar:  python ejercicios.py
Solo usa la biblioteca estandar de Python.
"""

from collections import deque
import random


# ---------------------------------------------------------------------------
# 1. N reinas
# ---------------------------------------------------------------------------
def n_reinas(n):
    """Devuelve todas las soluciones. Cada solucion es una lista: columna[fila]."""
    soluciones = []
    columnas, diag1, diag2 = set(), set(), set()
    actual = []

    def colocar(fila):
        if fila == n:
            soluciones.append(actual.copy())
            return
        for col in range(n):
            # Dos reinas se atacan si comparten columna o diagonal.
            # En una diagonal "\" fila-col es constante; en una "/" fila+col es constante.
            if col in columnas or (fila - col) in diag1 or (fila + col) in diag2:
                continue
            columnas.add(col); diag1.add(fila - col); diag2.add(fila + col)
            actual.append(col)
            colocar(fila + 1)
            actual.pop()
            columnas.discard(col); diag1.discard(fila - col); diag2.discard(fila + col)

    colocar(0)
    return soluciones


def tablero_reinas(solucion):
    n = len(solucion)
    return "\n".join(" ".join("R" if solucion[f] == c else "." for c in range(n)) for f in range(n))


# ---------------------------------------------------------------------------
# 2. Josephus: cada persona elimina a la siguiente viva y pasa el cuchillo
# ---------------------------------------------------------------------------
def josephus_simulado(n, inicio=1):
    """Simula el circulo. Las personas se numeran 1..n; 'inicio' tiene el cuchillo."""
    circulo = deque(range(1, n + 1))
    circulo.rotate(-(inicio - 1))  # dejamos al del cuchillo al frente
    while len(circulo) > 1:
        circulo.rotate(-1)   # el del cuchillo pasa atras...
        circulo.popleft()    # ...y elimina a quien sigue
    return circulo[0]


def josephus_formula(n, inicio=1):
    """Formula cerrada para k=2: si n = 2^m + L, sobrevive la posicion 2L+1 contando desde 'inicio'."""
    potencia = 1 << (n.bit_length() - 1)
    L = n - potencia
    relativo = 2 * L + 1
    return (inicio - 1 + relativo - 1) % n + 1


# ---------------------------------------------------------------------------
# 3. Monjes y canibales (3 y 3, bote de 2)
# ---------------------------------------------------------------------------
def estado_valido(m, c):
    """En una orilla con m monjes y c canibales, no puede haber mas canibales que monjes (si hay monjes)."""
    return m == 0 or m >= c


def monjes_canibales(total=3, capacidad=2):
    # Estado: (monjes en A, canibales en A, bote en A?)
    inicio = (total, total, True)
    meta = (0, 0, False)
    viajes = [(dm, dc) for dm in range(capacidad + 1) for dc in range(capacidad + 1)
              if 1 <= dm + dc <= capacidad and (dm == 0 or dm >= dc)]  # tampoco en el bote

    padre = {inicio: None}
    cola = deque([inicio])
    while cola:
        estado = cola.popleft()
        if estado == meta:
            break
        m, c, bote_en_a = estado
        signo = -1 if bote_en_a else 1
        for dm, dc in viajes:
            nm, nc = m + signo * dm, c + signo * dc
            if not (0 <= nm <= total and 0 <= nc <= total):
                continue
            if not (estado_valido(nm, nc) and estado_valido(total - nm, total - nc)):
                continue
            nuevo = (nm, nc, not bote_en_a)
            if nuevo not in padre:
                padre[nuevo] = estado
                cola.append(nuevo)

    camino = []
    e = meta
    while e is not None:
        camino.append(e)
        e = padre.get(e)
    return camino[::-1]


# ---------------------------------------------------------------------------
# 4. Laberinto (DFS recursivo con marcado de visitados)
# ---------------------------------------------------------------------------
def resolver_laberinto(laberinto, inicio):
    """'x' = pared, ' ' = libre, '1' = salida. Devuelve la lista de coordenadas (fila, col) del camino."""
    filas, cols = len(laberinto), len(laberinto[0])
    visitado = set()
    camino = []

    def dfs(f, c):
        if not (0 <= f < filas and 0 <= c < cols):
            return False
        if laberinto[f][c] == "x" or (f, c) in visitado:
            return False
        camino.append((f, c))
        if laberinto[f][c] == "1":
            return True
        visitado.add((f, c))
        for df, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):  # derecha, abajo, izquierda, arriba
            if dfs(f + df, c + dc):
                return True
        camino.pop()  # callejon sin salida: regresamos
        return False

    return camino if dfs(*inicio) else None


def dibujar_laberinto(laberinto, camino):
    pasos = set(camino or [])
    return "\n".join(
        "".join("*" if (f, c) in pasos and laberinto[f][c] == " " else
                {"x": "#", " ": ".", "1": "S"}[laberinto[f][c]]
                for c in range(len(laberinto[0])))
        for f in range(len(laberinto)))


# ---------------------------------------------------------------------------
# 5. Conteo de islas (4-conectividad)
# ---------------------------------------------------------------------------
def contar_islas(mapa):
    """No modifica el mapa original: usa un conjunto de visitados y una pila (sin recursion)."""
    if not mapa:
        return 0, []
    filas, cols = len(mapa), len(mapa[0])
    visitado = set()
    tamanos = []
    for i in range(filas):
        for j in range(cols):
            if mapa[i][j] != 1 or (i, j) in visitado:
                continue
            pila = [(i, j)]
            visitado.add((i, j))
            tam = 0
            while pila:
                f, c = pila.pop()
                tam += 1
                for nf, nc in ((f + 1, c), (f - 1, c), (f, c + 1), (f, c - 1)):
                    if 0 <= nf < filas and 0 <= nc < cols and mapa[nf][nc] == 1 and (nf, nc) not in visitado:
                        visitado.add((nf, nc))
                        pila.append((nf, nc))
            tamanos.append(tam)
    return len(tamanos), tamanos


# ---------------------------------------------------------------------------
def main():
    print("=" * 50, "\n1. N REINAS\n" + "=" * 50)
    sol4 = n_reinas(4)
    print(f"4 reinas: {len(sol4)} soluciones")
    for s in sol4:
        print(tablero_reinas(s), "\n")
    print("Soluciones para n=1..8:", [len(n_reinas(n)) for n in range(1, 9)])

    print("\n" + "=" * 50, "\n2. JOSEPHUS (k=2)\n" + "=" * 50)
    for n in (5, 7, 10, 41):
        print(f"n={n:>2}: sobrevive la posicion {josephus_simulado(n)} (formula: {josephus_formula(n)})")
    n, ini = 12, random.randint(1, 12)
    print(f"Inicio aleatorio: n={n}, cuchillo empieza en {ini} -> lugar seguro {josephus_simulado(n, ini)}")
    assert all(josephus_simulado(n, i) == josephus_formula(n, i) for n in range(1, 200) for i in range(1, n + 1))
    print("Formula verificada contra la simulacion para n=1..199 y todo inicio.")

    print("\n" + "=" * 50, "\n3. MONJES Y CANIBALES\n" + "=" * 50)
    camino = monjes_canibales()
    for paso, (m, c, bote_a) in enumerate(camino):
        a = "m" * m + "c" * c or "-"
        b = "m" * (3 - m) + "c" * (3 - c) or "-"
        sentido = "" if paso == 0 else ("<--" if bote_a else "-->")
        print(f"{paso:>2} | A: {a:<7}| B: {b:<7}| {sentido}")
    print(f"Viajes minimos: {len(camino) - 1}")

    print("\n" + "=" * 50, "\n4. LABERINTO\n" + "=" * 50)
    laberinto = [
        [" ", "x", " ", " ", " ", "x", " ", " "],
        [" ", "x", " ", "x", " ", " ", " ", "1"],
        [" ", " ", " ", "x", "x", "x", " ", "x"],
    ]
    camino = resolver_laberinto(laberinto, (0, 0))
    print("Camino:", camino)
    print(dibujar_laberinto(laberinto, camino))

    print("\n" + "=" * 50, "\n5. CONTEO DE ISLAS\n" + "=" * 50)
    mapa = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
    ]
    total, tamanos = contar_islas(mapa)
    print(f"Islas: {total}  (tamanos: {tamanos})")


if __name__ == "__main__":
    main()
