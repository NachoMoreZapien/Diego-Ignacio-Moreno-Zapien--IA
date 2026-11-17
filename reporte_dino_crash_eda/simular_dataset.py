"""
Dino run: simulador de datos de ejemplo para el EDA conceptual.
Autor: Diego Ignacio Moreno Zapien

No es el juego real: es una simulación simplificada (solo biblioteca estándar)
que genera filas con las mismas columnas propuestas en el README, para poder
revisar que las definiciones de cada escenario tienen sentido:

  1. ¿Morirá en el siguiente frame?   -> tabla por frame, etiqueta binaria
  2. ¿Cuántos puntos hará la partida?  -> tabla por partida, etiqueta numérica
  3. ¿Qué obstáculo viene después?     -> tabla por frame, etiqueta categórica

Uso:
  python simular_dataset.py                    # solo imprime el resumen
  python simular_dataset.py --csv carpeta      # además guarda frames.csv y partidas.csv
"""

import argparse
import csv
import os
import random
import statistics
from collections import Counter

FRAME_MS = 16                 # un frame cada 16 ms (~60 fps)
DINO_X, DINO_W = 50, 44       # posición y ancho del dinosaurio
DINO_H, DINO_H_AGACHADO = 47, 26
GRAVEDAD, IMPULSO = 0.6, 10.0  # física del salto (px/frame)

# tipo: (ancho, alto, altura sobre el suelo)
OBSTACULOS = {
    "cactus_chico":  (17, 35, 0),
    "cactus_grande": (25, 50, 0),
    "pajaro_medio":  (46, 30, 30),   # hay que agacharse o saltar
    "pajaro_alto":   (46, 30, 60),   # pasa por arriba: no hace falta hacer nada
}
PESOS_TIPO = [0.40, 0.30, 0.18, 0.12]
LIMITE_FRAMES = 15000          # tope de seguridad (~4 minutos)


def choca(dino_y, alto_dino, obs):
    """Colisión de cajas: se traslapan en X y en Y."""
    ancho, alto, elev = OBSTACULOS[obs["tipo"]]
    en_x = obs["x"] < DINO_X + DINO_W and obs["x"] + ancho > DINO_X
    # y = altura sobre el suelo; el dino ocupa [dino_y, dino_y + alto_dino]
    en_y = dino_y < elev + alto and dino_y + alto_dino > elev
    return en_x and en_y


def jugar(partida, rng):
    """Simula una partida con un 'jugador' imperfecto. Devuelve filas y resumen."""
    filas = []
    vel, dist = 6.0, 0.0
    dino_y, vy, agachado = 0.0, 0.0, False
    obstaculos = []
    siguiente_aparicion = 60
    superados = 0
    superados_10s = None          # obstáculos superados al llegar a los 10 s
    # Reflejos del jugador: cambia por partida para tener variedad.
    reaccion = rng.uniform(90, 160)
    descuido = rng.uniform(0.02, 0.12)   # prob. de no ver un obstáculo

    frame = 0
    while True:
        # --- aparecen obstáculos ---
        if frame >= siguiente_aparicion:
            tipo = rng.choices(list(OBSTACULOS), PESOS_TIPO)[0]
            # el jugador decide desde que aparece si lo va a ignorar (descuido)
            obstaculos.append({"tipo": tipo, "x": 650.0,
                               "ignorado": rng.random() < descuido})
            siguiente_aparicion = frame + rng.randint(45, 110)

        # --- obstáculo más cercano delante del dino (variables de entrada) ---
        delante = [o for o in obstaculos if o["x"] + OBSTACULOS[o["tipo"]][0] > DINO_X]
        prox = min(delante, key=lambda o: o["x"]) if delante else None
        distancia = round(prox["x"] - (DINO_X + DINO_W), 1) if prox else None

        # --- decisión del jugador ---
        if prox and dino_y == 0 and distancia is not None and distancia < reaccion * vel / 8:
            if not prox["ignorado"]:
                if prox["tipo"] == "pajaro_medio":
                    agachado = True
                elif prox["tipo"] != "pajaro_alto":
                    vy = IMPULSO
        if prox is None or (prox["tipo"] != "pajaro_medio"):
            agachado = False

        filas.append({
            "partida": partida,
            "frame": frame,
            "tiempo_ms": frame * FRAME_MS,
            "score": int(dist * 0.025),
            "velocidad": round(vel, 3),
            "dino_y": round(dino_y, 1),
            "agachado": int(agachado),
            "distancia_obstaculo": distancia,
            "obstaculo_tipo": prox["tipo"] if prox else "ninguno",
            "obstaculo_altura": OBSTACULOS[prox["tipo"]][2] if prox else None,
            "muere_siguiente_frame": 0,
            "siguiente_obstaculo": None,   # se llena al final
        })

        # --- avanzar un frame ---
        frame += 1
        dist += vel
        vel = min(13.0, vel + 0.002)
        if dino_y > 0 or vy > 0:
            dino_y = max(0.0, dino_y + vy)
            vy -= GRAVEDAD
            if dino_y == 0:
                vy = 0.0
        for o in obstaculos:
            o["x"] -= vel
        antes = len(obstaculos)
        obstaculos = [o for o in obstaculos if o["x"] + OBSTACULOS[o["tipo"]][0] > 0]
        superados += antes - len(obstaculos)
        if superados_10s is None and frame * FRAME_MS >= 10000:
            superados_10s = superados

        alto = DINO_H_AGACHADO if agachado else DINO_H
        if any(choca(dino_y, alto, o) for o in obstaculos):
            filas[-1]["muere_siguiente_frame"] = 1
            break
        if frame >= LIMITE_FRAMES:      # partida cortada: no cuenta como muerte
            break

    # Escenario 3: el obstáculo que aparecerá después del actual.
    secuencia = [f["obstaculo_tipo"] for f in filas]
    proximo = "ninguno"
    for i in range(len(filas) - 1, -1, -1):
        filas[i]["siguiente_obstaculo"] = proximo
        if i > 0 and secuencia[i - 1] != secuencia[i] and secuencia[i] != "ninguno":
            proximo = secuencia[i]

    resumen = {
        "partida": partida,
        "frames": frame,
        "duracion_s": round(frame * FRAME_MS / 1000, 2),
        "velocidad_final": round(vel, 3),
        "obstaculos_superados": superados,
        "puntos_finales": filas[-1]["score"],
        "murio": filas[-1]["muere_siguiente_frame"],
        "superados_a_los_10s": superados_10s,
    }
    return filas, resumen


def guardar(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidas", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--csv", help="carpeta donde guardar frames.csv y partidas.csv")
    args = ap.parse_args()

    rng = random.Random(args.semilla)
    frames, partidas = [], []
    for p in range(1, args.partidas + 1):
        f, r = jugar(p, rng)
        frames += f
        partidas.append(r)

    print(f"Partidas simuladas: {len(partidas)} | filas por frame: {len(frames)}")

    # Escenario 1: balance de clases
    pos = sum(f["muere_siguiente_frame"] for f in frames)
    print("\n[1] Morira en el siguiente frame")
    print(f"  positivos: {pos} de {len(frames)} ({100 * pos / len(frames):.3f} %)")
    print("  -> clases muy desbalanceadas: 1 positivo por partida.")

    # Escenario 2: puntos por partida
    puntos = [r["puntos_finales"] for r in partidas]
    print("\n[2] Puntos al morir (una fila por partida)")
    print(f"  min={min(puntos)}  media={statistics.mean(puntos):.1f}  max={max(puntos)}")
    dur = [r["duracion_s"] for r in partidas]
    print(f"  correlacion duracion vs puntos: {statistics.correlation(dur, puntos):.3f}")
    print("  -> la duracion final 'explica' los puntos: es fuga de informacion (leakage).")
    print(f"  partidas cortadas por el tope (sin muerte): {sum(1 - r['murio'] for r in partidas)}")
    # Variable 'honesta': lo que se sabe a los 10 s, sin ver el final de la partida.
    # El score a los 10 s no sirve: con velocidad fija depende solo del tiempo (es constante).
    tempranos = [(r["superados_a_los_10s"], r["puntos_finales"]) for r in partidas
                 if r["superados_a_los_10s"] is not None]
    if len(tempranos) > 2 and len(set(a for a, _ in tempranos)) > 1:
        a, b = zip(*tempranos)
        print(f"  partidas que llegaron a 10 s: {len(tempranos)}; correlacion "
              f"obstaculos superados a los 10 s vs puntos: {statistics.correlation(a, b):.3f}")

    # Escenario 3: distribución de obstáculos
    print("\n[3] Tipo de obstaculo mas cercano (por frame)")
    for tipo, n in Counter(f["obstaculo_tipo"] for f in frames).most_common():
        print(f"  {tipo:14s} {n:7d} ({100 * n / len(frames):5.1f} %)")
    muertes = Counter(f["obstaculo_tipo"] for f in frames if f["muere_siguiente_frame"])
    print("  obstaculo con el que se murio:", dict(muertes))

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        guardar(os.path.join(args.csv, "frames.csv"), frames)
        guardar(os.path.join(args.csv, "partidas.csv"), partidas)
        print(f"\nCSV guardados en {args.csv}")


if __name__ == "__main__":
    main()
