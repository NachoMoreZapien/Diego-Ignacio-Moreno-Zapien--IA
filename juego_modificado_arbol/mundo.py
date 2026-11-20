"""
Lógica del juego (sin gráficos) para el juego de la bala y el salto.

Autor: Diego Ignacio Moreno Zapien

Se separa de pygame a propósito: así la misma física la usan el juego con
ventana (juego_arbol.py) y la prueba automática sin ventana (prueba_juego.py).

Escena: un muñeco a la izquierda, una nave a la derecha que dispara una bala
horizontal a ras de suelo. El muñeco solo puede saltar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

ANCHO, ALTO = 1080, 720
SUELO_Y = ALTO - 100

JUGADOR_X = 50
JUGADOR_W, JUGADOR_H = 32, 48
BALA_W, BALA_H = 16, 16
BALA_Y = SUELO_Y + 10            # la bala va a la altura de las piernas
BALA_X0 = ANCHO - 50             # punto de salida de la bala

VEL_SALTO = 15                   # píxeles/frame al despegar
GRAVEDAD = 1                     # cuánto se reduce la velocidad cada frame
RAPIDEZ_MIN, RAPIDEZ_MAX = 6, 12  # píxeles/frame de la bala (va de derecha a izquierda)

NOMBRES_CARACTERISTICAS = ["rapidez_bala", "distancia"]


@dataclass
class Muestra:
    rapidez_bala: float   # píxeles/frame (positivo)
    distancia: float      # bala.x - jugador.x  (positivo = la bala aún viene)
    salto: int            # 1 = el jugador decidió saltar en este frame


def se_tocan(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    """Choque de dos rectángulos alineados a los ejes."""
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class Mundo:
    def __init__(self, semilla: Optional[int] = None) -> None:
        self.rng = random.Random(semilla)
        self.esquivadas = 0
        self.golpes = 0
        self.reiniciar()

    def reiniciar(self) -> None:
        self.jugador_y = SUELO_Y
        self.vel_y = 0
        self.en_suelo = True
        self.nueva_bala()

    def nueva_bala(self) -> None:
        self.bala_x = BALA_X0
        self.rapidez = self.rng.randint(RAPIDEZ_MIN, RAPIDEZ_MAX)

    # -------- lo que "ve" el modelo
    @property
    def distancia(self) -> int:
        # Con signo: si es negativa la bala ya pasó al jugador.
        return self.bala_x - JUGADOR_X

    def caracteristicas(self) -> List[float]:
        return [float(self.rapidez), float(self.distancia)]

    # -------- un frame de simulación
    def paso(self, saltar: bool) -> Optional[str]:
        """Avanza un frame. Devuelve 'golpe', 'esquivada' o None."""
        if saltar and self.en_suelo:
            self.en_suelo = False
            self.vel_y = VEL_SALTO

        # Física del salto (tiro vertical simple)
        if not self.en_suelo:
            self.jugador_y -= self.vel_y
            self.vel_y -= GRAVEDAD
            if self.jugador_y >= SUELO_Y:
                self.jugador_y = SUELO_Y
                self.en_suelo = True

        # La bala avanza hacia la izquierda
        self.bala_x -= self.rapidez

        if se_tocan(JUGADOR_X, self.jugador_y, JUGADOR_W, JUGADOR_H,
                    self.bala_x, BALA_Y, BALA_W, BALA_H):
            self.golpes += 1
            self.reiniciar()
            return "golpe"

        if self.bala_x < -BALA_W:
            self.esquivadas += 1
            self.nueva_bala()
            return "esquivada"
        return None
