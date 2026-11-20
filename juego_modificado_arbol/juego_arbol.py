"""
Juego de la bala y el salto controlado por un ÁRBOL DE DECISIÓN.

Autor: Diego Ignacio Moreno Zapien

Modo MANUAL: tú saltas con ESPACIO y el juego guarda, en cada frame en que el
            muñeco está en el suelo, (rapidez_bala, distancia, ¿saltaste?).
Entrenar:   con esos datos se construye un árbol de decisión (arbol_decision.py).
Modo AUTO:  el árbol decide en cada frame si saltar o no.

Teclas en el menú:
  M  jugar en manual (borra los datos y el modelo anteriores)
  A  jugar en automático (necesita un árbol entrenado)
  T  entrenar el árbol con los datos guardados
  R  mostrar las reglas del árbol (en la consola y en reglas_arbol.txt)
  C  exportar los datos a datos_arbol.csv
  L  cargar datos desde datos_arbol.csv
  Q  salir
Durante el juego: ESPACIO = saltar (manual), ESC = volver al menú.

Requiere: pygame y numpy  (pip install pygame numpy)
"""

from __future__ import annotations

import csv
import os
import sys
from typing import List, Optional

import pygame

# Permite importar los módulos vecinos aunque se ejecute con "python -I".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbol_decision import ArbolDecision, dividir_entrenamiento_prueba
from mundo import (ALTO, ANCHO, BALA_H, BALA_W, BALA_X0, BALA_Y, JUGADOR_H,
                   JUGADOR_W, JUGADOR_X, NOMBRES_CARACTERISTICAS, SUELO_Y,
                   Muestra, Mundo)

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA, "datos_arbol.csv")
RUTA_REGLAS = os.path.join(CARPETA, "reglas_arbol.txt")
MIN_MUESTRAS = 200   # mínimo de datos para entrenar
FPS = 45


# ---------------------------------------------------------------- datos / modelo
def entrenar_arbol(datos: List[Muestra]):
    """Entrena y devuelve (arbol, mensaje). arbol es None si no se pudo."""
    if len(datos) < MIN_MUESTRAS:
        return None, f"Faltan datos: {len(datos)}/{MIN_MUESTRAS}. Juega en MANUAL."
    X = [[m.rapidez_bala, m.distancia] for m in datos]
    y = [m.salto for m in datos]
    saltos = sum(y)
    if saltos < 5:
        return None, f"Solo hay {saltos} saltos registrados; salta más veces (>= 5)."
    X_ent, X_pru, y_ent, y_pru = dividir_entrenamiento_prueba(X, y, prueba=0.2)
    # Los saltos son muy pocos frente a los "no salto" (1 por cada bala),
    # por eso se balancean las clases al entrenar.
    arbol = ArbolDecision(profundidad_max=5, min_muestras_hoja=5, balancear_clases=True)
    arbol.entrenar(X_ent, y_ent)
    acc = arbol.exactitud(X_pru, y_pru)
    return arbol, (f"Árbol listo: {arbol.num_hojas()} hojas, profundidad {arbol.profundidad()}, "
                   f"exactitud en prueba = {acc:.3f}")


def exportar_csv(datos: List[Muestra]) -> str:
    if not datos:
        return "No hay datos para exportar."
    with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rapidez_bala", "distancia", "salto"])
        for m in datos:
            w.writerow([m.rapidez_bala, m.distancia, m.salto])
    return f"Guardado datos_arbol.csv ({len(datos)} filas)."


def cargar_csv() -> Optional[List[Muestra]]:
    if not os.path.exists(RUTA_CSV):
        return None
    with open(RUTA_CSV, newline="", encoding="utf-8") as f:
        return [Muestra(float(r["rapidez_bala"]), float(r["distancia"]), int(r["salto"]))
                for r in csv.DictReader(f)]


# ------------------------------------------------------------------- juego
class Juego:
    def __init__(self) -> None:
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Bala + salto controlado por árbol de decisión")
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("Arial", 24)
        self.fuente_chica = pygame.font.SysFont("Arial", 18)

        self.mundo = Mundo()
        self.datos: List[Muestra] = []
        self.arbol: Optional[ArbolDecision] = None
        self.modo_auto = False
        self.proba_salto: Optional[float] = None
        self.fondo_x = 0
        self.frame = 0
        self._cargar_imagenes()

    def _imagen(self, nombre: str, tam, color) -> pygame.Surface:
        """Carga una imagen; si falta, usa un rectángulo de color."""
        try:
            img = pygame.image.load(os.path.join(CARPETA, "assets", nombre)).convert_alpha()
            return pygame.transform.smoothscale(img, tam)
        except (pygame.error, FileNotFoundError):
            s = pygame.Surface(tam, pygame.SRCALPHA)
            s.fill(color)
            return s

    def _cargar_imagenes(self) -> None:
        self.img_jugador = [self._imagen(f"mono_frame_{i}.png", (JUGADOR_W, JUGADOR_H), (230, 230, 230))
                            for i in range(1, 5)]
        self.img_bala = self._imagen("purple_ball.png", (BALA_W, BALA_H), (160, 120, 255))
        self.img_nave = self._imagen("ufo.png", (64, 64), (140, 255, 200))
        self.img_fondo = self._imagen("fondo.jpg", (ANCHO, ALTO), (40, 40, 60))

    # ---------------------------------------------------------------- menú
    def menu(self) -> bool:
        """Devuelve True para jugar y False para salir."""
        msg = ""
        while True:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type != pygame.KEYDOWN:
                    continue
                if e.key == pygame.K_q:
                    return False
                if e.key == pygame.K_m:
                    self.datos.clear()
                    self.arbol = None
                    self.modo_auto = False
                    self.mundo = Mundo()
                    return True
                if e.key == pygame.K_a:
                    if self.arbol is None:
                        msg = "Primero entrena el árbol (T)."
                    else:
                        self.modo_auto = True
                        self.mundo = Mundo()
                        return True
                if e.key == pygame.K_t:
                    self.arbol, msg = entrenar_arbol(self.datos)
                if e.key == pygame.K_r:
                    msg = self._mostrar_reglas()
                if e.key == pygame.K_c:
                    msg = exportar_csv(self.datos)
                if e.key == pygame.K_l:
                    cargados = cargar_csv()
                    if cargados is None:
                        msg = "No existe datos_arbol.csv."
                    else:
                        self.datos = cargados
                        msg = f"Cargadas {len(cargados)} filas desde datos_arbol.csv."

    def _mostrar_reglas(self) -> str:
        if self.arbol is None:
            return "No hay árbol entrenado."
        texto = self.arbol.reglas(NOMBRES_CARACTERISTICAS)
        print("\n=== Reglas del árbol (clase 1 = saltar) ===\n" + texto)
        with open(RUTA_REGLAS, "w", encoding="utf-8") as f:
            f.write(texto + "\n")
        return "Reglas impresas en la consola y guardadas en reglas_arbol.txt."

    def _dibujar_menu(self, msg: str) -> None:
        self.pantalla.fill((0, 0, 0))
        y = 60
        for linea in ["MENÚ",
                      "M - Manual (borra datos y modelo)",
                      "A - Automático (lo controla el árbol)",
                      "T - Entrenar árbol",
                      "R - Ver reglas del árbol",
                      "C - Exportar datos a CSV",
                      "L - Cargar datos desde CSV",
                      "Q - Salir"]:
            self.pantalla.blit(self.fuente.render(linea, True, (255, 255, 255)), (80, y))
            y += 40
        estado = f"Datos: {len(self.datos)} (saltos: {sum(m.salto for m in self.datos)}) | " \
                 f"Árbol: {'sí' if self.arbol else 'no'}"
        self.pantalla.blit(self.fuente_chica.render(estado, True, (200, 200, 200)), (80, y + 10))
        if msg:
            self.pantalla.blit(self.fuente_chica.render(msg, True, (255, 220, 120)), (80, y + 45))
        pygame.display.flip()

    # --------------------------------------------------------------- partida
    def decidir_auto(self) -> bool:
        if not self.mundo.en_suelo:
            return False
        proba = self.arbol.predecir_proba([self.mundo.caracteristicas()])[0]
        clases = list(self.arbol.clases)
        self.proba_salto = float(proba[clases.index(1)]) if 1 in clases else 0.0
        return self.proba_salto >= 0.5

    def jugar(self) -> bool:
        """Bucle de una partida. True = volver al menú, False = cerrar."""
        while True:
            espacio = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return True
                    if e.key == pygame.K_SPACE:
                        espacio = True

            if self.modo_auto:
                saltar = self.decidir_auto()
            else:
                saltar = espacio
                # Solo se registra cuando el muñeco puede decidir (está en el suelo)
                # y la bala todavía no lo ha pasado.
                if self.mundo.en_suelo and self.mundo.distancia > -BALA_W:
                    rapidez, distancia = self.mundo.caracteristicas()
                    self.datos.append(Muestra(rapidez, distancia, int(saltar)))

            self.mundo.paso(saltar)
            self._dibujar()
            self.reloj.tick(FPS)

    def _dibujar(self) -> None:
        # Fondo que se desplaza
        self.fondo_x = (self.fondo_x - 2) % ANCHO
        self.pantalla.blit(self.img_fondo, (self.fondo_x - ANCHO, 0))
        self.pantalla.blit(self.img_fondo, (self.fondo_x, 0))

        self.frame += 1
        img = self.img_jugador[(self.frame // 10) % 4]
        self.pantalla.blit(img, (JUGADOR_X, self.mundo.jugador_y))
        self.pantalla.blit(self.img_nave, (BALA_X0 - 30, SUELO_Y - 16))
        self.pantalla.blit(self.img_bala, (self.mundo.bala_x, BALA_Y))

        modo = "AUTO (árbol)" if self.modo_auto else "MANUAL"
        info = f"{modo} | esquivadas: {self.mundo.esquivadas} | golpes: {self.mundo.golpes}"
        if self.modo_auto and self.proba_salto is not None:
            info += f" | p(saltar) = {self.proba_salto:.2f}"
        else:
            info += f" | datos: {len(self.datos)}"
        self.pantalla.blit(self.fuente_chica.render(info, True, (255, 255, 0)), (10, 10))
        pygame.display.flip()

    def correr(self) -> None:
        while self.menu() and self.jugar():
            pass
        pygame.quit()


if __name__ == "__main__":
    Juego().correr()
