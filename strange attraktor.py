import numpy as np
import pygame
from pygame.locals import *
import pygame.midi
import colorsys
from dataclasses import dataclass
import threading
import time

@dataclass
class AttractorParams:
    a: float = 10.0
    b: float = 28.0
    c: float = 8/3
    dt: float = 0.01
    
class ChaosMusicArtSystem:
    def __init__(self, width=1200, height=800):
        # Pygame und MIDI Setup
        pygame.init()
        pygame.midi.init()
        self.midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())
        
        # Fenster-Setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Chaos Music Art System')
        
        # Attractor-System Initialisierung
        self.attractors = []
        self.traces = []
        self.max_trace_length = 1000
        self.params = AttractorParams()
        
        # Musik-Parameter
        self.base_note = 60  # Mittleres C
        self.scale = [0, 2, 4, 5, 7, 9, 11]  # Dur-Skala
        
    def calculate_attractor(self, x, y, z, params):
        """Berechnet den nächsten Punkt für einen modifizierten Lorenz-Attraktor"""
        dx = params.a * (y - x) + np.sin(z)
        dy = x * (params.b - z) - y + np.cos(x)
        dz = x * y - params.c * z + np.sin(x*y)
        return dx, dy, dz
    
    def update_position(self, pos, params):
        """Aktualisiert die Position eines Attraktors"""
        x, y, z = pos
        dx, dy, dz = self.calculate_attractor(x, y, z, params)
        x += dx * params.dt
        y += dy * params.dt
        z += dz * params.dt
        return np.array([x, y, z])
    
    def get_color_from_velocity(self, velocity):
        """Erzeugt Farben basierend auf der Geschwindigkeit des Punktes"""
        speed = np.linalg.norm(velocity)
        hue = (speed % 50) / 50  # Zyklische Farbänderung
        return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
    
    def point_to_note(self, point):
        """Wandelt einen 3D-Punkt in MIDI-Noten und Velocity um"""
        x, y, z = point
        octave = int((z + 30) / 10) % 4
        scale_pos = int((x + 30) % len(self.scale))
        note = self.base_note + self.scale[scale_pos] + (12 * octave)
        velocity = min(127, max(30, int((y + 30) * 2)))
        return note, velocity
    
    def play_music(self):
        """Spielt Musik basierend auf den Attraktoren"""
        while True:
            if self.attractors:
                for attractor in self.attractors:
                    note, velocity = self.point_to_note(attractor[-1])
                    self.midi_out.note_off(note)
                    self.midi_out.note_on(note, velocity)
                    time.sleep(0.1)
    
    def run(self):
        """Hauptschleife des Systems"""
        music_thread = threading.Thread(target=self.play_music, daemon=True)
        music_thread.start()
        
        clock = pygame.time.Clock()
        running = True
        dragging = False
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    # Neuer Attraktor beim Klicken
                    x, y = event.pos
                    self.attractors.append(np.array([x/100 - 5, y/100 - 5, 0]))
                    self.traces.append([])
                elif event.type == KEYDOWN:
                    # Parameter-Steuerung
                    if event.key == K_a:
                        self.params.a += 1.0
                    elif event.key == K_s:
                        self.params.a -= 1.0
                    elif event.key == K_d:
                        self.params.b += 1.0
                    elif event.key == K_f:
                        self.params.b -= 1.0
            
            # Update und Zeichnen
            self.screen.fill((0, 0, 0))
            
            # Attraktoren aktualisieren
            for i, attractor in enumerate(self.attractors):
                new_pos = self.update_position(attractor, self.params)
                velocity = new_pos - attractor
                color = self.get_color_from_velocity(velocity)
                
                # Spur speichern
                self.traces[i].append((new_pos, color))
                if len(self.traces[i]) > self.max_trace_length:
                    self.traces[i].pop(0)
                
                # Spuren zeichnen
                for j in range(1, len(self.traces[i])):
                    pos1, col1 = self.traces[i][j-1]
                    pos2, col2 = self.traces[i][j]
                    pygame.draw.line(self.screen, col2,
                                  ((pos1[0] + 5) * 100, (pos1[1] + 5) * 100),
                                  ((pos2[0] + 5) * 100, (pos2[1] + 5) * 100), 2)
                
                self.attractors[i] = new_pos
            
            # Parameter-Anzeige
            font = pygame.font.Font(None, 36)
            text = f"a: {self.params.a:.1f} b: {self.params.b:.1f} c: {self.params.c:.1f}"
            text_surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        pygame.midi.quit()

if __name__ == "__main__":
    system = ChaosMusicArtSystem()
    system.run()