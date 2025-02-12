import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from dataclasses import dataclass
import colorsys

@dataclass
class AttractorParams:
    a: float = 10.0
    b: float = 28.0
    c: float = 8/3
    dt: float = 0.01

class ChaoticAttractorVis:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption('3D Chaos Visualization')
        
        # Camera parameters
        self.camera_distance = 50.0
        self.min_zoom = 5.0
        self.max_zoom = 300.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.last_mouse = None
        
        # Click-to-place parameters
        self.unproject_matrix = None
        self.projection_matrix = None
        self.modelview_matrix = None
        
        # Attractor parameters
        self.params = AttractorParams()
        self.trajectories = []  # List of lists for multiple trajectories
        self.max_points = 5000
        
        # View offset for model centering
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        # Initial trajectory
        self.add_trajectory(np.array([0.1, 0.1, 0.1]))
        
        # Setup OpenGL
        self.setup_gl()
    
    def setup_gl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5)
    
    def calculate_attractor(self, x, y, z, params):
        """Calculate next point for the Lorenz attractor"""
        dx = params.a * (y - x)
        dy = x * (params.b - z) - y
        dz = x * y - params.c * z
        return dx, dy, dz
    
    def add_trajectory(self, start_pos):
        """Add a new trajectory with given starting position and random direction"""
        # Add small random variations to the starting position
        random_direction = np.random.randn(3)
        random_direction = random_direction / np.linalg.norm(random_direction)
        variation = random_direction * 0.1
        
        # Create new starting position with variation
        new_pos = start_pos + variation
        
        self.trajectories.append({
            'points': [new_pos],
            'colors': [(1.0, 1.0, 1.0)],  # Start with white
            'current_pos': new_pos.copy()
        })
    
    def unproject_mouse(self, mouse_x, mouse_y):
        """Convert mouse coordinates to 3D world coordinates"""
        # Get matrices and viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Get window y coordinate (OpenGL has origin at bottom left)
        win_y = viewport[3] - mouse_y
        
        # Get 3D points for near and far plane
        near_point = gluUnProject(mouse_x, win_y, 0.0, modelview, projection, viewport)
        far_point = gluUnProject(mouse_x, win_y, 1.0, modelview, projection, viewport)
        
        # Get direction vector and normalize
        direction = np.array(far_point) - np.array(near_point)
        direction = direction / np.linalg.norm(direction)
        
        # Get point at a reasonable distance from camera
        point = np.array(near_point) + direction * 20
        return point

    def handle_input(self):
        """Handle user input for camera control"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    self.camera_distance = max(self.min_zoom, self.camera_distance - 5)
                elif event.button == 5:  # Mouse wheel down
                    self.camera_distance = min(self.max_zoom, self.camera_distance + 5)
                elif event.button == 1:  # Left click
                    x, y = event.pos
                    if pygame.key.get_mods() & KMOD_SHIFT:  # Shift + left click for rotation
                        self.last_mouse = (x, y)
                    else:  # Regular left click for new trajectory
                        new_point = self.unproject_mouse(x, y)
                        self.add_trajectory(new_point)
                elif event.button == 3:  # Right click - model movement
                    self.last_mouse = pygame.mouse.get_pos()
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1 and pygame.key.get_mods() & KMOD_SHIFT:
                    self.last_mouse = None
                elif event.button == 3:
                    self.last_mouse = None
            
            elif event.type == MOUSEMOTION and self.last_mouse is not None:
                x, y = pygame.mouse.get_pos()
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]
                
                if pygame.key.get_mods() & KMOD_SHIFT:  # Rotation
                    self.rot_y += dx * 0.5
                    self.rot_x += dy * 0.5
                elif pygame.mouse.get_pressed()[2]:  # Right button - model movement
                    self.offset_x += dx * 0.1
                    self.offset_y -= dy * 0.1
                
                self.last_mouse = (x, y)
            
            elif event.type == KEYDOWN:
                if event.key == K_r:  # Reset view
                    self.rot_x = 0.0
                    self.rot_y = 0.0
                    self.offset_x = 0.0
                    self.offset_y = 0.0
                    self.camera_distance = 50.0
                elif event.key == K_c:  # Clear trajectories
                    self.trajectories.clear()
                    self.add_trajectory(np.array([0.1, 0.1, 0.1]))
                elif event.key == K_a:
                    self.params.a += 1.0
                elif event.key == K_s:
                    self.params.a -= 1.0
                elif event.key == K_d:
                    self.params.b += 1.0
                elif event.key == K_f:
                    self.params.b -= 1.0
        
        return True

    def update_position(self):
        """Update the positions of all trajectories"""
        for traj in self.trajectories:
            x, y, z = traj['current_pos']
            dx, dy, dz = self.calculate_attractor(x, y, z, self.params)
            
            new_pos = traj['current_pos'] + np.array([dx, dy, dz]) * self.params.dt
            velocity = new_pos - traj['current_pos']
            
            # Generate color based on velocity
            speed = np.linalg.norm(velocity)
            hue = (speed % 10) / 10
            color = [c for c in colorsys.hsv_to_rgb(hue, 1.0, 0.8)]
            
            traj['points'].append(new_pos)
            traj['colors'].append(color)
            
            # Limit the number of points
            if len(traj['points']) > self.max_points:
                traj['points'].pop(0)
                traj['colors'].pop(0)
            
            traj['current_pos'] = new_pos

    def draw(self):
        """Render the visualization"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        gluPerspective(45, (self.width / self.height), 0.1, 200.0)
        glTranslatef(self.offset_x, self.offset_y, -self.camera_distance)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        # Draw all attractor trails
        for traj in self.trajectories:
            if len(traj['points']) > 1:
                glBegin(GL_LINE_STRIP)
                for point, color in zip(traj['points'], traj['colors']):
                    glColor3fv(color)
                    glVertex3fv(point)
                glEnd()
        
        # Draw parameters
        self.draw_text(f"a: {self.params.a:.1f} b: {self.params.b:.1f} c: {self.params.c:.1f}")
    
    def draw_text(self, text):
        """Draw text overlay"""
        pygame.display.get_surface().fill((0, 0, 0, 0))
        font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, (255, 255, 255))
        pygame.display.get_surface().blit(text_surface, (10, 10))
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_input()
            self.update_position()
            self.draw()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    vis = ChaoticAttractorVis()
    vis.run()
