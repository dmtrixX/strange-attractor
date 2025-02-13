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
    max_radius: float = 100.0
    color_scale: float = 0.5

class ChaoticAttractorVis:
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption('3D Chaos Visualization')
        self.color_offset = 0.0
        
        # Camera parameters
        self.camera_distance = 50.0
        self.min_zoom = 5.0
        self.max_zoom = 300.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.last_mouse = None
        
        # Dynamic camera parameters
        self.follow_mode = False
        self.camera_smoothing = 0.1  # Controls how smoothly the camera follows
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.current_camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.follow_distance = 30.0  # Distance from camera to followed point
        
        # Click-to-place parameters
        self.unproject_matrix = None
        self.projection_matrix = None
        self.modelview_matrix = None
        
        # Attractor parameters
        self.params = AttractorParams()
        self.trajectories = []
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

    def update_camera_follow(self):
        """Update camera position when in follow mode"""
        if not self.trajectories or not self.follow_mode:
            return

        # Get the latest point and its velocity
        current_traj = self.trajectories[-1]
        if len(current_traj['points']) < 2:
            return

        current_pos = np.array(current_traj['points'][-1])
        prev_pos = np.array(current_traj['points'][-2])
        
        # Calculate velocity direction
        velocity = current_pos - prev_pos
        velocity_norm = np.linalg.norm(velocity)
        
        if velocity_norm < 1e-6:
            return
            
        # Normalize velocity
        velocity = velocity / velocity_norm
        
        # Calculate desired camera position
        desired_camera_pos = current_pos - velocity * self.follow_distance
        
        # Smooth camera movement
        self.current_camera_pos += (desired_camera_pos - self.current_camera_pos) * self.camera_smoothing
        
        # Calculate up vector (try to keep it mostly vertical)
        right = np.cross(velocity, [0, 1, 0])
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
            self.camera_up = np.cross(right, velocity)
            self.camera_up = self.camera_up / np.linalg.norm(self.camera_up)

        # Update target position
        self.target_position = current_pos

    def draw(self):
        """Render the visualization"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        if self.follow_mode:
            # Set up camera for follow mode
            gluPerspective(45, (self.width / self.height), 0.1, 200.0)
            gluLookAt(
                self.current_camera_pos[0], self.current_camera_pos[1], self.current_camera_pos[2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                self.camera_up[0], self.camera_up[1], self.camera_up[2]
            )
        else:
            # Set up camera for free mode
            gluPerspective(45, (self.width / self.height), 0.1, 200.0)
            glTranslatef(self.offset_x, self.offset_y, -self.camera_distance)
            glRotatef(self.rot_x, 1, 0, 0)
            glRotatef(self.rot_y, 0, 1, 0)
        
        # Draw all attractor trails
        for traj in self.trajectories:
            if len(traj['points']) > 1:
                glBegin(GL_LINE_STRIP)
                for point, color in zip(traj['points'], traj['colors']):
                    if self.is_point_valid(point):
                        glColor3fv(color)
                        glVertex3fv(point)
                glEnd()
        
        # Draw parameters and camera mode
        mode_text = "Follow Mode" if self.follow_mode else "Free Mode"
        self.draw_text(f"a: {self.params.a:.1f} b: {self.params.b:.1f} c: {self.params.c:.1f} | {mode_text}")

    def handle_input(self):
        """Handle user input for camera control"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    if not self.follow_mode:
                        self.camera_distance = max(self.min_zoom, self.camera_distance - 5)
                    else:
                        self.follow_distance = max(self.min_zoom, self.follow_distance - 5)
                elif event.button == 5:  # Mouse wheel down
                    if not self.follow_mode:
                        self.camera_distance = min(self.max_zoom, self.camera_distance + 5)
                    else:
                        self.follow_distance = min(self.max_zoom, self.follow_distance + 5)
                elif event.button == 1:  # Left click
                    if not self.follow_mode:  # Only allow new trajectories in free mode
                        x, y = event.pos
                        if pygame.key.get_mods() & KMOD_SHIFT:  # Shift + left click for rotation
                            self.last_mouse = (x, y)
                        else:  # Regular left click for new trajectory
                            new_point = self.unproject_mouse(x, y)
                            self.add_trajectory(new_point)
                elif event.button == 3:  # Right click - model movement
                    if not self.follow_mode:
                        self.last_mouse = pygame.mouse.get_pos()
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1 and pygame.key.get_mods() & KMOD_SHIFT:
                    self.last_mouse = None
                elif event.button == 3:
                    self.last_mouse = None
            
            elif event.type == MOUSEMOTION and self.last_mouse is not None and not self.follow_mode:
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
                if event.key == K_v:  # Toggle view mode
                    self.follow_mode = not self.follow_mode
                    if self.follow_mode:
                        # Initialize follow camera position
                        if self.trajectories and len(self.trajectories[-1]['points']) > 0:
                            self.current_camera_pos = np.array(self.trajectories[-1]['points'][-1])
                            self.current_camera_pos[2] += self.follow_distance
                elif event.key == K_r:  # Reset view
                    self.rot_x = 0.0
                    self.rot_y = 0.0
                    self.offset_x = 0.0
                    self.offset_y = 0.0
                    self.camera_distance = 50.0
                    self.follow_distance = 30.0
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
        """Update the positions of all trajectories with stability checks"""
        for traj in self.trajectories:
            try:
                current_pos = traj['current_pos']
                if not self.is_point_valid(current_pos):
                    continue
                
                x, y, z = current_pos
                dx, dy, dz = self.calculate_attractor(x, y, z, self.params)
                
                new_pos = current_pos + np.array([dx, dy, dz]) * self.params.dt
                
                # Skip update if new position is invalid
                if not self.is_point_valid(new_pos):
                    continue
                
                velocity = new_pos - current_pos
                speed = np.linalg.norm(velocity)
                
                # Skip if velocity is too high
                if not np.isfinite(speed) or speed > self.params.max_radius:
                    continue
                
                # Generate color based on velocity
                hue = (speed % 10) / 10
                color = [c for c in colorsys.hsv_to_rgb(hue, 1.0, 0.8)]
                
                traj['points'].append(new_pos)
                traj['colors'].append(color)
                
                # Limit the number of points
                if len(traj['points']) > self.max_points:
                    traj['points'].pop(0)
                    traj['colors'].pop(0)
                
                traj['current_pos'] = new_pos
                
            except Exception as e:
                print(f"Error updating position: {e}")
                continue

        # Update camera if in follow mode
        if self.follow_mode:
            self.update_camera_follow()

    # Include the rest of the unchanged methods from the original code
    def is_point_valid(self, point):
        """Check if a point is within valid bounds and numerically stable"""
        if not np.all(np.isfinite(point)):  # Check for NaN and infinity
            return False
        if np.linalg.norm(point) > self.params.max_radius:
            return False
        return True
    
    def calculate_attractor(self, x, y, z, params):
        """Calculate next point for the Lorenz attractor with stability checks"""
        try:
            dx = params.a * (y - x)
            dy = x * (params.b - z) - y
            dz = x * y - params.c * z
            
            # Check for numerical stability
            if not all(np.isfinite(v) for v in (dx, dy, dz)):
                return 0.0, 0.0, 0.0
                
            # Limit maximum step size
            max_step = 100.0
            scale = min(1.0, max_step / (np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10))
            return dx * scale, dy * scale, dz * scale
            
        except Exception as e:
            print(f"Error in attractor calculation: {e}")
            return 0.0, 0.0, 0.0
    
    def add_trajectory(self, start_pos):
        """Add a new trajectory with given starting position and random direction"""
        try:
            # Ensure start_pos is valid
            if not self.is_point_valid(start_pos):
                start_pos = np.array([0.1, 0.1, 0.1])
            
            # Add small random variations to the starting position
            random_direction = np.random.randn(3)
            norm = np.linalg.norm(random_direction)
            if norm > 1e-10:  # Prevent division by zero
                random_direction = random_direction / norm
                variation = random_direction * 0.1
                new_pos = start_pos + variation
                
                # Verify new position is valid
                if not self.is_point_valid(new_pos):
                    new_pos = start_pos
            else:
                new_pos = start_pos
            
            self.trajectories.append({
                'points': [new_pos],
                'colors': [(1.0, 1.0, 1.0)],  # Start with white
                'current_pos': new_pos.copy()
            })
        except Exception as e:
            print(f"Error adding trajectory: {e}")
    
    def unproject_mouse(self, mouse_x, mouse_y):
        """Convert mouse coordinates to 3D world coordinates with safety checks"""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            win_y = viewport[3] - mouse_y
            
            near_point = gluUnProject(mouse_x, win_y, 0.0, modelview, projection, viewport)
            far_point = gluUnProject(mouse_x, win_y, 1.0, modelview, projection, viewport)
            
            if not (np.all(np.isfinite(near_point)) and np.all(np.isfinite(far_point))):
                return np.array([0.1, 0.1, 0.1])
            
            direction = np.array(far_point) - np.array(near_point)
            norm = np.linalg.norm(direction)
            
            if norm < 1e-10:  # Prevent division by zero
                return np.array([0.1, 0.1, 0.1])
                
            direction = direction / norm
            point = np.array(near_point) + direction * 20
            
            # Ensure point is within valid bounds
            if not self.is_point_valid(point):
                return np.array([0.1, 0.1, 0.1])
                
            return point
            
        except Exception as e:
            print(f"Error in unproject_mouse: {e}")
            return np.array([0.1, 0.1, 0.1])

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
