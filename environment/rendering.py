#!/usr/bin/env python3
"""
Visualization GUI components for warehouse environment
"""
import pygame
import math

def render_warehouse(env):
    """Render the warehouse environment with enhanced visuals"""
    if not hasattr(env, 'window') or env.window is None:
        pygame.init()
        pygame.display.init()
        
        try:
            env.window = pygame.display.set_mode((env.grid_size * 80 + 250, env.grid_size * 80 + 100))
            pygame.display.set_caption("🏭 Warehouse Robot")
            env.clock = pygame.time.Clock()
        except pygame.error:
            return None
    
    # Colors
    DARK_BLUE = (20, 30, 60)
    LIGHT_BLUE = (100, 150, 255)
    ROBOT_BLUE = (0, 100, 255)
    GOLD = (255, 215, 0)
    EMERALD = (0, 255, 127)
    RUBY = (255, 20, 60)
    SILVER = (192, 192, 192)
    WHITE = (255, 255, 255)
    
    cell_size = 80
    
    # Background gradient
    for y in range(env.grid_size * cell_size + 100):
        color_ratio = y / (env.grid_size * cell_size + 100)
        r = int(DARK_BLUE[0] * (1 - color_ratio) + LIGHT_BLUE[0] * color_ratio)
        g = int(DARK_BLUE[1] * (1 - color_ratio) + LIGHT_BLUE[1] * color_ratio)
        b = int(DARK_BLUE[2] * (1 - color_ratio) + LIGHT_BLUE[2] * color_ratio)
        pygame.draw.line(env.window, (r, g, b), (0, y), (env.grid_size * cell_size + 250, y))
    
    # Draw grid
    for x in range(env.grid_size + 1):
        for y in range(env.grid_size + 1):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(env.window, SILVER, rect, 2)
    
    # Draw obstacles
    for obs in env.obstacles:
        center_x = obs[0] * cell_size + cell_size // 2
        center_y = obs[1] * cell_size + cell_size // 2
        size = int(cell_size * 0.8)
        
        obstacle_rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
        pygame.draw.rect(env.window, RUBY, obstacle_rect)
        pygame.draw.rect(env.window, WHITE, obstacle_rect, 3)
    
    # Draw items
    for item in env.items:
        center_x = item[0] * cell_size + cell_size // 2
        center_y = item[1] * cell_size + cell_size // 2
        
        pygame.draw.circle(env.window, GOLD, (center_x, center_y), cell_size // 3)
        pygame.draw.circle(env.window, WHITE, (center_x, center_y), cell_size // 4)
    
    # Draw targets
    for target in env.targets:
        center_x = target[0] * cell_size + cell_size // 2
        center_y = target[1] * cell_size + cell_size // 2
        
        for ring in range(3):
            radius = cell_size // 3 + ring * 5
            pygame.draw.circle(env.window, EMERALD, (center_x, center_y), radius, 3)
    
    # Draw robot
    robot_center_x = env.robot_pos[0] * cell_size + cell_size // 2
    robot_center_y = env.robot_pos[1] * cell_size + cell_size // 2
    
    pygame.draw.circle(env.window, ROBOT_BLUE, (robot_center_x, robot_center_y), cell_size // 2)
    pygame.draw.circle(env.window, LIGHT_BLUE, (robot_center_x, robot_center_y), cell_size // 3)
    pygame.draw.circle(env.window, WHITE, (robot_center_x - 8, robot_center_y - 8), cell_size // 6)
    
    # Carrying indicator
    if env.carrying > 0:
        for i in range(env.carrying):
            carry_x = robot_center_x + (i - env.carrying/2) * 15
            carry_y = robot_center_y - 25
            pygame.draw.circle(env.window, GOLD, (int(carry_x), int(carry_y)), 8)
    
    # UI Panel
    ui_x = env.grid_size * cell_size + 20
    font_large = pygame.font.Font(None, 32)
    font_medium = pygame.font.Font(None, 24)
    
    title = font_large.render("🏭 WAREHOUSE ROBOT", True, WHITE)
    env.window.blit(title, (ui_x, 20))
    
    stats = [
        f"🔋 Steps: {env.steps}/{env.max_steps}",
        f"📦 Carrying: {env.carrying}/{env.max_inventory}",
        f"🎯 Delivered: {env.delivered_items}",
        f"📍 Items Left: {len(env.items)}",
        f"⚡ Status: {'LOADED' if env.carrying else 'ACTIVE'}"
    ]
    
    for i, stat in enumerate(stats):
        color = EMERALD if "ACTIVE" in stat or "LOADED" in stat else WHITE
        text = font_medium.render(stat, True, color)
        env.window.blit(text, (ui_x, 70 + i * 30))
    
    # Progress bar
    progress = (env.max_steps - env.steps) / env.max_steps
    bar_width = 180
    bar_height = 20
    
    pygame.draw.rect(env.window, (50, 50, 50), (ui_x, 220, bar_width, bar_height))
    
    fill_width = int(bar_width * progress)
    color = EMERALD if progress > 0.5 else GOLD if progress > 0.2 else RUBY
    pygame.draw.rect(env.window, color, (ui_x, 220, fill_width, bar_height))
    
    pygame.draw.rect(env.window, WHITE, (ui_x, 220, bar_width, bar_height), 2)
    
    pygame.display.flip()
    env.clock.tick(8)
    
    return None