import numpy as np
import matplotlib.pyplot as plt 


ARENA_SIZE = 100.0 
NUMBER_OF_ROBOTS = 50
STEP_SIZE= 1.0
DETECTION_RADIUS = 5.0
NUMBER_OF_STEPS = 1.0
RESOURCE_SITE  = np.array([75.0, 75.0]) 
RNG_SEED = 42
rng=np.random.default_rng(RNG_SEED)

def rand_robo_placement(number_of_robots, arena_size, rng):
    return rng.uniform(0, arena_size, size=(number_of_robots, 2))
def walk(positions, step_size, arena_size, rng):
    n = positions.shape[0]
    angles = rng.uniform(0, 2 * np.pi,size = n)
    dx = step_size * np.cos(angles)
    dy = step_size * np.sin(angles)
    new_pos = positions + np.stack([dx, dy], axis = 1)
    new_pos = np.clip(new_pos, 0, arena_size)
    return new_pos
def site_detected(positions, site_position, detection_radius):
    distances = np.linalg.norm(positions - site_position, axis = 1)
    if distances <= detection_radius:
        return True
    else: 
        return False
def run_base_environment():
    positions = rand_robo_placement(NUMBER_OF_ROBOTS, ARENA_SIZE, rng)
    has_discovered = np.zeros(NUMBER_OF_ROBOTS, dtype=bool)
    found_ratio = np.zeros(NUMBER_OF_STEPS)
    trajectory = []
    for traj in range(NUMBER_OF_STEPS):
        positions = walk(positions, STEP_SIZE, ARENA_SIZE, rng)
        curr_in_range = site_detected(positions, RESOURCE_SITE, DETECTION_RADIUS)
        has_discovered = has_discovered | curr_in_range
        found_ratio[traj] = has_discovered.mean()
        trajectory.append(positions[0].copy())
    return found_ratio, np.arary(trajectory), positions        

