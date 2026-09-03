import params as p
import numpy as np
# import matplotlib.pyplot as plt
from basic_environment import rand_robo_placement


def initialize_robots(number_of_robots, arena_size, rng):
    positions = rand_robo_placement(number_of_robots, arena_size, p.rng)
    states = np.full(number_of_robots, p.UNDECIDED, dtype=int)
    return positions, states 
def pheromone_gradient(p_grid, grid_unit):
    gradient_x, gradient_y = np.gradient(p_grid, grid_unit)
    return gradient_x, gradient_y
def convert_to_int_grid(positions, grid_unit, grid_size):
    index = (positions / grid_unit).astype(int)
    conv = np.clip(index, 0, grid_size-1)
    return conv
def committed_mask(states, support_value):
        
    return states==support_value

def random_walk_step(positions, states, p_grid, support_value,step_size, arena_size, gradient_bais, rng):
    n = positions.shape[0]
    angles=rng.uniform(0,2*np.pi, size=n)
    random_dx = np.cos(angles)
    random_dy = np.sin(angles)
    dx = random_dx.copy()
    dy = random_dy.copy()
    # committed_mask= []
    # for s in states:
    #     matched = (s==p.SUPPORT_A)
    #     committed_mask.append(matched)
        
    com_mask = committed_mask(states, support_value)
    if np.any(com_mask) and gradient_bais > 0:
        grad_x, grad_y = pheromone_gradient(p_grid, p.GRID_UNIT)
        index = convert_to_int_grid(positions[com_mask], p.GRID_UNIT, p.SIZE_OF_GRID)
        gx = grad_x[index[:, 0], index[:, 1]]
        gy = grad_y[index[:, 0], index[:, 1]]
        normalize = np.sqrt(gx ** 2 + gy **2)
        has_gradient = np.greater(normalize, 1e-8)            
        gx_unit = np.where(has_gradient, gx / np.where(normalize > 0, normalize, 1), 0)
        gy_unit = np.where(has_gradient, gy / np.where(normalize > 0, normalize, 1), 0)
        eff_bias = np.where(has_gradient, gradient_bais, 0.0)
        blended_dx = (1-eff_bias) * random_dx[com_mask] + eff_bias * gx_unit
        blended_dy = (1-eff_bias) * random_dy[com_mask] + eff_bias * gy_unit
        renormalize = np.sqrt(blended_dx ** 2 + blended_dy ** 2)
        renormalize = np.where(renormalize > 0, renormalize,1)
        dx[com_mask]=blended_dx/renormalize
        dy[com_mask]=blended_dy/renormalize
    new_pos = positions + step_size * np.stack([dx, dy], axis=1)
    rws= np.clip(new_pos,0,arena_size)
    return rws


def deposit_pheromone(p_grid, positions, states, support_value, deposit_amount):
    com_mask = committed_mask(states, support_value)
    if not np.any(com_mask):
        return None
    indices = convert_to_int_grid(positions[com_mask], p.GRID_UNIT,p.SIZE_OF_GRID)
    for index_x, index_y in indices:
        p_grid[index_x, index_y] += deposit_amount
    return deposit_amount

def sense_local_pheromone(p_grid, positions, sensing_radius):
    radius_units = max(1, int(round(sensing_radius/p.GRID_UNIT)))
    index = convert_to_int_grid(positions, p.GRID_UNIT, p.SIZE_OF_GRID)
    sensed = np.zeros(positions.shape[0])
    for i, (index_x,index_y) in enumerate(index):
        x_low, x_high = max(0,index_x - radius_units), min(p.SIZE_OF_GRID, index_x + radius_units +1)
        y_low, y_high = max(0,index_y- radius_units), min(p.SIZE_OF_GRID, index_y+radius_units +1) 
        neighbourhood = p_grid[x_low:x_high, y_low:y_high]
        if neighbourhood.size > 0:
            sensed[i] = neighbourhood.mean()
        else:
            0.0
    return sensed

def discover(states, positions, site_position, detection_radius, quality,support_value, rng):
    n=states.shape[0]
    distance_to_site = np.linalg.norm(positions - site_position, axis=1)
    in_range= distance_to_site <= detection_radius
    undecided = np.equal(states, p.UNDECIDED)
    discover_candidates = undecided & in_range
    discover_rolls = rng.uniform(0, 1, size=n)
    newly_discovered = discover_candidates & (discover_rolls < quality)
    states[newly_discovered] = support_value
    return states
def recruit(states, positions, p_grid, sensing_radius, recruitment_scale,support_value, rng):
    n=states.shape[0]
    
    still_undecided = np.equal(states, p.UNDECIDED)
    if np.any(still_undecided):
        sensed = sense_local_pheromone(p_grid, positions, sensing_radius)
        adopt_prob = np.clip(recruitment_scale * sensed, 0, 1)
        recruit_rolls = rng.uniform(0,1,size=n)
        newly_recruited = still_undecided & (recruit_rolls < adopt_prob)
        states[newly_recruited] = support_value
    return states
    
def abandon(states, support_value, abandon_rate, rng):
    n=states.shape[0]
    committed = np.equal(states, support_value)
    abandon_rolls = rng.uniform(0,1,size=n)
    abandoing = committed & (abandon_rolls < abandon_rate)
    states[abandoing] = p.UNDECIDED
    return states

def update(states, positions, p_grid, rng):
    states=discover(states, positions, p.RESOURCE_SITE_A, p.SENSING_RADIUS,p.QUALITY_A, p.SUPPORT_A, rng)
    states = recruit(states, positions, p_grid, p.SENSING_RADIUS,p.RECRUITMENT_SCALE, p.SUPPORT_A, rng)
    states = abandon(states, p.SUPPORT_A, p.ABANDON_RATE, rng)
    return states
def run_pheronome_environment():
    positions, states = initialize_robots(p.NUMBER_OF_ROBOTS, p.ARENA_SIZE, p.rng)
    p_grid = np.zeros((p.SIZE_OF_GRID, p.SIZE_OF_GRID))
    fraction_over_time_A = np.zeros(p.NUMBER_OF_STEPS)
    total_pheromone_over_time = np.zeros(p.NUMBER_OF_STEPS)
    for t in range(p.NUMBER_OF_STEPS):
        positions = random_walk_step(positions, states, p_grid, p.SUPPORT_A, p.STEP_SIZE, p.ARENA_SIZE, p.GRADIENT_BIAS, p.rng)
        states = update(states, positions, p_grid, p.rng)
        dp = deposit_pheromone(p_grid, positions, states, p.SUPPORT_A, p.DEPOSIT_AMOUNT_A)
        p_grid *= (1 - p.EVAPORATION_RATE)
        fraction_over_time_A[t] = np.mean(states==p.SUPPORT_A)
        total_pheromone_over_time[t] = p_grid.sum()
    return fraction_over_time_A, total_pheromone_over_time, p_grid, positions, states
    
        
    