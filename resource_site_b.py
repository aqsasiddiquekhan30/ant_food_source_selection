import params as p
import numpy as np
from basic_environment import rand_robo_placement
import resource_site_a as r

import resource_site_a as rsa

def random_walk_step_two_sites(positions, states, grid_A, grid_B, step_size,arena_size, gradient_bias, rng):
    pos_after_bias_A = rsa.random_walk_step(positions, states, grid_A, p.SUPPORT_A,step_size, arena_size, gradient_bias, rng)
    pos_after_bias_B = rsa.random_walk_step(positions, states, grid_A, p.SUPPORT_B,step_size, arena_size, gradient_bias, rng)
    mask_A = rsa.committed_mask(states, p.SUPPORT_A)
    mask_B = rsa.committed_mask(states, p.SUPPORT_B)
    mask_U = ~(mask_A | mask_B)
    new_pos = np.empty_like(positions)
    new_pos[mask_A]=pos_after_bias_A[mask_A]
    new_pos[mask_B]=pos_after_bias_B[mask_B]
    new_pos[mask_U]=pos_after_bias_A[mask_U]
    return new_pos

def update_two_sites(states,positions, grid_A, grid_B, rng):
    states = rsa.discover(states, positions, p.RESOURCE_SITE_A, p.DETECTION_RADIUS,
                         p.QUALITY_A, p.SUPPORT_A, rng)
    states = rsa.discover(states, positions, p.RESOURCE_SITE_B, p.DETECTION_RADIUS,
                         p.QUALITY_B, p.SUPPORT_B, rng)
 
    states = rsa.recruit(states, positions, grid_A, p.SENSING_RADIUS,
                        p.RECRUITMENT_SCALE, p.SUPPORT_A, rng)
    states = rsa.recruit(states, positions, grid_B, p.SENSING_RADIUS,
                        p.RECRUITMENT_SCALE, p.SUPPORT_B, rng)
 
    states = rsa.abandon(states, p.SUPPORT_A, p.ABANDON_RATE, rng)
    states = rsa.abandon(states, p.SUPPORT_B, p.ABANDON_RATE, rng)
    return states

def run_two_site_environment():
    positions, states = rsa.initialize_robots(p.NUMBER_OF_ROBOTS, p.ARENA_SIZE, p.rng)
    grid_A = np.zeros((p.SIZE_OF_GRID, p.SIZE_OF_GRID))
    grid_B = np.zeros((p.SIZE_OF_GRID, p.SIZE_OF_GRID))
 
    xA = np.zeros(p.NUMBER_OF_STEPS)
    xB = np.zeros(p.NUMBER_OF_STEPS)
    xU = np.zeros(p.NUMBER_OF_STEPS)
 
    for t in range(p.NUMBER_OF_STEPS):
        positions = random_walk_step_two_sites(positions, states, grid_A, grid_B,
                                                p.STEP_SIZE, p.ARENA_SIZE,
                                                p.GRADIENT_BIAS, p.rng)
        states = update_two_sites(states, positions, grid_A, grid_B, p.rng)
 
        rsa.deposit_pheromone(grid_A, positions, states, p.SUPPORT_A, p.DEPOSIT_AMOUNT_A)
        rsa.deposit_pheromone(grid_B, positions, states, p.SUPPORT_B, p.DEPOSIT_AMOUNT_B)
        grid_A *= (1 - p.EVAPORATION_RATE)
        grid_B *= (1 - p.EVAPORATION_RATE)
 
        xA[t] = np.mean(states == p.SUPPORT_A)
        xB[t] = np.mean(states == p.SUPPORT_B)
        xU[t] = np.mean(states == p.UNDECIDED)
 
    return xA, xB, xU, grid_A, grid_B, positions, states