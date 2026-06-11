"""
MuJoCo Simulation Starter Code for DOBOT.
Use this script to run your DOBOT experiments in the simulator.
"""

import argparse
import mujoco as mj
import numpy as np
import os
from dobot_sim_api import (
    SimDobotAPI,
    move_to_xyz,
    move_joint_angles,
    get_pose,
    engage_suction,
    release_suction,
    stop_pump,
    move_to_home,
    HOME_CTRL
)
from dobot_mujoco.env.dobot_pick_place import DobotPickPlace

# My link length constants from the lab manual (mm)
L1 = 138.0  # Standard Base height to J2
L2 = 135.0  # Bicep length
L3 = 147.0  # Forearm length


# ==============================================================================
# PART 1: DETERMINING FORWARD KINEMATICS (STEPS 1 & 2)
# ==============================================================================

def get_true_j3(j2_deg: float, pos6_deg: float) -> float:
    """Step 1A: Finding the true J3 angle by isolating it from the absolute forearm angle"""
    return pos6_deg - j2_deg

def get_forearm_angle(j2_deg: float, j3_true_deg: float) -> float:
    """Step 1B: Reversing the math to get the absolute angle needed for the robot's API"""
    return j3_true_deg + j2_deg

def calculate_forward_kinematics(j1_rad: float, j2_rad: float, j3_true_rad: float, L_eff_x: float, C_z: float) -> np.ndarray:
    """
    Step 2: My forward kinematics equations based on the 2D planar projection.
    """
    pos6_rad = j2_rad + j3_true_rad
    
    R = L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad) + L_eff_x
    
    # SIGN FLIP FIX: Subtracting L3*sin because a positive pos6 points DOWN to the table
    Z = C_z + L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad)
    
    X = R * np.cos(j1_rad)
    Y = R * np.sin(j1_rad)
    
    return np.array([X, Y, Z])


# ==============================================================================
# PART 1: OFFLINE VALIDATION (STEP 3)
# ==============================================================================

def run_offline_validation(filepath: str):
    print("\n" + "="*50)
    print(" PART 1 / STEP 3: OFFLINE DATA VALIDATION")
    print("="*50)
    
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find '{filepath}'.")
        return

    max_error = 0.0
    line_count = 0
    calibrated = False
    
    ta_L_eff_x = 0.0  
    ta_C_z = 0.0      

    with open(filepath, 'r') as file:
        for line in file:
            if not line.strip() or line.lower().startswith('x'):
                continue
                
            try:
                parts = [float(val) for val in line.replace(',', ' ').split()]
                if len(parts) < 6:
                    continue
                    
                target_x, target_y, target_z, j1_val, j2_val, pos6_val = parts[:6]
                
                j1_rad = np.radians(j1_val)
                j2_rad = np.radians(j2_val)
                pos6_rad = np.radians(pos6_val)
                
                if not calibrated:
                    R_target = np.hypot(target_x, target_y)
                    ta_L_eff_x = R_target - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
                    ta_C_z = target_z - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))
                    
                    print(f"Refined End-Effector X Offset from data: {ta_L_eff_x:.4f} mm")
                    print(f"Refined Base/Tool Z Offset from data: {ta_C_z:.4f} mm")
                    print("-" * 50)
                    calibrated = True
                
                j3_true_rad = pos6_rad - j2_rad
                calculated_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, ta_L_eff_x, ta_C_z)
                
                error = np.linalg.norm(np.array([target_x, target_y, target_z]) - calculated_xyz)
                if error > max_error:
                    max_error = error
                    
                line_count += 1
                
            except ValueError:
                continue 

    print(f"Successfully processed {line_count} lines of data.")
    print(f"Maximum Forward Kinematics Error: {max_error:.6f} mm")
    
    if max_error < 0.1:
        print("PASS! the math is sound.")
    else:
        print("FAIL. Error is still too high.")


# ==============================================================================
# PART 2: THE ROBOT'S WORKSPACE (STEP 5)
# ==============================================================================

def is_safe_move(start_xyz, target_xyz):
    # My exact workspace boundaries from Lab 1
    x1, y1, z1 = start_xyz
    x2, y2, z2 = target_xyz
    num_steps = 10 
    
    for i in range(num_steps + 1):
        fraction = i / num_steps
        current_x = x1 + (x2 - x1) * fraction
        current_y = y1 + (y2 - y1) * fraction
        current_z = z1 + (z2 - z1) * fraction
        
        if current_z > 0 or current_z < -120:
            return False
        if current_x < 0:
            return False
            
        radius = np.hypot(current_x, current_y)
        if radius < 140 or radius > 260:
            return False
            
    return True

def is_safe_joint_move(current_xyz, target_j1, target_j2, target_pos6, L_eff_x, C_z):
    # Takes current Cartesian position and desired joint angles.
    # Computes the theoretical target XYZ using Forward Kinematics, 
    # and verifies the 10-step path using the Lab 1 workspace validation.
    j1_rad = np.radians(target_j1)
    j2_rad = np.radians(target_j2)
    pos6_rad = np.radians(target_pos6)
    
    j3_true_rad = pos6_rad - j2_rad
    target_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, L_eff_x, C_z)
    
    return is_safe_move(current_xyz, target_xyz)


# ==============================================================================
# SIMULATOR INITIALIZATION
# ==============================================================================

def create_sim_api(seed: int, headless: bool) -> SimDobotAPI:
    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=seed)
    viewer = None
    if not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    dobot_body_id = env.model.body("dobot").id
    base_pos_mm = env.data.xpos[dobot_body_id].copy() * 1000.0
    api = SimDobotAPI(env=env, viewer=viewer, home_pos=np.zeros(3, dtype=np.float64), base_pos_mm=base_pos_mm)
    api.home_pos = api.current_xyz_mm()
    return api

def initialize_robot(api: SimDobotAPI) -> None:
    api.suction_on = False
    api.env.suction_activated = False
    api.env.data.ctrl[:4] = HOME_CTRL
    api.env.data.ctrl[4] = 0.0
    for step in range(250):
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
    api.home_pos = api.current_xyz_mm()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT Simulation Starter Script.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    # --- PART 1 EXECUTION ---
    run_offline_validation("Lab2DesignData.txt")

    print("\n" + "="*50)
    print(" LIVE SIMULATOR CALIBRATION")
    print("="*50)
    
    api = create_sim_api(seed=args.seed, headless=args.headless)

    try:
        initialize_robot(api)
        
        pose = get_pose(api)
        actual_xyz = pose[:3]
        
        j1_rad, j2_rad, pos6_rad = np.radians(pose[4]), np.radians(pose[5]), np.radians(pose[6])
        
        sim_R_actual = np.hypot(actual_xyz[0], actual_xyz[1])
        sim_L_eff_x = sim_R_actual - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
        sim_C_z = actual_xyz[2] - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))
        
        j3_true_rad = pos6_rad - j2_rad
        calculated_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, sim_L_eff_x, sim_C_z)
        
        print(f"Simulator's Actual XYZ:  [{actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}] mm")
        print(f"My Calculated FK XYZ:    [{calculated_xyz[0]:.4f}, {calculated_xyz[1]:.4f}, {calculated_xyz[2]:.4f}] mm")
        print(f"\n(Simulator internals use X-Offset: {sim_L_eff_x:.4f} mm, Z-Offset: {sim_C_z:.4f} mm)")

        # --- PART 2 EXECUTION ---
        print("\n" + "="*50)
        print(" PART 2 / STEP 4: WORKSPACE EXPLORATION")
        print("="*50)
        
        edge_cases = [
            [260, 0, 0,      "Max Radius, Max Height"],
            [260, 0, -120,   "Max Radius, Min Height"],
            [140, 0, 0,      "Min Radius, Max Height"],
            [140, 0, -120,   "Min Radius, Min Height"],
            [0, 200, 0,      "X-Boundary Limit (Y-Axis Reach)"]
        ]
        
        for case in edge_cases:
            target_x, target_y, target_z, description = case
            move_to_xyz(api, target_x, target_y, target_z)
            
            p = get_pose(api)
            print(f"Edge Case: {description}")
            print(f"  Cartesian Target: [X: {target_x}, Y: {target_y}, Z: {target_z}]")
            print(f"  Resulting Angles: [J1: {p[4]:.2f}°, J2: {p[5]:.2f}°, POS6: {p[6]:.2f}°]\n")

        # --- PART 3 EXECUTION ---
        print("\n" + "="*50)
        print(" PART 3 / STEP 6: HARDWARE VALIDATION TESTS")
        print("="*50)
        
        test_joints = [
            [0.0, 45.0, 0.0],     
            [45.0, 60.0, 20.0],   
            [-45.0, 10.0, -10.0], 
            [90.0, 45.0, 15.0]    
        ]
        
        # Grab the current position before the loop starts
        actual_pose = get_pose(api)
        current_xyz = actual_pose[:3]
        
        for idx, target_joints in enumerate(test_joints):
            j1_target, j2_target, pos6_target = target_joints
            
            print(f"\n--- Test {idx+1}: Joints [{j1_target}, {j2_target}, {pos6_target}] ---")
            
            if not is_safe_joint_move(current_xyz, j1_target, j2_target, pos6_target, sim_L_eff_x, sim_C_z):
                print("WARNING: This joint configuration (or the path to it) is OUT OF BOUNDS. Skipping.")
                continue
                
            print("Path SAFE. Moving robot...")
            
            move_joint_angles(api, j1_target, j2_target, pos6_target, 0)
            
            actual_pose = get_pose(api)
            actual_xyz = actual_pose[:3]
            
            # Update current position for the next iteration of the loop
            current_xyz = actual_xyz
            
            j1_rad, j2_rad, pos6_rad = np.radians(j1_target), np.radians(j2_target), np.radians(pos6_target)
            j3_true_rad = pos6_rad - j2_rad
            theoretical_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, sim_L_eff_x, sim_C_z)
            
            error = np.linalg.norm(actual_xyz - theoretical_xyz)
            print(f"Target Angles:   {target_joints}")
            print(f"Hardware XYZ:    [{actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}] mm")
            print(f"Theoretical XYZ: [{theoretical_xyz[0]:.4f}, {theoretical_xyz[1]:.4f}, {theoretical_xyz[2]:.4f}] mm")
            print(f"Positional Error: {error:.4f} mm")

    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()

if __name__ == "__main__":
    main()