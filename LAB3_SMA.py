"""
MuJoCo Simulation Starter Code for DOBOT.
ECE 486 Lab 3 - Inverse Kinematics Implementation (Parts 1 & 2)
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

# Pick and Place joint targets (measured from demo)
PICK_JOINTS = np.array([-24.3, 54.9, 39.9, 73.7], dtype=np.float64)
PLACE_LIFT_JOINTS = np.array([33.2, 30.9, 25.2, 40.1], dtype=np.float64)
PLACE_JOINTS = np.array([39.4, 43.2, 47.7, 38.9], dtype=np.float64)

# My link length constants from the lab manual (mm)
L1 = 138.0  # Standard Base height to J2
L2 = 135.0  # Bicep length
L3 = 147.0  # Forearm length

# ==============================================================================
# LAB 3 KINEMATICS EQUATIONS
# ==============================================================================

def calculate_forward_kinematics(j1_rad: float, j2_rad: float, j3_true_rad: float, L_eff_x: float, C_z: float) -> np.ndarray:
    """From Lab 2: Computes XYZ given joint angles."""
    pos6_rad = j2_rad + j3_true_rad
    R = L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad) + L_eff_x
    Z = C_z + L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad)
    X = R * np.cos(j1_rad)
    Y = R * np.sin(j1_rad)
    return np.array([X, Y, Z])

def calculate_inverse_kinematics(x: float, y: float, z: float, L_eff_x: float, C_z: float) -> np.ndarray:
    """Lab 3 Part 1: Computes Joint Angles given XYZ."""
    j1_rad = np.arctan2(y, x)

    R = np.hypot(x, y)
    R_arm = R - L_eff_x
    Z_arm = z - C_z

    D = R_arm**2 + Z_arm**2
    S = (D - L2**2 - L3**2) / (2 * L2 * L3)
    
    # Clip S to handle floating point edge cases strictly outside [-1, 1]
    S = np.clip(S, -1.0, 1.0)
    gamma = np.arcsin(S)

    A = L2 + L3 * np.sin(gamma)
    B = L3 * np.cos(gamma)

    sin_j2 = (A * R_arm - B * Z_arm) / D
    cos_j2 = (B * R_arm + A * Z_arm) / D

    j2_rad = np.arctan2(sin_j2, cos_j2)
    pos6_rad = j2_rad - gamma

    return np.array([j1_rad, j2_rad, pos6_rad])

# ==============================================================================
# LAB 3 VALIDATION ROUTINES
# ==============================================================================

def run_offline_ik_validation(filepath: str):
    """Lab 3 Step 3: Verifies IK derivation using offline validation file."""
    print("\n" + "="*50)
    print(" LAB 3 PART 1: OFFLINE IK DATA VALIDATION")
    print("="*50)
    
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find '{filepath}'. Please place it in the directory.")
        return

    max_ik_error = 0.0
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
                if len(parts) < 6: continue
                    
                target_x, target_y, target_z, j1_val, j2_val, pos6_val = parts[:6]
                j1_rad, j2_rad, pos6_rad = np.radians([j1_val, j2_val, pos6_val])
                
                # Calibrate offsets using the first line exactly like Lab 2
                if not calibrated:
                    R_target = np.hypot(target_x, target_y)
                    ta_L_eff_x = R_target - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
                    ta_C_z = target_z - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))
                    print(f"Calibrated X Offset: {ta_L_eff_x:.4f} mm | Z Offset: {ta_C_z:.4f} mm\n")
                    calibrated = True
                
                # Calculate IK based on XYZ
                calc_j1, calc_j2, calc_pos6 = calculate_inverse_kinematics(target_x, target_y, target_z, ta_L_eff_x, ta_C_z)
                calc_angles_deg = np.degrees([calc_j1, calc_j2, calc_pos6])
                expected_angles_deg = np.array([j1_val, j2_val, pos6_val])
                
                error = np.linalg.norm(expected_angles_deg - calc_angles_deg)
                if error > max_ik_error:
                    max_ik_error = error
                line_count += 1
                
            except ValueError:
                continue 

    print(f"Processed {line_count} points.")
    print(f"Maximum Inverse Kinematics Error: {max_ik_error:.6f} degrees")
    if max_ik_error < 0.1:
        print("PASS! The IK derivation accurately reverses the FK geometry.")
    else:
        print("FAIL. Review IK derivation.")

def is_safe_move(start_xyz, target_xyz):
    x1, y1, z1 = start_xyz
    x2, y2, z2 = target_xyz
    num_steps = 10 
    for i in range(num_steps + 1):
        fraction = i / num_steps
        current_x = x1 + (x2 - x1) * fraction
        current_y = y1 + (y2 - y1) * fraction
        current_z = z1 + (z2 - z1) * fraction
        if current_z > 0 or current_z < -120: return False
        if current_x < 0: return False
        radius = np.hypot(current_x, current_y)
        if radius < 140 or radius > 260: return False
    return True

# def is_safe_move(start_xyz, target_xyz):
#     x1, y1, z1 = start_xyz
#     x2, y2, z2 = target_xyz
#     num_steps = 10 
#     for i in range(num_steps + 1):
#         fraction = i / num_steps
#         current_x = x1 + (x2 - x1) * fraction
#         current_y = y1 + (y2 - y1) * fraction
#         current_z = z1 + (z2 - z1) * fraction
        
#         # FIX: Raised the upper Z limit to 50mm so the robot's home position doesn't instantly trigger a failure.
#         if current_z > 50 or current_z < -120: return False
        
#         if current_x < 0: return False
        
#         radius = np.hypot(current_x, current_y)
#         if radius < 140 or radius > 260: return False
        
#     return True

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

def print_status(api: SimDobotAPI, label: str) -> None:
    obs = api.env._get_obs()
    info = api.env._get_info(obs)
    cube_pos = api.env.data.body("pick_cube").xpos.copy() * 1000.0
    pose = get_pose(api)
    print(f"{label}: xyz_mm={pose[:3].round(1)} joints_deg={pose[4:].round(1)} ")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT Simulation Starter Script.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    # Part 1 Testing
    run_offline_ik_validation("Lab2DesignData.txt")

    api = create_sim_api(seed=args.seed, headless=args.headless)

    try:
        initialize_robot(api)
        print_status(api, "home")

        # # Starter Code Examples (Kept intact)
        # print("\n--- Cartesian Motion Demo ---")
        # move_to_xyz(api, 300, 0, 0)
        # print_status(api, "at_target_1")
        
        # move_to_xyz(api, 211, -31, -31)
        # print_status(api, "at_target_2")

        # print("\n--- Pick and Place Demo ---")
        # move_to_home(api)
        # move_joint_angles(api, *PICK_JOINTS)
        # engage_suction(api)
        # print_status(api, "grasped_cube")

        # move_joint_angles(api, *PLACE_LIFT_JOINTS)
        # move_joint_angles(api, *PLACE_JOINTS)
        # release_suction(api)
        # print_status(api, "released_cube")
        # move_to_home(api)
        
        # --- LAB 3 PART 2: LIVE IK/FK VALIDATION ---
        print("\n" + "="*50)
        print(" LAB 3 PART 2: SIMULTANEOUS IK & FK VALIDATION")
        print("="*50)
        
        # FIX: Move into the safe workspace BEFORE running the checks so step 0 of the check passes
        print("Pre-positioning to a safe workspace coordinate...")
        move_to_xyz(api, 200, 0, -20)
        
        # Recalibrate offsets dynamically from simulator home state
        actual_pose = get_pose(api)
        actual_xyz = actual_pose[:3]
        j1_rad, j2_rad, pos6_rad = np.radians(actual_pose[4:7])
        sim_R_actual = np.hypot(actual_xyz[0], actual_xyz[1])
        sim_L_eff_x = sim_R_actual - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
        sim_C_z = actual_xyz[2] - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))

        # Changed Z to -20 to avoid floating point > 0 errors
        test_xyz_targets = [
            [200, 0, -20],
            [190, 40, -20],
            [210, -40, -20],
            [180, 20, -20]
        ]
        
        for idx, target_xyz in enumerate(test_xyz_targets):
            print(f"\n--- Test {idx+1}: Desired XYZ {target_xyz} ---")
            
            # 1. Convert Desired Cartesian to Joint Angles (Inverse Kinematics)
            calc_j1, calc_j2, calc_pos6 = calculate_inverse_kinematics(target_xyz[0], target_xyz[1], target_xyz[2], sim_L_eff_x, sim_C_z)
            target_angles_deg = np.degrees([calc_j1, calc_j2, calc_pos6])
            
            # 2. Check if safe to move
            current_pose = get_pose(api)
            if not is_safe_move(current_pose[:3], target_xyz):
                print(f"WARNING: Target {target_xyz} is out of workspace. Skipping.")
                continue
                
            print(f"Calculated IK Angles: {target_angles_deg.round(2)}")
            
            # 3. Move via joint space
            move_joint_angles(api, target_angles_deg[0], target_angles_deg[1], target_angles_deg[2], 0)
            
            # 4. Get actual hardware pose
            actual_pose = get_pose(api)
            actual_xyz = actual_pose[:3]
            actual_j1, actual_j2, actual_pos6 = np.radians(actual_pose[4:7])
            
            # 5. Use Forward Kinematics to predict where we SHOULD be based on real joints
            expected_xyz = calculate_forward_kinematics(actual_j1, actual_j2, (actual_pos6 - actual_j2), sim_L_eff_x, sim_C_z)
            
            # 6. Validate Discrepancies 
            error = np.linalg.norm(actual_xyz - expected_xyz)
            print(f"Actual Robot XYZ:     {actual_xyz.round(4)} mm")
            print(f"Predicted FK XYZ:     {expected_xyz.round(4)} mm")
            print(f"Simultaneous Error:   {error:.4f} mm")

    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()

if __name__ == "__main__":
    main()
