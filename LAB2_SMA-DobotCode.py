"""
ECE 486 - Lab 2 Hardware Validation Script
This script runs the offline dataset math, initializes the physical DOBOT, 
auto-calibrates the tool offsets, maps the physical edge cases, and 
executes the automated hardware validation sequence.
"""

import DobotDllType as dType
import time
import threading
import numpy as np
import os

# ==============================================================================
# DOBOT API SETUP & CONSTANTS
# ==============================================================================

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

api = dType.load()
home_pos = [200, 100, 50]

# Link length constants from the lab manual (mm)
L1 = 138.0  
L2 = 135.0  
L3 = 147.0  

# ==============================================================================
# PART 1: DETERMINING FORWARD KINEMATICS
# ==============================================================================

def get_true_j3(j2_deg: float, pos6_deg: float) -> float:
    return pos6_deg - j2_deg

def get_forearm_angle(j2_deg: float, j3_true_deg: float) -> float:
    return j3_true_deg + j2_deg

def calculate_forward_kinematics(j1_rad: float, j2_rad: float, j3_true_rad: float, L_eff_x: float, C_z: float) -> np.ndarray:
    pos6_rad = j2_rad + j3_true_rad
    R = L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad) + L_eff_x
    Z = C_z + L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad)
    X = R * np.cos(j1_rad)
    Y = R * np.sin(j1_rad)
    return np.array([X, Y, Z])


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


# ==============================================================================
# PART 2: THE ROBOT'S WORKSPACE (STEP 5)
# ==============================================================================

def is_safe_move(start_xyz, target_xyz):
    """Lab 1 Workspace bounds and 10-step path interpolation."""
    x1, y1, z1 = start_xyz
    x2, y2, z2 = target_xyz
    num_steps = 10 
    
    for i in range(num_steps + 1):
        fraction = i / num_steps
        current_x = x1 + (x2 - x1) * fraction
        current_y = y1 + (y2 - y1) * fraction
        current_z = z1 + (z2 - z1) * fraction
        
        # Note: If your physical tests fail immediately, it might be because 
        # the real robot starts at Z=50, violating this current_z > 0 check!
        if current_z > 0 or current_z < -120:
            return False
        if current_x < 0:
            return False
            
        radius = np.hypot(current_x, current_y)
        if radius < 140 or radius > 260:
            return False
            
    return True

def is_safe_joint_move(current_xyz, target_j1, target_j2, target_pos6, L_eff_x, C_z):
    j1_rad = np.radians(target_j1)
    j2_rad = np.radians(target_j2)
    pos6_rad = np.radians(target_pos6)
    
    j3_true_rad = pos6_rad - j2_rad
    target_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, L_eff_x, C_z)
    
    return is_safe_move(current_xyz, target_xyz)


# ==============================================================================
# PHYSICAL ROBOT INITIALIZATION & MOTION FUNCTIONS
# ==============================================================================

def initialize_robot(api):
    com_port = dType.SearchDobot(api)
    print(f"Search results: {com_port}")
    if "COM" not in com_port[0]:
        print("Error: The robot either isn't on or isn't responding. Exiting now")
        exit()
    
    state = dType.DobotConnect.DobotConnect_NoError
    for i in range(0,len(com_port)):
        state_full = dType.ConnectDobot(api, com_port[i], 115200)
        state = state_full[0]
        if state == dType.DobotConnect.DobotConnect_NoError:
            print("Connected!")
            name = dType.GetDeviceName(api)
            if name[0] == "Not a dobot":
                dType.DisconnectDobot(api)
                continue
            else:
                break
            
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("Can not connect! Exiting")
        exit()    

    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)
    dType.SetHOMEParams(api, home_pos[0], home_pos[1], home_pos[2], 0, isQueued=1)
    
    execCmd = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]
    dType.SetQueuedCmdStartExec(api)
    
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_to_xyz(api, x, y, z):
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, 0, isQueued=1)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_joint_angles(api, J1, J2, J3, J4=0):
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, J1, J2, J3, J4, isQueued=1)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_to_home(api):
    move_to_xyz(api, home_pos[0], home_pos[1], home_pos[2])


# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # --- PART 1 EXECUTION ---
    run_offline_validation("Lab2DesignData.txt")

    print("\n" + "="*50)
    print(" CONNECTING TO PHYSICAL HARDWARE")
    print("="*50)
    
    initialize_robot(api)
    
    # Grab the robot's physical starting pose to auto-calibrate its exact tool offsets
    pose = dType.GetPose(api)
    actual_xyz = pose[0:3]
    j1_rad, j2_rad, pos6_rad = np.radians(pose[4]), np.radians(pose[5]), np.radians(pose[6])
    
    hw_R_actual = np.hypot(actual_xyz[0], actual_xyz[1])
    hw_L_eff_x = hw_R_actual - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
    hw_C_z = actual_xyz[2] - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))
    
    j3_true_rad = pos6_rad - j2_rad
    calculated_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, hw_L_eff_x, hw_C_z)
    
    print(f"Physical Robot Actual XYZ:  [{actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}] mm")
    print(f"My Calculated FK XYZ:       [{calculated_xyz[0]:.4f}, {calculated_xyz[1]:.4f}, {calculated_xyz[2]:.4f}] mm")
    print(f"\n(Hardware calibration offsets: X={hw_L_eff_x:.4f} mm, Z={hw_C_z:.4f} mm)")

    # --- PART 2 EXECUTION ---
    print("\n" + "="*50)
    print(" PART 2 / STEP 4: WORKSPACE EXPLORATION")
    print("="*50)
    
    edge_cases = [
        [260, 0, -10,    "Max Radius, Safest Max Height"],
        [260, 0, -120,   "Max Radius, Min Height"],
        [140, 0, -10,    "Min Radius, Safest Max Height"],
        [140, 0, -120,   "Min Radius, Min Height"],
        [0, 200, -50,    "X-Boundary Limit (Y-Axis Reach)"]
    ]
    
    for case in edge_cases:
        target_x, target_y, target_z, description = case
        move_to_xyz(api, target_x, target_y, target_z)
        
        p = dType.GetPose(api)
        print(f"Edge Case: {description}")
        print(f"  Cartesian Target: [X: {target_x}, Y: {target_y}, Z: {target_z}]")
        print(f"  Resulting Angles: [J1: {p[4]:.2f}°, J2: {p[5]:.2f}°, POS6: {p[6]:.2f}°]\n")

    # --- PART 3 EXECUTION ---
    print("\n" + "="*50)
    print(" PART 3 / STEP 6: HARDWARE VALIDATION TESTS")
    print("="*50)
    
    # Safe Joint Angles to test inside the Lab 1 envelope
    test_joints = [
        [-45.0, 40.0, 65.0],    # rotated right 
        [90.0, 35.0, 60.0],     # 90 twist
        [45.0, 45.0, 70.0],     # rotated left, nice tuck this time
        [15.0, 45.0, 20.0]      # safe stage, slightly left so it can go back home this time
    ]
    
    # ---------------------------------------------------------
    # THE FIX: Move down into the safe Z-zone BEFORE testing!
    # ---------------------------------------------------------
    print("Staging robot inside the safe bounding box...")
    move_to_xyz(api, 200, 0, -10) 

    # Now grab the current position, which is legally Z < 0
    actual_pose = dType.GetPose(api)
    current_xyz = actual_pose[0:3]
    
    for idx, target_joints in enumerate(test_joints):
        j1_target, j2_target, pos6_target = target_joints
        
        print(f"\n--- Test {idx+1}: Joints [{j1_target}, {j2_target}, {pos6_target}] ---")
        
        if not is_safe_joint_move(current_xyz, j1_target, j2_target, pos6_target, hw_L_eff_x, hw_C_z):
            print("WARNING: This joint configuration (or the path to it) is OUT OF BOUNDS. Skipping.")
            continue
            
        print("Path SAFE. Moving robot...")
        
        move_joint_angles(api, j1_target, j2_target, pos6_target, 0)
        
        actual_pose = dType.GetPose(api)
        actual_xyz = actual_pose[0:3]
        
        current_xyz = actual_xyz
        
        j1_rad, j2_rad, pos6_rad = np.radians(j1_target), np.radians(j2_target), np.radians(pos6_target)
        j3_true_rad = pos6_rad - j2_rad
        theoretical_xyz = calculate_forward_kinematics(j1_rad, j2_rad, j3_true_rad, hw_L_eff_x, hw_C_z)
        
        error = np.linalg.norm(np.array(actual_xyz) - theoretical_xyz)
        print(f"Target Angles:   {target_joints}")
        print(f"Hardware XYZ:    [{actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}] mm")
        print(f"Theoretical XYZ: [{theoretical_xyz[0]:.4f}, {theoretical_xyz[1]:.4f}, {theoretical_xyz[2]:.4f}] mm")
        print(f"Positional Error: {error:.4f} mm")

    print("\nAll tests complete! Moving back to home position.")
    move_to_home(api)
    
    # Wait for the physical arm to fully decelerate before dropping the serial connection
    print("Waiting for hardware to settle...")
    time.sleep(2)
    
    dType.DisconnectDobot(api)
