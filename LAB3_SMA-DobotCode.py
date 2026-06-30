import DobotDllType as dType
import time
import threading
import numpy as np
import os

# Useful global variables
# --- These are status strings that you might see, so we're defining them here ---
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

# always begin with this line, or you can't connect to the robot at all. Just don't
# remove this line and keep it at the top of your code
api = dType.load()

"""
These coordinates are to the left of the robot's x axis and slight above the xy plane, viewed from
the top. This is a useful home position when dealing with the vision labs, since it moves
the robot out of the way. You can change the coordinates here if you really want.
"""
home_pos = [200, 100, 50]

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

# ==============================================================================
# ROBOT API FUNCTIONS
# ==============================================================================

def initialize_robot(api):
    #detect the robot's com port
    com_port = dType.SearchDobot(api)
    print(dType.SearchDobot(api))
    #if we can't find it, then we can't continue, so exit
    if "COM" not in com_port[0]:
        print("Error: The robot either isn't on or isn't responding. Exiting now")
        exit()
    
    #we've found it, so let's try to connect
    state = dType.DobotConnect.DobotConnect_NoError
    for i in range(0,len(com_port)):
        state_full = dType.ConnectDobot(api, com_port[i], 115200)
        state = state_full[0]
        print("STATE FULL:")
        print(state_full)
        #If the connection failed at this point, we also can't proceed, so we need to exit
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
    
    #Set the robot's max speed and acceleration. We're keeping these to 50% of max for safety
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)
    
    #Set the home position
    dType.SetHOMEParams(api, home_pos[0], home_pos[1], home_pos[2], 0, isQueued=1)
    
    cmdIndx = -1
    execCmd = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]
    
    dType.SetQueuedCmdStartExec(api)
    
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_to_xyz(api,x,y,z):
    cmdIndx = -1
    execCmd = dType.SetPTPCmd(api,dType.PTPMode.PTPMOVLXYZMode,x,y,z,0,isQueued=0)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_joint_angles(api,J1,J2,J3,J4=0):
    cmdIndx = -1
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, J1, J2, J3, J4, isQueued = 0)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)

def move_to_home(api):
    move_to_xyz(api,home_pos[0],home_pos[1],home_pos[2])

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Part 1 Testing (Offline)
run_offline_ik_validation("Lab2DesignData.txt")

# Before running any physical commands, initialize
initialize_robot(api)

# --- LAB 3 PART 2: LIVE IK/FK VALIDATION (HARDWARE) ---
print("\n" + "="*50)
print(" LAB 3 PART 2: SIMULTANEOUS IK & FK VALIDATION (LIVE ROBOT)")
print("="*50)

# Move down into the safe workspace BEFORE running the checks 
# so step 0 of your path validation starts with a safe Z < 0
print("Pre-positioning to a safe workspace coordinate...")
move_to_xyz(api, 200, 0, -20)

# Recalibrate offsets dynamically from real hardware state
robot_pose = dType.GetPose(api)
actual_xyz = robot_pose[0:3]
j1_rad, j2_rad, pos6_rad = np.radians(robot_pose[4:7]) # Indices 4, 5, 6 are J1, J2, J3

sim_R_actual = np.hypot(actual_xyz[0], actual_xyz[1])
sim_L_eff_x = sim_R_actual - (L2 * np.sin(j2_rad) + L3 * np.cos(pos6_rad))
sim_C_z = actual_xyz[2] - (L2 * np.cos(j2_rad) - L3 * np.sin(pos6_rad))

# Safe coordinates well within the 140-260 radius and Z < 0
test_xyz_targets = [
    [150, 100, -20],    
    [220, 50, -20],   
    [150, -50, -20],  
    [220, -100, -20],  
    [150, -100, -10],    
    [220, -50, -10],   
    [150, 50, -10],  
    [220, 100, -10] 
]

for idx, target_xyz in enumerate(test_xyz_targets):
    print(f"\n--- Test {idx+1}: Desired XYZ {target_xyz} ---")
    
    # 1. Convert Desired Cartesian to Joint Angles (Inverse Kinematics)
    calc_j1, calc_j2, calc_pos6 = calculate_inverse_kinematics(target_xyz[0], target_xyz[1], target_xyz[2], sim_L_eff_x, sim_C_z)
    target_angles_deg = np.degrees([calc_j1, calc_j2, calc_pos6])
    
    # 2. Check if safe to move (using your original function)
    current_pose = dType.GetPose(api)
    if not is_safe_move(current_pose[0:3], target_xyz):
        print(f"WARNING: Target {target_xyz} is out of workspace. Skipping.")
        continue
        
    print(f"Calculated IK Angles: {target_angles_deg.round(2)}")
    
    # 3. Move via joint space
    move_joint_angles(api, target_angles_deg[0], target_angles_deg[1], target_angles_deg[2], 0)
    
    # 4. Get actual hardware pose
    actual_pose = dType.GetPose(api)
    actual_xyz = actual_pose[0:3]
    actual_j1, actual_j2, actual_pos6 = np.radians(actual_pose[4:7])
    
    # 5. Predict where we SHOULD be
    expected_xyz = calculate_forward_kinematics(actual_j1, actual_j2, (actual_pos6 - actual_j2), sim_L_eff_x, sim_C_z)
    
    # 6. Validate Discrepancies 
    error = np.linalg.norm(np.array(actual_xyz) - expected_xyz)
    print(f"Actual Robot XYZ:     [{actual_xyz[0]:.4f}, {actual_xyz[1]:.4f}, {actual_xyz[2]:.4f}] mm")
    print(f"Predicted FK XYZ:     [{expected_xyz[0]:.4f}, {expected_xyz[1]:.4f}, {expected_xyz[2]:.4f}] mm")
    print(f"Simultaneous Error:   {error:.4f} mm")

# Back to safe home when finished testing
print("\nTesting complete. Moving back to hardware home...")
move_to_home(api)
print("Done!")
dType.DisconnectDobot(api)
