import DobotDllType as dType
import time
import threading
import numpy as np
import csv

# Useful global variables
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

api = dType.load()
home_pos = [200, 100, 50]

def initialize_robot(api):
    com_port = dType.SearchDobot(api)
    print(dType.SearchDobot(api))
    if "COM" not in com_port[0]:
        print("Error: The robot either isn't on or isn't responding. Exiting now")
        exit()
    
    state = dType.DobotConnect.DobotConnect_NoError
    for i in range(0,len(com_port)):
        state_full = dType.ConnectDobot(api, com_port[i], 115200)
        state = state_full[0]
        print("STATE FULL:")
        print(state_full)
        if state == dType.DobotConnect.DobotConnect_NoError:
            print("Connected!")
            name = "Hermy"#dType.GetDeviceName(api)
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
    
# ==========================================
# YOUR CUSTOM LAB FUNCTIONS
# ==========================================

def is_safe_move(start_xyz, target_xyz):
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

def generate_circle_trajectory(center_x, center_y, z, radius, num_points=50):
    trajectory = []
    angles = np.linspace(0, 2 * np.pi, num_points)
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        trajectory.append([x, y, z])
    return trajectory


# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Connect and Home the Robot
initialize_robot(api)

# 2. PART 2: Workspace Validation
print("\n--- Part 2: Workspace Validation Tests ---")
print("Moving to safe start position...")
move_to_xyz(api, 200, 0, -10) 

# --- TEST 1: Valid Point ---
print("\n--- TEST 1: Valid Point ---")
current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
valid_target = [200, 0, -50] 
if is_safe_move(current_xyz, valid_target):
    print(f"Path to {valid_target} is SAFE. Moving robot...")
    move_to_xyz(api, valid_target[0], valid_target[1], valid_target[2])
else:
    print(f"WARNING: Path to {valid_target} rejected!")

# --- TEST 2: Z-Boundary Limit ---
print("\n--- TEST 2: Z-Boundary Limit ---")
current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
z_target = [200, 0, 5] 
if is_safe_move(current_xyz, z_target):
    print(f"Path to {z_target} is SAFE. Moving robot...")
    move_to_xyz(api, z_target[0], z_target[1], z_target[2])
else:
    print(f"WARNING: Path to {z_target} rejected! Z-height out of bounds.")

# --- TEST 3: X-Boundary Limit ---
print("\n--- TEST 3: X-Boundary Limit ---")
current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
x_target = [-10, 200, -50] 
if is_safe_move(current_xyz, x_target):
    print(f"Path to {x_target} is SAFE. Moving robot...")
    move_to_xyz(api, x_target[0], x_target[1], x_target[2])
else:
    print(f"WARNING: Path to {x_target} rejected! Cannot reach behind robot.")

# --- TEST 4: Maximum Radius ---
print("\n--- TEST 4: Maximum Radius ---")
current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
max_rad_target = [265, 0, -50] 
if is_safe_move(current_xyz, max_rad_target):
    print(f"Path to {max_rad_target} is SAFE. Moving robot...")
    move_to_xyz(api, max_rad_target[0], max_rad_target[1], max_rad_target[2])
else:
    print(f"WARNING: Path to {max_rad_target} rejected! Radius exceeds 260mm.")

# --- TEST 5: Minimum Radius ---
print("\n--- TEST 5: Minimum Radius ---")
current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
min_rad_target = [139, 0, -50] 
if is_safe_move(current_xyz, min_rad_target):
    print(f"Path to {min_rad_target} is SAFE. Moving robot...")
    move_to_xyz(api, min_rad_target[0], min_rad_target[1], min_rad_target[2])
else:
    print(f"WARNING: Path to {min_rad_target} rejected! Radius under 140mm.")

# --- TEST 6: Line Crossing Minimum Radius Zone ---
print("\n--- TEST 6: Line Crossing Minimum Radius Zone ---")
print("Setting up start point for line test...")
move_to_xyz(api, 0, 200, -50) 

current_pose = dType.GetPose(api) 
current_xyz = current_pose[0:3] 
cross_target = [0, -200, -50] 

if is_safe_move(current_xyz, cross_target):
    print(f"Path to {cross_target} is SAFE. Moving robot...")
    move_to_xyz(api, cross_target[0], cross_target[1], cross_target[2])
else:
    print(f"WARNING: Path to {cross_target} rejected! Path intersects restricted center zone.")

# 3. PART 3: Trajectory Execution
print("\n--- Part 3: Fun with Trajectories ---")
num_points = 50 
circle_path = generate_circle_trajectory(center_x=200, center_y=0, z=-10, radius=40, num_points=num_points)

report_data = [["Run", "Point_Index", "Target_X", "Target_Y", "Target_Z", "Actual_X", "Actual_Y", "Actual_Z"]]

for run in range(1, 11):
    print(f"\nExecuting Circle Run {run}/10...")
    
    start_pt = circle_path[0]
    move_to_xyz(api, start_pt[0], start_pt[1], start_pt[2])
    
    for idx, target_pt in enumerate(circle_path):
        if idx % 10 == 0: 
            print(f"  -> Moving to point {idx}/{num_points}...")
        
        move_to_xyz(api, target_pt[0], target_pt[1], target_pt[2])
        
        # Grab the hardware's real position using dType
        actual_pose = dType.GetPose(api)
        actual_xyz = actual_pose[0:3]
        
        report_data.append([
            run, idx, 
            target_pt[0], target_pt[1], target_pt[2],
            actual_xyz[0], actual_xyz[1], actual_xyz[2]
        ])

# 4. Save the physical data and finish
csv_filename = "physical_robot_trajectory_results.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(report_data)

print(f"\nSUCCESS! Physical robot data saved to {csv_filename}")

print("\nTests complete. Returning to home.")
move_to_home(api)

# Gracefully disconnect the hardware
dType.DisconnectDobot(api)
