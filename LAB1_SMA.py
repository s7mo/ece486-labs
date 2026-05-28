"""
MuJoCo Simulation Starter Code for DOBOT.
Use this script to run your DOBOT experiments in the simulator.
"""

import argparse
import mujoco as mj
import numpy as np
import csv
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


def create_sim_api(seed: int, headless: bool) -> SimDobotAPI:
    """Factory to create the simulator environment and API object."""
    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=seed)

    viewer = None
    if not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    dobot_body_id = env.model.body("dobot").id
    base_pos_mm = env.data.xpos[dobot_body_id].copy() * 1000.0

    api = SimDobotAPI(
        env=env,
        viewer=viewer,
        home_pos=np.zeros(3, dtype=np.float64),
        base_pos_mm=base_pos_mm,
    )
    api.home_pos = api.current_xyz_mm()
    return api


def initialize_robot(api: SimDobotAPI) -> None:
    """Initialize robot state and drive to home position."""
    api.suction_on = False
    api.env.suction_activated = False
    api.env.data.ctrl[:4] = HOME_CTRL
    api.env.data.ctrl[4] = 0.0
    for step in range(250):
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
    api.home_pos = api.current_xyz_mm()
    print(f"Simulator ready. home_pos = {api.home_pos.round(1).tolist()} mm")


def print_status(api: SimDobotAPI, label: str) -> None:
    """Print current robot and task status."""
    obs = api.env._get_obs()
    info = api.env._get_info(obs)
    cube_pos = api.env.data.body("pick_cube").xpos.copy() * 1000.0
    pose = get_pose(api)
    print(
        f"{label}: xyz_mm={pose[:3].round(1)} joints_deg={pose[4:].round(1)} "
        f"suction={api.suction_on} grasped={info['grasped']} success={info['is_success']} "
        f"cube_to_goal={info['cube_to_goal_distance']:.4f} cube_mm={cube_pos.round(1)}"
    )

import csv # Add this at the very top of your file with the other imports!

def generate_circle_trajectory(center_x, center_y, z, radius, num_points=50):
    """Generates a list of [x, y, z] points forming a circle."""
    trajectory = []
    # Generate 50 evenly spaced angles from 0 to 2*Pi
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        trajectory.append([x, y, z])
        
    return trajectory

"""---------------------ADDED CODE BELOW---------------------"""

def is_safe_move(start_xyz, target_xyz):
    """
    Checks if a straight-line path between start and target stays within 
    the restricted semi-annular workspace.
    """
    x1, y1, z1 = start_xyz
    x2, y2, z2 = target_xyz
    
    num_steps = 10 # Break the path into 10 chunks to catch line-crossing violations
    
    for i in range(num_steps + 1):
        # Calculate the intermediate coordinate along the line
        fraction = i / num_steps
        current_x = x1 + (x2 - x1) * fraction
        current_y = y1 + (y2 - y1) * fraction
        current_z = z1 + (z2 - z1) * fraction
        
        # Rule 1: Z coordinate must lie between 0mm and -120mm
        if current_z > 0 or current_z < -120:
            return False
            
        # Rule 4: X coordinate must never go below 0
        if current_x < 0:
            return False
            
        # Rules 2 & 3: Radius (x,y projection) must be between 140mm and 260mm
        radius = np.hypot(current_x, current_y)
        if radius < 140 or radius > 260:
            return False
            
    # If it survived all 10 checks without returning False, the line is safe
    return True

def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT Simulation Starter Script.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    api = create_sim_api(seed=args.seed, headless=args.headless)

    try:
        initialize_robot(api)
        print_status(api, "home")

        print("\n--- Part 2: Workspace Validation Tests ---")
        
        move_to_xyz(api, 200, 0, -10)
        
        current_pose = get_pose(api) 
        current_xyz = current_pose[0:3] 

        print("\n--- TEST 1: A Valid Point ---")
        valid_target = [200, 50, -50] 

        if is_safe_move(current_xyz, valid_target):
            print(f"Path to {valid_target} is SAFE. Moving robot...")
            move_to_xyz(api, valid_target[0], valid_target[1], valid_target[2])
            print_status(api, "at_valid_target")
        else:
            print(f"WARNING: Path to {valid_target} rejected!")

        current_pose = get_pose(api) 
        current_xyz = current_pose[0:3] 

        print("\n--- TEST 2: An Invalid Point (Violates Max Radius) ---")
        invalid_target = [300, 0, -50] 

        if is_safe_move(current_xyz, invalid_target):
            print(f"Path to {invalid_target} is SAFE. Moving robot...")
            move_to_xyz(api, invalid_target[0], invalid_target[1], invalid_target[2])
        else:
            print(f"WARNING: Path to {invalid_target} rejected! Radius too large.")

        print("\nTests complete. Returning to home.")

        print("\n--- Part 3: Fun with Trajectories ---")
        
        # 1. Generate the 50 points-20 for reduced time
# 1. Generate the points
        num_points = 20 
        circle_path = generate_circle_trajectory(center_x=200, center_y=0, z=-10, radius=40, num_points=num_points)
        
        # 2. Setup a list to hold our data for the CSV
        report_data = [["Run", "Point_Index", "Target_X", "Target_Y", "Target_Z", "Actual_X", "Actual_Y", "Actual_Z"]]
        
        # 3. Execute the trajectory 10 times
        for run in range(1, 11):
            print(f"\nExecuting Circle Run {run}/10...")
            
            # Safely move to the start of the circle first
            start_pt = circle_path[0]
            move_to_xyz(api, start_pt[0], start_pt[1], start_pt[2])
            
            # Trace the circle
            for idx, target_pt in enumerate(circle_path):
                
                # ---> ADD THIS PRINT STATEMENT <---
                if idx % 5 == 0: 
                    print(f"  -> Moving to point {idx}/{num_points}...")
                
                move_to_xyz(api, target_pt[0], target_pt[1], target_pt[2])
                
                # Immediately grab the physical robot's actual position
                actual_pose = get_pose(api)
                actual_xyz = actual_pose[0:3]
                
                # Save the theoretical target vs. the actual result
                report_data.append([
                    run, idx, 
                    target_pt[0], target_pt[1], target_pt[2],
                    actual_xyz[0], actual_xyz[1], actual_xyz[2]
                ])

        # 4. Save the data to a file for your report
        csv_filename = "trajectory_results.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(report_data)
        
        print(f"\nSUCCESS! All 500 data points saved to {csv_filename}")

        move_to_home(api)
        print_status(api, "back_home")
        
    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()

if __name__ == "__main__":
    main()