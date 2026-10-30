# One step of an action chunk.
# points[k].positions in a JointTrajectory use these fields as documented in
# docs/ARCHITECTURE.md; here a step stores an absolute joint target plus the
# raw action vector produced by the policy.
builtin_interfaces/Time stamp
uint32 step_index
string task_instruction
string[] joint_names
float64[] joint_positions
float64[] joint_velocities
geometry_msgs/Pose ee_pose
string ee_frame_id
float32[] action
uint8 action_space

uint8 ACTION_JOINT_POSITION_TARGET=0
uint8 ACTION_JOINT_DELTA=1
uint8 ACTION_EE_DELTA=2
