# Safety watchdog state. Ratios are normalised against configured limits:
# 1.0 means exactly at the limit. NaN means "unknown", not "safe".
std_msgs/Header header
bool is_safe
bool estop_latched
uint8 state
string failure_reason
bool obstacle_map_valid
float64 min_obstacle_distance_m
float64 max_joint_limit_ratio
float64 max_velocity_ratio
float64 max_acceleration_ratio
float64 max_jerk_ratio
float64 observation_age_s
float64 action_age_s

uint8 STATE_NOMINAL=0
uint8 STATE_DEGRADED=1
uint8 STATE_SAFE_STOP=2
uint8 STATE_ESTOP=3
