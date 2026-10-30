# Quality metrics and the accept/reject decision for one episode.
string episode_id
float64 dimensionless_squared_jerk
float64 sparc
float64 path_length_m
float64 min_joint_limit_margin
float64 min_velocity_limit_margin
uint32 collision_count
bool task_success
float64 acceptance_score
uint8 decision
string[] rejection_reasons
string metadata_json

uint8 DECISION_ACCEPT=0
uint8 DECISION_REJECT=1
uint8 DECISION_INVALID=2
