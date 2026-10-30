# One complete rollout episode. Images are NOT embedded: they stay on disk and
# are referenced from metadata_json. Metadata must be a bounded JSON object.
string episode_id
string schema_version
string backend
uint64 seed
string task_instruction
builtin_interfaces/Time start_stamp
float32 control_rate_hz
bool is_perturbed
uint8 outcome
string failure_reason
sim2real_msgs/TrajectoryStep[] steps
string metadata_json

uint8 OUTCOME_UNKNOWN=0
uint8 OUTCOME_SUCCESS=1
uint8 OUTCOME_FAILURE=2
uint8 OUTCOME_TIMEOUT=3
uint8 OUTCOME_ABORTED=4
