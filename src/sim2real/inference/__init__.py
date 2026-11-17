"""ROS-free VLA inference core.

Nothing in this package imports ``rclpy`` or ``torch``. The ROS adapter lives in
``ros2_ws/src/sim2real_vla`` and the ML environment is only reached through the
out-of-process policy server.
"""

from sim2real.inference.async_worker import (
    FAILED,
    MODEL_UNAVAILABLE,
    READY,
    RUNNING,
    STARTING,
    STOPPED,
    AsyncInferenceWorker,
    InferenceRequest,
    InferenceResult,
)
from sim2real.inference.chunk import (
    ChunkError,
    TrajectoryChunk,
    finger_out_of_range,
    make_chunk,
    select_actions,
    trim_for_latency,
)
from sim2real.inference.config import (
    InferenceConfig,
    InferenceConfigError,
    load_inference_config,
)
from sim2real.inference.fake_policy import FakeChunkPolicy, RawChunk
from sim2real.inference.observation import (
    ImageInput,
    JointStateInput,
    ObservationAssembler,
    ServerObservation,
)
from sim2real.inference.server_policy import (
    make_fake_policy_factory,
    make_policy_factory,
    make_server_policy_factory,
    resolve_repo_root,
    resolve_server_python,
)
from sim2real.inference.validation import ValidationError

__all__ = [
    "FAILED",
    "MODEL_UNAVAILABLE",
    "READY",
    "RUNNING",
    "STARTING",
    "STOPPED",
    "AsyncInferenceWorker",
    "ChunkError",
    "FakeChunkPolicy",
    "ImageInput",
    "InferenceConfig",
    "InferenceConfigError",
    "InferenceRequest",
    "InferenceResult",
    "JointStateInput",
    "ObservationAssembler",
    "RawChunk",
    "ServerObservation",
    "TrajectoryChunk",
    "ValidationError",
    "finger_out_of_range",
    "load_inference_config",
    "make_chunk",
    "make_fake_policy_factory",
    "make_policy_factory",
    "make_server_policy_factory",
    "resolve_repo_root",
    "resolve_server_python",
    "select_actions",
    "trim_for_latency",
]
