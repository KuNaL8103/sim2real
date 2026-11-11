"""ROS 2 adapter: closed-loop policy evaluation.

The backend runs **in-process**, like the collector.  This node is not a
simulator and does **not** publish ``/joint_states``; when ``sim_node`` runs it
remains the only ``/joint_states`` publisher.

Service ``/sim2real/run_eval`` (``std_srvs/Trigger``) runs one evaluation on a
worker thread (the executor keeps spinning) and publishes a JSON summary on
``/sim2real/eval_summary`` (``std_msgs/String``, reliable keep-last 10).

``auto_generate:=true`` fits a failure distribution from the paired evaluation
and, only if the flywheel guards permit, calls ``/sim2real/generate_rollouts``
with ``SAMPLING_FAILURE_TARGETED`` asynchronously with a timeout.  It never
starts training.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import rclpy

_bootstrap_root = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))
if importlib.util.find_spec("mujoco") is None and _bootstrap_root is not None:
    _venv_site = _bootstrap_root / ".venv-ros" / "lib" / "python3.12" / "site-packages"
    if _venv_site.is_dir():
        sys.path.append(str(_venv_site))

from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.evaluation import (  # noqa: E402
    CheckpointPolicy,
    FlywheelController,
    NoisyExpertPolicy,
    RandomPolicy,
    ScriptedExpertPolicy,
    ZeroPolicy,
    fit_failure_distribution,
    load_evaluation_config,
    run_eval,
    run_paired_eval,
)
from sim2real.evaluation.records import EvalRun  # noqa: E402
from sim2real.filtering.config import load_filter_config  # noqa: E402
from sim2real.sim import create_backend  # noqa: E402

try:
    from sim2real_msgs.srv import GenerateRollouts
except ImportError:  # pragma: no cover - the ROS build provides this
    GenerateRollouts = None  # type: ignore[assignment,misc]

EVAL_SUMMARY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class ClosedLoopEvalNode(Node):
    def __init__(
        self,
        injected_backend: Any | None = None,
        *,
        config_dir: str | None = None,
        expert_config_dir: str | None = None,
        randomization_dir: str | None = None,
        evaluation_config: str | None = None,
        filter_config: str | None = None,
        output_dir: str | None = None,
        policy_type: str | None = None,
        num_episodes: int | None = None,
        eval_seed: int | None = None,
        n_action_steps: int | None = None,
        auto_generate: bool | None = None,
    ) -> None:
        super().__init__("closed_loop_eval_node")
        self.declare_parameter("backend", "fake")
        self.declare_parameter("policy_type", "noisy_expert")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("num_episodes", 20)
        self.declare_parameter("eval_seed", 20000)
        self.declare_parameter("randomization_profile", "default")
        self.declare_parameter("output_dir", "data/eval")
        self.declare_parameter("n_action_steps", 10)
        self.declare_parameter("auto_generate", False)
        self.declare_parameter("config_dir", "")
        self.declare_parameter("expert_config_dir", "")
        self.declare_parameter("randomization_dir", "")
        self.declare_parameter("evaluation_config", "")
        self.declare_parameter("filter_config", "")
        self.declare_parameter("fake_policy_model", False)

        self._backend_name = str(self.get_parameter("backend").value)
        self._policy_type = policy_type or str(self.get_parameter("policy_type").value)
        self._checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self._num_episodes = (
            int(num_episodes)
            if num_episodes is not None
            else int(self.get_parameter("num_episodes").value)
        )
        self._eval_seed = (
            int(eval_seed) if eval_seed is not None else int(self.get_parameter("eval_seed").value)
        )
        self._profile = str(self.get_parameter("randomization_profile").value)
        self._output_dir = output_dir or str(self.get_parameter("output_dir").value)
        self._n_action_steps = (
            int(n_action_steps)
            if n_action_steps is not None
            else int(self.get_parameter("n_action_steps").value)
        )
        self._auto_generate = (
            bool(auto_generate)
            if auto_generate is not None
            else bool(self.get_parameter("auto_generate").value)
        )
        self._fake_policy_model = bool(self.get_parameter("fake_policy_model").value)

        root = _bootstrap_root or Path.cwd()
        self._config_dir = config_dir or str(self.get_parameter("config_dir").value)
        if not self._config_dir:
            try:
                from ament_index_python.packages import get_package_share_directory

                self._config_dir = os.path.join(
                    get_package_share_directory("sim2real_sim"), "config"
                )
            except Exception:  # noqa: BLE001 - source tree fallback
                self._config_dir = str(root / "configs" / "sim")
        self._expert_config_dir = expert_config_dir or str(
            self.get_parameter("expert_config_dir").value
        )
        if not self._expert_config_dir:
            self._expert_config_dir = str(root / "configs" / "expert")
        self._randomization_dir = randomization_dir or str(
            self.get_parameter("randomization_dir").value
        )
        if not self._randomization_dir:
            sibling = Path(self._config_dir).parent / "randomization"
            self._randomization_dir = (
                str(sibling) if sibling.is_dir() else str(root / "configs" / "randomization")
            )
        self._evaluation_config_path = evaluation_config or str(
            self.get_parameter("evaluation_config").value
        )
        if not self._evaluation_config_path:
            self._evaluation_config_path = str(root / "configs" / "evaluation" / "default.yaml")
        self._filter_config_path = filter_config or str(self.get_parameter("filter_config").value)
        if not self._filter_config_path:
            candidate = root / "configs" / "filtering" / f"{self._backend_name}.yaml"
            if not candidate.is_file():
                candidate = root / "configs" / "filtering" / "default.yaml"
            self._filter_config_path = str(candidate)

        self._backend = injected_backend
        self._backend_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._callback_group = ReentrantCallbackGroup()
        self._publisher = self.create_publisher(
            String, contracts.EVAL_SUMMARY_TOPIC, EVAL_SUMMARY_QOS
        )
        self._service = self.create_service(
            Trigger,
            contracts.RUN_EVAL_SERVICE,
            self._handle_run_eval,
            callback_group=self._callback_group,
        )
        self._generate_client = None
        if GenerateRollouts is not None:
            self._generate_client = self.create_client(
                GenerateRollouts, contracts.GENERATE_ROLLOUTS_SERVICE
            )
        self.get_logger().info(
            f"closed-loop eval ready; backend={self._backend_name} "
            f"policy={self._policy_type} output_dir={self._output_dir}"
        )

    def _get_backend(self) -> Any:
        with self._backend_lock:
            if self._backend is None:
                self._backend = create_backend(self._backend_name, self._config_dir)
            return self._backend

    def _make_policy(self, seed: int) -> Any:
        factory = lambda: create_backend(self._backend_name, self._config_dir)  # noqa: E731
        if self._policy_type == "zero":
            return ZeroPolicy()
        if self._policy_type == "random":
            return RandomPolicy(self._get_backend().joint_limits, seed=seed)
        expert = ScriptedExpertPolicy(
            backend_factory=factory,
            config_dir=self._config_dir,
            expert_config_dir=self._expert_config_dir,
            randomization_dir=self._randomization_dir,
            chunk_size=50,
        )
        if self._policy_type == "scripted":
            return expert
        if self._policy_type == "noisy_expert":
            return NoisyExpertPolicy(expert, seed=seed)
        if self._policy_type == "checkpoint":
            if not self._checkpoint_path:
                raise ValueError("checkpoint_path is required for policy_type=checkpoint")
            return CheckpointPolicy(
                self._checkpoint_path,
                fake_model=self._fake_policy_model,
                repo_root=_bootstrap_root,
            )
        raise ValueError(f"unknown policy_type {self._policy_type!r}")

    def _write_outputs(
        self,
        out_dir: Path,
        records: list[Any],
        run: EvalRun,
        pairs: list[dict[str, Any]] | None,
        paired_summary: dict[str, Any] | None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json() + "\n")
        manifest = run.to_dict()
        if paired_summary is not None:
            manifest["paired"] = paired_summary
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if pairs is not None:
            (out_dir / "pairs.json").write_text(
                json.dumps({"pairs": pairs}, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def _handle_run_eval(self, request: Any, response: Any) -> Any:
        del request
        if not self._run_lock.acquire(blocking=False):
            response.success = False
            response.message = "an evaluation is already running"
            return response
        try:
            done = threading.Event()
            holder: dict[str, Any] = {}

            def worker() -> None:
                try:
                    holder["summary"] = self._run_evaluation()
                except Exception as exc:  # noqa: BLE001 - report, do not kill the node
                    holder["error"] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=worker, name="closed_loop_eval", daemon=True)
            thread.start()
            timeout_s = max(120.0, float(self._num_episodes) * 120.0)
            if not done.wait(timeout_s):
                response.success = False
                response.message = f"evaluation did not finish within {timeout_s:.0f} s"
                return response
            if "error" in holder:
                raise holder["error"]
            summary = holder["summary"]
            payload = String()
            payload.data = json.dumps(summary, sort_keys=True, allow_nan=False)
            self._publisher.publish(payload)
            response.success = True
            response.message = (
                f"evaluated {summary['aggregates']['num_episodes']} episodes; "
                f"success={summary['aggregates']['num_success']}"
            )
            return response
        except Exception as exc:  # noqa: BLE001 - report, do not kill the node
            self.get_logger().error(f"run_eval failed: {exc}")
            response.success = False
            response.message = f"run_eval failed: {type(exc).__name__}: {exc}"
            return response
        finally:
            self._run_lock.release()

    def _run_evaluation(self) -> dict[str, Any]:
        config = load_evaluation_config(self._evaluation_config_path)
        from dataclasses import replace

        settings = replace(
            config.settings,
            num_episodes=self._num_episodes,
            eval_seed=self._eval_seed,
            randomization_profile=self._profile,
            n_action_steps=self._n_action_steps,
        )
        config = replace(config, settings=settings)
        filter_config = load_filter_config(self._filter_config_path)
        from sim2real.expert import load_expert_config

        expert_config = load_expert_config(self._expert_config_dir)
        backend = self._get_backend()
        policy = self._make_policy(self._eval_seed)
        run_id = f"ros_eval_{self._policy_type}_{self._eval_seed}_{self._num_episodes}"
        out_dir = Path(self._output_dir) / run_id
        pairs: list[dict[str, Any]] | None = None
        paired_summary: dict[str, Any] | None = None
        try:
            if self._auto_generate and self._policy_type != "scripted":
                expert_policy = ScriptedExpertPolicy(
                    backend_factory=lambda: create_backend(self._backend_name, self._config_dir),
                    config_dir=self._config_dir,
                    expert_config_dir=self._expert_config_dir,
                    randomization_dir=self._randomization_dir,
                    chunk_size=50,
                )
                try:
                    paired = run_paired_eval(
                        expert_backend_factory=None,
                        backend=backend,
                        expert_policy=expert_policy,
                        candidate_policy=policy,
                        config=config,
                        num_episodes=settings.num_episodes,
                        eval_seed=settings.eval_seed,
                        run_id=run_id,
                        config_hash=config.config_hash(),
                        randomization_dir=self._randomization_dir,
                        filter_config=filter_config,
                        expert_config=expert_config,
                    )
                finally:
                    expert_policy.close()
                records = paired["policy_records"]
                pairs = paired["pairs"]
                paired_summary = {
                    "expert": paired["expert"],
                    "policy_success_given_expert_success": paired[
                        "policy_success_given_expert_success"
                    ],
                    "policy_only_failures": paired["policy_only_failures"],
                }
                run = EvalRun(
                    run_id=run_id,
                    config_hash=config.config_hash(),
                    eval_seed=settings.eval_seed,
                    seeds=paired["seeds"],
                    policy_id=str(getattr(policy, "name", type(policy).__name__)),
                    policy_info=dict(getattr(policy, "info", {})),
                    randomization_profile=settings.randomization_profile,
                    n_action_steps=settings.n_action_steps,
                    renderer=records[0].renderer if records else settings.renderer,
                )
                run.aggregates = paired["policy"]["aggregates"]
                summary = {
                    "run": run.to_dict(),
                    "aggregates": run.aggregates,
                    "paired": paired_summary,
                }
            else:
                result = run_eval(
                    backend,
                    policy,
                    config,
                    num_episodes=settings.num_episodes,
                    eval_seed=settings.eval_seed,
                    run_id=run_id,
                    config_hash=config.config_hash(),
                    randomization_dir=self._randomization_dir,
                    filter_config=filter_config,
                    expert_config=expert_config,
                )
                records = result.records
                run = result.run
                summary = {"run": run.to_dict(), "aggregates": run.aggregates}
            self._write_outputs(out_dir, records, run, pairs, paired_summary)
            summary["output_dir"] = str(out_dir)
            summary["renderer"] = records[0].renderer if records else settings.renderer
            summary["auto_generate"] = self._attempt_auto_generate(
                config, run, records, pairs, paired_summary
            )
            return summary
        finally:
            close = getattr(policy, "close", None)
            if close is not None:
                close()

    def _attempt_auto_generate(
        self,
        config: Any,
        run: EvalRun,
        records: list[Any],
        pairs: list[dict[str, Any]] | None,
        paired_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self._auto_generate:
            return {"requested": False, "reason": "auto_generate=false"}
        if pairs is None:
            return {"requested": False, "reason": "no paired data; policy_only failures undefined"}
        try:
            distribution = fit_failure_distribution(
                records,
                config,
                source_eval_run_id=run.run_id,
                paired=pairs,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not crash the run
            return {"requested": False, "reason": f"failure fit failed: {exc}"}
        path = distribution.save(config.failure_analysis.output_dir)
        controller = FlywheelController(config)
        cycle = controller.decide(
            eval_run_id=run.run_id,
            success_rate=run.aggregates.get("success_rate", 0.0),
            expert_success_rate=(
                paired_summary.get("expert", {}).get("success_rate", 0.0) if paired_summary else 0.0
            ),
            n_failures=distribution.n_failures,
            policy_only_failures=(
                paired_summary.get("policy_only_failures", 0) if paired_summary else 0
            ),
            checkpoint_hash=run.policy_info.get("checkpoint_hash", "none"),
            failure_distribution_id=distribution.distribution_id,
            episode_ids=[record.episode_id for record in records],
            episode_hash=run.config_hash,
        )
        if cycle.decision != "REQUEST_TARGETED_GENERATION":
            return {
                "requested": False,
                "decision": cycle.decision,
                "reason": cycle.reason,
                "failure_distribution": str(path),
            }
        if self._generate_client is None:
            return {"requested": False, "reason": "GenerateRollouts service is unavailable"}
        return {
            "requested": True,
            "decision": cycle.decision,
            "failure_distribution": str(path),
            "num_rollouts": cycle.rollouts_requested,
            "service_result": self._call_generate(cycle.rollouts_requested, path, config),
        }

    def _call_generate(self, count: int, path: Path, config: Any) -> dict[str, Any]:
        if not self._generate_client.wait_for_service(timeout_sec=5.0):
            return {"success": False, "message": "generator service unavailable"}
        request = GenerateRollouts.Request()
        request.num_rollouts = int(count)
        request.seed = int(config.settings.targeted_seed_range[0])
        request.task_instruction = config.settings.task_instruction
        request.sampling_mode = contracts.SAMPLING_FAILURE_TARGETED
        request.failure_distribution_ref = str(path)
        request.include_perturbed = False
        future = self._generate_client.call_async(request)
        deadline = time.monotonic() + 120.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"success": False, "message": "generator call timed out"}
        response = future.result()
        return {
            "success": bool(response.success),
            "num_generated": int(response.num_generated),
            "num_failed": int(response.num_failed),
            "message": str(response.message),
        }

    def destroy_node(self) -> None:
        with self._backend_lock:
            if self._backend is not None:
                self._backend.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ClosedLoopEvalNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
