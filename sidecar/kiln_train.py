#!/usr/bin/env python3
"""Kiln training sidecar — receives training jobs from the Rust server via unix socket.

Protocol: JSON-line (one JSON object per line, newline-delimited) over a unix domain socket.

Request types (Rust -> Python):
  sft_request   - Submit an SFT training job
  grpo_request  - Submit a GRPO training job
  status_query  - Query status of a running/completed job
  shutdown      - Gracefully shut down the sidecar

Response types (Python -> Rust):
  job_accepted  - Job was queued
  job_status    - Current status of a job
  job_complete  - Job finished, adapter written to disk
  error         - Something went wrong
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("kiln-train")


class JobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    job_id: str
    job_type: str  # "sft" or "grpo"
    request: dict
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    epoch: int = 0
    loss: Optional[float] = None
    adapter_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class TrainingSidecar:
    def __init__(self, socket_path: str, adapter_dir: str):
        self.socket_path = socket_path
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, TrainingJob] = {}
        self.job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._shutdown = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the unix socket server and training worker."""
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = await asyncio.start_unix_server(
            self._handle_connection, path=self.socket_path
        )
        os.chmod(self.socket_path, 0o660)
        log.info("listening on %s", self.socket_path)

        self._worker_task = asyncio.create_task(self._training_worker())

        # Handle signals for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: self._shutdown.set())

        try:
            async with server:
                await self._shutdown.wait()
        finally:
            log.info("shutting down")
            if self._worker_task:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle a single connection — read JSON lines and respond."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError as e:
                    resp = {"type": "error", "job_id": "", "message": f"invalid JSON: {e}"}
                    writer.write((json.dumps(resp) + "\n").encode())
                    await writer.drain()
                    continue

                resp = self._dispatch(msg)
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("connection handler error: %s", e)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _dispatch(self, msg: dict) -> dict:
        """Route an incoming message to the appropriate handler."""
        msg_type = msg.get("type", "")

        if msg_type == "sft_request":
            return self._handle_sft_request(msg)
        elif msg_type == "grpo_request":
            return self._handle_grpo_request(msg)
        elif msg_type == "status_query":
            return self._handle_status_query(msg)
        elif msg_type == "shutdown":
            self._shutdown.set()
            return {"type": "job_accepted", "job_id": "", "message": "shutting down"}
        else:
            return {
                "type": "error",
                "job_id": msg.get("job_id", ""),
                "message": f"unknown message type: {msg_type}",
            }

    def _handle_sft_request(self, msg: dict) -> dict:
        job_id = msg.get("job_id", str(uuid.uuid4()))
        job = TrainingJob(
            job_id=job_id,
            job_type="sft",
            request=msg,
        )
        self.jobs[job_id] = job
        self.job_queue.put_nowait(job_id)
        log.info(
            "queued SFT job %s (%d examples, %d epochs)",
            job_id,
            len(msg.get("examples", [])),
            msg.get("epochs", 3),
        )
        return {"type": "job_accepted", "job_id": job_id}

    def _handle_grpo_request(self, msg: dict) -> dict:
        job_id = msg.get("job_id", str(uuid.uuid4()))
        job = TrainingJob(
            job_id=job_id,
            job_type="grpo",
            request=msg,
        )
        self.jobs[job_id] = job
        self.job_queue.put_nowait(job_id)
        log.info(
            "queued GRPO job %s (%d groups)",
            job_id,
            len(msg.get("groups", [])),
        )
        return {"type": "job_accepted", "job_id": job_id}

    def _handle_status_query(self, msg: dict) -> dict:
        job_id = msg.get("job_id", "")
        job = self.jobs.get(job_id)
        if job is None:
            return {
                "type": "error",
                "job_id": job_id,
                "message": f"unknown job: {job_id}",
            }
        resp: dict = {
            "type": "job_status",
            "job_id": job_id,
            "state": job.state.value,
            "progress": job.progress,
            "epoch": job.epoch,
        }
        if job.loss is not None:
            resp["loss"] = job.loss
        if job.adapter_path is not None:
            resp["adapter_path"] = job.adapter_path
        if job.error_message is not None:
            resp["message"] = job.error_message
        return resp

    async def _training_worker(self):
        """Process training jobs from the queue one at a time."""
        while True:
            try:
                job_id = await self.job_queue.get()
            except asyncio.CancelledError:
                return

            job = self.jobs.get(job_id)
            if job is None:
                continue

            job.state = JobState.RUNNING
            log.info("starting %s job %s", job.job_type, job_id)

            try:
                if job.job_type == "sft":
                    await self._run_sft(job)
                elif job.job_type == "grpo":
                    await self._run_grpo(job)

                job.state = JobState.COMPLETED
                job.progress = 1.0
                log.info("completed %s job %s", job.job_type, job_id)
            except asyncio.CancelledError:
                job.state = JobState.FAILED
                job.error_message = "cancelled"
                raise
            except Exception as e:
                job.state = JobState.FAILED
                job.error_message = str(e)
                log.exception("job %s failed: %s", job_id, e)

    async def _run_sft(self, job: TrainingJob):
        """Mock SFT training loop. Will be replaced with real PEFT training."""
        req = job.request
        epochs = req.get("epochs", 3)
        num_examples = len(req.get("examples", []))
        adapter_name = req.get("adapter_name", f"sft-{job.job_id[:8]}")

        # Simulate training: one step per example per epoch
        total_steps = max(epochs * max(num_examples, 1), 1)
        step = 0
        mock_loss = 2.5

        for epoch in range(epochs):
            job.epoch = epoch + 1
            for _ex in range(max(num_examples, 1)):
                step += 1
                job.progress = step / total_steps
                mock_loss *= 0.85  # loss decreases over time
                job.loss = round(mock_loss, 4)
                await asyncio.sleep(0.1)  # simulate compute time

        # Write a placeholder adapter directory
        adapter_path = self.adapter_dir / adapter_name
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model": "Qwen/Qwen3.5-4B",
                    "lora_rank": req.get("lora_rank", 16),
                    "lora_alpha": req.get("lora_alpha", 32.0),
                    "target_modules": "all-linear",
                    "task_type": "CAUSAL_LM",
                },
                indent=2,
            )
        )
        job.adapter_path = str(adapter_path)
        log.info("wrote mock adapter to %s", adapter_path)

    async def _run_grpo(self, job: TrainingJob):
        """Mock GRPO training loop. Will be replaced with real GRPO training."""
        req = job.request
        num_groups = len(req.get("groups", []))
        adapter_name = req.get("adapter_name", f"grpo-{job.job_id[:8]}")

        # Simulate training: a few steps per group
        total_steps = max(num_groups * 3, 1)
        step = 0
        mock_loss = 1.8

        for _g in range(max(num_groups, 1)):
            for _s in range(3):
                step += 1
                job.progress = step / total_steps
                mock_loss *= 0.90
                job.loss = round(mock_loss, 4)
                await asyncio.sleep(0.1)

        adapter_path = self.adapter_dir / adapter_name
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model": "Qwen/Qwen3.5-4B",
                    "lora_rank": req.get("lora_rank", 16),
                    "lora_alpha": req.get("lora_alpha", 32.0),
                    "target_modules": "all-linear",
                    "task_type": "CAUSAL_LM",
                    "training_type": "grpo",
                },
                indent=2,
            )
        )
        job.adapter_path = str(adapter_path)
        log.info("wrote mock adapter to %s", adapter_path)


def main():
    parser = argparse.ArgumentParser(description="Kiln training sidecar")
    parser.add_argument(
        "--socket-path",
        default="/tmp/kiln-train.sock",
        help="Path to the unix domain socket (default: /tmp/kiln-train.sock)",
    )
    parser.add_argument(
        "--adapter-dir",
        default="adapters",
        help="Directory to store trained adapter weights (default: adapters)",
    )
    args = parser.parse_args()

    sidecar = TrainingSidecar(
        socket_path=args.socket_path,
        adapter_dir=args.adapter_dir,
    )
    asyncio.run(sidecar.start())


if __name__ == "__main__":
    main()
