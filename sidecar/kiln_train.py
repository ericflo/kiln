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
import functools
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

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # Lazy-loaded base model and tokenizer (cached across jobs)
        self._base_model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_path: Optional[str] = None

    def _ensure_model_loaded(self, model_path: Optional[str] = None) -> None:
        """Lazy-load the base model and tokenizer, reusing across jobs."""
        path = model_path or os.environ.get("KILN_MODEL_PATH")
        if path is None:
            raise RuntimeError(
                "No model path provided. Set KILN_MODEL_PATH or include "
                "'model_path' in the training request."
            )

        # If already loaded with the same path, reuse
        if self._base_model is not None and self._model_path == path:
            return

        log.info("loading base model from %s", path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._base_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model_path = path
        log.info("base model loaded on %s (%s)", device, dtype)

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

    def _tokenize_examples(
        self, examples: list[dict], max_length: int = 2048
    ) -> list[dict]:
        """Tokenize SFT examples into input_ids and labels tensors.

        Each example should have 'messages' (chat format) or 'input'/'output' pairs.
        Labels are set to -100 for the prompt portion so loss is only on completions.
        """
        tokenizer = self._tokenizer
        tokenized = []

        for ex in examples:
            if "messages" in ex:
                # Chat format: [{"role": "user", "content": "..."}, ...]
                # Tokenize the full conversation, mask everything except assistant turns
                text = tokenizer.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False
                )
                full_ids = tokenizer.encode(text, add_special_tokens=False)

                # Build labels: find assistant content and unmask only those tokens
                # Simple approach: tokenize without last assistant turn for prompt length
                prompt_messages = []
                for msg in ex["messages"]:
                    if msg["role"] == "assistant":
                        # Tokenize everything up to this point as prompt
                        prompt_text = tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        prompt_ids = tokenizer.encode(
                            prompt_text, add_special_tokens=False
                        )
                        break
                    prompt_messages.append(msg)
                else:
                    # No assistant message found, skip
                    continue

                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            elif "input" in ex and "output" in ex:
                # Simple input/output pair
                prompt_ids = tokenizer.encode(ex["input"], add_special_tokens=True)
                output_ids = tokenizer.encode(ex["output"], add_special_tokens=False)
                full_ids = prompt_ids + output_ids
                labels = [-100] * len(prompt_ids) + output_ids
            else:
                continue

            # Truncate to max_length
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

            tokenized.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        return tokenized

    async def _run_sft(self, job: TrainingJob):
        """Run real SFT training with PEFT LoRA."""
        req = job.request
        epochs = req.get("epochs", 3)
        examples = req.get("examples", [])
        adapter_name = req.get("adapter_name", f"sft-{job.job_id[:8]}")
        lora_rank = req.get("lora_rank", 16)
        lora_alpha = req.get("lora_alpha", 32.0)
        learning_rate = req.get("learning_rate", 1e-4)
        max_length = req.get("max_length", 2048)
        model_path = req.get("model_path")

        if not examples:
            raise ValueError("SFT job requires at least one example")

        # Load base model (lazy, cached across jobs)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, functools.partial(self._ensure_model_loaded, model_path)
        )

        device = next(self._base_model.parameters()).device

        # Tokenize examples
        tokenized = await loop.run_in_executor(
            None,
            functools.partial(self._tokenize_examples, examples, max_length),
        )
        if not tokenized:
            raise ValueError("No valid examples after tokenization")

        log.info(
            "SFT job %s: %d examples tokenized, %d epochs, rank=%d",
            job.job_id,
            len(tokenized),
            epochs,
            lora_rank,
        )

        # Create LoRA config and wrap model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        def _train():
            peft_model = get_peft_model(self._base_model, lora_config)
            peft_model.train()
            trainable = sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            )
            log.info("trainable parameters: %d", trainable)

            optimizer = torch.optim.AdamW(
                peft_model.parameters(), lr=learning_rate
            )

            total_steps = epochs * len(tokenized)
            step = 0

            for epoch in range(epochs):
                job.epoch = epoch + 1
                epoch_loss = 0.0

                for ex in tokenized:
                    input_ids = ex["input_ids"].unsqueeze(0).to(device)
                    labels = ex["labels"].unsqueeze(0).to(device)

                    outputs = peft_model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    step += 1
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    job.progress = step / total_steps
                    job.loss = round(loss_val, 4)

                avg_loss = epoch_loss / len(tokenized)
                log.info(
                    "job %s epoch %d/%d avg_loss=%.4f",
                    job.job_id,
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

            # Save the trained adapter
            adapter_path = self.adapter_dir / adapter_name
            adapter_path.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(adapter_path))
            job.adapter_path = str(adapter_path)
            log.info("saved LoRA adapter to %s", adapter_path)

            # Clean up: remove LoRA from the base model so it's reusable
            peft_model.unload()

        await loop.run_in_executor(None, _train)

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
