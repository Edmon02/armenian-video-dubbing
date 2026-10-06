#!/usr/bin/env python3
"""
FastAPI Backend for Dubbing Service — Production

Features:
  - Async video processing with job queue
  - API key authentication
  - Prometheus metrics (/metrics)
  - Rate limiting awareness (nginx handles actual limits)
  - CORS, health checks, structured logging

Usage:
    python src/api/fastapi_server.py [--port 8000]
    uvicorn src.api.fastapi_server:create_app --factory --host 0.0.0.0 --port 8000

API Endpoints:
  POST   /api/v1/dub              - Submit dubbing job
  GET    /api/v1/jobs/{job_id}     - Get job status
  DELETE /api/v1/jobs/{job_id}     - Cancel job
  GET    /api/v1/jobs              - List jobs
  GET    /api/v1/results/{job_id}  - Download result
  GET    /api/v1/health            - Health check
  GET    /metrics                  - Prometheus metrics
"""

import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    BackgroundTasks,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.pipeline import DubbingPipeline
from src.utils.logger import setup_logger


# ============================================================================
# Data Models
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DubbingJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    input_file: str
    output_file: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[dict] = None


# ============================================================================
# Metrics (Prometheus-compatible)
# ============================================================================

class Metrics:
    """Simple Prometheus-compatible metrics collector."""

    def __init__(self):
        self.requests_total = 0
        self.requests_by_status: dict[int, int] = {}
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.active_jobs = 0
        self.request_duration_sum = 0.0
        self.request_count = 0

    def record_request(self, status_code: int, duration: float):
        self.requests_total += 1
        self.requests_by_status[status_code] = self.requests_by_status.get(status_code, 0) + 1
        self.request_duration_sum += duration
        self.request_count += 1

    def to_prometheus(self) -> str:
        lines = []
        lines.append("# HELP armtts_requests_total Total HTTP requests")
        lines.append("# TYPE armtts_requests_total counter")
        lines.append(f"armtts_requests_total {self.requests_total}")

        lines.append("# HELP armtts_requests_by_status HTTP requests by status code")
        lines.append("# TYPE armtts_requests_by_status counter")
        for code, count in sorted(self.requests_by_status.items()):
            lines.append(f'armtts_requests_by_status{{code="{code}"}} {count}')

        lines.append("# HELP armtts_jobs_submitted_total Total dubbing jobs submitted")
        lines.append("# TYPE armtts_jobs_submitted_total counter")
        lines.append(f"armtts_jobs_submitted_total {self.jobs_submitted}")

        lines.append("# HELP armtts_jobs_completed_total Total dubbing jobs completed")
        lines.append("# TYPE armtts_jobs_completed_total counter")
        lines.append(f"armtts_jobs_completed_total {self.jobs_completed}")

        lines.append("# HELP armtts_jobs_failed_total Total dubbing jobs failed")
        lines.append("# TYPE armtts_jobs_failed_total counter")
        lines.append(f"armtts_jobs_failed_total {self.jobs_failed}")

        lines.append("# HELP armtts_active_jobs Currently processing jobs")
        lines.append("# TYPE armtts_active_jobs gauge")
        lines.append(f"armtts_active_jobs {self.active_jobs}")

        avg_duration = (self.request_duration_sum / self.request_count) if self.request_count else 0
        lines.append("# HELP armtts_request_duration_avg Average request duration seconds")
        lines.append("# TYPE armtts_request_duration_avg gauge")
        lines.append(f"armtts_request_duration_avg {avg_duration:.4f}")

        gpu_available = 1 if torch.cuda.is_available() else 0
        lines.append("# HELP armtts_gpu_available GPU availability (1=yes, 0=no)")
        lines.append("# TYPE armtts_gpu_available gauge")
        lines.append(f"armtts_gpu_available {gpu_available}")

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append("# HELP armtts_gpu_memory_used_gb GPU memory used in GB")
            lines.append("# TYPE armtts_gpu_memory_used_gb gauge")
            lines.append(f"armtts_gpu_memory_used_gb {mem_used:.2f}")
            lines.append("# HELP armtts_gpu_memory_total_gb GPU memory total in GB")
            lines.append("# TYPE armtts_gpu_memory_total_gb gauge")
            lines.append(f"armtts_gpu_memory_total_gb {mem_total:.2f}")

        return "\n".join(lines) + "\n"


# ============================================================================
# Authentication
# ============================================================================

def get_api_key():
    """Get configured API key from environment."""
    return os.environ.get("ARMTTS_API_KEY", "")


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if one is configured.

    If ARMTTS_API_KEY env var is not set, authentication is disabled.
    """
    configured_key = get_api_key()
    if not configured_key:
        return  # Auth disabled

    if not x_api_key or x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ============================================================================
# Job Queue Manager
# ============================================================================

class JobQueue:
    """In-memory job queue for dubbing tasks."""

    def __init__(self):
        self.jobs: dict[str, DubbingJob] = {}

    def create_job(self, input_file: str) -> DubbingJob:
        job_id = str(uuid.uuid4())
        job = DubbingJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now().isoformat(),
            input_file=input_file,
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[DubbingJob]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            self.update_job(job_id, status=JobStatus.CANCELLED)
            return True
        return False

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 10) -> List[DubbingJob]:
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Armenian Video Dubbing API",
        description="Production API for video dubbing with ASR, translation, TTS, and lip-sync",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ARMTTS_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_logger()
    pipeline = DubbingPipeline()
    job_queue = JobQueue()
    metrics = Metrics()
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Middleware — request timing for metrics
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        metrics.record_request(response.status_code, duration)
        return response

    # ------------------------------------------------------------------
    # Public endpoints (no auth)
    # ------------------------------------------------------------------

    @app.get("/api/v1/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "gpu": "available" if torch.cuda.is_available() else "unavailable",
            "active_jobs": metrics.active_jobs,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return metrics.to_prometheus()

    # ------------------------------------------------------------------
    # Protected endpoints (API key required if configured)
    # ------------------------------------------------------------------

    @app.post("/api/v1/dub", dependencies=[Depends(verify_api_key)])
    async def submit_job(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        speaker: Optional[UploadFile] = File(None),
        emotion: str = Form("neutral"),
        src_lang: str = Form("eng"),
        tgt_lang: str = Form("hye"),
        dialect: str = Form("eastern"),
        skip_lipsync: bool = Form(False),
        keep_background: bool = Form(True),
        voice_consent: bool = Form(False),
    ):
        """Submit a video dubbing job."""
        try:
            video_path = upload_dir / f"{uuid.uuid4()}_{video.filename}"
            with open(video_path, "wb") as f:
                f.write(await video.read())

            speaker_path = None
            if speaker:
                speaker_path = str(upload_dir / f"{uuid.uuid4()}_{speaker.filename}")
                with open(speaker_path, "wb") as f:
                    f.write(await speaker.read())

            job = job_queue.create_job(str(video_path))
            metrics.jobs_submitted += 1

            background_tasks.add_task(
                _process_job,
                job_id=job.job_id,
                pipeline=pipeline,
                job_queue=job_queue,
                metrics=metrics,
                emotion=emotion,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dialect=dialect,
                speaker_path=speaker_path,
                skip_lipsync=skip_lipsync,
                keep_background=keep_background,
                voice_consent=voice_consent,
            )

            logger.info("Created job {} for {}", job.job_id, video.filename)

            return {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
            }

        except Exception as e:
            logger.error("Failed to submit job: {}", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    async def get_job_status(job_id: str):
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump()

    @app.delete("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    async def cancel_job(job_id: str):
        if job_queue.cancel_job(job_id):
            return {"status": "cancelled"}
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

    @app.get("/api/v1/jobs", dependencies=[Depends(verify_api_key)])
    async def list_jobs(
        status: Optional[JobStatus] = Query(None),
        limit: int = Query(10, ge=1, le=100),
    ):
        jobs = job_queue.list_jobs(status=status, limit=limit)
        return {"jobs": [j.model_dump() for j in jobs]}

    @app.get("/api/v1/results/{job_id}", dependencies=[Depends(verify_api_key)])
    async def download_result(job_id: str):
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job not completed")
        if not job.output_file or not Path(job.output_file).exists():
            raise HTTPException(status_code=404, detail="Output file not found")

        return FileResponse(
            job.output_file,
            filename=Path(job.output_file).name,
            media_type="video/mp4",
        )

    return app


# ============================================================================
# Background Processing
# ============================================================================

def _process_job(
    job_id: str,
    pipeline: DubbingPipeline,
    job_queue: JobQueue,
    metrics: Metrics,
    emotion: str,
    src_lang: str,
    tgt_lang: str,
    dialect: str = "eastern",
    speaker_path: Optional[str] = None,
    skip_lipsync: bool = False,
    keep_background: bool = True,
    voice_consent: bool = False,
):
    """Process dubbing job synchronously in background thread."""
    job = job_queue.get_job(job_id)
    if not job:
        return

    metrics.active_jobs += 1

    try:
        job_queue.update_job(job_id, status=JobStatus.PROCESSING, progress=0.1)
        logger.info("Processing job: {}", job_id)

        output_path = Path(f"outputs/video/dubbed_{job_id}.mp4")

        result = pipeline.dub_video(
            video_path=job.input_file,
            reference_speaker_audio=speaker_path,
            emotion=emotion,
            output_path=str(output_path),
            keep_background=keep_background,
            skip_lipsync=skip_lipsync,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dialect=dialect,
            voice_consent=voice_consent,
        )

        if "error" in result:
            job_queue.update_job(job_id, status=JobStatus.FAILED, error=result["error"])
            metrics.jobs_failed += 1
        else:
            job_queue.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                output_file=str(output_path),
                result=result,
                progress=1.0,
            )
            metrics.jobs_completed += 1
            logger.info("Completed job: {}", job_id)

    except Exception as e:
        logger.error("Job {} failed: {}", job_id, e)
        job_queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))
        metrics.jobs_failed += 1

    finally:
        metrics.active_jobs -= 1


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="FastAPI Dubbing Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    app = create_app()

    logger.info("Starting FastAPI server on {}:{}", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
