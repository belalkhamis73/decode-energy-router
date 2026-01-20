"""
Training API Routes
Handles model training operations for all energy prediction models
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import asyncio
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.ml_service import MLService
from app.models.training import TrainingJob, TrainingStatus, ModelType

router = APIRouter(prefix="/training", tags=["training"])


# Pydantic Models
class TrainingConfig(BaseModel):
    """Configuration for training jobs"""
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)
    early_stopping_patience: int = Field(default=10, ge=1, le=50)
    use_gpu: bool = Field(default=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "use_gpu": False
            }
        }


class TrainingRequest(BaseModel):
    """Request to start training"""
    config: Optional[TrainingConfig] = None
    data_start_date: Optional[datetime] = None
    data_end_date: Optional[datetime] = None
    retrain: bool = Field(default=False, description="Force retrain even if model exists")


class TrainingResponse(BaseModel):
    """Response for training job creation"""
    job_id: str
    model_type: str
    status: str
    message: str
    created_at: datetime


class TrainingStatusResponse(BaseModel):
    """Detailed training status"""
    job_id: str
    model_type: str
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class TrainingMetrics(BaseModel):
    """Training metrics response"""
    job_id: str
    model_type: str
    metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    best_epoch: Optional[int] = None
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None


class DeploymentRequest(BaseModel):
    """Request to deploy model"""
    replace_current: bool = Field(default=True, description="Replace current production model")
    backup_current: bool = Field(default=True, description="Backup current model before replacing")


class DeploymentResponse(BaseModel):
    """Deployment response"""
    job_id: str
    model_type: str
    deployed: bool
    version: str
    deployed_at: datetime
    message: str


# Background task functions
async def run_training_job(
    job_id: str,
    model_type: str,
    config: TrainingConfig,
    db: Session
):
    """Background task to run model training"""
    ml_service = MLService()
    
    try:
        # Update job status to running
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            db.commit()
        
        # Run training based on model type
        if model_type == ModelType.SOLAR:
            result = await ml_service.train_solar_model(config.dict())
        elif model_type == ModelType.WIND:
            result = await ml_service.train_wind_model(config.dict())
        elif model_type == ModelType.BATTERY:
            result = await ml_service.train_battery_model(config.dict())
        elif model_type == ModelType.GRID:
            result = await ml_service.train_grid_model(config.dict())
        elif model_type == ModelType.ENSEMBLE:
            result = await ml_service.train_ensemble_model(config.dict())
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Update job with results
        if job:
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.metrics = result.get('metrics', {})
            job.model_path = result.get('model_path')
            db.commit()
            
    except Exception as e:
        # Update job with error
        if job:
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            db.commit()


# Route Endpoints
@router.post("/solar/start", response_model=TrainingResponse)
async def start_solar_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start solar power prediction model training"""
    config = request.config or TrainingConfig()
    
    # Create training job
    job = TrainingJob(
        model_type=ModelType.SOLAR,
        status=TrainingStatus.PENDING,
        config=config.dict(),
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background training
    background_tasks.add_task(
        run_training_job,
        job_id=str(job.id),
        model_type=ModelType.SOLAR,
        config=config,
        db=db
    )
    
    return TrainingResponse(
        job_id=str(job.id),
        model_type=ModelType.SOLAR,
        status=TrainingStatus.PENDING,
        message="Solar model training started",
        created_at=job.created_at
    )


@router.post("/wind/start", response_model=TrainingResponse)
async def start_wind_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start wind power prediction model training"""
    config = request.config or TrainingConfig()
    
    job = TrainingJob(
        model_type=ModelType.WIND,
        status=TrainingStatus.PENDING,
        config=config.dict(),
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(
        run_training_job,
        job_id=str(job.id),
        model_type=ModelType.WIND,
        config=config,
        db=db
    )
    
    return TrainingResponse(
        job_id=str(job.id),
        model_type=ModelType.WIND,
        status=TrainingStatus.PENDING,
        message="Wind model training started",
        created_at=job.created_at
    )


@router.post("/battery/start", response_model=TrainingResponse)
async def start_battery_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start battery optimization model training"""
    config = request.config or TrainingConfig()
    
    job = TrainingJob(
        model_type=ModelType.BATTERY,
        status=TrainingStatus.PENDING,
        config=config.dict(),
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(
        run_training_job,
        job_id=str(job.id),
        model_type=ModelType.BATTERY,
        config=config,
        db=db
    )
    
    return TrainingResponse(
        job_id=str(job.id),
        model_type=ModelType.BATTERY,
        status=TrainingStatus.PENDING,
        message="Battery model training started",
        created_at=job.created_at
    )


@router.post("/grid/start", response_model=TrainingResponse)
async def start_grid_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start grid demand prediction model training"""
    config = request.config or TrainingConfig()
    
    job = TrainingJob(
        model_type=ModelType.GRID,
        status=TrainingStatus.PENDING,
        config=config.dict(),
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(
        run_training_job,
        job_id=str(job.id),
        model_type=ModelType.GRID,
        config=config,
        db=db
    )
    
    return TrainingResponse(
        job_id=str(job.id),
        model_type=ModelType.GRID,
        status=TrainingStatus.PENDING,
        message="Grid model training started",
        created_at=job.created_at
    )


@router.post("/ensemble/start", response_model=TrainingResponse)
async def start_ensemble_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Train ensemble meta-learner combining all models"""
    config = request.config or TrainingConfig()
    
    job = TrainingJob(
        model_type=ModelType.ENSEMBLE,
        status=TrainingStatus.PENDING,
        config=config.dict(),
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(
        run_training_job,
        job_id=str(job.id),
        model_type=ModelType.ENSEMBLE,
        config=config,
        db=db
    )
    
    return TrainingResponse(
        job_id=str(job.id),
        model_type=ModelType.ENSEMBLE,
        status=TrainingStatus.PENDING,
        message="Ensemble model training started",
        created_at=job.created_at
    )


@router.get("/{job_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get status of a training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Calculate progress
    progress = 0.0
    if job.status == TrainingStatus.COMPLETED:
        progress = 100.0
    elif job.status == TrainingStatus.RUNNING and job.metrics:
        current_epoch = job.metrics.get('current_epoch', 0)
        total_epochs = job.config.get('epochs', 100)
        progress = (current_epoch / total_epochs) * 100
    
    return TrainingStatusResponse(
        job_id=str(job.id),
        model_type=job.model_type,
        status=job.status,
        progress=progress,
        current_epoch=job.metrics.get('current_epoch') if job.metrics else None,
        total_epochs=job.config.get('epochs'),
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        estimated_completion=None  # Could calculate based on current progress
    )


@router.get("/{job_id}/metrics", response_model=TrainingMetrics)
async def get_training_metrics(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed training metrics for a job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Training job not completed. Current status: {job.status}"
        )
    
    metrics = job.metrics or {}
    
    return TrainingMetrics(
        job_id=str(job.id),
        model_type=job.model_type,
        metrics=metrics.get('train', {}),
        validation_metrics=metrics.get('validation', {}),
        best_epoch=metrics.get('best_epoch'),
        final_loss=metrics.get('final_loss'),
        final_val_loss=metrics.get('final_val_loss')
    )


@router.post("/{job_id}/deploy", response_model=DeploymentResponse)
async def deploy_model(
    job_id: str,
    request: DeploymentRequest,
    db: Session = Depends(get_db)
):
    """Deploy trained model to production"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot deploy incomplete model. Status: {job.status}"
        )
    
    if not job.model_path:
        raise HTTPException(status_code=400, detail="No model file found")
    
    try:
        ml_service = MLService()
        
        # Deploy the model
        deployment_result = await ml_service.deploy_model(
            model_type=job.model_type,
            model_path=job.model_path,
            replace_current=request.replace_current,
            backup_current=request.backup_current
        )
        
        # Update job deployment status
        job.deployed = True
        job.deployed_at = datetime.utcnow()
        db.commit()
        
        return DeploymentResponse(
            job_id=str(job.id),
            model_type=job.model_type,
            deployed=True,
            version=deployment_result.get('version', '1.0.0'),
            deployed_at=job.deployed_at,
            message=f"{job.model_type} model deployed successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deployment failed: {str(e)}"
        )


@router.get("/jobs", response_model=List[TrainingStatusResponse])
async def list_training_jobs(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all training jobs with optional filters"""
    query = db.query(TrainingJob)
    
    if model_type:
        query = query.filter(TrainingJob.model_type == model_type)
    if status:
        query = query.filter(TrainingJob.status == status)
    
    jobs = query.order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        TrainingStatusResponse(
            job_id=str(job.id),
            model_type=job.model_type,
            status=job.status,
            progress=100.0 if job.status == TrainingStatus.COMPLETED else 0.0,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )
        for job in jobs
    ]


@router.delete("/{job_id}")
async def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Cancel a running training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status not in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    job.status = TrainingStatus.FAILED
    job.error_message = "Job cancelled by user"
    job.completed_at = datetime.utcnow()
    db.commit()
    
    return {"message": f"Training job {job_id} cancelled", "status": "cancelled"}
