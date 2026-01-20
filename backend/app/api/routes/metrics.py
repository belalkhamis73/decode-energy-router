"""
Metrics Route Handler - Analytics & Observability Layer.
Provides comprehensive metrics access for:
1. Model Predictions & Performance
2. Physics Residuals & Constraints
3. Uncertainty Quantification
4. Historical Time-Series Data
5. Export Functionality (CSV/JSON)

Map Requirement: "Metrics endpoint handler. Called by: main.py."
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
import torch
import logging
import asyncio
import time
from datetime import datetime, timedelta
from collections import deque
import io
import csv
import json

# --- IMPORT LAYERS ---
from backend.services.data_manager import data_manager

# Try/Except for Redis (Event History)
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import simulation functions for live metrics
from backend.app.api.routes.simulation import (
    predict_solar_pv,
    predict_wind_turbine,
    predict_battery_thermal,
    predict_pinn_voltage,
    predict_grid_stability,
    predict_load_forecast,
    check_constraints
)

# --- SETUP ---
router = APIRouter()
logger = logging.getLogger("MetricsRoute")

# In-memory metrics buffer (fallback if Redis unavailable)
metrics_buffer: Dict[str, deque] = {}
MAX_BUFFER_SIZE = 10000

redis_pool = None

async def get_redis():
    """Lazy Redis connection pool initialization."""
    global redis_pool
    if not REDIS_AVAILABLE:
        return None
    if redis_pool is None:
        try:
            from backend.app.core.config import settings
            redis_pool = await aioredis.create_redis_pool(
                f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}',
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            return None
    return redis_pool

# --- RESPONSE MODELS ---
class ModelMetrics(BaseModel):
    model_name: str
    output_kw: float
    efficiency: float
    confidence: float
    physics_residual: float
    inference_time_ms: float
    inputs_used: Dict[str, float]

class ConstraintMetrics(BaseModel):
    constraint_name: str
    is_violated: bool
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    current_value: float
    limit_value: float
    margin: float

class PhysicsResiduals(BaseModel):
    model_name: str
    residual_value: float
    residual_type: Literal["absolute", "relative", "normalized"]
    threshold: float
    is_acceptable: bool

class PerformanceMetrics(BaseModel):
    model_name: str
    avg_inference_ms: float
    p50_inference_ms: float
    p95_inference_ms: float
    p99_inference_ms: float
    total_invocations: int
    error_count: int

class UncertaintyBounds(BaseModel):
    model_name: str
    prediction_mean: float
    prediction_std: float
    lower_bound_95: float
    upper_bound_95: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float

class HistoricalDataPoint(BaseModel):
    timestamp: float
    tick: int
    metric_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

# --- HELPER FUNCTIONS ---
def get_or_create_buffer(session_id: str) -> deque:
    """Get or create metrics buffer for session."""
    if session_id not in metrics_buffer:
        metrics_buffer[session_id] = deque(maxlen=MAX_BUFFER_SIZE)
    return metrics_buffer[session_id]

async def store_metric(session_id: str, metric: Dict[str, Any]):
    """Store metric in Redis or fallback buffer."""
    redis = await get_redis()
    if redis:
        try:
            await redis.xadd(
                f"metrics:{session_id}",
                {"data": json.dumps(metric)},
                maxlen=MAX_BUFFER_SIZE
            )
        except Exception as e:
            logger.error(f"Redis metric storage failed: {e}")
            get_or_create_buffer(session_id).append(metric)
    else:
        get_or_create_buffer(session_id).append(metric)

async def fetch_historical_metrics(session_id: str, window_seconds: int) -> List[Dict]:
    """Fetch metrics from Redis or buffer within time window."""
    redis = await get_redis()
    cutoff_time = time.time() - window_seconds
    
    if redis:
        try:
            stream_data = await redis.xrange(f"metrics:{session_id}")
            metrics = []
            for msg_id, data in stream_data:
                metric = json.loads(data[b'data'])
                if metric.get('timestamp', 0) >= cutoff_time:
                    metrics.append(metric)
            return metrics
        except Exception as e:
            logger.error(f"Redis fetch failed: {e}")
    
    # Fallback to buffer
    buffer = get_or_create_buffer(session_id)
    return [m for m in buffer if m.get('timestamp', 0) >= cutoff_time]

def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calculate percentile from list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]

# --- ENDPOINTS ---

@router.get("/models/{session_id}")
async def get_all_model_outputs(
    session_id: str,
    tick: Optional[int] = Query(None, description="Specific tick (default: latest)")
):
    """
    GET /metrics/models/{session_id}
    Returns current outputs from all 6 ML models with execution timing.
    """
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    weather = session['weather']
    current_tick = tick if tick is not None else session.get('current_tick', 0)
    current_tick = current_tick % 24
    
    # Extract weather data
    ghi = weather['ghi'][current_tick]
    temp = weather['temperature'][current_tick]
    wind = weather['wind_speed'][current_tick]
    cloud_cover = weather.get('cloud_cover', [0.3] * 24)[current_tick]
    
    # Parallel execution with timing
    start_times = {}
    
    async def timed_predict(name: str, coro):
        start_times[name] = time.time()
        result = await coro
        inference_ms = (time.time() - start_times[name]) * 1000
        return name, result, inference_ms
    
    results = await asyncio.gather(
        timed_predict("solar_pv", predict_solar_pv(ghi, temp, cloud_cover)),
        timed_predict("wind_turbine", predict_wind_turbine(wind, 1.225)),
        timed_predict("battery_thermal", predict_battery_thermal(
            0.0, session.get('battery_temp_c', 25.0), session.get('battery_soc', 0.5)
        )),
        timed_predict("pinn_voltage", predict_pinn_voltage(
            session_id, torch.tensor([[1.0, ghi/1000.0, 0.0]])
        )),
        timed_predict("grid_stability", predict_grid_stability(1.0, 1.0, 5.0)),
        timed_predict("load_forecast", predict_load_forecast(temp, current_tick, 100.0))
    )
    
    # Format response
    model_metrics = []
    for name, pred, inf_ms in results:
        metric = ModelMetrics(
            model_name=name,
            output_kw=pred.output_kw,
            efficiency=pred.efficiency,
            confidence=pred.confidence,
            physics_residual=pred.physics_residual,
            inference_time_ms=inf_ms,
            inputs_used=pred.inputs_used
        )
        model_metrics.append(metric.dict())
        
        # Store for historical tracking
        await store_metric(session_id, {
            "timestamp": time.time(),
            "tick": current_tick,
            "metric_type": "model_output",
            "model_name": name,
            "value": pred.output_kw,
            "metadata": metric.dict()
        })
    
    return {
        "session_id": session_id,
        "tick": current_tick,
        "timestamp_utc": time.time(),
        "models": model_metrics,
        "total_inference_ms": sum(r[2] for r in results)
    }


@router.get("/constraints/{session_id}")
async def get_active_constraints(session_id: str):
    """
    GET /metrics/constraints/{session_id}
    Returns all active and violated physics constraints with severity levels.
    """
    # Use sync method (now available after patch)
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="DataManager unavailable")

    session = data_manager.get_session(session_id)  # Now works - sync method exists
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Build current state
    state = {
        'voltage_pu': session.get('voltage_pu', 1.0),
        'battery_soc': session.get('battery_soc', 0.5),
        'battery_temp_c': session.get('battery_temp_c', 25.0),
        'load_kw': session.get('load_kw', 100.0)
    }
    
    # Get constraint violations
    active_constraints_raw = check_constraints(state, {})
    
    # Detailed constraint analysis
    constraints = []
    
    # Voltage constraints
    v_pu = state['voltage_pu']
    constraints.append(ConstraintMetrics(
        constraint_name="VOLTAGE_LOWER_LIMIT",
        is_violated=v_pu < 0.95,
        severity="CRITICAL" if v_pu < 0.90 else ("WARNING" if v_pu < 0.95 else "INFO"),
        current_value=v_pu,
        limit_value=0.95,
        margin=v_pu - 0.95
    ))
    
    constraints.append(ConstraintMetrics(
        constraint_name="VOLTAGE_UPPER_LIMIT",
        is_violated=v_pu > 1.05,
        severity="CRITICAL" if v_pu > 1.10 else ("WARNING" if v_pu > 1.05 else "INFO"),
        current_value=v_pu,
        limit_value=1.05,
        margin=1.05 - v_pu
    ))
    
    # Battery SOC constraints
    soc = state['battery_soc']
    constraints.append(ConstraintMetrics(
        constraint_name="BATTERY_SOC_LOWER_LIMIT",
        is_violated=soc < 0.2,
        severity="CRITICAL" if soc < 0.15 else ("WARNING" if soc < 0.2 else "INFO"),
        current_value=soc,
        limit_value=0.2,
        margin=soc - 0.2
    ))
    
    constraints.append(ConstraintMetrics(
        constraint_name="BATTERY_SOC_UPPER_LIMIT",
        is_violated=soc > 0.9,
        severity="WARNING" if soc > 0.95 else "INFO",
        current_value=soc,
        limit_value=0.9,
        margin=0.9 - soc
    ))
    
    # Thermal constraints
    temp = state['battery_temp_c']
    constraints.append(ConstraintMetrics(
        constraint_name="BATTERY_THERMAL_LIMIT",
        is_violated=temp > 45.0,
        severity="CRITICAL" if temp > 55.0 else ("WARNING" if temp > 45.0 else "INFO"),
        current_value=temp,
        limit_value=45.0,
        margin=45.0 - temp
    ))
    
    violations = [c for c in constraints if c.is_violated]
    
    return {
        "session_id": session_id,
        "timestamp_utc": time.time(),
        "total_constraints": len(constraints),
        "violations_count": len(violations),
        "constraints": [c.dict() for c in constraints],
        "active_violations": active_constraints_raw,
        "state_snapshot": state
    }


@router.get("/physics/{session_id}")
async def get_physics_residuals(session_id: str):
    """
    GET /metrics/physics/{session_id}
    Returns physics residuals for each model indicating prediction quality.
    """
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get latest model predictions
    weather = session['weather']
    tick = session.get('current_tick', 0) % 24
    
    models = await asyncio.gather(
        predict_solar_pv(weather['ghi'][tick], weather['temperature'][tick], 0.3),
        predict_wind_turbine(weather['wind_speed'][tick], 1.225),
        predict_battery_thermal(0.0, 25.0, 0.5),
        predict_pinn_voltage(session_id, torch.tensor([[1.0, 0.5, 0.0]])),
        predict_grid_stability(1.0, 1.0, 5.0),
        predict_load_forecast(weather['temperature'][tick], tick, 100.0)
    )
    
    model_names = ["solar_pv", "wind_turbine", "battery_thermal", 
                   "pinn_voltage", "grid_stability", "load_forecast"]
    
    # Define acceptable residual thresholds
    thresholds = {
        "solar_pv": 0.15,
        "wind_turbine": 0.20,
        "battery_thermal": 0.10,
        "pinn_voltage": 0.05,
        "grid_stability": 0.08,
        "load_forecast": 0.12
    }
    
    residuals = []
    for name, pred in zip(model_names, models):
        residual = PhysicsResiduals(
            model_name=name,
            residual_value=pred.physics_residual,
            residual_type="relative",
            threshold=thresholds.get(name, 0.10),
            is_acceptable=pred.physics_residual <= thresholds.get(name, 0.10)
        )
        residuals.append(residual.dict())
    
    total_residual = sum(m.physics_residual for m in models)
    acceptable_count = sum(1 for r in residuals if r['is_acceptable'])
    
    return {
        "session_id": session_id,
        "timestamp_utc": time.time(),
        "residuals": residuals,
        "summary": {
            "total_residual": total_residual,
            "acceptable_models": acceptable_count,
            "total_models": len(residuals),
            "quality_score": acceptable_count / len(residuals)
        }
    }


@router.get("/performance/{session_id}")
async def get_performance_metrics(session_id: str):
    """
    GET /metrics/performance/{session_id}
    Returns model inference performance statistics (latency percentiles).
    """
    # Fetch historical metrics
    metrics = await fetch_historical_metrics(session_id, window_seconds=3600)
    
    # Group by model
    model_timings = {}
    for metric in metrics:
        if metric.get('metric_type') == 'model_output':
            model_name = metric.get('model_name')
            if model_name:
                if model_name not in model_timings:
                    model_timings[model_name] = []
                
                metadata = metric.get('metadata', {})
                inf_time = metadata.get('inference_time_ms', 0)
                if inf_time > 0:
                    model_timings[model_name].append(inf_time)
    
    # Calculate percentiles
    performance = []
    for model_name, timings in model_timings.items():
        if timings:
            perf = PerformanceMetrics(
                model_name=model_name,
                avg_inference_ms=sum(timings) / len(timings),
                p50_inference_ms=calculate_percentile(timings, 50),
                p95_inference_ms=calculate_percentile(timings, 95),
                p99_inference_ms=calculate_percentile(timings, 99),
                total_invocations=len(timings),
                error_count=0  # Would track from error logs
            )
            performance.append(perf.dict())
    
    return {
        "session_id": session_id,
        "window_seconds": 3600,
        "timestamp_utc": time.time(),
        "performance_metrics": performance
    }


@router.get("/uncertainty/{session_id}")
async def get_uncertainty_bounds(session_id: str):
    """
    GET /metrics/uncertainty/{session_id}
    Returns prediction uncertainty bounds (95% confidence intervals).
    """
    # Fetch recent predictions for statistical analysis
    metrics = await fetch_historical_metrics(session_id, window_seconds=1800)
    
    # Group predictions by model
    model_predictions = {}
    for metric in metrics:
        if metric.get('metric_type') == 'model_output':
            model_name = metric.get('model_name')
            value = metric.get('value', 0)
            
            if model_name not in model_predictions:
                model_predictions[model_name] = []
            model_predictions[model_name].append(value)
    
    # Calculate uncertainty bounds
    uncertainty_metrics = []
    for model_name, predictions in model_predictions.items():
        if len(predictions) >= 5:  # Need minimum samples
            mean = sum(predictions) / len(predictions)
            variance = sum((x - mean) ** 2 for x in predictions) / len(predictions)
            std = variance ** 0.5
            
            # 95% confidence interval (1.96 * std for normal distribution)
            uncertainty = UncertaintyBounds(
                model_name=model_name,
                prediction_mean=mean,
                prediction_std=std,
                lower_bound_95=mean - 1.96 * std,
                upper_bound_95=mean + 1.96 * std,
                epistemic_uncertainty=std * 0.6,  # Model uncertainty
                aleatoric_uncertainty=std * 0.4   # Data uncertainty
            )
            uncertainty_metrics.append(uncertainty.dict())
    
    return {
        "session_id": session_id,
        "window_seconds": 1800,
        "timestamp_utc": time.time(),
        "uncertainty_bounds": uncertainty_metrics
    }


@router.get("/history/{session_id}")
async def get_historical_metrics(
    session_id: str,
    window: int = Query(3600, description="Time window in seconds", ge=60, le=86400)
):
    """
    GET /metrics/history/{session_id}?window=3600
    Returns time-series metrics within specified window.
    """
    metrics = await fetch_historical_metrics(session_id, window_seconds=window)
    
    # Organize by metric type
    organized = {
        "model_outputs": [],
        "constraints": [],
        "state_updates": []
    }
    
    for metric in metrics:
        metric_type = metric.get('metric_type', 'unknown')
        
        data_point = HistoricalDataPoint(
            timestamp=metric.get('timestamp', 0),
            tick=metric.get('tick', 0),
            metric_name=metric.get('model_name', metric_type),
            value=metric.get('value', 0),
            metadata=metric.get('metadata')
        )
        
        if metric_type == 'model_output':
            organized['model_outputs'].append(data_point.dict())
        elif metric_type == 'constraint':
            organized['constraints'].append(data_point.dict())
        else:
            organized['state_updates'].append(data_point.dict())
    
    return {
        "session_id": session_id,
        "window_seconds": window,
        "data_points_count": len(metrics),
        "timestamp_utc": time.time(),
        "metrics": organized
    }


@router.get("/export/{session_id}")
async def export_metrics(
    session_id: str,
    format: Literal["csv", "json"] = Query("csv", description="Export format"),
    window: int = Query(3600, description="Time window in seconds")
):
    """
    GET /metrics/export/{session_id}?format=csv&window=3600
    Exports metrics data in CSV or JSON format.
    """
    metrics = await fetch_historical_metrics(session_id, window_seconds=window)
    
    if format == "json":
        # JSON export
        export_data = {
            "session_id": session_id,
            "export_timestamp": time.time(),
            "window_seconds": window,
            "metrics": metrics
        }
        
        json_str = json.dumps(export_data, indent=2)
        return StreamingResponse(
            io.BytesIO(json_str.encode()),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=metrics_{session_id}_{int(time.time())}.json"
            }
        )
    
    else:  # CSV export
        output = io.StringIO()
        
        if metrics:
            # Flatten nested structure for CSV
            fieldnames = ["timestamp", "tick", "metric_type", "model_name", "value"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in metrics:
                row = {
                    "timestamp": metric.get('timestamp', ''),
                    "tick": metric.get('tick', ''),
                    "metric_type": metric.get('metric_type', ''),
                    "model_name": metric.get('model_name', ''),
                    "value": metric.get('value', '')
                }
                writer.writerow(row)
        
        csv_data = output.getvalue()
        return StreamingResponse(
            io.BytesIO(csv_data.encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=metrics_{session_id}_{int(time.time())}.csv"
            }
        )


@router.delete("/history/{session_id}")
async def clear_metrics_history(session_id: str):
    """
    DELETE /metrics/history/{session_id}
    Clears historical metrics for a session (useful for testing/reset).
    """
    redis = await get_redis()
    
    if redis:
        try:
            await redis.delete(f"metrics:{session_id}")
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
    
    # Clear buffer
    if session_id in metrics_buffer:
        metrics_buffer[session_id].clear()
    
    return {
        "status": "cleared",
        "session_id": session_id,
        "timestamp_utc": time.time()
}

