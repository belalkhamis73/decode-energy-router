"""
Metrics Publisher Service

PURPOSE: Publish all internal states to monitoring systems for observability,
debugging, and real-time monitoring of the energy management system.

METRICS PUBLISHED:
- Model predictions (all 6 models with timestamps)
- Physics constraint violations
- User override parameters
- Routing decisions with explanations
- Model inference latencies
- Asset state variables (SOC, temp, voltage, frequency)
- Economic metrics (cost, CO2, savings)
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

# Simulated external dependencies
try:
    import psycopg2
    from psycopg2.extras import execute_values
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we publish"""
    PREDICTION = "prediction"
    CONSTRAINT = "constraint"
    OVERRIDE = "override"
    ROUTING = "routing"
    LATENCY = "latency"
    STATE = "state"
    ECONOMIC = "economic"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    metric_type: MetricType
    metric_name: str
    value: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelPrediction:
    """Model prediction metrics"""
    model_name: str
    timestamp: float
    prediction_value: float
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None


@dataclass
class ConstraintViolation:
    """Physics constraint violation"""
    constraint_name: str
    timestamp: float
    expected_value: float
    actual_value: float
    violation_percentage: float
    severity: str  # "warning", "critical"


@dataclass
class RoutingDecision:
    """Routing decision with explanation"""
    timestamp: float
    selected_model: str
    confidence_score: float
    all_model_scores: Dict[str, float]
    explanation: str
    override_applied: bool


class TimescaleDBWriter:
    """Write time-series metrics to TimescaleDB"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        self.enabled = TIMESCALE_AVAILABLE
        
        if not self.enabled:
            logger.warning("TimescaleDB dependencies not available, writer disabled")
    
    def connect(self):
        """Establish database connection"""
        if not self.enabled:
            return
        
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self._create_tables()
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            self.enabled = False
    
    def _create_tables(self):
        """Create hypertables if they don't exist"""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            # Main metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    time TIMESTAMPTZ NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    value DOUBLE PRECISION,
                    tags JSONB,
                    metadata JSONB
                );
            """)
            
            # Convert to hypertable if not already
            cur.execute("""
                SELECT create_hypertable('metrics', 'time', 
                    if_not_exists => TRUE);
            """)
            
            # Model predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    time TIMESTAMPTZ NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    prediction_value DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION,
                    inference_time_ms DOUBLE PRECISION
                );
            """)
            
            cur.execute("""
                SELECT create_hypertable('model_predictions', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Constraint violations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS constraint_violations (
                    time TIMESTAMPTZ NOT NULL,
                    constraint_name VARCHAR(100) NOT NULL,
                    expected_value DOUBLE PRECISION,
                    actual_value DOUBLE PRECISION,
                    violation_percentage DOUBLE PRECISION,
                    severity VARCHAR(20)
                );
            """)
            
            cur.execute("""
                SELECT create_hypertable('constraint_violations', 'time',
                    if_not_exists => TRUE);
            """)
            
            # Routing decisions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    time TIMESTAMPTZ NOT NULL,
                    selected_model VARCHAR(100) NOT NULL,
                    confidence_score DOUBLE PRECISION,
                    all_model_scores JSONB,
                    explanation TEXT,
                    override_applied BOOLEAN
                );
            """)
            
            cur.execute("""
                SELECT create_hypertable('routing_decisions', 'time',
                    if_not_exists => TRUE);
            """)
            
            self.conn.commit()
    
    def write_metrics(self, metrics: List[MetricPoint]):
        """Write batch of metrics"""
        if not self.enabled or not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                values = [
                    (
                        datetime.fromtimestamp(m.timestamp),
                        m.metric_type.value,
                        m.metric_name,
                        m.value,
                        json.dumps(m.tags),
                        json.dumps(m.metadata) if m.metadata else None
                    )
                    for m in metrics
                ]
                
                execute_values(
                    cur,
                    """
                    INSERT INTO metrics 
                    (time, metric_type, metric_name, value, tags, metadata)
                    VALUES %s
                    """,
                    values
                )
                
                self.conn.commit()
                logger.debug(f"Wrote {len(metrics)} metrics to TimescaleDB")
        
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
            self.conn.rollback()
    
    def write_predictions(self, predictions: List[ModelPrediction]):
        """Write model predictions"""
        if not self.enabled or not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                values = [
                    (
                        datetime.fromtimestamp(p.timestamp),
                        p.model_name,
                        p.prediction_value,
                        p.confidence,
                        p.inference_time_ms
                    )
                    for p in predictions
                ]
                
                execute_values(
                    cur,
                    """
                    INSERT INTO model_predictions
                    (time, model_name, prediction_value, confidence, inference_time_ms)
                    VALUES %s
                    """,
                    values
                )
                
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to write predictions: {e}")
            self.conn.rollback()
    
    def write_violations(self, violations: List[ConstraintViolation]):
        """Write constraint violations"""
        if not self.enabled or not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                values = [
                    (
                        datetime.fromtimestamp(v.timestamp),
                        v.constraint_name,
                        v.expected_value,
                        v.actual_value,
                        v.violation_percentage,
                        v.severity
                    )
                    for v in violations
                ]
                
                execute_values(
                    cur,
                    """
                    INSERT INTO constraint_violations
                    (time, constraint_name, expected_value, actual_value, 
                     violation_percentage, severity)
                    VALUES %s
                    """,
                    values
                )
                
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to write violations: {e}")
            self.conn.rollback()
    
    def write_routing_decision(self, decision: RoutingDecision):
        """Write routing decision"""
        if not self.enabled or not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO routing_decisions
                    (time, selected_model, confidence_score, all_model_scores,
                     explanation, override_applied)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.fromtimestamp(decision.timestamp),
                        decision.selected_model,
                        decision.confidence_score,
                        json.dumps(decision.all_model_scores),
                        decision.explanation,
                        decision.override_applied
                    )
                )
                
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to write routing decision: {e}")
            self.conn.rollback()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class RedisStreamPublisher:
    """Publish metrics to Redis streams for real-time consumption"""
    
    def __init__(self, redis_url: str, stream_prefix: str = "metrics"):
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix
        self.redis_client = None
        self.enabled = REDIS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Redis dependencies not available, publisher disabled")
    
    async def connect(self):
        """Connect to Redis"""
        if not self.enabled:
            return
        
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
    
    async def publish_metric(self, metric: MetricPoint):
        """Publish single metric to stream"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            stream_name = f"{self.stream_prefix}:{metric.metric_type.value}"
            
            data = {
                "timestamp": str(metric.timestamp),
                "metric_name": metric.metric_name,
                "value": str(metric.value),
                "tags": json.dumps(metric.tags)
            }
            
            if metric.metadata:
                data["metadata"] = json.dumps(metric.metadata)
            
            await self.redis_client.xadd(stream_name, data, maxlen=10000)
        
        except Exception as e:
            logger.error(f"Failed to publish metric to Redis: {e}")
    
    async def publish_prediction(self, prediction: ModelPrediction):
        """Publish model prediction"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            stream_name = f"{self.stream_prefix}:predictions"
            
            data = {
                "timestamp": str(prediction.timestamp),
                "model_name": prediction.model_name,
                "prediction_value": str(prediction.prediction_value),
                "confidence": str(prediction.confidence) if prediction.confidence else "",
                "inference_time_ms": str(prediction.inference_time_ms) if prediction.inference_time_ms else ""
            }
            
            await self.redis_client.xadd(stream_name, data, maxlen=10000)
        
        except Exception as e:
            logger.error(f"Failed to publish prediction to Redis: {e}")
    
    async def publish_violation(self, violation: ConstraintViolation):
        """Publish constraint violation"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            stream_name = f"{self.stream_prefix}:violations"
            
            data = {
                "timestamp": str(violation.timestamp),
                "constraint_name": violation.constraint_name,
                "expected_value": str(violation.expected_value),
                "actual_value": str(violation.actual_value),
                "violation_percentage": str(violation.violation_percentage),
                "severity": violation.severity
            }
            
            await self.redis_client.xadd(stream_name, data, maxlen=1000)
            
            # Also publish to alert channel if critical
            if violation.severity == "critical":
                await self.redis_client.publish(
                    "alerts:critical",
                    json.dumps(asdict(violation))
                )
        
        except Exception as e:
            logger.error(f"Failed to publish violation to Redis: {e}")
    
    async def publish_routing_decision(self, decision: RoutingDecision):
        """Publish routing decision"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            stream_name = f"{self.stream_prefix}:routing"
            
            data = {
                "timestamp": str(decision.timestamp),
                "selected_model": decision.selected_model,
                "confidence_score": str(decision.confidence_score),
                "all_model_scores": json.dumps(decision.all_model_scores),
                "explanation": decision.explanation,
                "override_applied": str(decision.override_applied)
            }
            
            await self.redis_client.xadd(stream_name, data, maxlen=5000)
        
        except Exception as e:
            logger.error(f"Failed to publish routing decision to Redis: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()


class PrometheusExporter:
    """Export metrics in Prometheus format"""
    
    def __init__(self, pushgateway_url: Optional[str] = None, job_name: str = "energy_mgmt"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus dependencies not available, exporter disabled")
            return
        
        # Define metrics
        self.prediction_gauge = Gauge(
            'model_prediction_value',
            'Current prediction value from each model',
            ['model_name'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_inference_latency_ms',
            'Model inference latency in milliseconds',
            ['model_name'],
            registry=self.registry
        )
        
        self.constraint_violations = Counter(
            'constraint_violations_total',
            'Total number of constraint violations',
            ['constraint_name', 'severity'],
            registry=self.registry
        )
        
        self.routing_decisions = Counter(
            'routing_decisions_total',
            'Total routing decisions made',
            ['selected_model'],
            registry=self.registry
        )
        
        self.asset_state = Gauge(
            'asset_state_value',
            'Current asset state values',
            ['asset_type', 'metric_name'],
            registry=self.registry
        )
        
        self.economic_metric = Gauge(
            'economic_metric_value',
            'Economic metrics (cost, CO2, savings)',
            ['metric_type'],
            registry=self.registry
        )
        
        self.override_active = Gauge(
            'user_override_active',
            'Whether user override is currently active',
            ['parameter_name'],
            registry=self.registry
        )
    
    def record_prediction(self, prediction: ModelPrediction):
        """Record model prediction"""
        if not self.enabled:
            return
        
        self.prediction_gauge.labels(model_name=prediction.model_name).set(
            prediction.prediction_value
        )
        
        if prediction.inference_time_ms is not None:
            self.prediction_latency.labels(model_name=prediction.model_name).observe(
                prediction.inference_time_ms
            )
    
    def record_violation(self, violation: ConstraintViolation):
        """Record constraint violation"""
        if not self.enabled:
            return
        
        self.constraint_violations.labels(
            constraint_name=violation.constraint_name,
            severity=violation.severity
        ).inc()
    
    def record_routing_decision(self, decision: RoutingDecision):
        """Record routing decision"""
        if not self.enabled:
            return
        
        self.routing_decisions.labels(
            selected_model=decision.selected_model
        ).inc()
    
    def record_asset_state(self, asset_type: str, metric_name: str, value: float):
        """Record asset state metric"""
        if not self.enabled:
            return
        
        self.asset_state.labels(
            asset_type=asset_type,
            metric_name=metric_name
        ).set(value)
    
    def record_economic_metric(self, metric_type: str, value: float):
        """Record economic metric"""
        if not self.enabled:
            return
        
        self.economic_metric.labels(metric_type=metric_type).set(value)
    
    def record_override(self, parameter_name: str, active: bool):
        """Record user override status"""
        if not self.enabled:
            return
        
        self.override_active.labels(parameter_name=parameter_name).set(
            1.0 if active else 0.0
        )
    
    def push_to_gateway(self):
        """Push metrics to Prometheus Pushgateway"""
        if not self.enabled or not self.pushgateway_url:
            return
        
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.debug("Pushed metrics to Prometheus")
        except Exception as e:
            logger.error(f"Failed to push to Prometheus: {e}")


class MetricsPublisher:
    """Main metrics publishing engine - orchestrates all publishers"""
    
    def __init__(
        self,
        timescale_conn: Optional[str] = None,
        redis_url: Optional[str] = None,
        prometheus_gateway: Optional[str] = None
    ):
        # Initialize all publishers
        self.timescale = TimescaleDBWriter(timescale_conn) if timescale_conn else None
        self.redis = RedisStreamPublisher(redis_url) if redis_url else None
        self.prometheus = PrometheusExporter(prometheus_gateway) if prometheus_gateway else None
        
        # Metric buffers for batching
        self.metric_buffer: List[MetricPoint] = []
        self.prediction_buffer: List[ModelPrediction] = []
        self.violation_buffer: List[ConstraintViolation] = []
        
        self.buffer_size = 100
        self.flush_interval = 5.0  # seconds
        
        self._running = False
        self._flush_task = None
    
    async def start(self):
        """Start the publisher"""
        # Connect to backends
        if self.timescale:
            self.timescale.connect()
        
        if self.redis:
            await self.redis.connect()
        
        # Start periodic flush
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info("MetricsPublisher started")
    
    async def stop(self):
        """Stop the publisher"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        
        # Close connections
        if self.timescale:
            self.timescale.close()
        
        if self.redis:
            await self.redis.close()
        
        logger.info("MetricsPublisher stopped")
    
    async def _periodic_flush(self):
        """Periodically flush buffers"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def flush(self):
        """Flush all buffered metrics"""
        if self.metric_buffer and self.timescale:
            self.timescale.write_metrics(self.metric_buffer)
            self.metric_buffer.clear()
        
        if self.prediction_buffer and self.timescale:
            self.timescale.write_predictions(self.prediction_buffer)
            self.prediction_buffer.clear()
        
        if self.violation_buffer and self.timescale:
            self.timescale.write_violations(self.violation_buffer)
            self.violation_buffer.clear()
        
        if self.prometheus:
            self.prometheus.push_to_gateway()
    
    def publish_prediction(self, prediction: ModelPrediction):
        """Publish model prediction to all backends"""
        # Buffer for TimescaleDB
        self.prediction_buffer.append(prediction)
        
        # Real-time to Redis
        if self.redis:
            asyncio.create_task(self.redis.publish_prediction(prediction))
        
        # Prometheus gauges
        if self.prometheus:
            self.prometheus.record_prediction(prediction)
        
        # Auto-flush if buffer full
        if len(self.prediction_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())
    
    def publish_violation(self, violation: ConstraintViolation):
        """Publish constraint violation"""
        # Buffer for TimescaleDB
        self.violation_buffer.append(violation)
        
        # Real-time to Redis
        if self.redis:
            asyncio.create_task(self.redis.publish_violation(violation))
        
        # Prometheus counter
        if self.prometheus:
            self.prometheus.record_violation(violation)
        
        # Auto-flush if buffer full
        if len(self.violation_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())
    
    def publish_routing_decision(self, decision: RoutingDecision):
        """Publish routing decision"""
        # Write directly to TimescaleDB (important decisions)
        if self.timescale:
            self.timescale.write_routing_decision(decision)
        
        # Real-time to Redis
        if self.redis:
            asyncio.create_task(self.redis.publish_routing_decision(decision))
        
        # Prometheus counter
        if self.prometheus:
            self.prometheus.record_routing_decision(decision)
    
    def publish_asset_state(self, asset_type: str, states: Dict[str, float]):
        """Publish asset state variables (SOC, temp, voltage, etc.)"""
        timestamp = time.time()
        
        for metric_name, value in states.items():
            # Create metric point
            metric = MetricPoint(
                timestamp=timestamp,
                metric_type=MetricType.STATE,
                metric_name=metric_name,
                value=value,
                tags={"asset_type": asset_type}
            )
            
            self.metric_buffer.append(metric)
            
            # Prometheus gauge
            if self.prometheus:
                self.prometheus.record_asset_state(asset_type, metric_name, value)
        
        # Auto-flush if buffer full
        if len(self.metric_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())
    
    def publish_economic_metrics(self, metrics: Dict[str, float]):
        """Publish economic metrics (cost, CO2, savings)"""
        timestamp = time.time()
        
        for metric_type, value in metrics.items():
            # Create metric point
            metric = MetricPoint(
                timestamp=timestamp,
                metric_type=MetricType.ECONOMIC,
                metric_name=metric_type,
                value=value,
                tags={"category": "economic"}
            )
            
            self.metric_buffer.append(metric)
            
            # Prometheus gauge
            if self.prometheus:
                self.prometheus.record_economic_metric(metric_type, value)
        
        # Auto-flush if buffer full
        if len(self.metric_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())
    
    def publish_override(self, parameter_name: str, value: Any, active: bool):
        """Publish user override parameter"""
        metric = MetricPoint(
            timestamp=time.time(),
            metric_type=MetricType.OVERRIDE,
            metric_name=parameter_name,
            value=float(value) if isinstance(value, (int, float)) else 1.0,
            tags={"parameter": parameter_name, "active": str(active)},
            metadata={"original_value": str(value)}
        )
        
        self.metric_buffer.append(metric)
        
        # Prometheus gauge
        if self.prometheus:
            self.prometheus.record_override(parameter_name, active)
        
        # Auto-flush if buffer full
        if len(self.metric_buffer) >= self.buffer_size:
            asyncio.create_task(self.flush())


# Example usage
async def main():
    """Example usage of MetricsPublisher"""
    
    # Initialize publisher with all backends
    publisher = MetricsPublisher(
        timescale_conn="postgresql://user:pass@localhost/metrics",
        redis_url="redis://localhost:6379",
        prometheus_gateway="http://localhost:9091"
    )
    
    await publisher.start()
    
    try:
        # Publish model predictions
        publisher.publish_prediction(ModelPrediction(
            model_name="lstm_ensemble",
            timestamp=time.time(),
            prediction_value=1250.5,
            confidence=0.92,
            inference_time_ms=45.3
        ))
        
        # Publish constraint violation
        publisher.publish_violation(ConstraintViolation(
            constraint_name="battery_soc_range",
            timestamp=time.time(),
            expected_value=90.0,
            actual_value=95.2,
            violation_percentage=5.8,
            severity="warning"
        ))
        
        # Publish routing decision
        publisher.publish_routing_decision(RoutingDecision(
            timestamp=time.time(),
            selected_model="transformer",
            confidence_score=0.89,
            all_model_scores={
                "lstm": 0.76,
                "transformer": 0.89,
                "prophet": 0.71
            },
            explanation="Transformer selected due to high confidence and recent accuracy",
            override_applied=False
        ))
        
        # Publish asset states
        publisher.publish_asset_state("battery", {
            "soc_percentage": 85.5,
            "temperature_c": 24.3,
            "voltage_v": 52.1,
            "current_a": 15.2
        })
        
        # Publish economic metrics
        publisher.publish_economic_metrics({
            "cost_usd": 125.50,
            "co2_kg": 45.2,
            "savings_usd": 23.75
        })
        
        # Publish user override
        publisher.publish_override("max_charge_rate", 50.0, True)
        
        # Wait for flush
        await asyncio.sleep(6)
    
    finally:
        await publisher.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
