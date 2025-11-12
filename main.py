import time
import json
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# --- OpenTelemetry for tracing ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# --- Logging Setup ---
logger = logging.getLogger("iris-api-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    json.dumps({
        "severity": "%(levelname)s",
        "message": "%(message)s",
        "timestamp": "%(asctime)s"
    })
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- FastAPI App ---
app = FastAPI(title="Iris Classifier API with Logging and Monitoring")

# --- Application State ---
app_state = {
    "is_ready": False,
    "is_alive": True
}

# Global model variable (loaded once)
model = None
class_names = ["setosa", "versicolor", "virginica"]  # standard iris labels


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global model
    time.sleep(2)  # Simulate loading delay
    model = joblib.load("model.joblib")
    app_state["is_ready"] = True
    logger.info("✅ Iris model loaded and API ready.")


# --- Liveness Probe ---
@app.get("/live-check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- Readiness Probe ---
@app.get("/ready-check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    else:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# --- Middleware: Measure Request Time ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response


# --- Exception Handler ---
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "trace_id": trace_id
        }
    )


# --- Root Route ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}


# --- Input Schema ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict_species(input: IrisInput, request: Request):
    start_time = time.time()
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    try:
        # Prepare input data
        input_df = pd.DataFrame([input.dict()])

        # Make prediction
        prediction = model.predict(input_df)[0]
        predicted_class = prediction

        # Optional confidence using predict_proba
        try:
            probs = model.predict_proba(input_df)[0]
            confidence = float(np.max(probs))
        except Exception:
            confidence = None

        latency = round((time.time() - start_time) * 1000, 2)

        # Logging
        logger.info(json.dumps({
            "event": "prediction",
            "trace_id": trace_id,
            "input": input.dict(),
            "latency_ms": latency,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "status": "success"
        }))

        # Response
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "trace_id": trace_id
        }

    except Exception as e:
        logger.exception(json.dumps({
            "event": "prediction_error",
            "trace_id": trace_id,
            "error": str(e)
        }))
        raise HTTPException(status_code=500, detail="Prediction failed")
