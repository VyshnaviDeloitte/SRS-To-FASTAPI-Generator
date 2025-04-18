from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

analyzer_llm = Groq(api_key=api_key)

# Import your compiled LangGraph app and state definition
try:
    from graph_builder import app_graph
    from project_state import ProjectState
except ImportError as e:
    fatal_error_message = f"""
---! FATAL ERROR: Could not import 'app_graph' from graph_builder.py: {e} !---
---! Ensure graph_builder.py exists and compiles the graph correctly. !---
"""
    print(fatal_error_message)
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Move these to config file or env vars ideally
UPLOAD_DIR = "temp_uploads"
GENERATED_PROJECTS_DIR = "generated_projects"
MAX_DEBUG_ITERATIONS = 3 # Max times the debug loop can run

# --- App Lifecycle (for setup/teardown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create directories if they don't exist
    logger.info("Generator service starting up...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(GENERATED_PROJECTS_DIR, exist_ok=True)
    logger.info(f"Upload directory: {os.path.abspath(UPLOAD_DIR)}")
    logger.info(f"Generated projects directory: {os.path.abspath(GENERATED_PROJECTS_DIR)}")
    yield
    # Shutdown: Optional cleanup (e.g., remove old temp files)
    logger.info("Generator service shutting down.")

app = FastAPI(lifespan=lifespan, title="SRS-to-FastAPI Generator Service")

# --- Background Task Function ---
def run_generation_workflow(initial_state: ProjectState, temp_srs_path: str):
    """Runs the LangGraph workflow in the background."""
    logger.info(f"Starting background generation for {initial_state.get('project_root')}")
    final_state = None
    try:
        # Invoke the graph with a recursion limit
        # The stream() method might be better for observing progress
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 25}) # Adjust limit as needed
        logger.info(f"Workflow completed for {initial_state.get('project_root')}")
    except Exception as e:
        logger.error(f"Error during background graph execution: {e}", exc_info=True)
        # TODO: Implement status tracking/reporting for background task failure
        # Maybe update a database record or a status file associated with the run ID
    finally:
        # Always clean up the uploaded SRS file
        if os.path.exists(temp_srs_path):
            try:
                os.remove(temp_srs_path)
                logger.info(f"Cleaned up temporary file: {temp_srs_path}")
            except OSError as e:
                logger.error(f"Error cleaning up temporary file {temp_srs_path}: {e}")
        # Log final state info if available
        if final_state:
            logger.info(f"Final state info for {initial_state.get('project_root')}: "
                        f"Zip={final_state.get('final_zip_path')}, "
                        f"LangSmith={final_state.get('langsmith_run_url')}, "
                        f"Errors={len(final_state.get('error_log', []))}")

# --- API Endpoint ---
@app.post("/generate-from-srs/", status_code=202) # 202 Accepted for background tasks
async def generate_project_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts an SRS (.docx) file, triggers the generation workflow in the background,
    and returns an initial acceptance response.
    """
    logger.info(f"Received request to generate project from: {file.filename}")
    if not file.filename or not file.filename.endswith(".docx"):
        logger.warning(f"Invalid file format received: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file format. Only .docx is supported.")

    # Generate unique ID for this run
    run_id = str(uuid.uuid4())
    project_folder_name = f"{run_id}_{os.path.splitext(file.filename)[0]}"
    project_output_root = os.path.abspath(os.path.join(GENERATED_PROJECTS_DIR, project_folder_name))
    temp_srs_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{run_id}_{file.filename}"))

    # Save uploaded file temporarily
    logger.info(f"Saving uploaded file to: {temp_srs_path}")
    try:
        with open(temp_srs_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await file.close() # Ensure file handle is closed

    # Define initial state for the graph
    # All values must be serializable if using certain LangGraph persistence backends
    initial_state: ProjectState = {
        "srs_path": temp_srs_path,
        "project_root": project_output_root,
        "max_debug_iterations": MAX_DEBUG_ITERATIONS,
        "debug_iterations": 0,
        # Initialize other fields as None or empty lists/dicts
        "srs_text": None, "srs_image_path": None, "requirements": None,
        "generated_files": [], "test_results": None, "validation_errors": None,
        "error_log": [], "persistent_context": {}, "langsmith_run_url": None,
        "final_zip_path": None, "documentation_files": [],
    }

    # Add the generation task to run in the background
    logger.info(f"Adding generation task to background for run ID: {run_id}")
    background_tasks.add_task(run_generation_workflow, initial_state, temp_srs_path)

    # Return an immediate response acknowledging the request
    return JSONResponse(
        status_code=202, # Accepted
        content={
            "message": "SRS received. Project generation started in the background.",
            "run_id": run_id,
            "estimated_output_path": project_output_root,
            # Provide a way to check status later (requires more implementation)
            # "status_check_url": f"/generation-status/{run_id}"
        }
    )

# TODO: Add endpoint to check status / retrieve results based on run_id
# @app.get("/generation-status/{run_id}")
# async def get_generation_status(run_id: str):
#     # Check status file, database record, etc. associated with run_id
#     pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GENERATOR_PORT", 8001))
    logger.info(f"Starting generator service on port {port}")
    # Note: Reload=True is helpful for development but may cause issues with background tasks
    # if the main process restarts unexpectedly. Use with care for background jobs.
    uvicorn.run("generator_service:app", host="0.0.0.0", port=port, reload=False)