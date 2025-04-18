# project_state.py
from typing import List, Dict, Optional, TypedDict, Any

# Define structures for clarity (can be Pydantic models too)
class EndpointDetail(TypedDict):
    path: str
    method: str
    description: Optional[str]
    request_body_schema: Optional[Dict]
    response_schema: Optional[Dict]
    auth_required: bool
    roles: List[str] # e.g., ['user', 'manager']

class TableSchema(TypedDict):
    name: str
    columns: List[Dict] # e.g., [{'name': 'id', 'type': 'INTEGER', 'primary_key': True}, ...]
    relationships: Optional[List[Dict]]

class DatabaseSchema(TypedDict):
    tables: List[TableSchema]

class ExtractedRequirements(TypedDict):
    endpoints: List[EndpointDetail]
    database_schema: DatabaseSchema
    business_logic: List[str]
    auth_requirements: Dict

class GenerationResult(TypedDict):
    file_path: str
    code: Optional[str] # Store code optionally if needed in state, otherwise just write to file
    status: str # e.g., 'generated', 'written', 'error', 'skipped'
    description: Optional[str] # e.g., 'User model', 'LMS routes'

class TestResult(TypedDict):
    passed: bool
    output: str
    coverage: Optional[float]

# The main state passed between nodes
class ProjectState(TypedDict):
    # Input & Setup
    srs_path: str
    project_root: str # Root directory for the generated project
    max_debug_iterations: int

    # Extracted Info
    srs_text: Optional[str]
    srs_image_path: Optional[str] # Path if schema is in an image
    requirements: Optional[ExtractedRequirements]

    # Generation & Validation Artefacts
    generated_files: List[GenerationResult] # Track files and their status
    test_results: Optional[TestResult]
    validation_errors: Optional[str] # Store linting/test errors for debugging

    # Iteration & Debugging Control
    error_log: List[str] # Accumulates errors across steps if needed
    debug_iterations: int # Counter for debug loop

    # Consistency & Context
    persistent_context: Dict[str, Any] # Store key names, structures generated so far

    # Output & Logging
    langsmith_run_url: Optional[str]
    final_zip_path: Optional[str]
    documentation_files: List[str]