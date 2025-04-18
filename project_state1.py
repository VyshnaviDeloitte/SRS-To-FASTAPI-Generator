# test_node_functions.py
import os
import shutil
import logging
import traceback
import json
import ast
import subprocess # Import needed for mocking subprocess.CompletedProcess
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock # Use MagicMock for flexibility

# --- Step 1: Load Environment Variables ---
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("Warning: .env file not found. Relying on system environment variables.")

# --- Step 2: Import Project Components ---
try:
    # Import the state definition and expected types
    from project_state import ProjectState, GenerationResult, TestResult
    print("Successfully imported state definitions from 'project_state.py'.")
except ImportError as e:
    print(f"---! FATAL ERROR: Could not import from 'project_state.py': {e} !---")
    print("---! Ensure 'project_state.py' exists and defines ProjectState etc. !---")
    exit()

try:
    # Import the node functions we are testing
    import node_functions
    print("Successfully imported 'node_functions.py'.")
except ImportError as e:
    print(f"---! FATAL ERROR: Could not import 'node_functions.py': {e}\n{traceback.format_exc()} !---")
    print("---! Ensure 'node_functions.py' uses direct imports (not 'from .') !---")
    exit()
except Exception as e:
    print(f"---! FATAL ERROR: An unexpected error occurred during 'import node_functions': {e} !---")
    traceback.print_exc()
    exit()

try:
    # Import the modules nodes depend on
    import tools
    import llm_calls
    print("Successfully imported 'tools.py' and 'llm_calls.py'.")
except ImportError as e:
    print(f"---! FATAL ERROR: Could not import 'tools.py' or 'llm_calls.py': {e} !---")
    print("---! Ensure these files exist and have no syntax errors. !---")
    exit()
except RuntimeError as e: # Catch LLM init errors specifically
     print(f"---! FATAL ERROR: Failed to initialize LLMs in 'llm_calls.py': {e} !---")
     print("---! Please check your GROQ_API_KEY in the .env file and model names. !---")
     exit()
except Exception as e:
    print(f"---! FATAL ERROR: An unexpected error occurred during tool/llm imports: {e} !---")
    traceback.print_exc()
    exit()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Artifacts Setup ---
TEST_ARTIFACTS_DIR = "temp_node_test_artifacts"

def setup_test_environment(test_name: str):
    """Creates the temporary directory for test artifacts for a specific test."""
    print(f"\n--- Setup {test_name}: Creating '{TEST_ARTIFACTS_DIR}' ---")
    # Ensure clean slate
    if os.path.exists(TEST_ARTIFACTS_DIR):
        try: shutil.rmtree(TEST_ARTIFACTS_DIR)
        except OSError as e: print(f"[WARN] Could not remove previous dir '{TEST_ARTIFACTS_DIR}': {e}")
    try: os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    except OSError as e: print(f"[FATAL ERROR] Could not create dir '{TEST_ARTIFACTS_DIR}': {e}"); raise

def teardown_test_environment(test_name: str):
    """Deletes the temporary directory."""
    print(f"\n--- Teardown {test_name}: Deleting '{TEST_ARTIFACTS_DIR}' ---")
    if os.path.exists(TEST_ARTIFACTS_DIR):
        try: shutil.rmtree(TEST_ARTIFACTS_DIR); print(f"[INFO] Removed: {TEST_ARTIFACTS_DIR}")
        except OSError as e: print(f"[WARN] Error removing dir {TEST_ARTIFACTS_DIR}: {e}")
    else: print(f"[INFO] Dir {TEST_ARTIFACTS_DIR} not found, no cleanup needed.")

# --- Sample Data ---
sample_srs_text_for_nodes = "Feature: User Login. POST /api/auth/login needs email/password, returns JWT. Needs users table (id, email, hashed_password)."
sample_requirements_for_nodes = { # Output of analyze_srs
    "endpoints": [{"path": "/api/auth/login", "method": "POST", "description": "User Login", "request_body_schema": {"email": "string", "password": "string"}, "response_schema": {"token": "string"}, "auth_required": False, "roles": []}],
    "database_schema": {"tables": [{"name": "users", "columns": [{"name": "id", "type": "INTEGER", "primary_key": True}, {"name": "email", "type": "VARCHAR", "unique": True}, {"name": "hashed_password", "type": "VARCHAR"}], "relationships": []}]},
    "business_logic": ["Hash password.", "Validate email."], "auth_requirements": {"type": "JWT"} }
sample_impl_requirements = { # Requirements for impl test
    "database_schema": {"tables": [{"name": "users", "columns": [{"name": "id", "type": "INTEGER", "primary_key": True}, {"name": "username", "type": "VARCHAR", "unique": True}, {"name": "email", "type": "VARCHAR"}, {"name": "hashed_password", "type": "VARCHAR"}], "relationships": []}]},
    "endpoints": [{"path": "/api/auth/register", "method": "POST", "description": "Register user", "request_body_schema": {"username": "str", "email": "str", "password": "str"}, "response_schema": {"id": "int", "username": "str"}, "auth_required": False, "roles": []},
                  {"path": "/api/auth/login", "method": "POST", "description": "User login", "request_body_schema": {"username": "str", "password": "str"}, "response_schema": {"access_token": "str"}, "auth_required": False, "roles": []}],
    "business_logic": ["Hash passwords using bcrypt.", "Login checks username/password.", "Default logic example."],
    "auth_requirements": {"type": "JWT"} }

# --- Base Initial State (Helper) ---
def create_base_initial_state() -> ProjectState:
     """Creates a base state dictionary with default values."""
     return {"srs_path": "", "project_root": "", "max_debug_iterations": 3, "srs_text": None, "srs_image_path": None, "requirements": None,
             "generated_files": [], "test_results": None, "validation_errors": None, "error_log": [], "debug_iterations": 0,
             "persistent_context": {}, "langsmith_run_url": None, "final_zip_path": None, "documentation_files": [], "passed_validation": False} # Added passed_validation


# --- Mock Data & Expected Code (for generate_implementation_code test) ---
MOCK_USER_MODEL_CODE = """from app.database import Base
from sqlalchemy import Column, Integer, String
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
"""
MOCK_AUTH_SERVICE_CODE = """from sqlalchemy.orm import Session
import app.models.models as models
import app.models.models as schemas # Using models for schemas too
from passlib.context import CryptContext
import logging
logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
class AuthService:
    def get_user_by_username(self, db: Session, username: str): return None
    def verify_password(self, plain_password: str, hashed_password: str) -> bool: return True
    def get_password_hash(self, password: str) -> str: return pwd_context.hash(password)
# auth_service = AuthService()
"""
MOCK_DEFAULT_SERVICE_CODE = """# Default logic service code
import logging
logger = logging.getLogger(__name__)
def perform_default_logic(): logger.info("Default logic."); return {"status": "default ok"}
"""
MOCK_AUTH_ROUTES_CODE = """from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session; from app.database import get_db
from app.services.auth_service import AuthService
import app.models.models as schemas; from pydantic import BaseModel
auth_service = AuthService(); router = APIRouter()
class UserCreate(BaseModel): username: str; email: str; password: str
class UserLogin(BaseModel): username: str; password: str
class Token(BaseModel): access_token: str; token_type: str = "bearer"
class UserResponse(BaseModel): id: int; username: str; email: str
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    return {"id": 1, "username": user_data.username, "email": user_data.email}
@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin, db: Session = Depends(get_db)):
    user = {"hashed_password": "mock_hash"}
    if not user or not auth_service.verify_password(form_data.password, user["hashed_password"]): raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {"access_token": f"fake-token", "token_type": "bearer"}
"""
INITIAL_MAIN_PY_CONTENT = "from fastapi import FastAPI\napp = FastAPI()\n# Placeholder"


# --- Node Test Functions ---

def test_load_srs_node():
    """Tests the load_srs node function."""
    test_name = "test_load_srs_node"; print("\n" + "="*10 + f" Testing Node: {test_name} " + "="*10)
    setup_test_environment(test_name)
    dummy_docx_path = os.path.join(TEST_ARTIFACTS_DIR, "test_srs.docx"); dummy_content = "SRS Content."
    docx_created = False
    try: import docx; doc = docx.Document(); doc.add_paragraph(dummy_content); doc.save(dummy_docx_path); print(f"[SETUP] Created: {dummy_docx_path}"); assert os.path.exists(dummy_docx_path); docx_created = True
    except ImportError: print("\n---! SKIPPING: python-docx not installed !---\n"); teardown_test_environment(test_name); return
    except Exception as e: teardown_test_environment(test_name); assert False, f"Dummy docx creation failed: {e}"
    if not docx_created: teardown_test_environment(test_name); return
    initial_state = create_base_initial_state(); initial_state["srs_path"] = dummy_docx_path
    try: output_state = node_functions.load_srs(initial_state.copy())
    except Exception as e: teardown_test_environment(test_name); assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state..."); assert output_state is not None; assert isinstance(output_state, dict)
    assert output_state.get("srs_text") is not None; assert dummy_content in output_state.get("srs_text", "")
    assert not output_state.get("error_log"), f"Errors: {output_state.get('error_log')}"; assert output_state.get("srs_path") == dummy_docx_path
    print(f"[PASS] Node '{test_name}' OK."); teardown_test_environment(test_name)


def test_analyze_srs_node():
    """Tests the analyze_srs node function (makes LLM call)."""
    test_name = "test_analyze_srs_node"; print("\n" + "="*10 + f" Testing Node: {test_name} " + "="*10)
    setup_test_environment(test_name); initial_state = create_base_initial_state(); initial_state["srs_text"] = sample_srs_text_for_nodes
    print("[ACTION] Calling analyze_srs (LLM call)...");
    try: output_state = node_functions.analyze_srs(initial_state.copy())
    except Exception as e: teardown_test_environment(test_name); assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state..."); assert output_state is not None
    critical_errors = [e for e in output_state.get("error_log", []) if "failed" in e.lower()]
    assert not critical_errors, f"Node reported critical errors: {critical_errors}"
    assert output_state.get("requirements") is not None; requirements_data = output_state.get("requirements")
    print(f"[INFO] Requirements type: {type(requirements_data)}"); assert isinstance(requirements_data, dict)
    assert "endpoints" in requirements_data; assert "database_schema" in requirements_data
    assert output_state.get("persistent_context", {}).get("requirements_summary")
    print(f"[PASS] Node '{test_name}' OK."); print("[INFO] Manual check needed:"); print(json.dumps(requirements_data, indent=2))
    teardown_test_environment(test_name)


def test_setup_project_structure_node():
    """Tests the setup_project_structure node function."""
    test_name = "test_setup_project_structure_node"; print("\n" + "="*10 + f" Testing Node: {test_name} " + "="*10)
    setup_test_environment(test_name); test_project_root = os.path.join(TEST_ARTIFACTS_DIR, "my_generated_project")
    initial_state = create_base_initial_state(); initial_state["project_root"] = test_project_root
    print(f"[ACTION] Calling setup_project_structure: {test_project_root}")
    try: output_state = node_functions.setup_project_structure(initial_state.copy())
    except Exception as e: teardown_test_environment(test_name); assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state and file system..."); assert output_state is not None; assert not output_state.get("error_log")
    assert os.path.isdir(test_project_root)
    expected_dirs = [os.path.join("app","api","routes"), os.path.join("tests"), os.path.join("alembic","versions")]
    for d in expected_dirs: assert os.path.isdir(os.path.join(test_project_root, d)), f"Missing dir '{d}'"
    expected_files = [os.path.join("app","main.py"), "requirements.txt", "alembic.ini", ".gitignore"]
    for f in expected_files: assert os.path.isfile(os.path.join(test_project_root, f)), f"Missing file '{f}'"
    assert len(output_state.get("generated_files", [])) > 5; assert output_state.get("persistent_context", {}).get("project_structure_summary")
    print(f"[PASS] Node '{test_name}' OK."); teardown_test_environment(test_name)


def test_generate_unit_tests_node():
    """Tests the generate_unit_tests node function (makes LLM calls)."""
    test_name = "test_generate_unit_tests_node"; print("\n" + "="*10 + f" Testing Node: {test_name} " + "="*10)
    setup_test_environment(test_name); test_project_root = os.path.join(TEST_ARTIFACTS_DIR, "project_for_tests")
    print("[SETUP] Creating minimal structure..."); tools.create_directory(os.path.join(test_project_root, "tests", "routes"))
    print("[SETUP] Using sample requirements..."); initial_state = create_base_initial_state(); initial_state["project_root"] = test_project_root
    initial_state["requirements"] = sample_requirements_for_nodes # Uses 'auth' module
    print("[ACTION] Calling generate_unit_tests (LLM call)...");
    try: output_state = node_functions.generate_unit_tests(initial_state.copy())
    except Exception as e: teardown_test_environment(test_name); assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state and file system..."); assert output_state is not None
    syntax_errors_logged = [e for e in output_state.get("error_log", []) if "Syntax error in generated tests" in e]
    if syntax_errors_logged: print(f"[WARN] Node logged expected syntax errors from LLM: {syntax_errors_logged}")
    unexpected_errors = [e for e in output_state.get("error_log", []) if "Syntax error" not in e]
    assert not unexpected_errors, f"Node reported unexpected errors: {unexpected_errors}"
    expected_test_file = os.path.join(test_project_root, "tests", "routes", "test_auth_routes.py")
    if not os.path.isfile(expected_test_file): print(f"[INFO] Expected test file '{expected_test_file}' not created (likely due to syntax error).") # Check node logs for why
    else: print(f"[PASS] Test file found: {expected_test_file}"); test_content = tools.read_file(expected_test_file); assert test_content and "def test_" in test_content
    assert any(os.path.normpath(expected_test_file) in os.path.normpath(f['file_path']) for f in output_state.get("generated_files", [])), "Test file path not tracked"
    assert output_state.get("persistent_context", {}).get("auth_tests_generated") is True, "'auth_tests_generated' flag not set"
    print(f"[PASS] Node '{test_name}' OK."); teardown_test_environment(test_name)


# <<< Corrected Test Function Using Mocks >>>
def test_generate_implementation_code_node():
    """Tests the generate_implementation_code node function (using mocks)."""
    test_name = "test_generate_implementation_code_node"
    print("\n" + "="*10 + f" Testing Node: {test_name} " + "="*10)
    setup_test_environment(test_name)
    try:
        project_root = TEST_ARTIFACTS_DIR

        # --- 1. Setup Test Environment ---
        print("[SETUP] Creating prerequisite directories and files...")
        app_dir=os.path.join(project_root,"app"); models_dir=os.path.join(app_dir,"models")
        services_dir=os.path.join(app_dir,"services"); api_dir=os.path.join(app_dir,"api")
        routes_dir=os.path.join(api_dir,"routes"); alembic_dir=os.path.join(project_root,"alembic")
        alembic_versions_dir=os.path.join(alembic_dir,"versions")
        tools.create_directory(models_dir); tools.create_directory(services_dir)
        tools.create_directory(routes_dir); tools.create_directory(alembic_versions_dir)
        tools.write_file(os.path.join(app_dir,"__init__.py"),""); tools.write_file(os.path.join(api_dir,"__init__.py"),"")
        tools.write_file(os.path.join(models_dir,"__init__.py"),""); tools.write_file(os.path.join(services_dir,"__init__.py"),"")
        tools.write_file(os.path.join(routes_dir,"__init__.py"),"")
        main_py_path=os.path.join(app_dir,"main.py"); tools.write_file(main_py_path, INITIAL_MAIN_PY_CONTENT)
        # Ensure alembic.ini and env.py exist for the node's checks
        tools.write_file(os.path.join(project_root,"alembic.ini"), "[alembic]\nscript_location = alembic\n")
        tools.write_file(os.path.join(alembic_dir,"env.py"), "# Dummy env.py\n")

        initial_state = create_base_initial_state()
        initial_state["project_root"] = project_root
        initial_state["requirements"] = sample_impl_requirements # Use specific requirements for this test
        initial_state["persistent_context"] = {"generated_files": [], "errors": [], "error_log": []} # Ensure keys exist

        # --- 2. Mock External Calls ---
        print("[SETUP] Mocking LLM calls and run_shell_command...")
        mock_llm = MagicMock(spec=llm_calls.generate_code_llm)
        def llm_side_effect(requirements, target_file_path, tests_code, context):
             logging.debug(f"Mock LLM received target: {target_file_path}")
             norm_path = os.path.normpath(target_file_path)
             if norm_path.endswith(os.path.join("models", "models.py")): logger.info("Mock LLM: Return model"); return MOCK_USER_MODEL_CODE
             elif norm_path.endswith(os.path.join("services", "auth_service.py")): logger.info("Mock LLM: Return auth service"); return MOCK_AUTH_SERVICE_CODE
             elif norm_path.endswith(os.path.join("services", "default_logic_service.py")): logger.info("Mock LLM: Return default service"); return MOCK_DEFAULT_SERVICE_CODE
             elif norm_path.endswith(os.path.join("api", "routes", "auth_routes.py")): logger.info("Mock LLM: Return auth routes"); return MOCK_AUTH_ROUTES_CODE
             logger.warning(f"Mock LLM: No specific mock for {target_file_path}."); return "# default code\npass"
        mock_llm.side_effect = llm_side_effect

        # <<< Corrected mocking target to tools.run_shell_command >>>
        mock_run_shell = MagicMock(spec=tools.run_shell_command)
        alembic_command_str = 'alembic revision --autogenerate -m "AI generated model changes"'
        dummy_migration_filename = "abc123_mock_migration.py"
        dummy_migration_path_in_stdout = os.path.join("alembic", "versions", dummy_migration_filename).replace("\\","/")
        success_alembic_result = {"command": alembic_command_str, "return_code": 0, "stdout": f"Generating {dummy_migration_path_in_stdout} ... done", "stderr": "", "success": True}

        def run_shell_side_effect(command, cwd=None, timeout=300):
            logger.info(f"Mocked run_shell_command received: '{command}'")
            if command.strip().startswith("alembic revision --autogenerate"):
                logger.info("Mocking successful Alembic run.")
                # The node function might parse stdout, so we return the success dict
                return success_alembic_result
            logger.warning(f"Unexpected shell command mocked: {command}")
            return {"command": command, "return_code": 1, "stderr": "Unexpected cmd", "success": False}
        mock_run_shell.side_effect = run_shell_side_effect

        # --- 3. Execute Node Function ---
        print(f"[ACTION] Calling {node_functions.generate_implementation_code.__name__}...")
        # Patch the correct targets within the node_functions module
        with patch.object(node_functions.llm_calls, 'generate_code_llm', mock_llm), \
             patch.object(node_functions.tools, 'run_shell_command', mock_run_shell):
             output_state = node_functions.generate_implementation_code(initial_state.copy())

        # --- 4. Assertions ---
        print("[ASSERT] Checking output state and file system...")
        assert output_state is not None, "Node function returned None"
        # <<< Check correct error log key >>>
        node_errors = output_state.get("error_log", [])
        assert not node_errors, f"Node reported errors: {node_errors}"

        # <<< Corrected expected call count >>>
        assert mock_llm.call_count == 4, f"Expected 4 LLM calls (model, svc_auth, svc_default, route_auth), got {mock_llm.call_count}"

        expected_model_file=os.path.join(project_root,"app","models","models.py")
        expected_service_file=os.path.join(project_root,"app","services","auth_service.py")
        expected_default_service_file=os.path.join(project_root,"app","services","default_logic_service.py")
        expected_route_file=os.path.join(project_root,"app","api","routes","auth_routes.py")
        expected_main_py=main_py_path

        assert os.path.exists(expected_model_file), f"Model file missing: {expected_model_file}"
        assert os.path.exists(expected_service_file), f"Auth Service file missing: {expected_service_file}"
        assert os.path.exists(expected_default_service_file), f"Default Service file missing: {expected_default_service_file}"
        assert os.path.exists(expected_route_file), f"Auth Route file missing: {expected_route_file}"
        assert os.path.exists(expected_main_py), "main.py missing"

        model_content = tools.read_file(expected_model_file); assert model_content is not None
        # <<< Corrected assertion for SQLAlchemy model >>>
        assert "class User(Base):" in model_content, "Model content mismatch (User class)"
        assert "__tablename__ = \"users\"" in model_content, "Model content mismatch (__tablename__)"

        service_content = tools.read_file(expected_service_file); assert service_content is not None
        assert "class AuthService:" in service_content, "Service content mismatch"

        route_content = tools.read_file(expected_route_file); assert route_content is not None
        assert "router = APIRouter()" in route_content, "Route content missing router instance"
        assert "@router.post(\"/login\"" in route_content or "@router.post('/login'" in route_content, "Route content missing login route"

        main_content = tools.read_file(expected_main_py); assert main_content is not None
        print(f"--- Updated main.py content ---\n{main_content}\n-----------------------------")
        # <<< Corrected assertion for aliased import >>>
        assert "from app.api.routes.auth_routes import router as auth_router" in main_content, "main.py missing router import"
        assert "app.include_router(auth_router" in main_content.replace(" ", ""), "main.py missing include_router call"

        # Check Alembic mock was called correctly
        mock_run_shell.assert_called_once_with(alembic_command_str, cwd=project_root)

        # Check generated files list in state
        generated_files_list = output_state.get("generated_files", [])
        expected_paths_in_state = [os.path.normpath(p) for p in [expected_model_file, expected_service_file, expected_route_file, expected_default_service_file, expected_main_py]]
        found_files_in_state = [os.path.normpath(f['file_path']) for f in generated_files_list]
        for expected in expected_paths_in_state: assert expected in found_files_in_state, f"Expected file {expected} not in state list"

        # Check migration file tracking based on mock stdout parsing
        expected_migration_abs = os.path.join(project_root, "alembic", "versions", dummy_migration_filename)
        found_migration = any(os.path.normpath(f['file_path']) == os.path.normpath(expected_migration_abs) for f in generated_files_list)
        assert found_migration, "Migration file path not tracked in state generated_files list"

        p_context = output_state.get("persistent_context", {})
        assert p_context.get("models_generated") is True, "Context 'models_generated' flag not set"
        assert p_context.get("auth_service_generated") is True, "Context 'auth_service_generated' flag not set"
        assert p_context.get("default_logic_service_generated") is True, "Context 'default_logic_service_generated' flag not set" # Added check
        assert p_context.get("auth_routes_generated") is True, "Context 'auth_routes_generated' flag not set"

        print(f"[PASS] Node '{test_name}' completed successfully.")

    finally:
        # --- 5. Teardown ---
        teardown_test_environment(test_name) # Ensure cleanup even on failure


# --- TODO: Add Test Functions for other nodes ---
# test_run_tests_and_validate_node
# test_debug_code_node
# test_generate_documentation_node
# test_generate_deployment_package_node
# test_finalize_workflow_node

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n" + "="*60); print("--- Starting Node Function Tests ---"); print("="*60)
    node_tests_to_run = [
        test_load_srs_node,
        test_analyze_srs_node,
        test_setup_project_structure_node,
        test_generate_unit_tests_node,
        test_generate_implementation_code_node, # Keep this test
        # Add other test function names here when implemented
    ]
    failed_node_tests = []; passed_node_tests = []
    for test_func in node_tests_to_run:
        test_name = test_func.__name__
        try: test_func(); passed_node_tests.append(test_name)
        except AssertionError as e: print(f"\n---! ASSERTION FAILED in {test_name}: {e} !---"); failed_node_tests.append(test_name)
        except Exception as e: print(f"\n---! UNEXPECTED ERROR in {test_name}: {e} !---"); traceback.print_exc(); failed_node_tests.append(test_name)

    print("\n" + "="*60); print("--- Node Function Test Summary ---"); print(f"Total Executed: {len(node_tests_to_run)}");
    print(f"Passed: {len(passed_node_tests)}"); print(f"Failed/Errored: {len(failed_node_tests)}")
    if failed_node_tests: print(f"Failed/Errored Tests: {', '.join(failed_node_tests)}"); print("--- Please review errors. ---")
    else: print("--- All tests passed! Review LLM quality & implement remaining nodes/tests. ---")
    print("="*60)
    # if failed_node_tests: exit(1)
