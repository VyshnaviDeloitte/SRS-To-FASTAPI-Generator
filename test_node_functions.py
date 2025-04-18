# test_node_functions.py
import os
import shutil
import logging
import traceback
import json
import ast
import subprocess  # Import needed for mocking
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock  # Use MagicMock for flexibility

# --- Load Env Vars & Imports ---
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("Warning: .env file not found.")
try:
    from project_state import ProjectState, GenerationResult, TestResult

    print("Imported state.")
except ImportError as e:
    print(f"FATAL: Import project_state failed: {e}");
    exit()
try:
    import node_functions

    print("Imported node_functions.")
except ImportError as e:
    print(f"FATAL: Import node_functions failed: {e}\n{traceback.format_exc()}");
    exit()  # Print traceback
except Exception as e:
    print(f"FATAL: Unexpected error importing node_functions: {e}");
    traceback.print_exc();
    exit()
try:
    import tools
    import llm_calls

    print("Imported tools & llm_calls.")
except ImportError as e:
    print(f"FATAL: Import tools/llm_calls failed: {e}");
    exit()
except RuntimeError as e:
    print(f"FATAL: LLM init failed: {e}");
    exit()
except Exception as e:
    print(f"FATAL: Import error: {e}");
    traceback.print_exc();
    exit()

# --- Logging & Test Env Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)
TEST_ARTIFACTS_DIR = "temp_node_test_artifacts"


def setup_test_environment(test_name: str):
    print(f"\n--- Setup {test_name} ---");
    shutil.rmtree(TEST_ARTIFACTS_DIR, ignore_errors=True);
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)


def teardown_test_environment(test_name: str):
    print(f"\n--- Teardown {test_name} ---");
    shutil.rmtree(TEST_ARTIFACTS_DIR, ignore_errors=True)


# --- Sample Data ---
sample_srs_text_for_nodes = "Feature: User Login. POST /api/auth/login needs email/password, returns JWT. Needs users table (id, email, hashed_password)."
sample_requirements_for_nodes = {  # Output of analyze_srs
    "endpoints": [{"path": "/api/auth/login", "method": "POST", "description": "User Login",
                   "request_body_schema": {"email": "string", "password": "string"},
                   "response_schema": {"token": "string"}, "auth_required": False, "roles": []}],
    "database_schema": {
        "tables": [{"name": "users",
                    "columns": [{"name": "id", "type": "INTEGER", "primary_key": True},
                                {"name": "email", "type": "VARCHAR", "unique": True},
                                {"name": "hashed_password", "type": "VARCHAR"}], "relationships": []}]},
    "business_logic": ["Hash password.", "Validate email."], "auth_requirements": {"type": "JWT"}}
sample_impl_requirements = {  # Requirements for impl test
    "database_schema": {
        "tables": [{"name": "users",
                    "columns": [{"name": "id", "type": "INTEGER", "primary_key": True},
                                {"name": "username", "type": "VARCHAR", "unique": True},
                                {"name": "email", "type": "VARCHAR"},
                                {"name": "hashed_password", "type": "VARCHAR"}], "relationships": []}]},
    "endpoints": [{"path": "/api/auth/register", "method": "POST", "description": "Register user",
                   "request_body_schema": {"username": "str", "email": "str", "password": "str"},
                   "response_schema": {"id": "int", "username": "str"}, "auth_required": False, "roles": []},
                  {"path": "/api/auth/login", "method": "POST", "description": "User login",
                   "request_body_schema": {"username": "str", "password": "str"},
                   "response_schema": {"access_token": "str"}, "auth_required": False, "roles": []}],
    "business_logic": ["Hash passwords using bcrypt.", "Login checks username/password."],
    "auth_requirements": {"type": "JWT"}}

# --- Base Initial State (Helper) ---
def create_base_initial_state() -> ProjectState:
    return {"srs_path": "", "project_root": "", "max_debug_iterations": 3, "srs_text": None, "srs_image_path": None,
            "requirements": None,
            "generated_files": [], "test_results": None, "validation_errors": None, "error_log": [], "debug_iterations": 0,
            "persistent_context": {}, "langsmith_run_url": None, "final_zip_path": None, "documentation_files": []}

# --- Mock Data & Expected Code (for generate_implementation_code test) ---
MOCK_USER_MODEL_CODE = """
from app.database import Base
from sqlalchemy import Column, Integer, String
from pydantic import BaseModel

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    email = Column(String)
    hashed_password = Column(String)

class UserSchema(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        orm_mode = True
"""
MOCK_AUTH_SERVICE_CODE = """
from sqlalchemy.orm import Session
# Assume models/schemas are imported correctly where needed
import app.models.models as models
import app.models.models as schemas # If schemas in models.py
from passlib.context import CryptContext
import logging
logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
class AuthService:
    def get_user(self, db: Session, username: str): return None # Simple mock
    def verify_password(self, plain_password: str, hashed_password: str) -> bool: return True # Simple mock
    def get_password_hash(self, password: str) -> str: return pwd_context.hash(password)
auth_service = AuthService()
"""
MOCK_AUTH_ROUTES_CODE = """
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
import app.services.auth_service as service
import app.models.models as schemas # Assuming schemas in models.py
router = APIRouter()
class UserCreate(schemas.BaseModel): username: str; email: str; password: str
class UserLogin(schemas.BaseModel): username: str; password: str
class Token(schemas.BaseModel): access_token: str; token_type: str = "bearer"
@router.post("/register", response_model=schemas.UserSchema, status_code=status.HTTP_201_CREATED) # Specify response model & status
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Mock registering user {user_data.username}")
    hashed_pw = service.auth_service.get_password_hash(user_data.password)
    return {"id": 1, "username": user_data.username, "email": user_data.email} # Match UserSchema
@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin, db: Session = Depends(get_db)):
    # user = service.auth_service.get_user(db, username=form_data.username) # Simplified mock
    if form_data.username == "test" and form_data.password == "pass": # Dummy check
        access_token = f"fake-jwt-token-for-{form_data.username}"
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
"""
INITIAL_MAIN_PY_CONTENT = "from fastapi import FastAPI\napp = FastAPI()\n# Placeholder"

# --- Node Test Functions ---

def test_load_srs_node():
    test_name = "test_load_srs_node";
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name);
    dummy_docx_path = os.path.join(TEST_ARTIFACTS_DIR, "test_srs.docx");
    dummy_content = "SRS Content."
    docx_created = False
    try:
        import docx

        doc = docx.Document();
        doc.add_paragraph(dummy_content);
        doc.save(dummy_docx_path);
        print(f"[SETUP] Created: {dummy_docx_path}");
        assert os.path.exists(dummy_docx_path);
        docx_created = True
    except ImportError:
        print("\n---! SKIPPING: python-docx not installed !---\n");
        teardown_test_environment(test_name);
        return
    except Exception as e:
        teardown_test_environment(test_name);
        assert False, f"Dummy docx creation failed: {e}"
    if not docx_created:
        teardown_test_environment(test_name);
        return
    initial_state = create_base_initial_state();
    initial_state["srs_path"] = dummy_docx_path
    try:
        output_state = node_functions.load_srs(initial_state.copy())
    except Exception as e:
        teardown_test_environment(test_name);
        assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state...");
    assert output_state is not None;
    assert isinstance(output_state, dict)
    assert output_state.get("srs_text") is not None;
    assert dummy_content in output_state.get("srs_text", "")
    assert not output_state.get("error_log"), f"Errors: {output_state.get('error_log')}";
    assert output_state.get("srs_path") == dummy_docx_path
    print(f"[PASS] Node '{test_name}' OK.");
    teardown_test_environment(test_name)

def test_analyze_srs_node():
    test_name = "test_analyze_srs_node";
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name);
    initial_state = create_base_initial_state();
    initial_state["srs_text"] = sample_srs_text_for_nodes
    print("[ACTION] Calling analyze_srs (LLM call)...");
    try:
        output_state = node_functions.analyze_srs(initial_state.copy())
    except Exception as e:
        teardown_test_environment(test_name);
        assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state...");
    assert output_state is not None
    critical_errors = [e for e in output_state.get("error_log", []) if "failed" in e.lower()]
    assert not critical_errors, f"Node reported critical errors: {critical_errors}"
    assert output_state.get("requirements") is not None;
    requirements_data = output_state.get("requirements")
    print(f"[INFO] Requirements type: {type(requirements_data)}");
    assert isinstance(requirements_data, dict)
    assert "endpoints" in requirements_data;
    assert "database_schema" in requirements_data
    assert output_state.get("persistent_context", {}).get("requirements_summary")
    print(f"[PASS] Node '{test_name}' OK.");
    print("[INFO] Manual check needed:");
    print(json.dumps(requirements_data, indent=2))
    teardown_test_environment(test_name)

def test_setup_project_structure_node():
    test_name = "test_setup_project_structure_node";
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name);
    test_project_root = os.path.join(TEST_ARTIFACTS_DIR, "my_generated_project")
    initial_state = create_base_initial_state();
    initial_state["project_root"] = test_project_root
    print(f"[ACTION] Calling setup_project_structure: {test_project_root}")
    try:
        output_state = node_functions.setup_project_structure(initial_state.copy())
    except Exception as e:
        teardown_test_environment(test_name);
        assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state and file system...");
    assert output_state is not None;
    assert not output_state.get("error_log")
    assert os.path.isdir(test_project_root)
    expected_dirs = [os.path.join("app", "api", "routes"), os.path.join("tests"),
                     os.path.join("alembic", "versions")]
    for d in expected_dirs:
        assert os.path.isdir(os.path.join(test_project_root, d)), f"Missing dir '{d}'"
    expected_files = [os.path.join("app", "main.py"), "requirements.txt", "alembic.ini",
                      ".gitignore"]
    for f in expected_files:
        assert os.path.isfile(os.path.join(test_project_root, f)), f"Missing file '{f}'"
    assert len(output_state.get("generated_files", [])) > 5;
    assert output_state.get("persistent_context", {}).get("project_structure_summary")
    print(f"[PASS] Node '{test_name}' OK.");
    teardown_test_environment(test_name)

def test_generate_unit_tests_node():
    test_name = "test_generate_unit_tests_node";
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name);
    test_project_root = os.path.join(TEST_ARTIFACTS_DIR, "project_for_tests")
    print("[SETUP] Creating minimal structure...");
    tools.create_directory(os.path.join(test_project_root, "tests", "routes"))
    print("[SETUP] Using sample requirements...");
    initial_state = create_base_initial_state();
    initial_state["project_root"] = test_project_root
    initial_state["requirements"] = sample_requirements_for_nodes  # Uses 'auth' module
    print("[ACTION] Calling generate_unit_tests (LLM call)...");
    try:
        output_state = node_functions.generate_unit_tests(initial_state.copy())
    except Exception as e:
        teardown_test_environment(test_name);
        assert False, f"Node '{test_name}' raised: {e}\n{traceback.format_exc()}"
    print("[ASSERT] Checking output state and file system...");
    assert output_state is not None
    syntax_errors_logged = [e for e in output_state.get("error_log", []) if
                            "Syntax error in generated tests" in e]
    if syntax_errors_logged:
        print(f"[WARN] Node logged expected syntax errors from LLM: {syntax_errors_logged}")
    unexpected_errors = [e for e in output_state.get("error_log", []) if "Syntax error" not in e]
    assert not unexpected_errors, f"Node reported unexpected errors: {unexpected_errors}"
    expected_test_file = os.path.join(test_project_root, "tests", "routes", "test_auth_routes.py")
    # Check if file exists - node might skip writing if syntax error found
    if not os.path.isfile(expected_test_file):
        print(
            f"[INFO] Expected test file '{expected_test_file}' not created (likely due to generated code syntax error inside the node).")
        assert syntax_errors_logged, "Test file not created, but no syntax error was logged"
    else:
        print(f"[PASS] Test file found: {expected_test_file}");
        test_content = tools.read_file(expected_test_file);
        assert test_content and "def test_" in test_content
        print(f"[INFO] Content of {expected_test_file} (first 500 chars):\n```python\n{test_content[:500]}...\n```")
    assert any(
        os.path.normpath(expected_test_file) in os.path.normpath(f['file_path']) for f in
        output_state.get("generated_files", [])), "Test file path not tracked"
    assert output_state.get("persistent_context", {}).get("auth_tests_generated") is True, "'auth_tests_generated' flag not set"
    print(f"[PASS] Node '{test_name}' OK.");
    teardown_test_environment(test_name)

def test_generate_implementation_code_node():
    """Tests the generate_implementation_code node function (using mocks)."""
    test_name = "test_generate_implementation_code_node"
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    # Use try...finally for setup/teardown within the test
    setup_test_environment(test_name)
    try:
        project_root = TEST_ARTIFACTS_DIR

        # --- 1. Setup Test Env ---
        print("[SETUP] Creating prerequisite directories and files...")
        app_dir = os.path.join(project_root, "app");
        models_dir = os.path.join(app_dir, "models")
        services_dir = os.path.join(app_dir, "services");
        api_dir = os.path.join(app_dir, "api")
        routes_dir = os.path.join(api_dir, "routes");
        alembic_dir = os.path.join(project_root, "alembic")
        alembic_versions_dir = os.path.join(alembic_dir, "versions")
        tools.create_directory(models_dir);
        tools.create_directory(services_dir)
        tools.create_directory(routes_dir);
        tools.create_directory(alembic_versions_dir)
        tools.write_file(os.path.join(app_dir, "__init__.py"), "");
        tools.write_file(os.path.join(api_dir, "__init__.py"), "")
        tools.write_file(os.path.join(models_dir, "__init__.py"), "");
        tools.write_file(os.path.join(services_dir, "__init__.py"), "")
        tools.write_file(os.path.join(routes_dir, "__init__.py"), "")
        main_py_path = os.path.join(app_dir, "main.py");
        tools.write_file(main_py_path, INITIAL_MAIN_PY_CONTENT)
        tools.write_file(os.path.join(project_root, "alembic.ini"), "[alembic]\nscript_location = alembic\n")
        tools.write_file(os.path.join(alembic_dir, "env.py"), "# Dummy env.py\n")

        initial_state = create_base_initial_state()
        initial_state["project_root"] = project_root
        # <<< Use sample_impl_requirements defined earlier >>>
        initial_state["requirements"] = sample_impl_requirements
        initial_state["persistent_context"] = {"generated_files": [], "errors": []}  # Start clean

        # --- 2. Mock External Calls ---
        print("[SETUP] Mocking LLM calls and subprocess...")
        mock_llm = MagicMock(spec=llm_calls.generate_code_llm)

        def llm_side_effect(requirements, target_file_path, tests_code, context):
            # Check requirements to simulate different outputs
            is_model = "database_schema" in requirements
            is_service = "business_logic" in requirements
            is_route = "endpoints" in requirements

            if is_model and "User" in requirements["database_schema"]["tables"][0]["name"]:
                logger.info("Mock LLM: Returning model code.");
                return MOCK_USER_MODEL_CODE
            elif is_service and "Auth" in target_file_path:
                logger.info("Mock LLM: Returning service code.");
                return MOCK_AUTH_SERVICE_CODE
            elif is_route and "Auth" in target_file_path:
                logger.info("Mock LLM: Returning routes code.");
                return MOCK_AUTH_ROUTES_CODE
            logger.warning(f"Mock LLM: No specific mock for {target_file_path}.");
            return "# Default mock code\npass\n"

        mock_llm.side_effect = llm_side_effect

        mock_subprocess_run = MagicMock(spec=subprocess.run)
        dummy_migration_filename = "abc123_generate_models_structure.py"
        dummy_migration_path_in_stdout = os.path.join("alembic", "versions",
                                                    dummy_migration_filename).replace("\\",
                                                                                               "/")  # Path as it might appear in stdout
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=["alembic", "..."],
                                                                      returncode=0,
                                                                      stdout=f"Generating {dummy_migration_path_in_stdout} ... done",
                                                                      stderr="")

        # --- 3. Execute Node Function ---
        print(f"[ACTION] Calling {node_functions.generate_implementation_code.__name__}...")
        with patch.object(node_functions.llm_calls, 'generate_code_llm', mock_llm), \
             patch.object(node_functions, 'subprocess') as mock_subprocess_module:
            mock_subprocess_module.run = mock_subprocess_run
            output_state = node_functions.generate_implementation_code(initial_state.copy())

        # --- 4. Assertions ---
        print("[ASSERT] Checking output state and file system...")
        assert output_state is not None, "Node function returned None"
        # Check errors specifically added by *this* node run
        impl_errors = [e for e in output_state.get("error_log", []) if
                       "generate_implementation_code:" in e]  # Check errors logged by this node specifically
        assert not impl_errors, f"Node reported errors during execution: {impl_errors}"

        assert mock_llm.call_count == 3, f"Expected 3 LLM calls, got {mock_llm.call_count}"

        expected_model_file = os.path.join(models_dir, "models.py")
        expected_service_file = os.path.join(services_dir, "auth_service.py")  # Check name based on reqs
        expected_route_file = os.path.join(routes_dir, "auth_routes.py")  # Check name based on reqs
        expected_main_py = main_py_path
        # Check if *actual* migration file was created (it shouldn't be with mock)
        actual_migration_files = os.listdir(alembic_versions_dir)
        assert not any(f.endswith(".py") for f in actual_migration_files), f"Mock shouldn't create real migration file, but found: {actual_migration_files}"

        assert os.path.exists(expected_model_file), "Model file was not created"
        assert os.path.exists(expected_service_file), "Service file was not created"
        assert os.path.exists(expected_route_file), "Route file was not created"
        assert os.path.exists(expected_main_py), "main.py should still exist"

        model_content = tools.read_file(expected_model_file);
        assert model_content is not None
        # Check content based on MOCK data
        assert "class User(Base):" in model_content or "class User(BaseModel):" in model_content, "Model content mismatch"

        service_content = tools.read_file(expected_service_file);
        assert service_content is not None
        assert "class AuthService:" in service_content, "Service content mismatch"

        route_content = tools.read_file(expected_route_file);
        assert route_content is not None
        assert "router = APIRouter()" in route_content, "Route content mismatch - router instance"
        assert "@router.post(\"/login\"" in route_content or "@router.post('/login'" in route_content, "Route content mismatch - login route"

        main_content = tools.read_file(expected_main_py);
        assert main_content is not None
        print(f"--- Updated main.py content ---\n{main_content}\n-----------------------------")
        assert "from app.api.routes.auth_routes import auth_router" in main_content, "main.py missing import (check alias used)"
        assert "app.include_router(auth_router" in main_content.replace(" ", ""), "main.py missing include_router call"

        mock_subprocess_run.assert_called_once()  # Check alembic mock was called
        call_args, call_kwargs = mock_subprocess_run.call_args
        assert call_args[0][2] == "--autogenerate", "Alembic command wasn't '--autogenerate'"
        assert call_kwargs.get("cwd") == project_root, "Alembic wasn't run in project_root"

        generated_files_list = output_state.get("generated_files", [])
        assert any(
            os.path.normpath(f['file_path']) == os.path.normpath(expected_model_file) for f in
            generated_files_list), "Model file missing from state list"
        assert any(
            os.path.normpath(f['file_path']) == os.path.normpath(expected_service_file) for f in
            generated_files_list), "Service file missing from state list"
        assert any(
            os.path.normpath(f['file_path']) == os.path.normpath(expected_route_file) for f in
            generated_files_list), "Route file missing from state list"
        # Note: Migration file path from mock stdout might not be added by node logic, adjust if needed

        p_context = output_state.get("persistent_context", {})
        assert p_context.get("models_generated") is True, "Context 'models_generated' flag not set"
        # Check flags based on requirements used
        assert p_context.get("auth_service_generated") is True, "Context 'auth_service_generated' flag not set"
        assert p_context.get("auth_routes_generated") is True, "Context 'auth_routes_generated' flag not set"

        print(f"[PASS] Node '{test_name}' completed successfully.")

    finally:
        # --- 5. Teardown ---
        teardown_test_environment(test_name)  # Ensure cleanup even on failure

def test_run_tests_and_validate_node():
    test_name = "test_run_tests_and_validate_node"
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name)
    project_root = TEST_ARTIFACTS_DIR
    initial_state = create_base_initial_state()
    initial_state["project_root"] = project_root

    # --- 1. Mock tools.run_shell_command ---
    mock_run_shell_command = MagicMock(spec=tools.run_shell_command)

    def run_shell_side_effect(command, cwd):
        if "ruff" in command:
            return {
                "success": True,
                "stdout": "No issues found.",
                "stderr": "",
            }  # Simulate ruff success
        elif "pytest" in command:
            return {
                "success": True,
                "stdout": "1 passed in 0.10s",
                "stderr": "",
            }  # Simulate pytest success
        return {"success": False, "stdout": "", "stderr": "Unknown command"}

    mock_run_shell_command.side_effect = run_shell_side_effect

    # --- 2. Execute the Node ---
    with patch.object(tools, "run_shell_command", mock_run_shell_command):
        output_state = node_functions.run_tests_and_validate(initial_state.copy())

    # --- 3. Assertions ---
    print("[ASSERT] Checking output state...")
    assert output_state is not None
    assert output_state.get("test_results") is not None
    assert output_state["test_results"]["passed"] is True
    assert "passed" in output_state["test_results"]["output"]

    assert output_state.get("validation_errors") is None

    # --- 4. Test Failure Scenario ---
    # Simulate pytest failure
    mock_run_shell_command.side_effect = lambda command, cwd: {
        "success": False,
        "stdout": "1 failed",
        "stderr": "AssertionError: ...",
    } if "pytest" in command else run_shell_side_effect(command, cwd)

    with patch.object(tools, "run_shell_command", mock_run_shell_command):
        output_state = node_functions.run_tests_and_validate(initial_state.copy())

    assert output_state["test_results"]["passed"] is False

    print(f"[PASS] Node '{test_name}' OK.")
    teardown_test_environment(test_name)

def test_debug_code_node():
    test_name = "test_debug_code_node"
    print("\n" + "=" * 10 + f" Testing Node: {test_name} " + "=" * 10)
    setup_test_environment(test_name)
    project_root = TEST_ARTIFACTS_DIR
    initial_state = create_base_initial_state()
    initial_state["project_root"] = project_root
    test_file_path = os.path.join(project_root, "test_file.py")
    tools.write_file(test_file_path, "def my_func():\n    print('Hello')\n")  # Create a dummy file

    # --- 1. Mocking ---
    mock_read_file = MagicMock(spec=tools.read_file)
    mock_read_file.return_value = "def my_func():\n    print('Hello')\n"

    mock_llm_call = MagicMock(spec=llm_calls.generate_code_llm)
    mock_llm_call.return_value = "def my_func():\n    print('Hello')\n    # Fixed\n"  # Simulate a code fix

    mock_find_and_replace = MagicMock(spec=node_functions._find_and_replace_code_block)
    mock_find_and_replace.return_value = "def my_func():\n    print('Hello')\n    # Fixed\n"

    # --- 2. Set Validation Errors ---
    initial_state["validation_errors"] = [
        {
            "file_path": test_file_path,
            "error_message": "IndentationError: unexpected indent",
            "line_number": 2,
        }
    ]

    # --- 3. Execute Node ---
    with patch.object(tools, "read_file", mock_read_file), \
        patch.object(llm_calls, "generate_code_llm", mock_llm_call), \
        patch.object(node_functions, "_find_and_replace_code_block", mock_find_and_replace):
        output_state = node_functions.debug_code(initial_state.copy())

    # --- 4. Assertions ---
    print("[ASSERT] Checking output state...")
    assert output_state is not None
    assert output_state.get("validation_errors") is None  # Errors should be cleared

    mock_read_file.assert_called_once_with(test_file_path)
    mock_llm_call.assert_called_once()
    mock_find_and_replace.assert_called_once()

    print(f"[PASS] Node '{test_name}' OK.")
    teardown_test_environment(test_name)

def test_generate_documentation_node():
    test_name = "test_generate_documentation_node"
    print("\n" + "=" * 10 + f" Testing Node: {test_name}")
