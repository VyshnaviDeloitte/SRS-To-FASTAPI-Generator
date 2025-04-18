# llm_calls.py
import os
import base64
import logging
import json
from typing import Dict, List, Optional, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
# from project_state import ExtractedRequirements # Uncomment if using Pydantic validation

logger = logging.getLogger(__name__)

# --- Configure Groq LLMs ---
try:
    analyzer_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)
    test_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    code_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)
    debugger_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1)
    doc_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    logger.info("Groq LLM clients initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM clients: {e}", exc_info=True)
    raise RuntimeError(f"Could not initialize Groq LLMs: {e}") from e

def analyze_srs_llm(srs_content: str, image_path: Optional[str] = None) -> Optional[Dict]:
    """
    Uses Groq LLM(s) to analyze SRS text (and potentially an image)
    and extract structured requirements.
    """
    logger.info("--- Calling Groq LLM to Analyze SRS ---")
    if not analyzer_llm:
        logger.error("Analyzer LLM not initialized.")
        return None

    # --- Vision Model Handling (Placeholder) ---
    image_analysis_results = None
    if image_path:
        logger.warning(f"Image analysis requested for {image_path}, but not implemented.")
        image_analysis_results = {"comment": "Image analysis not implemented"}

    image_context_str = ""
    if image_analysis_results:
        image_context_str = f"\n\nAnalysis results from an accompanying image (if provided):\n```json\n{json.dumps(image_analysis_results, indent=2)}\n```\nUse these results to supplement the text analysis, particularly for the database schema."

    # --- Text Analysis Prompt ---
    # <<< FIXED: Define template with explicit input variables and escaped literals >>>
    prompt_template_text = """
    You are an expert software requirements analyst using Llama 3. Analyze the following Software Requirements Specification (SRS) text:
    --- SRS TEXT ---
    {srs_content}
    --- END SRS TEXT ---
    {image_context}

    Extract the following information and structure it as a JSON object:
    1.  `endpoints`: A list of API endpoints including path (e.g., "/api/lms/leaves/apply"), method (e.g., "POST"), description (if any), request body schema (describe structure, e.g., {{{{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "reason": "string" }}}}), response schema (describe structure, e.g., {{{{ "message": "string", "status": "string" }}}}), auth_required (boolean), and roles (list of strings like ['user', 'manager'], derive from context). For paths with path parameters like `/api/lms/leaves/{{leave_id}}/approve`, represent the parameter clearly. Example Body: {{{{ status: 'approved' | 'rejected', comment }}}}. Example Response: {{{{ message: string, status: string }}}}. Example Response for balance: {{{{ paid_balance: int, sick_balance: int }}}}.
    2.  `database_schema`: An object containing a list of `tables`. Each table should have a name (e.g., "leaves"), a list of `columns` (with name, type - use standard SQL types like VARCHAR, INTEGER, TEXT, BOOLEAN, TIMESTAMP, DATE, etc., primary_key=True/False, foreign_key details if any, e.g., {{{{ "name": "user_id", "type": "INTEGER", "foreign_key": "users.id" }}}}), and optional `relationships` descriptions (e.g., "User has many Leaves"). **Prioritize details found in image analysis results if provided.**
    3.  `business_logic`: A list of strings, each describing a key business rule or computation required (e.g., "Leave balance decreases upon approval", "Managers can only approve leaves for their team").
    4.  `auth_requirements`: An object describing authentication (e.g., {{{{ "type": "JWT" }}}}) and authorization rules (e.g., {{{{ "rbac_summary": "Managers can access all user APIs plus specific manager APIs" }}}}).

    **Important:** Provide the output ONLY as a valid JSON object enclosed in triple backticks (```json ... ```). Do not include any explanatory text before or after the JSON. Ensure JSON keys and string values use double quotes. Ensure the JSON is well-formed. If certain information isn't found, use null or empty lists/objects as appropriate within the JSON structure.
    """

    # Create the prompt template expecting the variables
    prompt = ChatPromptTemplate.from_template(prompt_template_text)
    parser = JsonOutputParser()
    chain = prompt | analyzer_llm | parser

    # Prepare the input dictionary for the template variables
    input_data = {
        "srs_content": srs_content,
        "image_context": image_context_str
    }

    try:
        logger.info("Invoking Groq Analyzer LLM (Llama3-70b)...")
        # <<< FIXED: Pass the input_data dictionary to invoke >>>
        result = chain.invoke(input_data)

        # TODO: Add Pydantic validation here if desired

        logger.info("Groq LLM Analysis successful (structure validation may be needed).")
        return result

    except Exception as e:
        logger.error(f"Error during Groq LLM analysis: {e}", exc_info=True)
        # Attempt to log raw output on failure
        try:
            logger.info("Attempting to get raw output on failure...")
            raw_output_chain = prompt | analyzer_llm | StrOutputParser()
            raw_output = raw_output_chain.invoke(input_data) # Pass input here too
            logger.debug(f"Raw output from failed analysis: {raw_output}")
        except Exception as inner_e:
            logger.debug(f"Could not get raw output after initial failure: {inner_e}")
        return None


# --- Other functions (generate_tests_llm, generate_code_llm, etc.) ---
# Keep the versions from the previous response (they correctly passed input variables)

def generate_tests_llm(requirements: Dict, file_path: str, context: Dict) -> Optional[str]:
    """
    Generates pytest unit tests based on requirements using Groq LLM.
    """
    logger.info(f"--- Calling Groq LLM to Generate Tests for: {file_path} ---")
    if not test_llm:
        logger.error("Test LLM (Groq) not initialized.")
        return None

    prompt_template_text = """
    You are an expert Python Test Engineer using pytest and FastAPI's TestClient, powered by Llama 3.
    Generate pytest unit tests for the following requirements, targeting the file `{file_path}`.
    Assume standard FastAPI project structure and pytest fixtures (like a `client` fixture defined in `tests/conftest.py`).

    Relevant Requirements:
    ```json
    {requirements_str}
    ```

    Persistent Context (Existing elements like model names or other generated code):
    ```json
    {context_str}
    ```

    Generate comprehensive tests covering:
    - Success cases (status code 200/201) with valid inputs.
    - Input validation errors (status code 422) with invalid inputs.
    - Authentication/Authorization errors (status codes 401, 403 based on roles if specified) by simulating requests with/without valid tokens/roles.
    - Not found errors (status code 404) when applicable.
    - Edge cases relevant to the logic.

    **CRITICAL INSTRUCTIONS:**
    1.  **Provide ONLY valid Python code.**
    2.  **Do NOT include any explanatory text, comments, or markdown formatting (like ```python) before or after the code.**
    3.  **Start the response directly with the necessary Python import statements.**
    4.  **Ensure the generated code is syntactically correct and complete.**
    5.  Assume necessary fixtures like `client = TestClient(app)` are available via `conftest.py`.
    6.  Use placeholder data for request bodies where appropriate (e.g., valid dates, sample IDs).

    Example Structure:
    ```python
    import pytest
    from fastapi.testclient import TestClient
    # from app.main import app # Import may not be needed if client fixture handles it
    # Import necessary schemas if needed for request bodies

    def test_endpoint_success(client): # Assuming client fixture
        response = client.post("/api/lms/leaves/apply", json={{{{...valid_data...}}}}) # Escaped braces in example
        assert response.status_code == 200 # Or 201, etc.
        # Add assertions for response body content

    def test_endpoint_validation_error(client):
        response = client.post("/api/lms/leaves/apply", json={{{{...invalid_data...}}}}) # Escaped braces in example
        assert response.status_code == 422
    # ... more test functions ...
    ```

    Now, generate the tests based on the requirements provided above:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_text)
    parser = StrOutputParser()
    chain = prompt | test_llm | parser

    input_data = {
        "file_path": file_path,
        "requirements_str": json.dumps(requirements, indent=2),
        "context_str": json.dumps(context, indent=2)
    }

    try:
        logger.info(f"Invoking Groq Test Generation LLM (Llama3-70b) for {file_path}...")
        code = chain.invoke(input_data)
        code = code.strip()
        if not code or not ("import " in code or "def test_" in code):
             logger.warning(f"LLM response for test generation doesn't look like Python code: {code[:200]}...")
             return code
        logger.info(f"Groq LLM Test Generation successful for {file_path}. Raw response received (length: {len(code)}).")
        return code
    except Exception as e:
        logger.error(f"Error during Groq LLM test generation for {file_path}: {e}", exc_info=True)
        return None


def generate_code_llm(requirements: Dict, target_file_path: str, tests_code: Optional[str], context: Dict) -> Optional[str]:
    """
    Generates implementation code using Groq LLM.
    """
    logger.info(f"--- Calling Groq LLM to Generate Code for: {target_file_path} ---")
    if not code_llm:
        logger.error("Code LLM (Groq) not initialized.")
        return None

    test_guidance = f"Use the following pytest tests as a guide for the implementation (ensure the generated code passes these tests):\n```python\n{tests_code}\n```" if tests_code else ""

    prompt_template_text = """
    You are an expert Python FastAPI developer using Llama 3, adhering to best practices.
    Generate the complete implementation code for the Python module file `{target_file_path}` based on the following requirements.

    Relevant Requirements:
    ```json
    {requirements_str}
    ```

    Persistent Context (Existing elements like models, function names you might need to reference):
    ```json
    {context_str}
    ```

    {test_guidance_str}

    **CRITICAL INSTRUCTIONS:**
    1.  **Provide ONLY valid, complete, and syntactically correct Python code for the entire file content.**
    2.  **Do NOT include any explanatory text, comments preceding the code, or markdown formatting (like ```python).**
    3.  **Start the response directly with the necessary Python import statements.**
    4.  Include necessary imports (e.g., `SQLAlchemy`, `FastAPI`, `Depends`, Models, Schemas).
    5.  Implement type hints for function arguments and return values.
    6.  Add clear docstrings explaining classes and functions.
    7.  Implement proper error handling (e.g., raising `HTTPException` with appropriate status codes for API routes).
    8.  Use FastAPI's Dependency Injection where appropriate (e.g., for database sessions (`db: Session = Depends(get_db)`), services in routes).
    9.  Ensure adherence to the provided requirements and consistency with the context (e.g., use correct model names from context if provided).
    10. If generating routes, include the `router = APIRouter()` instance.

    Generate the Python code now:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_text)
    parser = StrOutputParser()
    chain = prompt | code_llm | parser

    input_data = {
        "target_file_path": target_file_path,
        "requirements_str": json.dumps(requirements, indent=2),
        "context_str": json.dumps(context, indent=2),
        "test_guidance_str": test_guidance
    }

    try:
        logger.info(f"Invoking Groq Code Generation LLM (Llama3-70b) for {target_file_path}...")
        code = chain.invoke(input_data)
        code = code.strip()
        if not code or not ("import " in code or "def " in code or "class " in code):
             logger.warning(f"LLM response for code generation doesn't look like Python code: {code[:200]}...")
             return code
        logger.info(f"Groq LLM Code Generation successful for {target_file_path}. Raw response received (length: {len(code)}).")
        return code
    except Exception as e:
        logger.error(f"Error during Groq LLM code generation for {target_file_path}: {e}", exc_info=True)
        return None


def debug_code_llm(error_message: str, code_snippet: str, file_context: str, requirements: Dict, context: Dict) -> Optional[str]:
    """
    Attempts to debug code based on an error message using Groq LLM.
    """
    logger.info("--- Calling Groq LLM to Debug Code ---")
    if not debugger_llm:
        logger.error("Debugger LLM (Groq) not initialized.")
        return None

    prompt_template_text = """
    You are an expert Python code debugger using Llama 3. Analyze the following error message and the associated code.
    Identify the cause of the error and provide the corrected version of the relevant code section (e.g., the specific function or class definition where the error occurred).

    Error Message:
    ```
    {error_message}
    ```

    Potentially Faulty Code Snippet (focus your correction here if possible):
    ```python
    {code_snippet}
    ```

    Full File Content (for context):
    ```python
    {file_context}
    ```

    Relevant Requirements (if applicable to the bug):
    ```json
    {requirements_str}
    ```

    Persistent Context (if applicable):
    ```json
    {context_str}
    ```

    **CRITICAL INSTRUCTIONS:**
    1. Analyze the error and the code provided.
    2. Determine the necessary fix within the code snippet or its surrounding function/class.
    3. **Provide ONLY the corrected, complete Python code section (e.g., the entire fixed function `def ...:` or `class ...:`).**
    4. **Do NOT include explanations, apologies, or any text other than the Python code itself.**
    5. Ensure the corrected code is syntactically valid Python.
    6. If you cannot determine a fix or the error is unclear, respond with only the text: `# No fix identified.`

    Corrected Code Section:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_text)
    parser = StrOutputParser()
    chain = prompt | debugger_llm | parser

    input_data = {
        "error_message": error_message,
        "code_snippet": code_snippet,
        "file_context": file_context,
        "requirements_str": json.dumps(requirements, indent=2),
        "context_str": json.dumps(context, indent=2)
    }

    try:
        logger.info("Invoking Groq Debugger LLM (Llama3-8b)...")
        corrected_code = chain.invoke(input_data)
        corrected_code = corrected_code.strip()
        if corrected_code.startswith("```python"):
            corrected_code = corrected_code[len("```python"):].strip()
        if corrected_code.endswith("```"):
            corrected_code = corrected_code[:-len("```")].strip()
        if not corrected_code or corrected_code == "# No fix identified.":
             logger.warning("Debugger LLM did not identify a fix or returned an empty response.")
             return None
        logger.info("Groq LLM Debugging successful. Suggested fix received.")
        return corrected_code
    except Exception as e:
        logger.error(f"Error during Groq LLM debugging: {e}", exc_info=True)
        return None


def generate_documentation_llm(doc_type: str, context: Dict) -> Optional[str]:
    """
    Generates documentation (e.g., README, API docs) using Groq LLM.
    """
    logger.info(f"--- Calling Groq LLM to Generate Documentation: {doc_type} ---")
    if not doc_llm:
        logger.error("Documentation LLM (Groq) not initialized.")
        return None

    if doc_type == 'readme':
        prompt_template_text = """
        You are a technical writer using Llama 3. Generate a comprehensive README.md file for a FastAPI project with the following characteristics:

        Project Requirements Summary:
        ```json
        {requirements_summary_str}
        ```

        Project Structure Summary:
        ```
        {project_structure_summary_str}
        ```

        Setup Instructions:
        - Requires Python 3.9+
        - Install dependencies: `pip install -r requirements.txt`
        - Database: PostgreSQL (details in `.env`)
        - Run migrations: `alembic upgrade head`
        - Set up `.env` file based on `.env.example`.
        - Run the server: `uvicorn app.main:app --reload`

        API Endpoint Overview:
        ```json
        {endpoints_summary_str}
        ```

        **INSTRUCTIONS:**
        1. Generate the content for the README.md file in Markdown format ONLY.
        2. Do NOT include ```markdown fences around the output.
        3. Include sections for: Introduction, Features, Project Structure, Setup, Running the Application, API Endpoints Overview.
        4. Use appropriate Markdown formatting (headers, lists, code blocks).
        """
        input_data = {
            "requirements_summary_str": json.dumps(context.get('requirements_summary', 'N/A'), indent=2),
            "project_structure_summary_str": context.get('project_structure_summary', 'Standard FastAPI structure'),
            "endpoints_summary_str": json.dumps(context.get('endpoints_summary', []), indent=2)
        }

    elif doc_type == 'api_markdown':
        prompt_template_text = """
        You are a technical writer using Llama 3. Generate API documentation in Markdown format based on the following extracted endpoint details.
        For each endpoint, describe the Path, Method, Description, Request Body (if any, show structure or example using JSON code blocks), Response Format (show structure or example using JSON code blocks), Authentication requirement, and required Roles.

        Endpoint Details:
        ```json
        {endpoints_str}
        ```

        **INSTRUCTIONS:**
        1. Generate the content for an API documentation file (e.g., API_DOCS.md) in Markdown format ONLY.
        2. Do NOT include ```markdown fences around the output.
        3. Use clear headings (e.g., `### POST /api/lms/leaves/apply`), bullet points, and JSON code blocks for examples.
        4. Be accurate based on the provided Endpoint Details.
        """
        input_data = {
            "endpoints_str": json.dumps(context.get('requirements', {}).get('endpoints', []), indent=2)
        }
    else:
        logger.warning(f"Unknown documentation type requested: {doc_type}")
        return None

    prompt = ChatPromptTemplate.from_template(prompt_template_text)
    parser = StrOutputParser()
    chain = prompt | doc_llm | parser

    try:
        logger.info(f"Invoking Groq Documentation LLM (Llama3-70b) for {doc_type}...")
        doc_content = chain.invoke(input_data)
        logger.info(f"Groq LLM Documentation generation successful for {doc_type}.")
        doc_content = doc_content.strip()
        if doc_content.startswith("```markdown"):
             doc_content = doc_content[len("```markdown"):].strip()
        elif doc_content.startswith("```"):
             doc_content = doc_content[len("```"):].strip()
        if doc_content.endswith("```"):
             doc_content = doc_content[:-len("```")].strip()
        return doc_content
    except Exception as e:
        logger.error(f"Error during Groq LLM documentation generation for {doc_type}: {e}", exc_info=True)
        return None
