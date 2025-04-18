# tools.py
import os
import subprocess
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- File System Tools ---

# <<< FUNCTION ADDED >>>
def create_directory(path: str) -> bool:
    """Creates a directory if it doesn't exist, including parent directories."""
    try:
        # os.makedirs creates parent directories as needed (like mkdir -p)
        # exist_ok=True means it won't raise an error if the directory already exists
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created or already exists: {path}")
        return True
    except OSError as e:
        # Log specific OS errors like permission denied
        logger.error(f"Error creating directory {path}: {e}")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred creating directory {path}: {e}", exc_info=True)
        return False

# <<< RENAMED from _file to write_file >>>
def write_file(file_path: str, content: str) -> bool:
    """Writes content to a file, creating parent directories if needed."""
    try:
        # Ensure parent directory exists using the newly added function
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            # <<< FIXED: Call the existing create_directory function >>>
            if not create_directory(parent_dir):
                # If creating the parent directory failed, we can't write the file
                logger.error(f"Could not create parent directory for {file_path}. Aborting write.")
                return False

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote file: {file_path}")
        return True
    except IOError as e: # Catch file-related errors
        logger.error(f"Error writing file {file_path}: {e}")
        return False
    except Exception as e: # Catch other unexpected errors
        logger.error(f"An unexpected error occurred writing file {file_path}: {e}", exc_info=True)
        return False

def read_file(file_path: str) -> Optional[str]:
    """Reads content from a file."""
    # Check if file exists first for a clearer log message
    if not os.path.exists(file_path):
        logger.warning(f"File not found when attempting to read: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except FileNotFoundError:
         # This case should ideally be caught by the check above, but good to keep
         logger.error(f"File not found during read operation (unexpected): {file_path}")
         return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return None


# --- Shell Execution Tool ---
# <<< REMOVED redundant 'writetry' function >>>

def run_shell_command(command: str, cwd: str = None, timeout: int = 300) -> Dict[str, Any]:
    """Runs a shell command and captures its output."""
    logger.info(f"Running command: '{command}' in '{cwd or os.getcwd()}'")
    try:
        process = subprocess.run(
            command,
            shell=True,        # Use shell=True cautiously
            check=False,       # Don't raise exception on non-zero exit code immediately
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,         # Capture output as text
            cwd=cwd,           # Set working directory if provided
            timeout=timeout    # Use the timeout parameter
        )
        output = {
            "command": command,
            "return_code": process.returncode,
            "stdout": process.stdout.strip() if process.stdout else "", # Handle None stdout/stderr
            "stderr": process.stderr.strip() if process.stderr else "",
            "success": process.returncode == 0
        }
        if output["success"]:
            logger.info(f"Command successful: {command}")
            if output["stdout"]: logger.debug(f"STDOUT:\n{output['stdout']}")
        else:
            logger.warning(f"Command failed with code {output['return_code']}: {command}")
            # Log stderr first if it exists, otherwise stdout
            if output["stderr"]: logger.warning(f"STDERR:\n{output['stderr']}")
            elif output["stdout"]: logger.info(f"STDOUT (on failure):\n{output['stdout']}")

        return output
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command}' timed out after {timeout} seconds.")
        return {"command": command, "return_code": -1, "stdout": "", "stderr": f"TimeoutExpired after {timeout} seconds", "success": False}
    except FileNotFoundError: # Usually requires shell=False, but can happen in complex shell=True cases
         logger.error(f"Command not found (or path issue with shell=True): {command.split()[0]}")
         return {"command": command, "return_code": -1, "stdout": "", "stderr": "FileNotFoundError (or bad command path)", "success": False}
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return {
             "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

# --- Archiving Tool ---

def create_zip_archive(source_dir: str, output_zip_path: str) -> bool:
    """Creates a zip archive of the source directory."""
    logger.info(f"Creating zip archive for '{source_dir}' at '{output_zip_path}'")
    source_path = Path(source_dir)
    output_path = Path(output_zip_path)

    # Ensure source directory exists before trying to zip it
    if not source_path.is_dir():
        logger.error(f"Source directory for zipping not found or is not a directory: {source_dir}")
        return False

    # Ensure parent directory for the zip file exists
    if output_path.parent:
        # <<< FIXED: Call the existing create_directory function >>>
        if not create_directory(str(output_path.parent)):
            logger.error(f"Could not create parent directory for zip file {output_zip_path}. Aborting zip.")
            return False

    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Use Path.rglob to recursively find all files within source_dir
            for item_path in source_path.rglob('*'):
                # Calculate path relative to the source directory itself for storing in zip
                relative_path = item_path.relative_to(source_path)
                # Pass the absolute path to write and the relative path for the arcname
                zipf.write(item_path, arcname=relative_path)
        logger.info(f"Successfully created zip archive: {output_zip_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating zip archive for {source_dir}: {e}", exc_info=True)
        return False

# --- Document Parsing Tool ---
def parse_docx(file_path: str) -> Optional[str]:
    """Parses text content from a .docx file."""
    # Check if file exists first
    if not os.path.exists(file_path):
        logger.warning(f"File not found when attempting to parse docx: {file_path}")
        return None
    try:
        # Import locally: ensures error occurs here if docx not installed
        import docx
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs if para.text] # Filter empty paragraphs
        logger.info(f"Successfully parsed docx file: {file_path}")
        return '\n'.join(full_text)
    except ImportError:
        logger.error("ImportError: 'python-docx' library not installed. Cannot parse .docx file. Run: pip install python-docx")
        return None # Return None if library is missing
    except Exception as e:
        # Catch potential errors from the docx library itself (e.g., corrupted file)
        logger.error(f"Error parsing docx file {file_path}: {e}", exc_info=True)
        return None
