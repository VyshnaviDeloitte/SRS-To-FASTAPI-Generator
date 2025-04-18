import os       # For checking paths, deleting files
import shutil   # For deleting directories
import zipfile  # For potentially inspecting zip files (optional advanced check)
import logging  # To potentially configure logging level for tests if needed
from typing import Optional, Dict, Any

# Import the functions you want to test from your tools.py file
import tools

# --- Optional: Configure logging for more detailed test output ---
# logging.basicConfig(level=logging.DEBUG) # Show DEBUG level messages from tools
# logger = logging.getLogger(__name__)
# logger.info("Starting tool tests...")

print("--- Starting Tool Tests ---") # Simple print statement is also fine

# --- Define paths for test artifacts ---
TEST_DIR = "temp_test_artifacts"
TEST_FILE = os.path.join(TEST_DIR, "my_test_file.txt")
TEST_SUBDIR = os.path.join(TEST_DIR, "subdir")
TEST_NESTED_FILE = os.path.join(TEST_SUBDIR, "nested.txt")
TEST_ZIP_SOURCE_DIR = os.path.join(TEST_DIR, "zip_source")
TEST_ZIP_FILE = os.path.join(TEST_DIR, "test_archive.zip")
# You'll need to create this sample docx file manually first
SAMPLE_DOCX = "sample_test_doc.docx"

def test_directory_creation():
    print("\n--- Testing Directory Creation ---")
    # Test basic creation
    print(f"Attempting to create directory: {TEST_DIR}")
    success = tools.create_directory(TEST_DIR)
    assert success is True, "create_directory failed on first attempt"
    assert os.path.exists(TEST_DIR), f"Directory {TEST_DIR} was not actually created."
    print(f"Directory {TEST_DIR} created successfully.")

    # Test creating it again (should succeed due to exist_ok=True)
    print(f"Attempting to create directory again: {TEST_DIR}")
    success_again = tools.create_directory(TEST_DIR)
    assert success_again is True, "create_directory failed on second attempt (exist_ok)"
    print("Creating existing directory succeeded as expected.")

    # Test nested creation
    print(f"Attempting to create nested directory: {TEST_SUBDIR}")
    success_nested = tools.create_directory(TEST_SUBDIR)
    assert success_nested is True, "create_directory failed for nested path"
    assert os.path.exists(TEST_SUBDIR), f"Nested directory {TEST_SUBDIR} was not created."
    print(f"Nested directory {TEST_SUBDIR} created successfully.")
    print("--- Directory Creation Test PASSED ---")

def test_file_write_read():
    print("\n--- Testing File Write/Read ---")
    test_content = "Hello from the test script!\nLine 2."

    # Test writing
    print(f"Attempting to write file: {TEST_FILE}")
    success_write = tools.write_file(TEST_FILE, test_content)
    assert success_write is True, f"write_file failed for {TEST_FILE}"
    assert os.path.exists(TEST_FILE), f"File {TEST_FILE} was not created."
    print("File written successfully.")

    # Test reading
    print(f"Attempting to read file: {TEST_FILE}")
    read_content = tools.read_file(TEST_FILE)
    assert read_content is not None, f"read_file returned None for existing file {TEST_FILE}"
    assert read_content == test_content, "Read content does not match written content."
    print("File read successfully and content matches.")

    # Test reading non-existent file
    non_existent_file = os.path.join(TEST_DIR, "does_not_exist.txt")
    print(f"Attempting to read non-existent file: {non_existent_file}")
    read_none = tools.read_file(non_existent_file)
    assert read_none is None, "read_file did not return None for non-existent file."
    print("Reading non-existent file handled correctly.")

    # Test writing to a nested path (directory should be created)
    print(f"Attempting to write nested file: {TEST_NESTED_FILE}")
    success_nested_write = tools.write_file(TEST_NESTED_FILE, "Nested content.")
    assert success_nested_write is True, f"write_file failed for nested path {TEST_NESTED_FILE}"
    assert os.path.exists(TEST_NESTED_FILE), f"Nested file {TEST_NESTED_FILE} was not created."
    print("Nested file written successfully.")
    print("--- File Write/Read Test PASSED ---")

def test_shell_commands():
    print("\n--- Testing Shell Commands ---")

    # Test simple successful command (echo)
    print("Testing successful command: echo 'Shell test successful'")
    result_echo = tools.run_shell_command("echo 'Shell test successful'")
    print(f"Result: {result_echo}")
    assert result_echo["success"] is True, "Simple echo command failed"
    assert result_echo["return_code"] == 0, "Echo command return code not 0"
    assert "Shell test successful" in result_echo["stdout"], "Echo output missing expected text"
    assert not result_echo["stderr"], "Echo command produced unexpected stderr" # Check stderr is empty
    print("Successful command test PASSED.")

    # Test command with specific CWD (list files in the test dir)
    # Use 'dir' for Windows, 'ls -la' for Linux/macOS - adjust as needed
    list_cmd = "dir" if os.name == 'nt' else "ls -la"
    print(f"Testing command with cwd={TEST_DIR}: {list_cmd}")
    result_ls = tools.run_shell_command(list_cmd, cwd=TEST_DIR)
    print(f"Result (stdout subset): {result_ls.get('stdout', '')[:100]}...") # Print subset
    assert result_ls["success"] is True, f"'{list_cmd}' command failed in {TEST_DIR}"
    assert os.path.basename(TEST_FILE) in result_ls["stdout"], f"'{list_cmd}' output missing {TEST_FILE}" # Check if created file is listed
    print("Command with CWD test PASSED.")

    # Test failing command
    fail_cmd = "non_existent_command_abc123"
    print(f"Testing failing command: {fail_cmd}")
    result_fail = tools.run_shell_command(fail_cmd)
    print(f"Result: {result_fail}")
    assert result_fail["success"] is False, "Failing command reported success"
    assert result_fail["return_code"] != 0, "Failing command return code was 0"
    assert result_fail["stderr"], "Failing command did not produce stderr" # Expecting error message
    print("Failing command test PASSED.")

    # Test command timeout (optional - uncomment to test)
    # print("Testing command timeout (will take ~5 seconds)...")
    # timeout_cmd = "sleep 5" if os.name != 'nt' else "timeout /t 5 /nobreak > NUL" # Sleep command varies by OS
    # result_timeout = tools.run_shell_command(timeout_cmd, timeout=2) # Set timeout shorter than command duration
    # print(f"Result: {result_timeout}")
    # assert result_timeout["success"] is False, "Timeout command reported success"
    # assert "TimeoutExpired" in result_timeout["stderr"], "Timeout command stderr missing 'TimeoutExpired'"
    # print("Command timeout test PASSED.")

    print("--- Shell Commands Test PASSED ---")

def test_zip_creation():
    print("\n--- Testing Zip Creation ---")
    # 1. Create source directory and dummy files
    print(f"Creating source directory for zipping: {TEST_ZIP_SOURCE_DIR}")
    tools.create_directory(TEST_ZIP_SOURCE_DIR)
    tools.write_file(os.path.join(TEST_ZIP_SOURCE_DIR, "file1.txt"), "Zip content 1")
    tools.create_directory(os.path.join(TEST_ZIP_SOURCE_DIR, "subfolder"))
    tools.write_file(os.path.join(TEST_ZIP_SOURCE_DIR, "subfolder", "file2.txt"), "Zip content 2")
    print("Source directory and files created.")

    # 2. Create the zip archive
    print(f"Attempting to create zip file: {TEST_ZIP_FILE}")
    success_zip = tools.create_zip_archive(TEST_ZIP_SOURCE_DIR, TEST_ZIP_FILE)
    assert success_zip is True, f"create_zip_archive failed for {TEST_ZIP_SOURCE_DIR}"
    assert os.path.exists(TEST_ZIP_FILE), f"Zip file {TEST_ZIP_FILE} was not created."
    print("Zip file created successfully.")

    # 3. (Optional Advanced Check) Verify zip content
    try:
        with zipfile.ZipFile(TEST_ZIP_FILE, 'r') as zf:
            namelist = zf.namelist()
            print(f"Files found in zip: {namelist}")
            # Check if expected files/folders are present (adjust paths based on how create_zip works)
            # Assumes create_zip includes the source dir name itself in the archive paths
            source_dir_name = os.path.basename(TEST_ZIP_SOURCE_DIR)
            assert f"{source_dir_name}/file1.txt" in namelist
            assert f"{source_dir_name}/subfolder/file2.txt" in namelist
            print("Zip content verified.")
    except Exception as e:
        print(f"Could not verify zip content: {e}")
        # Decide if this should fail the test

    print("--- Zip Creation Test PASSED ---")

def test_docx_parsing():
    print("\n--- Testing Docx Parsing ---")
    # 1. Check if sample docx exists
    if not os.path.exists(SAMPLE_DOCX):
        print(f"SKIPPING Docx test: Please create a sample file named '{SAMPLE_DOCX}' with some text.")
        return # Skip the test if the file isn't there

    # 2. Test parsing the sample docx
    print(f"Attempting to parse docx file: {SAMPLE_DOCX}")
    parsed_text = tools.parse_docx(SAMPLE_DOCX)
    assert parsed_text is not None, f"parse_docx returned None for existing file {SAMPLE_DOCX}"
    # Add a check for expected content based on your sample file
    # Example: assert "some specific text from your docx" in parsed_text.lower()
    print(f"Docx parsed successfully. Content subset: '{parsed_text[:100]}...'")
    print(f"Length of parsed text: {len(parsed_text)}")

    # 3. Test parsing non-existent file
    non_existent_docx = "no_such_document.docx"
    print(f"Attempting to parse non-existent docx: {non_existent_docx}")
    parsed_none = tools.parse_docx(non_existent_docx)
    assert parsed_none is None, "parse_docx did not return None for non-existent file."
    print("Parsing non-existent docx handled correctly.")

    print("--- Docx Parsing Test PASSED ---")

def cleanup_test_artifacts():
    """Deletes files and directories created during tests."""
    print("\n--- Cleaning up test artifacts ---")
    if os.path.exists(TEST_DIR):
        try:
            shutil.rmtree(TEST_DIR) # Recursively delete the main test directory
            print(f"Removed test directory: {TEST_DIR}")
        except OSError as e:
            print(f"Error removing directory {TEST_DIR}: {e}")
    else:
        print(f"Test directory {TEST_DIR} not found, no cleanup needed.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure clean slate before starting
    cleanup_test_artifacts()

    # Run tests within a try...finally block to ensure cleanup happens
    try:
        test_directory_creation()
        test_file_write_read()
        test_shell_commands()
        test_zip_creation()
        test_docx_parsing()

        print("\n--- ALL TOOL TESTS COMPLETED (See output for PASS/FAIL details) ---")

    except AssertionError as e:
        print(f"\n--- TEST FAILED: {e} ---")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR DURING TESTS: {e} ---")
        # You might want to log the full traceback here in a real test suite
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup after tests run, even if errors occurred
        cleanup_test_artifacts()
