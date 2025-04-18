# graph_builder.py
import os
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import state definition and node functions
from project_state import ProjectState
from node_functions import (
    load_srs,
    analyze_srs,
    setup_project_structure,
    generate_unit_tests,
    generate_implementation_code,
    run_tests_and_validate,
    debug_code,
    generate_documentation,
    generate_deployment_package, # Optional node
    finalize_workflow
)

# Load environment variables (for API keys, LangSmith)
load_dotenv()

# --- Graph Definition ---

# Conditional logic for the debugging loop
def should_debug(state: ProjectState) -> str:
    """Determines if the graph should enter the debug loop or proceed."""
    print("--- Condition: should_debug ---")
    errors = state.get('validation_errors')
    debug_iter = state.get('debug_iterations', 0)
    max_iters = state.get('max_debug_iterations', 3) # Default max 3 iterations

    if errors and debug_iter < max_iters:
        print(f"Validation failed (Iteration {debug_iter + 1}/{max_iters}). Routing to debug.")
        return "debug" # Go to debug node
    elif errors:
        print(f"Validation failed, but max debug iterations ({max_iters}) reached. Proceeding with errors.")
        # Decide path when debugging fails permanently
        return "package_or_document" # Proceed despite errors
    else:
        print("Validation passed. Proceeding...")
        return "package_or_document" # Proceed normally

# Conditional logic for packaging vs documentation first
def package_or_document_route(state: ProjectState) -> str:
    """ Routes to packaging or documentation based on preference/state. """
    # Example: Always package if possible, then document
    print("--- Condition: package_or_document_route ---")
    if state.get("project_root"): # Check if project exists to be packaged
        print("Routing to package.")
        return "package"
    else:
        print("Project root missing, skipping package, routing to document.")
        return "document" # Fallback if packaging not possible


# Create the state graph
workflow = StateGraph(ProjectState)

# Add nodes to the graph
workflow.add_node("load_srs", load_srs)
workflow.add_node("analyze_srs", analyze_srs)
workflow.add_node("setup_project", setup_project_structure)
workflow.add_node("generate_tests", generate_unit_tests)
workflow.add_node("generate_code", generate_implementation_code)
workflow.add_node("validate", run_tests_and_validate)
workflow.add_node("debug", debug_code)
workflow.add_node("package", generate_deployment_package) # Milestone 5 node
workflow.add_node("document", generate_documentation)     # Milestone 6 node
workflow.add_node("finalize", finalize_workflow)          # Final reporting node

# --- Define Edges ---

# Linear flow until validation
workflow.set_entry_point("load_srs")
workflow.add_edge("load_srs", "analyze_srs")
workflow.add_edge("analyze_srs", "setup_project")
workflow.add_edge("setup_project", "generate_tests")
workflow.add_edge("generate_tests", "generate_code")
workflow.add_edge("generate_code", "validate")

# Conditional debugging loop
workflow.add_conditional_edges(
    "validate",
    should_debug,
    {
        "debug": "debug", # If errors and iterations left, go to debug
        "package_or_document": "package" # Proceed if validation ok or max iterations reached (default to package first)
        # We route to "package" here, and "package" will route to "document"
    }
)

# Loop back from debug to regenerate/revalidate
# Regenerating code after debug might be safer than just re-validating
workflow.add_edge("debug", "generate_code")

# Flow after validation/debugging is complete
workflow.add_edge("package", "document") # After packaging, generate docs
workflow.add_edge("document", "finalize") # After docs, finalize
workflow.add_edge("finalize", END) # End the graph

# Compile the graph
app_graph = workflow.compile()

# Optional: Generate a visualization (if pygraphviz installed)
try:
    output_png = "graph_visualization.png"
    app_graph.get_graph().draw_png(output_png)
    print(f"Graph visualization saved to {output_png}")
except Exception as e:
    print(f"Could not generate graph visualization (pygraphviz likely not installed or error): {e}")