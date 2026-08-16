from langgraph.graph import END,START, StateGraph
from langgraph.checkpoint.memory  import InMemorySaver

from nodes.inspect_dataset_node import inspect_dataset_note
from nodes.analyze_dataset_node import analyze_dataset_note
from nodes.propose_problem_node import propose_problem_node
from nodes.review_problem_node import review_problem_node
from nodes.analyze_target_node import analyze_target_node
from nodes.plan_notebook_node import plan_notebook_node
from nodes.validate_plan_node import validate_plan_node
from nodes.fix_plan_node import fix_plan_node
from nodes.generate_section_node import generate_section_node
from nodes.prepare_generation_node import prepare_generation_node
from nodes.validate_cells_node import validate_cells_node
from nodes.fix_cells_node import fix_cells_node
from nodes.notebook_builder_node import notebook_builder
from route.route_after_validation_cell import route_after_validation_cell
from route.route_after_section_generation import (
    route_after_section_generation,
)
from route.route_after_review_proplem import route_after_review_proplem
from route.route_after_plan_validation import route_after_plan_validation
from state import State

def build_graph():
    builder = StateGraph(State)

    builder.add_node("inspect_data",inspect_dataset_note)
    builder.add_node("analyze_data",analyze_dataset_note)
    builder.add_node("propose_problem",propose_problem_node)
    builder.add_node("review_problem",review_problem_node)
    builder.add_node("analyze_target",analyze_target_node)
    builder.add_node("plan_notebook",plan_notebook_node)
    builder.add_node("validate_plan_node",validate_plan_node)
    builder.add_node("fix_plan_node",fix_plan_node)
    builder.add_node("prepare_generation",prepare_generation_node)
    builder.add_node("generate_section",generate_section_node)
    builder.add_node("validate_cells",validate_cells_node)
    builder.add_node("fix_cells",fix_cells_node)
    builder.add_node("notebook_builder",notebook_builder)
    builder.add_edge(START,"inspect_data")
    builder.add_edge("inspect_data","analyze_data")
    builder.add_edge("analyze_data","propose_problem")
    builder.add_edge("propose_problem","review_problem")
    builder.add_conditional_edges("review_problem",route_after_review_proplem,{"approved":"analyze_target","rejected":END})
    builder.add_edge("analyze_target","plan_notebook")
    builder.add_edge("plan_notebook","validate_plan_node")
    builder.add_conditional_edges("validate_plan_node",route_after_plan_validation,{"valid":"prepare_generation","fix":"fix_plan_node","failed":END})
    builder.add_edge("fix_plan_node","validate_plan_node")
    builder.add_edge("prepare_generation","generate_section")
    builder.add_conditional_edges("generate_section",route_after_section_generation,{"continue":"generate_section","complete":"validate_cells","failed":END})
    builder.add_conditional_edges("validate_cells",route_after_validation_cell,{"valid":"notebook_builder","fix":"fix_cells","failed":END})
    builder.add_edge("fix_cells","validate_cells",)
    builder.add_edge("notebook_builder",END)
    graph = builder.compile(checkpointer=InMemorySaver())

    return graph
