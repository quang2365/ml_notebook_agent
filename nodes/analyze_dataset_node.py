from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from state import State
from model.model import llm
def analyze_dataset_note(state:State) -> dict:
    # full dataset analysis
    dataset_summary = state.get("summary")
    if not dataset_summary:
        return {"error":"I did not receive the summary of the Dataset",
                "messages": [AIMessage(content = "I did not receive the summary of the Dataset")]}


    analysis_context = (
        dataset_summary.get("analysis_context")
        or dataset_summary
    )
    system_prompt = """
    You are an expert in summarizing and analyzing data.
    You will be provided with data related to the dataset; summarize that content for the user.
    Present only the information provided; do not fabricate or infer.
    
    General conventions:
    When presenting monetary amounts, clearly distinguish the thousands separator and
    the decimal separator. You may write amounts in thousands of USD to avoid confusion.
    
        Please analyze:

        1. Dataset size and structure.
        2. Numeric columns and categorical columns.
        3. Missing values and duplicates.
        4. Skewed distributions and outliers.
        5. Notable correlations.
        6. Columns that are likely IDs or not useful.
        7. Propose a target and the type of Machine Learning problem.
        8. Next steps that need to be taken.

        Rules:
        - Only use the provided data.
        - Do not assert a target unless the user has confirmed it.
        - Outliers according to IQR are only statistical indicators; do not assume they are incorrect data.
        - Correlation does not imply causation.
        - Present using Markdown in Vietnamese.
    """
    message=f"""You are a Data Scientist. This is the dataset information:
                    {analysis_context}"""

    try:
        result = llm.invoke([SystemMessage(content= system_prompt),HumanMessage(message)])
        return {
            "messages": [result],
            "summary_llm": result.content,
            "error": None,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"An error occurred while I was trying to read the dataset: {str(exc)}")]
        }
