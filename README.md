# ML Notebook Agent

**ML Notebook Agent** is a project that builds an **AI Agent capable of automatically analyzing datasets and creating Jupyter Notebooks for Machine Learning problems**.

The system is developed according to a workflow architecture using **LangGraph**, in which each stage of the Machine Learning process is separated into independent nodes such as data inspection, dataset analysis, problem proposal, target confirmation, notebook planning, code generation, error checking, and automatic error correction.

The goal of the project is to build an agent that can receive an input dataset and gradually create a Machine Learning notebook with a clear, consistent structure and the ability to be verified before execution.

---

## Project Goal

ML Notebook Agent aims to automate the process:

```text
Dataset
   ↓
Analyze data
   ↓
Propose Machine Learning problem
   ↓
User confirms Target / Problem Type
   ↓
Analyze Target
   ↓
Plan Notebook
   ↓
Generate each section
   ↓
Merge Notebook Cells
   ↓
Static Validation
   ↓
Dependency Validation
   ↓
Automatic Repair
   ↓
Jupyter Notebook
```

Instead of asking the LLM to generate the entire notebook in a single call, the system splits the notebook into multiple **small sections** and generates each section separately. This design reduces the risk of timeout, reduces the size of each request, and makes errors easier to control.

---

## Overall Architecture

Current main workflow:

```text
START
  ↓
inspect_data
  ↓
analyze_data
  ↓
propose_problem
  ↓
review_problem
  ├── rejected ─────────────→ END
  │
  └── approved
        ↓
  analyze_target
        ↓
  plan_notebook
        ↓
  generate_cells
        ↓
  route_after_generation
        ├── failed ─────────→ END
        │
        └── success
              ↓
        validate_cells
              ↓
        route_after_validation
          ├── valid ────────→ END
          │
          ├── invalid
          │      ↓
          │   fix_cells
          │      ↓
          └──── validate_cells
```

The workflow is managed by **LangGraph StateGraph** and uses a **checkpointer** to support Human-in-the-Loop.

---

## Main Components

### 1. Dataset Inspection

The `inspect_data` node is responsible for reading the dataset and collecting basic information such as:

- the number of rows and columns;
- column names;
- data types;
- numeric columns;
- categorical columns;
- missing values;
- possible ID columns;
- necessary overview information for the downstream nodes.

### 2. Dataset Analysis with LLM

The `analyze_data` node uses an LLM to analyze the dataset information that has been collected. This result is used as context for determining the appropriate Machine Learning problem.

### 3. Problem Proposal

The `propose_problem` node automatically proposes:

- `target_column`;
- `problem_type`;
- the reason for the proposal;
- the number of distinct values of the target;
- the suitability level of the candidate.

Example:

```json
{
  "target_column": "median_house_value",
  "problem_type": "regression",
  "candidate_score": 3,
  "unique_count": 3842,
  "reasons": [
    "Column name containing keyword: value"
  ]
}
```

### 4. Human-in-the-Loop

The system does not automatically accept the target proposed by the AI.

The `review_problem` node uses LangGraph's `interrupt()` to request user confirmation. The user can approve, edit, or reject the proposal.

The graph is resumed by:

```python
Command(resume=decision)
```

using the same `thread_id`.

---

## Notebook Planner

After the target is confirmed, the system creates a `NotebookPlan`.

The notebook is currently limited to about **8–10 sections** to avoid generating an overly long notebook and to reduce the number of requests to the LLM.

A typical notebook structure:

```text
section_1   Setup
section_2   Data Loading
section_3   Exploratory Data Analysis
section_4   Train/Test Split
section_5   Preprocessing
section_6   Baseline Model
section_7   Candidate Model 1
section_8   Candidate Model 2
section_9   Model Evaluation
section_10  Conclusion
```

The planner must ensure a proper Machine Learning order, especially:

```text
Train/Test Split
        ↓
Fit Preprocessing on Train
        ↓
Model Training
        ↓
Evaluation
```

to limit the risk of **data leakage**.

---

## Per-Section Notebook Generation

One of the main points of the project is not to generate the entire notebook with a single request.

Instead:

```text
NotebookPlan
    ↓
section_1 → LLM
section_2 → LLM
section_3 → LLM
...
section_N → LLM
    ↓
Merge
    ↓
notebook_cells
```

Each section is generated independently using structured output.

Example of a notebook cell:

```json
{
  "cell_id": "section_4_code_1",
  "section_id": "section_4",
  "cell_type": "code",
  "title": "Train/Test Split",
  "source": "X_train, X_test, y_train, y_test = ...",
  "purpose": "Split data into train and test",
  "expected_output": null
}
```

---

## Variable Contract

Because the sections are generated by multiple independent LLM calls, the project uses a **Variable Contract** to maintain continuity between sections.

Some standard variables:

```python
df

target_column

X
y

X_train
X_test
y_train
y_test

preprocessor

trained_models = {}
predictions = {}
model_results = []

RANDOM_STATE = 42
```

The agent is instructed not to rename the standard variables to other names such as:

```text
train_X
X_training
housing_df
X_train_data
```

This helps reduce dependency errors between sections.

---

## Structured Output

Important LLM outputs are defined using **Pydantic models**.

Example:

```python
class NotebookCell(BaseModel):
    cell_id: str
    section_id: str
    cell_type: Literal["markdown", "code"]
    title: str
    source: str
    purpose: str
    expected_output: str | None
```

Using structured output helps control format better than asking the LLM to return free-form JSON.

---

## Static Cell Validation

After merging all sections, the cells are passed to the validator.

The validator currently checks for errors such as:

- missing `cell_id`;
- duplicate `cell_id`;
- sai `cell_type`;
- source is not a string;
- Markdown code fence trong Python cell;
- Python syntax error;
- missing dataset path;
- missing confirmed target.

Example:

```python
compile(
    source,
    filename=cell_id,
    mode="exec",
)
```

is used to check Python syntax without needing to execute the notebook.

---

## Dependency Validation with AST

Correct syntax does not mean the notebook can run.

Example:

```python
model.fit(
    X_train_processed,
    y_train,
)
```

still compiles successfully even though `X_train_processed` has never been created.

Therefore, the project uses Python's `ast` to analyze code in notebook order.

The dependency validator tracks:

```text
defined variables
used variables
imports
functions
classes
```

and detects errors such as:

```json
{
  "cell_id": "section_6_code_2",
  "error_type": "undefined_variable",
  "variable": "X_train_processed",
  "message": "Variable `X_train_processed` is used before it is defined or imported."
}
```

---

## Automatic Cell Repair

When the validator detects an error:

```text
validate_cells
      ↓
invalid
      ↓
fix_cells
      ↓
validate_cells
```

`fix_cells` fixes each cell instead of regenerating the entire notebook.

The system currently supports repairing error groups such as:

```text
syntax_error
undefined_variable
```

The fixer can use:

- validation errors;
- variable contract;
- previous code context;
- available variable names;
- source of the current cell.

The goal is to keep changes as minimal as possible and not create dummy variables just to make the validator pass.

---

## Retry and Rate Limit Handling

Generating each section creates many small requests to the LLM.

The project has a dedicated retry for each section:

```text
section_1 ✅
section_2 ✅
section_3 ❌
          ↓
      retry section_3
          ↓
section_3 ✅
```

For the `429` rate-limit error, the system uses exponential backoff and has pauses between sections to reduce request frequency.

---

## LLM Provider

The current version uses the model through NVIDIA NIM with an OpenAI-compatible API.

Example configuration:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="minimaxai/minimax-m3",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)
```

The API key is stored in `.env`:

```env
NVIDIA_API_KEY=your_api_key_here
```

Do not commit `.env` to the repository.

---

## Technologies Used

Main technologies:

```text
Python
LangChain
LangGraph
Pydantic
Python AST
Jupyter Notebook
NVIDIA NIM
MiniMax M3
```

The Machine Learning libraries that the notebook agent can use depending on the dataset and notebook plan, for example:

```text
pandas
numpy
scikit-learn
matplotlib
xgboost
```

---

## Project Structure

A reference directory structure:

```text
ml_notebook_agent/
│
├── graph.py
├── state.py
│
├── model/
│   ├── capabilities.py
│   ├── model.py
│   └── structured_output.py
│
├── nodes/
│   ├── inspect_dataset_node.py
│   ├── analyze_dataset_node.py
│   ├── propose_problem_node.py
│   ├── review_problem_node.py
│   ├── analyze_target_node.py
│   ├── plan_notebook_node.py
│   ├── prepare_generation_node.py
│   ├── generate_section_node.py
│   ├── validate_cells_node.py
│   └── fix_cells_node.py
│
├── route/
│   ├── route_after_review.py
│   ├── route_after_generate.py
│   └── route_after_validation.py
│
├── schemas/
│   ├── notebook_plan_schema.py
│   ├── notebook_cell_schema.py
│   └── fixed_cell_schema.py
│
├── validators/
│   ├── __init__.py
│   └── dependency_validator.py
│
├── config/
│   ├── __init__.py
│   └── notebook_contract.py
│
├── tools/
├── data/
├── .env
├── .gitignore
└── README.md
```

The actual structure may change during development.

---

## LangGraph State

A part of the current state:

```python
class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    dataset_path: str | None

    summary: dict | None
    summary_llm: str | None

    problem_proposal: dict | None

    target_column: str | None
    problem_type: Literal[
        "regression",
        "classification",
    ] | None

    approval_status: Literal[
        "pending",
        "approved",
        "rejected",
    ] | None

    target_analysis: dict | None

    notebook_plan: dict | None
    notebook_cells: list[dict] | None

    section_generation_status: Literal[
        "pending",
        "success",
        "failed",
    ] | None

    validation_cell_status: Literal[
        "pending",
        "valid",
        "invalid",
    ] | None

    validation_errors: list[dict] | None

    fix_attempts: int
    fixed_cell_ids: list[str] | None
    fix_failures: list[dict] | None

    error: str | None
```

---

## Installation

Clone repository:

```bash
git clone <repository-url>
cd ml_notebook_agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project and its dependencies:

```bash
pip install -e .
```

Create the `.env` file:

```env
NVIDIA_API_KEY=your_api_key_here
```

---

## Running the Project

Configure the provider and API key, then start QIU:

```bash
qiu setup
qiu
```

The graph will run up to the Human Review step. After the user confirms the target and problem type, the graph is resumed and continues:

```text
analyze_target
→ plan_notebook
→ generate_cells
→ validate_cells
→ fix_cells if needed
```

---

## Example Result

A successful run may return:

```text
Target              : median_house_value
Problem Type        : regression
Generation Status   : success
Notebook Cells      : 39
Validation Status   : valid
Validation Errors   : 0
Fix Attempts        : 0
```

---

## Development Status

### Implemented

- [x] Dataset inspection
- [x] Dataset analysis using LLM
- [x] Problem proposal
- [x] Target confirmation with Human-in-the-Loop
- [x] Target analysis
- [x] Notebook planner
- [x] Per-section notebook generation
- [x] Structured output
- [x] Generation routing
- [x] Retry for each section
- [x] Rate-limit backoff
- [x] Variable contract
- [x] Static syntax validation
- [x] AST dependency validator
- [x] Basic repair loop

### In Development

- [ ] Finalize dependency-aware cell repair
- [ ] Notebook Builder `.ipynb`
- [ ] Notebook Executor
- [ ] Runtime error detection
- [ ] Runtime debugger / repair loop
- [ ] Semantic Machine Learning validation
- [ ] Advanced data leakage checks
- [ ] RAG for Machine Learning knowledge
- [ ] Multi-agent architecture

---

## Roadmap

```text
Notebook Cells
      ↓
Dependency Validation
      ↓
Dependency Repair
      ↓
Notebook Builder
      ↓
.ipynb
      ↓
Notebook Executor
      ↓
Runtime Error
      ↓
Debugger Agent
      ↓
Repair
      ↓
Execute Again
```


## Design Principles

**Human control** — AI proposes the problem, but the user decides the final target.

**Small LLM calls** — Generate the notebook section by section instead of one large request.

**Structured output** — Use Pydantic to constrain the LLM's output.

**Deterministic validation** — Errors that can be checked with Python are checked with Python instead of handing everything over to the LLM.

**Repair instead of regenerate** — When a cell fails, prioritize fixing that cell rather than regenerating the entire notebook.

**Machine Learning safety** — The workflow attempts to mitigate data leakage, out-of-order preprocessing, unintended target changes, fake metrics, undefined variables, and inconsistent code between sections.


---

## Author

The project is built with the goal of researching and practicing the following topics:

- AI Agent;
- LangChain;
- LangGraph;
- Human-in-the-Loop;
- Automated Machine Learning Workflow;
- Code Generation;
- Static Analysis;
- AI-assisted Debugging.
