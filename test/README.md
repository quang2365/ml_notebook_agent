# Offline test suite

Test suite for the workflow without sending requests to NVIDIA or a real LLM.
`FakeRunnable` accepts a prompt like a LangChain Runnable, records calls, and
returns a pre-prepared `AIMessage` or Pydantic structured output.

## Running tests

Run the full test suite from the project root directory:

```powershell
.\.venv314\Scripts\python.exe -m unittest discover -s test -p "test_*.py" -v
```

Run each group individually:

```powershell
.\.venv314\Scripts\python.exe -m unittest test.test_llm_nodes_offline -v
.\.venv314\Scripts\python.exe -m unittest test.test_prepare_generation -v
.\.venv314\Scripts\python.exe -m unittest test.test_generate_section -v
.\.venv314\Scripts\python.exe -m unittest test.test_validation_and_routes -v
.\.venv314\Scripts\python.exe -m unittest test.test_notebook_builder -v
.\.venv314\Scripts\python.exe -m unittest test.test_full_rendered_notebook -v
.\.venv314\Scripts\python.exe -m unittest test.test_model_selection -v
.\.venv314\Scripts\python.exe -m unittest test.test_execute_notebook -v
```

## `test_llm_nodes_offline.py`

Tests nodes that normally need to call an LLM, substituting fake data for the real LLM.

- `test_analyze_dataset_uses_fake_llm`: verifies the dataset analysis node calls the fake LLM
  exactly once and stores the response in `summary_llm`.
- `test_plan_node_accepts_structured_fake_response`: verifies the planning node
  accepts the `NotebookPlan` structured output and creates all required sections.
- `test_fix_plan_receives_old_plan_and_errors`: verifies the fixer receives the old plan along with
  all validation errors, returns a new plan, and increments `fix_plan_attempts`.
- `test_generate_all_sections_without_network`: simulates calling
  `generate_section_node` multiple times, verifies sections are generated sequentially, cells
  are accumulated, and no network requests are made.
- `test_fix_without_validation_errors_uses_cell_status`: tests the fixer branch
  with no input errors still returns the correct cell status and increments the fix count.
- `test_fix_syntax_error_with_fake_structured_response`: feeds code cell errors to the
  fake LLM and verifies the source is replaced with the fixed code.
- `test_fix_undefined_best_model_with_previous_context`: reproduces the `best_model` error,
  verifies the fixer receives the previous code/available names and fixes it using
  the model stored in `trained_models`.
- `test_rejects_fix_that_keeps_undefined_variable`: verifies a fix that compiles
  but still contains an undefined variable is not recorded as a successful fix.
- `test_build_dataset_context`: verifies each section context only contains the necessary
  dataset and target information.
- `test_build_dataset_context_with_empty_state`: verifies the context builder does not
  error when `summary` and `target_analysis` are not yet available.

## `test_prepare_generation.py`

Tests the node that initializes a new generation session before the section loop.

- `test_resets_old_generation_data`: verifies cells, errors, index, section ID, and
  retries from the previous session are reset together.
- `test_fails_without_notebook_plan`: verifies generation stops before calling the LLM
  if there is no notebook plan.
- `test_fails_with_empty_sections`: verifies a plan with `sections=[]` reports an error
  instead of entering the generation loop.

## `test_generate_section.py`

Tests the node that generates one section directly and the route controlling the graph loop.

- `test_generates_one_section`: verifies each call generates only one section, increments the
  index by one, and keeps `pending` while sections remain.
- `test_appends_to_existing_cells`: verifies new cells are appended to existing cells,
  does not reset the previous section, and transitions to `success` on the last section.
- `test_failure_preserves_progress`: simulates a failing section, verifies existing cells,
  index, and completed sections are retained so execution can resume.
- `test_continue_when_sections_remain`: route returns `continue` when sections still remain.
- `test_complete_after_last_section`: route returns `complete` after all sections have been generated so
  the graph moves to `validate_cells`.
- `test_failed_generation_stops`: route returns `failed` when generation fails,
  avoiding continuation with an incomplete notebook.
- `test_retry_failed_section_when_attempts_remain`: verifies the route returns `retry`
  and keeps the current section unchanged while retry attempts remain.
- `test_stop_after_section_retry_limit`: verifies the route returns `failed` when the number of
  attempts reaches the limit, preventing an infinite generation loop.

## `test_validation_and_routes.py`

Tests that the validator runs in Python and that the route limits the fix loop.

- `test_valid_plan`: verifies a valid notebook plan does not produce validation errors.
- `test_collects_multiple_plan_errors`: verifies the validator aggregates multiple plan
  errors in one pass instead of stopping at the first error.
- `test_missing_cells_uses_cell_validation_status`: verifies missing cells return
  `validation_cell_status="invalid"` along with `missing_cells` errors.
- `test_valid_cells`: verifies cells with correct schema, syntax, and dependencies are valid.
- `test_invalid_syntax`: verifies Python syntax errors are detected.
- `test_plan_route`: tests the `valid`, `fix`, and `failed` branches and the limit on the number of
  plan fix loops.
- `test_cell_route`: tests the `valid`, `fix`, and `failed` branches and the limit on the number of
  cell fix loops.

## `test_notebook_builder.py`

Tests converting the agent's JSON into a Jupyter Notebook structure.

- `test_object_to_code_cell`: converts an agent dictionary into a standard Jupyter code cell,
  with `execution_count` and `outputs`.
- `test_json_string_with_cells_wrapper`: parses a JSON string with a `cells` wrapper and
  converts all elements into notebook cells.
- `test_builder_writes_valid_ipynb`: verifies the builder creates a `.ipynb` file with correct
  `nbformat`, cell count, and build status.

## `test_full_rendered_notebook.py`

Simulates a complete Machine Learning notebook with 10 sections and 30 cells.

- `test_complete_render_contains_30_cells`: verifies 10 fake LLM turns generate exactly 30
  cells, including 20 code cells and 10 markdown cells.
- `test_complete_render_passes_static_and_dependency_validation`: verifies a
  large notebook passes syntax and dependency validation.
- `test_complete_render_builds_30_cell_ipynb`: verifies 30 cells are written into
  a complete `.ipynb` file.
- `test_validator_detects_error_in_large_render`: injects a syntax error into the large
  notebook and verifies the validator detects the correct error.

## Supporting data

- `fakes.py`: provides `FakeRunnable`, a notebook plan, and fake structured responses.
- `json_samples/`: contains sample JSON for JSON-to-cell and notebook builder.

## `test_model_selection.py`

- `test_create_deepseek_model_when_selected`: verifies selecting DeepSeek uses
  the `deepseek-v4-flash` model, the DeepSeek endpoint, and `DEEPSEEK_API_KEY`.
- `test_keep_nvidia_model_when_deepseek_not_selected`: verifies the default
  selection still uses NVIDIA Nemotron and `NVIDIA_API_KEY`.
- `test_prompt_accepts_deepseek`: verifies a `yes` answer enables DeepSeek.
- `test_prompt_defaults_to_current_model`: verifies pressing Enter keeps the current model.

## `test_execute_notebook.py`

- `test_missing_notebook_path`: verifies the node reports errors when State does not yet have a notebook
  path.
- `test_notebook_file_does_not_exist`: verifies the node reports errors when the notebook file
  does not exist.
- `test_successful_execution`: simulates successful kernel execution and verifies the
  node updates `execution_status=success`.
- `test_cell_execution_error`: simulates runtime errors in the cell and verifies the node
  stores errors under `execution_error`.

If the offline tests pass but `main.py` encounters `429` or a timeout, the cause is
usually the real API. If the offline tests fail, fix the node, route, schema,
or validator before calling the LLM.
