# ML Notebook Agent

**ML Notebook Agent** là một dự án xây dựng **AI Agent có khả năng tự động phân tích dataset và tạo Jupyter Notebook cho bài toán Machine Learning**.

Hệ thống được phát triển theo kiến trúc workflow bằng **LangGraph**, trong đó mỗi giai đoạn của quy trình Machine Learning được tách thành các node riêng biệt như kiểm tra dữ liệu, phân tích dataset, đề xuất bài toán, xác nhận target, lập kế hoạch notebook, sinh code, kiểm tra lỗi và tự động sửa lỗi.

Mục tiêu của dự án là xây dựng một agent có thể nhận một dataset đầu vào và từng bước tạo ra một notebook Machine Learning có cấu trúc rõ ràng, nhất quán và có khả năng được kiểm tra trước khi thực thi.

---

## Mục tiêu của dự án

ML Notebook Agent hướng tới việc tự động hóa quy trình:

```text
Dataset
   ↓
Phân tích dữ liệu
   ↓
Đề xuất bài toán Machine Learning
   ↓
Người dùng xác nhận Target / Problem Type
   ↓
Phân tích Target
   ↓
Lập kế hoạch Notebook
   ↓
Sinh từng Section
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

Thay vì yêu cầu LLM sinh toàn bộ notebook trong một lần gọi, hệ thống chia notebook thành nhiều **section nhỏ** và generate từng section riêng biệt. Thiết kế này giúp giảm nguy cơ timeout, giảm kích thước mỗi request và dễ kiểm soát lỗi hơn.

---

## Kiến trúc tổng quan

Workflow chính hiện tại:

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

Workflow được quản lý bằng **LangGraph StateGraph** và sử dụng **checkpointer** để hỗ trợ Human-in-the-Loop.

---

## Các thành phần chính

### 1. Dataset Inspection

Node `inspect_data` chịu trách nhiệm đọc dataset và thu thập các thông tin cơ bản như:

- số dòng và số cột;
- tên các cột;
- kiểu dữ liệu;
- numeric columns;
- categorical columns;
- missing values;
- possible ID columns;
- các thông tin tổng quan cần thiết cho các node phía sau.

### 2. Dataset Analysis bằng LLM

Node `analyze_data` sử dụng LLM để phân tích thông tin dataset đã được thu thập. Kết quả này được dùng làm context cho việc xác định bài toán Machine Learning phù hợp.

### 3. Problem Proposal

Node `propose_problem` tự động đề xuất:

- `target_column`;
- `problem_type`;
- lý do đề xuất;
- số lượng giá trị khác nhau của target;
- mức độ phù hợp của candidate.

Ví dụ:

```json
{
  "target_column": "median_house_value",
  "problem_type": "regression",
  "candidate_score": 3,
  "unique_count": 3842,
  "reasons": [
    "Tên cột chứa từ khóa: value"
  ]
}
```

### 4. Human-in-the-Loop

Hệ thống không tự động chấp nhận target do AI đề xuất.

Node `review_problem` sử dụng `interrupt()` của LangGraph để yêu cầu người dùng xác nhận. Người dùng có thể approve, edit hoặc reject đề xuất.

Graph được resume bằng:

```python
Command(resume=decision)
```

với cùng `thread_id`.

---

## Notebook Planner

Sau khi target được xác nhận, hệ thống tạo một `NotebookPlan`.

Notebook hiện được giới hạn khoảng **8–10 section** để tránh sinh notebook quá dài và giảm số lượng request tới LLM.

Một cấu trúc notebook điển hình:

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

Planner phải đảm bảo thứ tự Machine Learning hợp lý, đặc biệt:

```text
Train/Test Split
        ↓
Fit Preprocessing trên Train
        ↓
Model Training
        ↓
Evaluation
```

nhằm hạn chế nguy cơ **data leakage**.

---

## Per-Section Notebook Generation

Một trong những điểm chính của dự án là không generate toàn bộ notebook bằng một request duy nhất.

Thay vào đó:

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

Mỗi section được sinh độc lập bằng structured output.

Ví dụ một notebook cell:

```json
{
  "cell_id": "section_4_code_1",
  "section_id": "section_4",
  "cell_type": "code",
  "title": "Train/Test Split",
  "source": "X_train, X_test, y_train, y_test = ...",
  "purpose": "Chia dữ liệu thành train và test",
  "expected_output": null
}
```

---

## Variable Contract

Vì các section được generate bằng nhiều LLM call độc lập, dự án sử dụng một **Variable Contract** để giữ continuity giữa các section.

Một số biến chuẩn:

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

Agent được yêu cầu không tự đổi các biến chuẩn sang những tên khác như:

```text
train_X
X_training
housing_df
X_train_data
```

Điều này giúp giảm lỗi dependency giữa các section.

---

## Structured Output

Các output quan trọng của LLM được định nghĩa bằng **Pydantic models**.

Ví dụ:

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

Việc sử dụng structured output giúp kiểm soát format tốt hơn so với việc yêu cầu LLM trả về JSON tự do.

---

## Static Cell Validation

Sau khi merge toàn bộ section, các cell được đưa vào validator.

Validator hiện kiểm tra các lỗi như:

- thiếu `cell_id`;
- duplicate `cell_id`;
- sai `cell_type`;
- source không phải string;
- Markdown code fence trong Python cell;
- Python syntax error;
- thiếu dataset path;
- thiếu target đã xác nhận.

Ví dụ:

```python
compile(
    source,
    filename=cell_id,
    mode="exec",
)
```

được sử dụng để kiểm tra cú pháp Python mà chưa cần thực thi notebook.

---

## Dependency Validation bằng AST

Syntax đúng không đồng nghĩa với notebook có thể chạy.

Ví dụ:

```python
model.fit(
    X_train_processed,
    y_train,
)
```

vẫn compile thành công ngay cả khi `X_train_processed` chưa từng được tạo.

Do đó dự án sử dụng `ast` của Python để phân tích code theo thứ tự notebook.

Dependency validator theo dõi:

```text
defined variables
used variables
imports
functions
classes
```

và phát hiện lỗi như:

```json
{
  "cell_id": "section_6_code_2",
  "error_type": "undefined_variable",
  "variable": "X_train_processed",
  "message": "Biến `X_train_processed` được sử dụng trước khi được định nghĩa hoặc import."
}
```

---

## Automatic Cell Repair

Khi validator phát hiện lỗi:

```text
validate_cells
      ↓
invalid
      ↓
fix_cells
      ↓
validate_cells
```

`fix_cells` sửa từng cell thay vì regenerate toàn bộ notebook.

Hệ thống đang hỗ trợ repair các nhóm lỗi như:

```text
syntax_error
undefined_variable
```

Fixer có thể sử dụng:

- validation errors;
- variable contract;
- previous code context;
- available variable names;
- source của cell hiện tại.

Mục tiêu là giữ thay đổi nhỏ nhất có thể và không tự tạo biến giả chỉ để làm validator hết lỗi.

---

## Retry và Rate Limit Handling

Việc generate từng section tạo ra nhiều request nhỏ tới LLM.

Dự án có retry riêng cho từng section:

```text
section_1 ✅
section_2 ✅
section_3 ❌
          ↓
      retry section_3
          ↓
section_3 ✅
```

Đối với lỗi rate limit `429`, hệ thống sử dụng exponential backoff và có khoảng nghỉ giữa các section để giảm tần suất request.

---

## LLM Provider

Phiên bản hiện tại sử dụng model thông qua NVIDIA NIM với OpenAI-compatible API.

Ví dụ cấu hình:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="minimaxai/minimax-m3",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)
```

API key được lưu trong `.env`:

```env
NVIDIA_API_KEY=your_api_key_here
```

Không nên commit `.env` lên repository.

---

## Công nghệ sử dụng

Các công nghệ chính:

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

Các thư viện Machine Learning được notebook agent có thể sử dụng tùy theo dataset và notebook plan, ví dụ:

```text
pandas
numpy
scikit-learn
matplotlib
xgboost
```

---

## Cấu trúc dự án

Một cấu trúc thư mục tham khảo:

```text
ml_notebook_agent/
│
├── main.py
├── graph.py
├── state.py
│
├── model/
│   └── model.py
│
├── nodes/
│   ├── inspect_dataset_node.py
│   ├── analyze_dataset_node.py
│   ├── propose_problem_node.py
│   ├── review_problem_node.py
│   ├── analyze_target_node.py
│   ├── plan_notebook_node.py
│   ├── generate_cells_node.py
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

Cấu trúc thực tế có thể thay đổi trong quá trình phát triển.

---

## State của LangGraph

Một phần state hiện tại:

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

    generation_status: Literal[
        "pending",
        "success",
        "failed",
    ] | None

    validation_status: Literal[
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

## Cài đặt

Clone repository:

```bash
git clone <repository-url>
cd ml_notebook_agent
```

Tạo virtual environment:

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

Cài dependency:

```bash
pip install -r requirements.txt
```

Tạo file `.env`:

```env
NVIDIA_API_KEY=your_api_key_here
```

---

## Chạy dự án

Chạy:

```bash
python main.py
```

Graph sẽ chạy cho tới bước Human Review. Sau khi người dùng xác nhận target và problem type, graph được resume và tiếp tục:

```text
analyze_target
→ plan_notebook
→ generate_cells
→ validate_cells
→ fix_cells nếu cần
```

---

## Ví dụ kết quả

Một lần chạy thành công có thể trả về:

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

## Trạng thái phát triển

### Đã triển khai

- [x] Dataset inspection
- [x] Dataset analysis bằng LLM
- [x] Problem proposal
- [x] Target confirmation bằng Human-in-the-Loop
- [x] Target analysis
- [x] Notebook planner
- [x] Per-section notebook generation
- [x] Structured output
- [x] Generation routing
- [x] Retry cho từng section
- [x] Rate-limit backoff
- [x] Variable contract
- [x] Static syntax validation
- [x] AST dependency validator
- [x] Repair loop cơ bản

### Đang phát triển

- [ ] Hoàn thiện dependency-aware cell repair
- [ ] Notebook Builder `.ipynb`
- [ ] Notebook Executor
- [ ] Runtime error detection
- [ ] Runtime debugger / repair loop
- [ ] Semantic Machine Learning validation
- [ ] Kiểm tra data leakage nâng cao
- [ ] RAG cho kiến thức Machine Learning
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


## Nguyên tắc thiết kế

**Human control** — AI đề xuất bài toán nhưng người dùng quyết định target cuối cùng.

**Small LLM calls** — Sinh notebook theo từng section thay vì một request lớn.

**Structured output** — Dùng Pydantic để giới hạn output của LLM.

**Deterministic validation** — Các lỗi có thể kiểm tra bằng Python sẽ được kiểm tra bằng Python thay vì giao toàn bộ cho LLM.

**Repair instead of regenerate** — Khi một cell lỗi, ưu tiên sửa cell đó thay vì generate lại toàn bộ notebook.

**Machine Learning safety** — Workflow cố gắng hạn chế data leakage, preprocessing sai thứ tự, target thay đổi ngoài ý muốn, fake metrics, undefined variables và code không nhất quán giữa các section.


---

## Tác giả

Dự án được xây dựng với mục tiêu nghiên cứu và thực hành các chủ đề:

- AI Agent;
- LangChain;
- LangGraph;
- Human-in-the-Loop;
- Automated Machine Learning Workflow;
- Code Generation;
- Static Analysis;
- AI-assisted Debugging.
