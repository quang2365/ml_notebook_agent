# Offline test suite

Các test trong thư mục này mô phỏng LangChain Runnable bằng `FakeRunnable`.
`FakeRunnable.invoke()` nhận prompt giống LLM thật, ghi lại call và trả về
`AIMessage` hoặc Pydantic structured output đã chuẩn bị trước. Không test nào
gửi request tới NVIDIA.

Chạy toàn bộ test từ thư mục gốc project:

```powershell
.\.venv314\Scripts\python.exe -m unittest discover -s test -p "test_*.py" -v
```

Chạy riêng từng nhóm:

```powershell
.\.venv314\Scripts\python.exe -m unittest test.test_llm_nodes_offline -v
.\.venv314\Scripts\python.exe -m unittest test.test_validation_and_routes -v
.\.venv314\Scripts\python.exe -m unittest test.test_notebook_builder -v
```

Các phần được kiểm tra:

- dataset analysis với fake chat response;
- notebook planning với fake `NotebookPlan`;
- plan repair với plan cũ và validation errors;
- per-section generation với 8 fake `GeneratedSection` responses;
- syntax repair với fake `FixedCell`;
- plan/cell validation và routing;
- JSON-to-cell conversion và ghi file `.ipynb`.
- full render 10 sections/30 cells, dependency validation và notebook build.
