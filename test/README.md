# Offline test suite

Bộ test kiểm tra workflow mà không gửi request tới NVIDIA hoặc LLM thật.
`FakeRunnable` nhận prompt giống một LangChain Runnable, ghi lại các lần gọi và
trả về `AIMessage` hoặc Pydantic structured output đã chuẩn bị trước.

## Chạy test

Chạy toàn bộ test từ thư mục gốc project:

```powershell
.\.venv314\Scripts\python.exe -m unittest discover -s test -p "test_*.py" -v
```

Chạy riêng từng nhóm:

```powershell
.\.venv314\Scripts\python.exe -m unittest test.test_llm_nodes_offline -v
.\.venv314\Scripts\python.exe -m unittest test.test_prepare_generation -v
.\.venv314\Scripts\python.exe -m unittest test.test_generate_section -v
.\.venv314\Scripts\python.exe -m unittest test.test_validation_and_routes -v
.\.venv314\Scripts\python.exe -m unittest test.test_notebook_builder -v
.\.venv314\Scripts\python.exe -m unittest test.test_full_rendered_notebook -v
```

## `test_llm_nodes_offline.py`

Kiểm tra các node thường cần gọi LLM nhưng thay LLM thật bằng dữ liệu giả lập.

- `test_analyze_dataset_uses_fake_llm`: xác nhận node phân tích dataset gọi LLM
  giả đúng một lần và lưu phản hồi vào `summary_llm`.
- `test_plan_node_accepts_structured_fake_response`: xác nhận node lập kế hoạch
  chấp nhận structured output `NotebookPlan` và tạo đủ section.
- `test_fix_plan_receives_old_plan_and_errors`: xác nhận fixer nhận plan cũ cùng
  toàn bộ lỗi validation, trả plan mới và tăng `fix_plan_attempts`.
- `test_generate_all_sections_without_network`: mô phỏng gọi
  `generate_section_node` nhiều lượt, xác nhận section được sinh tuần tự, cells
  được cộng dồn và không có request mạng.
- `test_fix_without_validation_errors_uses_cell_status`: kiểm tra nhánh fixer
  không có lỗi đầu vào vẫn trả đúng cell status và tăng số lần fix.
- `test_fix_syntax_error_with_fake_structured_response`: đưa code cell lỗi cho
  LLM giả và xác nhận source được thay bằng code đã sửa.
- `test_build_dataset_context`: xác nhận context mỗi section chỉ chứa thông tin
  dataset và target cần thiết.
- `test_build_dataset_context_with_empty_state`: xác nhận context builder không
  lỗi khi `summary` và `target_analysis` chưa có.

## `test_prepare_generation.py`

Kiểm tra node khởi tạo một phiên generation mới trước vòng lặp section.

- `test_resets_old_generation_data`: xác nhận cells, lỗi, index, section ID và
  retry từ phiên cũ được reset đồng bộ.
- `test_fails_without_notebook_plan`: xác nhận generation dừng trước khi gọi LLM
  nếu không có notebook plan.
- `test_fails_with_empty_sections`: xác nhận plan có `sections=[]` được báo lỗi
  thay vì đi vào vòng generate.

## `test_generate_section.py`

Kiểm tra trực tiếp node sinh một section và route điều khiển vòng lặp graph.

- `test_generates_one_section`: xác nhận mỗi lần gọi chỉ sinh một section, tăng
  index một đơn vị và giữ `pending` khi vẫn còn section.
- `test_appends_to_existing_cells`: xác nhận cells mới được nối vào cells cũ,
  không reset section trước và chuyển `success` ở section cuối.
- `test_failure_preserves_progress`: giả lập section lỗi, xác nhận cells cũ,
  index và section đã hoàn thành vẫn được giữ để có thể resume.
- `test_continue_when_sections_remain`: route trả `continue` khi vẫn còn section.
- `test_complete_after_last_section`: route trả `complete` khi đã sinh hết để
  graph chuyển sang `validate_cells`.
- `test_failed_generation_stops`: route trả `failed` khi generation thất bại,
  tránh tiếp tục với notebook chưa hoàn chỉnh.

## `test_validation_and_routes.py`

Kiểm tra validator chạy bằng Python và route giới hạn vòng sửa.

- `test_valid_plan`: xác nhận notebook plan hợp lệ không tạo lỗi validation.
- `test_collects_multiple_plan_errors`: xác nhận validator tổng hợp nhiều lỗi
  plan trong một lượt thay vì dừng ở lỗi đầu tiên.
- `test_missing_cells_uses_cell_validation_status`: xác nhận thiếu cells trả
  `validation_cell_status="invalid"` cùng lỗi `missing_cells`.
- `test_valid_cells`: xác nhận cells đúng schema, syntax và dependency là valid.
- `test_invalid_syntax`: xác nhận Python syntax error được phát hiện.
- `test_plan_route`: kiểm tra các nhánh `valid`, `fix`, `failed` và giới hạn số
  vòng sửa plan.
- `test_cell_route`: kiểm tra các nhánh `valid`, `fix`, `failed` và giới hạn số
  vòng sửa cell.

## `test_notebook_builder.py`

Kiểm tra chuyển JSON của agent thành cấu trúc Jupyter Notebook.

- `test_object_to_code_cell`: chuyển dictionary agent thành code cell đúng chuẩn
  Jupyter, có `execution_count` và `outputs`.
- `test_json_string_with_cells_wrapper`: parse JSON string có wrapper `cells` và
  chuyển toàn bộ phần tử thành notebook cells.
- `test_builder_writes_valid_ipynb`: xác nhận builder tạo `.ipynb` đúng
  `nbformat`, số cell và trạng thái build.

## `test_full_rendered_notebook.py`

Giả lập notebook Machine Learning hoàn chỉnh gồm 10 section và 30 cells.

- `test_complete_render_contains_30_cells`: xác nhận 10 lượt LLM giả tạo đúng 30
  cells, gồm 20 code cells và 10 markdown cells.
- `test_complete_render_passes_static_and_dependency_validation`: xác nhận
  notebook lớn vượt qua syntax và dependency validation.
- `test_complete_render_builds_30_cell_ipynb`: xác nhận 30 cells được ghi thành
  file `.ipynb` hoàn chỉnh.
- `test_validator_detects_error_in_large_render`: chèn syntax error vào notebook
  lớn và xác nhận validator phát hiện đúng lỗi.

## Dữ liệu hỗ trợ

- `fakes.py`: cung cấp `FakeRunnable`, notebook plan và structured response giả.
- `json_samples/`: chứa JSON mẫu cho JSON-to-cell và notebook builder.

Nếu test offline thành công nhưng `main.py` gặp `429` hoặc timeout, nguyên nhân
thường thuộc API thật. Nếu test offline thất bại, cần sửa node, route, schema
hoặc validator trước khi gọi LLM.
