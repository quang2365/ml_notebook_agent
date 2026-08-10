import ast
import builtins


BUILTIN_NAMES = set(
    dir(builtins)
)


def validate_dependencies(
    cells: list[dict],
) -> list[dict]:
    """
    Kiểm tra code cell theo đúng thứ tự notebook.

    Phát hiện biến được sử dụng trước
    khi được định nghĩa/import.
    """

    errors: list[dict] = []

    # Những name đã tồn tại từ
    # các cell trước
    defined_names: set[str] = set(
        BUILTIN_NAMES
    )

    for cell_index, cell in enumerate(
        cells
    ):
        if (
            cell.get("cell_type")
            != "code"
        ):
            continue

        cell_id = cell.get(
            "cell_id"
        )

        source = (
            cell.get("source")
            or ""
        )

        try:
            tree = ast.parse(
                source
            )

        except SyntaxError:
            # Syntax validator đã xử lý.
            # Không phân tích dependency
            # trên AST lỗi.
            continue

        analyzer = CellDependencyAnalyzer()

        analyzer.visit(tree)

        # ==========================
        # 1. NAME ĐƯỢC SỬ DỤNG
        # ==========================

        available_names = (
            defined_names
            | analyzer.defined_names
        )

        undefined_names = (
            analyzer.used_names
            - available_names
        )

        for name in sorted(
            undefined_names
        ):
            errors.append(
                {
                    "cell_index":
                        cell_index,

                    "cell_id":
                        cell_id,

                    "error_type":
                        "undefined_variable",

                    "variable":
                        name,

                    "message": (
                        f"Biến `{name}` "
                        "được sử dụng trước "
                        "khi được định nghĩa "
                        "hoặc import."
                    ),
                }
            )

        # ==========================
        # 2. CẬP NHẬT CONTEXT
        # ==========================

        defined_names.update(
            analyzer.defined_names
        )

    return errors


class CellDependencyAnalyzer(
    ast.NodeVisitor
):
    """
    Thu thập name được định nghĩa và
    name được sử dụng trong một code cell.

    Đây là validator continuity cấp notebook,
    không phải static analyzer Python hoàn chỉnh.
    """

    def __init__(self) -> None:
        self.defined_names: set[str] = set()
        self.used_names: set[str] = set()

    # ==============================
    # NAME
    # ==============================

    def visit_Name(
        self,
        node: ast.Name,
    ) -> None:

        if isinstance(
            node.ctx,
            ast.Store,
        ):
            self.defined_names.add(
                node.id
            )

        elif isinstance(
            node.ctx,
            ast.Load,
        ):
            self.used_names.add(
                node.id
            )

    # ==============================
    # IMPORT
    # ==============================

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:

        for alias in node.names:
            name = (
                alias.asname
                or alias.name.split(
                    "."
                )[0]
            )

            self.defined_names.add(
                name
            )

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:

        for alias in node.names:
            if alias.name == "*":
                continue

            name = (
                alias.asname
                or alias.name
            )

            self.defined_names.add(
                name
            )

    # ==============================
    # FUNCTION
    # ==============================

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        """
        Function name tồn tại sau cell.

        Không đi sâu vào function body
        ở validator continuity version 1,
        vì biến bên trong function có scope riêng.
        """

        self.defined_names.add(
            node.name
        )

        # Default arguments có thể dùng
        # biến global.
        for default in node.args.defaults:
            self.visit(default)

        for default in (
            node.args.kw_defaults
        ):
            if default is not None:
                self.visit(default)

        for decorator in (
            node.decorator_list
        ):
            self.visit(decorator)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:

        self.visit_FunctionDef(node)

    # ==============================
    # CLASS
    # ==============================

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:

        self.defined_names.add(
            node.name
        )

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(
                keyword.value
            )

        for decorator in (
            node.decorator_list
        ):
            self.visit(decorator)

    # ==============================
    # EXCEPTION
    # ==============================

    def visit_ExceptHandler(
        self,
        node: ast.ExceptHandler,
    ) -> None:

        if node.type:
            self.visit(node.type)

        if node.name:
            self.defined_names.add(
                node.name
            )

        for statement in node.body:
            self.visit(statement)