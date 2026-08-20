import ast
import builtins


BUILTIN_NAMES = set(
    dir(builtins)
)


def validate_dependencies(
    cells: list[dict],
) -> list[dict]:
    errors: list[dict] = []

    # Names already defined by
    # previous cells
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
            continue

        analyzer = CellDependencyAnalyzer()

        analyzer.visit(tree)

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
                        f"Variable `{name}` "
                        "is used before "
                        "being defined "
                        "or imported."
                    ),
                }
            )

        defined_names.update(
            analyzer.defined_names
        )

    return errors


class CellDependencyAnalyzer(
    ast.NodeVisitor
):

    def __init__(self) -> None:
        self.defined_names: set[str] = set()
        self.used_names: set[str] = set()

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


    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self.defined_names.add(
            node.name
        )
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
