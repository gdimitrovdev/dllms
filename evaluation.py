import ast
import multiprocessing

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from zss import distance

PY_LANGUAGE = Language(tspython.language())
parser = Parser()
parser.language = PY_LANGUAGE


class ZssNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def get_children(self):
        return self.children


def tree_sitter_to_zss(node):
    children = [tree_sitter_to_zss(child) for child in node.children]
    return ZssNode(node.type, children)


def compute_ted_with_zss(tree_prefix, tree_suffix):
    root_prefix = tree_sitter_to_zss(tree_prefix.root_node)
    root_suffix = tree_sitter_to_zss(tree_suffix.root_node)

    return distance(
        root_prefix,
        root_suffix,
        get_children=ZssNode.get_children,
        insert_cost=lambda node: 1,
        remove_cost=lambda node: 1,
        update_cost=lambda a, b: 0 if a.label == b.label else 1,
    )

def get_ast_node_count(node):
    count = 1
    for child in node.children:
        count += get_ast_node_count(child)
    return count


def parse_python(code):
    return parser.parse(bytes(code, "utf8"))


def is_valid_python(code):
    if not code.strip():
        return False

    tree = parse_python(code)
    return not tree.root_node.has_error

def calculate_ast_deviation(code_prefix, code_suffix):
    tree_prefix = parse_python(code_prefix)
    tree_suffix = parse_python(code_suffix)

    nodes_p = get_ast_node_count(tree_prefix.root_node)
    nodes_s = get_ast_node_count(tree_suffix.root_node)
    

    edit_distance = compute_ted_with_zss(tree_prefix, tree_suffix)
    
    max_nodes = max(nodes_p, nodes_s)
    normalized_diff = edit_distance / max_nodes if max_nodes > 0 else 0
    
    return max(0.0, 1 - normalized_diff)


def _normalize_test_body(tests):
    if not tests or not tests.strip():
        return []

    module = ast.parse(tests)
    body = module.body
    if len(body) == 1 and isinstance(body[0], ast.If):
        test = body[0].test
        if isinstance(test, ast.Constant) and test.value is True:
            body = body[0].body
    return body


def _statement_contains_assert(statement):
    return any(isinstance(node, ast.Assert) for node in ast.walk(statement))


def count_canitedit_test_cases(tests):
    try:
        body = _normalize_test_body(tests)
    except SyntaxError:
        return 0
    return sum(1 for statement in body if _statement_contains_assert(statement))


def _execute_statement(statement, namespace):
    module = ast.Module(body=[statement], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<canitedit-tests>", "exec"), namespace)


def _run_canitedit_tests_worker(code, tests, queue):
    total = count_canitedit_test_cases(tests)

    try:
        body = _normalize_test_body(tests)
        namespace = {}
        exec(compile(code, "<candidate>", "exec"), namespace)

        passed = 0
        aborted = False
        for statement in body:
            is_test_case = _statement_contains_assert(statement)
            if aborted:
                continue

            try:
                _execute_statement(statement, namespace)
                if is_test_case:
                    passed += 1
            except Exception:
                if not is_test_case:
                    aborted = True

        queue.put({
            "passed": passed,
            "total": total,
            "suite_passed": total > 0 and passed == total and not aborted,
            "timed_out": False,
            "errored": False,
        })
    except Exception:
        queue.put({
            "passed": 0,
            "total": total,
            "suite_passed": False,
            "timed_out": False,
            "errored": True,
        })


def run_canitedit_tests(code, tests, timeout_seconds=5):
    total = count_canitedit_test_cases(tests)
    if total == 0:
        return {
            "passed": 0,
            "total": 0,
            "pass_rate": None,
            "suite_passed": False,
            "timed_out": False,
            "errored": False,
        }

    if not code or not code.strip() or not is_valid_python(code):
        return {
            "passed": 0,
            "total": total,
            "pass_rate": 0.0,
            "suite_passed": False,
            "timed_out": False,
            "errored": False,
        }

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_canitedit_tests_worker, args=(code, tests, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "passed": 0,
            "total": total,
            "pass_rate": 0.0,
            "suite_passed": False,
            "timed_out": True,
            "errored": False,
        }

    result = {
        "passed": 0,
        "total": total,
        "suite_passed": False,
        "timed_out": False,
        "errored": process.exitcode not in (0, None),
    }
    if not queue.empty():
        result.update(queue.get())

    result["pass_rate"] = result["passed"] / total if total > 0 else None
    return result
