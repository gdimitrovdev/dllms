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
