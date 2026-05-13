from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
parser = Parser()
parser.language = PY_LANGUAGE

def get_ast_node_count(node):
    count = 1
    for child in node.children:
        count += get_ast_node_count(child)
    return count

def calculate_ast_deviation(code_prefix, code_suffix):
    tree_prefix = parser.parse(bytes(code_prefix, "utf8"))
    tree_suffix = parser.parse(bytes(code_suffix, "utf8"))

    nodes_p = get_ast_node_count(tree_prefix.root_node)
    nodes_s = get_ast_node_count(tree_suffix.root_node)
    

    edit_distance = compute_ted_with_zss(tree_prefix, tree_suffix)
    
    max_nodes = max(nodes_p, nodes_s)
    normalized_diff = edit_distance / max_nodes if max_nodes > 0 else 0
    
    return 1 - normalized_diff
