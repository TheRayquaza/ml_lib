import pygraphviz as pgv


def convert_tree_to_dot(root, dot, parent_node=None, parent_label=None, side=None):
    if root is not None:
        node_label = f"Feature N°{root.feature} (value > {root.value})"
        dot.add_node(node_label)
        if parent_node is not None:
            dot.add_edge(parent_label, node_label, label=side)
        convert_tree_to_dot(root.left, dot, node_label, node_label, "L")
        convert_tree_to_dot(root.right, dot, node_label, node_label, "R")


def visualize_tree(root):
    dot = pgv.AGraph(directed=True)
    convert_tree_to_dot(root, dot)
    filename = "binary_tree_graphviz"
    dot.write(filename + ".dot")
    dot.draw(filename + ".png", format="png", prog="dot")
