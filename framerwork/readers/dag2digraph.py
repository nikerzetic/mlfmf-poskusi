"""
For reading .dag files to networkx.Digraphs with silmultaneous data transformation
"""

import networkx as nx
from typing import *


def reader(
    file_path: str,
    transformer: Callable = lambda s: s,
) -> nx.DiGraph:
    """
    Reads a .dag file (TODO: as in mlfmf-data), execute transformations of it's nodes, and returns a networkx.Digraph.

    ## Parameters
    
    file_path:

    transformer:

    ## Returns

    ## Examples
    ```
    def transformer(G, node_id, node_type, node_description, node_children):
        node_type.replace(":", "")
        helpers.replace_unicode_with_latex(format_as_label(node_description))
    ```
    """
    children = {}
    G = nx.DiGraph()
    # leaves = []
    # name = None
    file = open(file_path, "r", encoding="utf-8")
    if not str(file).endswith(".dag"):
        file.readline()  # Skip column names id, type, description, children ids
    for line in file:
        parts = line.split("\t")
        node_id = int(parts[0])
        node_type = parts[1]
        node_description = parts[2]
        node_children = eval(parts[3])
        children[node_id] = node_children
        transformer(G, node_id, node_type, node_description, node_children)
        G.add_node(
            node_id,
            type=node_type,
            desc=node_description,
            # type=node_type.replace(":", ""),
            # desc=helpers.replace_unicode_with_latex(format_as_label(node_description)),
        )
        # is_leaf_node = not node_children
        # is_appropriate_type = node_type in [
        #     ":name",
        #     ":bound",
        #     ":pattern-var",
        # ]  # XXX: but :bound and :arg-var are actual code, but not leaves
        # if is_leaf_node and is_appropriate_type:
        #     leaves.append(node_id)
        # if name:
        #     continue
        # # TODO make sure the first :name node is the name of the function
        # if node_type == ":name":
        #     name = helpers.replace_unicode_with_latex(format_as_label(node_description))
        #     G.nodes[node_id]["desc"] = "METHOD_NAME"  # HACK: anonimising in place
    file.close()
    for node_id, children_ids in children.items():
        for child_id in children_ids:
            G.add_edge(node_id, child_id)
    return G
