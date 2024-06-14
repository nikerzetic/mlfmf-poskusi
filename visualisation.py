import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import helpers
import typing as tp


def provide_format(function: tp.Callable[[nx.DiGraph], tp.Optional[tp.Any]]):
    func_input_type = list(function.__annotations__.values())
    if len(func_input_type) > 2:
        raise ValueError("Too many input arguments")  # HACK: make this more robust

    def applied_to_appropriate_format(
        input: tp.Union[str, helpers.Entry, nx.DiGraph],
    ) -> tp.Optional[tp.Any]:
        if isinstance(input, str) and not func_input_type is str:
            input = helpers.load_entry(input)
        if isinstance(input, helpers.Entry) and not func_input_type is helpers.Entry:
            input = entry_to_graph(input)
        if not isinstance(input, nx.DiGraph):
            raise AttributeError(
                f"Attribute {input} has type {type(input)}, but only str, helpers.Entry and networkx.DiGraph are allowed."
            )
        if not func_input_type is nx.DiGraph:
            raise AttributeError(
                f"Function {function} has input type {func_input_type}, but only str, helpers.Entry and networkx.DiGraph are allowed."
            )
        return function(input)

    return applied_to_appropriate_format


def add_children_to_graph(G: nx.DiGraph, e: helpers.EntryNode):
    for child in e.children:
        add_children_to_graph(G, child)

    G.add_node(e.id, type=e.type.replace(":", ""), desc=e.description)

    for child in e.children:
        G.add_edge(e.id, child.id)


def draw_syntax_tree(G: nx.DiGraph, width=None, heigth=None):
    type_labels = nx.get_node_attributes(G, "type")
    desc_labels = nx.get_node_attributes(G, "desc")

    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    state_pos = {n: (x, y - 8) for n, (x, y) in pos.items()}

    size = min(
        G.number_of_nodes() // np.ceil(np.power(G.number_of_nodes(), 0.25)), 200
    )  # HACK: hardcoded upper limmit of 2000
    width, heigth = size, size
    plt.figure(figsize=(width, heigth), dpi=100)
    # prop = mfm.FontProperties(fname=font_path)
    nx.draw_networkx(G, pos, labels=type_labels, node_size=1000, font_size=9)
    nx.draw_networkx_labels(
        G, state_pos, labels=desc_labels, font_color="red", font_size=9
    )


def change_nodes_encoding(G: nx.DiGraph):
    """
    Replaces Unicode symbols with "?", because Networkx does not work with Unicode symbols found in Agda.
    """
    for node in G.nodes:
        data = G.nodes[node]
        data["type"] = data["type"].encode("ascii", "replace").decode("utf-8")
        data["desc"] = data["desc"].encode("ascii", "replace").decode("utf-8")


def entry_to_graph(entry: helpers.Entry):
    G = nx.DiGraph()
    add_children_to_graph(G, entry.root)
    return G


def draw_syntax_tree_of_entry(entry: helpers.Entry):
    G = entry_to_graph(entry)
    try:
        draw_syntax_tree(G)
    except UnicodeEncodeError:
        change_nodes_encoding(G)
        draw_syntax_tree(G)


def draw_from_dag_file(path):
    entry = helpers.load_entry(path)
    draw_syntax_tree_of_entry(entry)


def get_dag_leaves(path):
    entry = helpers.load_entry(path)
    G = entry_to_graph(entry)
    leaves = []
    for node in G.nodes:
        if not G.succ[node]:
            leaves.append(G.nodes[node]["desc"])
    return leaves


def print_dag_file_leaves(path):
    entry = helpers.load_entry(path)
    print(get_dag_leaves(entry))
