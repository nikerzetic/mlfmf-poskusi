import networkx as nx
import tqdm
import os
import zipfile


class EntryNode:
    def __init__(self, node_id: int, node_type: str, node_description: str):
        self.id = node_id
        self.type = node_type
        self.description = node_description.strip('"')
        self.parents: list["EntryNode"] = []
        self.children: list["EntryNode"] = []

    def __repr__(self) -> str:
        return f"EntryNode({self.id}, {self.type}, {self.description})"

    def add_children(self, children: list["EntryNode"]):
        for child in children:
            if child not in self.children:
                self.children.append(child)
            if self not in child.parents:
                child.parents.append(self)


class Entry:
    def __init__(self, name: str, root: EntryNode):
        self.name = name
        self.root = root
        self.is_tree = self.compute_is_tree()

    def __repr__(self) -> str:
        return f"Entry({self.name}, {self.root})"

    def compute_is_tree(self):
        stack = [self.root]
        processed_nodes = {self.root.id}
        while stack:
            current = stack.pop()
            if len(current.parents) > 1:
                return False
            for child in current.children:
                c_id = child.id
                if c_id in processed_nodes:
                    continue
                stack.append(child)
                processed_nodes.add(c_id)
        return True


def load_entry(entry_file):
    nodes = {}
    root: EntryNode | None = None
    with open(entry_file, "r", encoding="utf-8") as f:
        f.readline()  # id, type, description, children ids
        for line in f:
            parts = line.split("\t")
            node_id = int(parts[0])
            node_type = parts[1]
            node_description = parts[2]
            node_children = eval(parts[3])
            node = EntryNode(node_id, node_type, node_description)
            nodes[node_id] = (node, node_children)
            if root is None:
                root = node
    assert root is not None
    for node, children_ids in nodes.values():
        node.add_children([nodes[c][0] for c in children_ids])
    name = root.children[0].description
    return Entry(name, root)


def load_entries(entry_dir):
    entries = []
    print("Loading entries ...")
    for file in tqdm.tqdm(os.listdir(entry_dir)):
        if not file.endswith(".dag"):
            continue
        entry_file = os.path.join(entry_dir, file)
        entries.append(load_entry(entry_file))
    print(f"Loaded {len(entries)} entries.")
    return entries


def load_graph(graph_file):
    graph = nx.MultiDiGraph()
    print("Loading network ...")
    with open(graph_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")[1:]
            parts = [d.strip() for d in parts]  # check this
            if line.startswith("node"):
                node, properties = parts
                properties = eval(properties)
                graph.add_node(node, **properties)
            else:
                source, sink, edge_type, properties = parts
                properties = eval(properties)
                graph.add_edge(source, sink, edge_type, **properties)
    print(f"Loaded G(V, E) where (|V|, |E|) = ({len(graph.nodes)}, {len(graph.edges)})")
    return graph


def try_unzip(zip_file, entry_dir):
    with zipfile.ZipFile(zip_file, 'r') as z:
        for file_info in tqdm.tqdm(z.infolist()):
            if file_info.filename.endswith('.dag'):
                file_info.filename = os.path.basename(file_info.filename)
                z.extract(file_info, entry_dir)


def load_library(library_name):
    entry_dir = f"{library_name}/entries"
    zip_file = f"{library_name}/entries.zip"
    network_file = f"{library_name}/network.csv"
    if not os.path.exists(entry_dir):
        if os.path.exists(zip_file):
            print(f"Did not found {entry_dir}, but {zip_file} exists. Unzipping ...")
            try_unzip(zip_file, entry_dir)
        else:
            print(f"Did not found {entry_dir} nor {zip_file}.")
    bad = False	
    if not os.path.exists(entry_dir):
        print(f"Did not found {entry_dir}.")
        bad = True
    if not os.path.exists(network_file):
        print(f"Did not found {network_file}.")
        bad = True
    if bad:
        return [], nx.MultiDiGraph()
    return load_entries(entry_dir), load_graph(network_file)


if __name__ == "__main__":
    for lib in ["stdlib", "TypeTopology", "unimath", "mathlib"]:
        print(lib)
        entries, network = load_library(lib)
        print()