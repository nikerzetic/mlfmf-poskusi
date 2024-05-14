import networkx as nx
import re
import tqdm
import os
import zipfile
import shutil
import random
import datetime
import sys


class EntryNode:
    def __init__(self, node_id: int, node_type: str, node_description: str):
        self.id = node_id
        self.type = node_type
        self.description = node_description.strip('"')
        self.parents: list["EntryNode"] = []
        self.children: list["EntryNode"] = []

    def __repr__(self) -> str:
        return f"EntryNode({self.id}, {self.type}, {self.description})"

    def __str__(self) -> str:
        return f"{self.id}\t{self.type}\t{self.description}\t{[child.id for child in self.children]}"

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

    def __str__(self) -> str:
        lines = ["NODE ID\tNODE TYPE\tNODE DESCRIPTION\tCHILDREN IDS"]
        lines += [str(node) for node in self.to_list()]
        return "\n".join(lines)

    def to_list(self) -> list[EntryNode]:
        def append_entry_node(lst: list[EntryNode], node: EntryNode):
            lst.append(node)
            for child in node.children:
                append_entry_node(lst, child)

        lst = []
        append_entry_node(lst, self.root)
        return lst

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
    with zipfile.ZipFile(zip_file, "r") as z:
        for file_info in tqdm.tqdm(z.infolist()):
            if file_info.filename.endswith(".dag"):
                file_info.filename = os.path.basename(file_info.filename)
                z.extract(file_info, entry_dir)


def _library_paths(library_name: str):
    entry_dir = f"{library_name}/entries"
    zip_file = f"{library_name}/entries.zip"
    network_file = f"{library_name}/network.csv"
    return entry_dir, zip_file, network_file


def _library_invalid(library_name):
    entry_dir, _, network_file = _library_paths(library_name)
    bad = False
    if not os.path.exists(entry_dir):
        print(f"Did not find {entry_dir}.")
        bad = True
    if not os.path.exists(network_file):
        print(f"Did not find {network_file}.")
        bad = True
    return bad


def load_library(library_name):
    entry_dir, zip_file, network_file = _library_paths(library_name)
    if not os.path.exists(entry_dir):
        if os.path.exists(zip_file):
            print(f"Did not find {entry_dir}, but {zip_file} exists. Unzipping ...")
            try_unzip(zip_file, entry_dir)
        else:
            print(f"Did not find {entry_dir} nor {zip_file}.")

    if _library_invalid(library_name):
        return [], nx.MultiDiGraph()
    return load_entries(entry_dir), load_graph(network_file)


def split_network_into_nodes_and_links(network_file_path):
    network_directory = os.path.dirname(os.path.abspath(network_file_path))
    nodes_file = open(
        os.path.join(network_directory, "nodes.tsv"), "w", encoding="utf-8"
    )
    links_file = open(
        os.path.join(network_directory, "links.tsv"), "w", encoding="utf-8"
    )
    nodes_file.write("node\tproperties")
    links_file.write("source\tsink\tedge_type\tproperties")

    print("Splitting the network file into separate nodes and links files...")
    with open(network_file_path, encoding="utf-8") as network_file:
        for line in network_file:
            parts = line.split("\t")[1:]
            parts = [d.strip() for d in parts]  # check this
            if line.startswith("node"):
                node, properties = parts
                nodes_file.write(f"\n{node}\t{properties}")
            else:
                source, sink, edge_type, properties = parts
                links_file.write(f"\n{source}\t{sink}\t{edge_type}\t{properties}")

    nodes_file.close()
    links_file.close()


def _check_dirs_exist_or_create(dirs_list):
    for dir in dirs_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def _code2vec_train_val_test_dirs(path):
    new_dir = os.path.join(path, "code2vec")
    train_dir = os.path.join(new_dir, "train")
    val_dir = os.path.join(new_dir, "val")
    test_dir = os.path.join(new_dir, "test")
    return train_dir, val_dir, test_dir


def copy_entries_into_train_val_test_directories(
    library_name: str,
    train_mask: list[int],
    val_mask: list[int],
    test_mask: list[int],
):
    """Copy entries of the specified library into separate train, val and test directories, as required by code2vec."""
    # TODO: test for overlap of masks
    # TODO: ensure ordered lists
    entry_dir, _, _ = _library_paths(library_name)
    train_dir, val_dir, test_dir = _code2vec_train_val_test_dirs(
        os.path.dirname(os.path.abspath(entry_dir))
    )
    _check_dirs_exist_or_create([train_dir, val_dir, test_dir])
    train_pointer, val_pointer, test_pointer = 0, 0, 0  # we only traverse lists once
    for i, entry in tqdm.tqdm(enumerate(os.listdir(entry_dir))):
        if i in train_mask[train_pointer]:
            shutil.copy(entry, train_dir)
            train_pointer += 1
        if i in val_mask[val_pointer]:
            shutil.copy(entry, val_dir)
            val_pointer += 1
        if i in test_mask[test_pointer]:
            shutil.copy(entry, test_dir)
            test_pointer += 1


def probibalistic_copy_entries_into_train_val_test_directories(
    library_name: str,
    val_probability: float,
    test_probability: float,
):
    """Randomly copy entries of the specified library into separate train, val and test directories, as required by code2vec."""
    if val_probability + test_probability > 1:
        raise ValueError(
            "The probabilities sum up to {}".format(val_probability + test_probability)
        )
    entry_dir, _, _ = _library_paths(library_name)
    train_dir, val_dir, test_dir = _code2vec_train_val_test_dirs(
        os.path.dirname(os.path.abspath(entry_dir))
    )
    _check_dirs_exist_or_create([train_dir, val_dir, test_dir])
    train_counter, val_counter, test_counter = 0, 0, 0

    print("Splitting entries into separate train, val, test directories...")
    for entry in tqdm.tqdm(os.listdir(entry_dir)):
        entry = os.path.join(entry_dir, entry)
        u = random.random()
        if u <= val_probability:
            shutil.copy(entry, val_dir)
            val_counter += 1
        elif u <= val_probability + test_probability:
            shutil.copy(entry, test_dir)
            test_counter += 1
        else:
            shutil.copy(entry, train_dir)
            train_counter += 1
    print(
        "\nTotal count:",
        "\n\tTrain: ",
        train_counter,
        "\n\tVal: ",
        val_counter,
        "\n\tTest: ",
        test_counter,
    )


def entry_to_dag(entry: Entry):
    pass


def write_entry_node(node: EntryNode, id: int):
    child_ids = [child.id for child in node.children]
    return f"{id}\t{node.type}\t{node.description}\t{child_ids}", node.children


def reindex_entry(entry: Entry, starting_id: int):
    current_id = starting_id
    entry.root.id = current_id
    queue = [entry.root]
    while queue:
        current_node = queue.pop(0)
        for child in current_node.children:
            current_id += 1
            child.id = current_id
            queue.append(child)


def reindex_library_asts(library_name):
    print(f"Reindexing library {library_name}...")
    entry_dir, _, _ = _library_paths(library_name)
    for file in tqdm.tqdm(os.listdir(entry_dir)):
        if not file.endswith(".dag"):
            continue
        entry = load_entry(os.path.join(entry_dir, file))
        current_id = entry.root.children[0].id
        reindex_entry(entry, current_id)
        with open(os.path.join(entry_dir, file), "w", encoding="utf-8") as f:
            f.write(str(entry))


def generate_report():
    import math
    import pandas as pd

    leaves = 0
    entries = 0
    paths = 0
    name_paths = 0
    max_leaves = 0
    max_entry = None
    functions = 0
    function_paths = 0
    library = "D:\\Nik\\Projects\\mlfmf-poskusi\\stdlib"
    entries_dir = os.path.join(library, "entries")
    log = open(os.path.join(library, "entries_stats.tsv"), "w", encoding="utf-8")
    log.write("file_name\tentry_type\tnum_nodes\tnum_edges\tnum_leaves\tnum_names")

    labels_dict = {}
    with open(os.path.join(library, "nodes.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.split("\t")
            labels_dict[str(parts[0])] = eval(parts[1])

    print("Calculating total number of paths...")
    for file in tqdm.tqdm(os.listdir(entries_dir)):
        if not file.endswith(".dag"):
            continue
        entries += 1
        entry_nodes = 0
        entry_edges = 0
        entry_leaves = 0
        entry_names = 0
        with open(os.path.join(entries_dir, file), "r", encoding="utf-8") as f:
            f.readline()  # id, type, description, children ids
            name = None
            for line in f:
                entry_nodes += 1
                parts = line.split("\t")
                node_type = parts[1]
                node_children = eval(parts[3])

                if not node_children:
                    leaves += 1
                    entry_leaves += 1
                else:
                    entry_edges += len(node_children)
                if node_type == ":name":
                    if not name:
                        name = str(parts[2]).replace('"', "")
                    entry_names += 1

        entry_type = labels_dict[name]["label"].replace(":", "")

        if entry_type == "function":
            functions += 1
            function_paths += math.factorial(entry_leaves - 1)
            # if function_paths_carry >= BILLION:
            #     function_paths_billions += function_paths_carry // BILLION
            #     function_paths_carry = function_paths_carry % BILLION

        log.write(
            f"\n{file}\t{entry_type}\t{entry_nodes}\t{entry_edges}\t{entry_leaves}\t{entry_names}"
        )
        paths += math.factorial(entry_leaves - 1)
        name_paths += math.factorial(entry_names - 1)
        # if paths_carry >= BILLION:
        #     paths_billions += paths_carry // BILLION
        #     paths_carry = paths_carry % BILLION
        # if name_paths_carry >= BILLION:
        #     name_paths_billions += name_paths_carry // BILLION
        #     name_paths_carry = name_paths_carry % BILLION
        if entry_leaves > max_leaves:
            max_leaves = entry_leaves
            max_entry = file

    log.close()

    print("Making numbers printable...")
    paths_power = 0
    name_paths_power = 0
    function_paths_power = 0
    while paths > 10:
        paths_power += 1
        paths = paths // 10
    while name_paths > 10:
        name_paths_power += 1
        name_paths = name_paths // 10
    while function_paths > 10:
        function_paths_power += 1
        function_paths = function_paths // 10

    print(
        "-------REPORT-------",
        "\nTotal number of leaves: ",
        leaves,
        f"\n\tOn average {leaves / entries} leaves for each entry.",
        "\n\tMax leaves: ",
        max_leaves,
        "\n\tEntry with most leaves: ",
        max_entry,
        f"\nTotal paths (billions): {paths} * 10^{paths_power}",
        f"\nTotal paths between :name entries (billions): {name_paths} * 10^{name_paths_power}",
        f"\nTotal paths in :function entries (billions): {function_paths} * 10^{function_paths_power}",
        "\n\tTotal :function entries: ",
        functions,
    )


def write_log(message, log=os.path.abspath("./logs/main.txt")):
    with open(log, "a", encoding="utf-8") as LOG:
        print(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M]: "), message, file=LOG)


def string_contains_invalid_context(s: str, pattern: re.Pattern):
    return pattern.search(s)


def invalid_contexts_in_file(path):
    invalid_expression = re.compile("\S+,\S+,\S+,\S+")
    invalid_lines = {}
    num = 0
    print("Validating contexts...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            num += 1
            if string_contains_invalid_context(line, invalid_expression):
                invalid_lines[num] = invalid_expression
            if num % 200 == 0:
                print(".", end="", flush=True)


def add_to_missing_symbols_file(s: str):
    file_name = "missing_symbols.txt"
    with open(file_name, "a", encoding="utf-8") as file:
        print(f"{s}", file=file)

def consolidate_missing_symbols_to_dict():
    missing_symbols = {}
    file_name = "missing_symbols.txt"
    with open(file_name, "r", encoding="utf-8") as file:
        for c in file:
            c = c.replace("\n", "")
            if not c in missing_symbols:
                missing_symbols[c] = c.encode("unicode-escape")
    return missing_symbols

from unicode_to_latex import unicode_to_latex


def replace_unicode_with_latex(s: str):
    """
    Replaces a unicode character with its latex representation.

    Designed to be used after entries_extractor.format_as_label().
    """
    new_s = []
    for c in s:
        if c.isascii():
            new_s.append(c)
        else:
            try:
                new_c = unicode_to_latex[c]
            except KeyError:
                new_c = "???"
                add_to_missing_symbols_file(c)
            new_s.append(new_c.strip())
    return "".join(new_s)


if __name__ == "__main__":
    # for lib in ["stdlib", "TypeTopology", "unimath", "mathlib"]:
    #     print(lib)
    #     entries, network = load_library(lib)
    #     print()
    # split_network_into_nodes_and_links(
    #     f"D:\\Nik\\Projects\\mlfmf-poskusi\\{lib}\\network.csv")

    # print(invalid_contexts_in_file("D:\\Nik\\Projects\\mlfmf-poskusi\\data\\stdlib\\stdlib.train.raw.txt"))

    # _, _ = load_library("stdlib")
    # probibalistic_copy_entries_into_train_val_test_directories("stdlib", 0.2, 0.2)
    # reindex_library_asts("stdlib")

    missing = consolidate_missing_symbols_to_dict()
    for key, value in missing.items():
        decoded_value = value.decode("ascii", "backslashreplace")
        print(f'u"{decoded_value}": "{key}",')