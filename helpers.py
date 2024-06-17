import networkx as nx
import re
import tqdm
import os
import zipfile
import shutil
import random
import datetime
import sys
import json
import entries_extractor as ee
import multiprocessing as mp
import numpy as np


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


def load_entries(entry_dir: str) -> list[Entry]:
    entries = []
    print("Loading entries ...")
    for file in tqdm.tqdm(os.listdir(entry_dir)):
        if not file.endswith(".dag"):
            continue
        entry_file = os.path.join(entry_dir, file)
        entries.append(load_entry(entry_file))
    print(f"Loaded {len(entries)} entries.")
    return entries


def load_graph(graph_file: str) -> nx.MultiDiGraph:
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
    """
    Returns the following paths (using os.path.join):
    - "data/raw/library_name/entries"
    - "data/raw/library_name/entries.zip"
    - "data/raw/library_name/network.csv"
    """
    dir = os.path.abspath(os.path.join("data", "raw", library_name))
    entry_dir = os.path.join(dir, "entries")
    zip_file = os.path.join(dir, "entries.zip")
    network_file = os.path.join(dir, "network.csv")
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


def load_library(library_name: str) -> tuple[list[Entry], nx.MultiDiGraph]:
    """
    ## Returns
    - entries
    - network
    """
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


def add_children_to_graph(G: nx.DiGraph, e: EntryNode):
    for child in e.children:
        add_children_to_graph(G, child)

    G.add_node(e.id, type=e.type.replace(":", ""), desc=e.description)

    for child in e.children:
        G.add_edge(e.id, child.id)


def entry_to_dag(entry: Entry) -> nx.DiGraph:
    G = nx.DiGraph()
    add_children_to_graph(G, entry.root)
    return G


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


def generate_report(library_path: str):
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
    entries_dir = os.path.join(library_path, "entries")
    log = open(os.path.join(library_path, "entries_stats.tsv"), "w", encoding="utf-8")
    log.write("file_name\tentry_type\tnum_nodes\tnum_edges\tnum_leaves\tnum_names")

    labels_dict = {}
    with open(os.path.join(library_path, "nodes.tsv"), "r", encoding="utf-8") as f:
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
    log_dir = os.path.dirname(log)
    os.makedirs(log_dir, exist_ok=True)
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


def create_dictionaries(library_name: str, save_to_file: bool = False):
    """
    Creates dictionaries corresponding to `entries_extractor.format_as_label`

    ## Parameters
    - library_name: the name of the library in root folder
    - save_to_file: if set to true, saves dictionaries to
        - `library_name/dictionaries/raw2label.json` and
        - `library_name/dictionaries/label2raw.json`

    ## Returns
    - raw2label: dictionary for converting original names to labels
    - label2raw: dictionary for converting labels to original names
    """
    print(f"Creating dictionaries for {library_name}...")
    raw2label = {}
    label2raw = {}
    with open(os.path.join(library_name, "nodes.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            name = line.split("\t")[0]
            label = replace_unicode_with_latex(name).split(" ")[0]
            raw2label[name] = label
            label2raw[label] = name
    if save_to_file:
        DICT_PATH = os.path.join(library_name, "dictionaries")
        os.makedirs(DICT_PATH, exist_ok=True)
        with open(
            os.path.join(DICT_PATH, "raw2label.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(raw2label, f)
        with open(
            os.path.join(DICT_PATH, "label2raw.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(label2raw, f)
    return raw2label, label2raw


def tokenization(library: str, save_to_file: bool = False):
    """
    Creates tokenization dictionaries for converting types to id.
    
    ## Parameters
    - library: library name; path automatically configured to ./data/raw/library
    - save_to_file: default False; if set to True, the dictionaries will be saved to
        - ./library/dictionaries/type2id.json
        - ./library/dictionaries/id2type.json

    ## Returns
    - type2id
    - id2type
    """
    counter = 0
    type2id = {}
    id2type = {}
    entries, _ = load_library(library)
    for entry in tqdm.tqdm(entries):
        G = entry_to_dag(entry)
        for node in G.nodes:
            type = G.nodes[node]["type"]
            if not type in type2id:
                counter += 1
                type2id[type] = counter
                id2type[counter] = type
    if save_to_file:
        dict_path = os.path.join("data", "raw", library, "dictionaries")
        os.makedirs(dict_path, exist_ok=True)
        with open(os.path.join(dict_path, "type2id.json"), "w", encoding="utf-8") as f:
            json.dump(type2id, f)
        with open(os.path.join(dict_path, "id2type.json"), "w", encoding="utf-8") as f:
            json.dump(id2type, f)
    return type2id, id2type


def extract_tokens_from_dag(entry: Entry, D: dict[dict[str, int]]):
    G = entry_to_dag(entry)
    for node in G.nodes:
        node_type = G.nodes[node]["type"]
        if not node_type in D["type2id"]:
            D["counter"] += 1
            id = D["counter"]
            D["type2id"][node_type] = id
            D["id2type"][id] = node_type


# Somehow, this is slower than tokenization
def create_tokenization_dictionaries(library_name: str, save_to_file: bool = False):
    """
    Creates a tokenization dictionary that replaces each token with a unique id

    ## Parameters
    - library_name: the name of the library in root folder
    - save_to_file: if set to true, saves dictionaries to
        - `library_name/dictionaries/type2id.json` and
        - `library_name/dictionaries/id2type.json`

    ## Returns
    - type2id:
    - id2type:
    """
    entries_dir = os.path.join(library_name, "entries")
    entries = load_entries(entries_dir)
    print(f"Creating token dictionaries for {library_name}...")
    manager = mp.Manager()
    dictionaries = manager.dict()
    type2id = manager.dict()
    id2type = manager.dict()
    dictionaries["counter"] = 0
    dictionaries["type2id"] = type2id
    dictionaries["id2type"] = id2type
    pool = mp.get_context("spawn").Pool(32)

    jobs = []
    for entry in entries:
        job = pool.apply_async(extract_tokens_from_dag, (entry, dictionaries))
        jobs.append(job)

    for job in tqdm.tqdm(jobs):
        job.get()

    pool.close()
    pool.join()

    print(f"Unique types: {dictionaries['counter']}")

    type2id = dictionaries["type2id"]
    id2type = dictionaries["id2type"]

    if save_to_file:
        DICT_PATH = os.path.join(library_name, "dictionaries")
        os.makedirs(DICT_PATH, exist_ok=True)
        with open(os.path.join(DICT_PATH, "type2id.json"), "w", encoding="utf-8") as f:
            json.dump(type2id, f)
        with open(os.path.join(DICT_PATH, "id2type.json"), "w", encoding="utf-8") as f:
            json.dump(id2type, f)
    return type2id, id2type


def count_tokens_in_file(file_path: str):
    max_parts = {"label": 0, "token": 0, "path": 0}
    max_values = {"label": "", "token": "", "path": ""}
    total_each = {"label": [], "token": [], "path": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            parts = line.strip("\n").split(" ")
            label_parts = parts[0].count("|")
            total_each["label"].append(label_parts)
            if label_parts > max_parts["label"]:
                max_parts["label"] = label_parts
                max_values["label"] = parts[0]
            for context in parts[1:]:
                if context == "":
                    break
                subparts = context.split(",")

                token_parts = subparts[0].count("|")
                if token_parts > max_parts["token"]:
                    max_parts["token"] = token_parts
                    max_values["token"] = subparts[0]
                total_each["token"].append(token_parts)

                path_nodes = subparts[1].count("|")
                if path_nodes > max_parts["path"]:
                    max_parts["path"] = path_nodes
                    max_values["path"] = subparts[1]
                total_each["path"].append(path_nodes)

                token_parts = subparts[2].count("|")
                if token_parts > max_parts["token"]:
                    max_parts["token"] = token_parts
                    max_values["token"] = subparts[2]
                total_each["token"].append(token_parts)
    print(
        "Max parts:",
        "\n\tLabel:", max_parts["label"],
        "\n\t\t", max_values["label"],
        "\n\tToken:", max_parts["token"],
        "\n\t\t", max_values["token"],
        "\n\tPath:", max_parts["path"],
        "\n\t\t", max_values["path"],
        "\nMean parts:",
        "\n\tLabel:", np.mean(total_each["label"]),
        "\n\tToken:", np.mean(total_each["token"]),
        "\n\tPath:", np.mean(total_each["path"]),
        "\nMedian parts:",
        "\n\tLabel:", np.median(total_each["label"]),
        "\n\tToken:", np.median(total_each["token"]),
        "\n\tPath:", np.median(total_each["path"]),
          )



if __name__ == "__main__":
    # for lib in ["stdlib", "TypeTopology", "unimath", "mathlib"]:
    #     print(lib)
    #     entries, network = load_library(lib)
    #     print()
        # split_network_into_nodes_and_links(
        #     f"D:\\Nik\\Projects\\mlfmf-poskusi\\{lib}\\network.csv")

    # print(invalid_contexts_in_file("D:\\Nik\\Projects\\mlfmf-poskusi\\data\\stdlib\\stdlib.train.raw.txt"))

    # _, _ = load_library("stdlib")
    # reindex_library_asts("stdlib")
    # probibalistic_copy_entries_into_train_val_test_directories("stdlib", 0.1, 0.1)

    # missing = consolidate_missing_symbols_to_dict()
    # for key, value in missing.items():
    #     decoded_value = value.decode("ascii", "backslashreplace")
    #     print(f'u"{decoded_value}": "{key}",')

    # create_dictionaries("stdlib", True)
    # tokenization("stdlib", True)
    # count_tokens_in_file("data/code2seq/stdlib/predict.c2s")
    split_network_into_nodes_and_links("data/raw/stdlib/network.csv")
    generate_report("data/raw/stdlib")

