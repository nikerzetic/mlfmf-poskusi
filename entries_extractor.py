import multiprocessing
import itertools
import os
import sys
import datetime
import helpers
import tqdm
import time
import networkx as nx
from argparse import ArgumentParser
from extract import concatenate_dir_files


def hash_string(s: str):
    """
    Mirrors Java String.hashCode() method. To be consistent with original code.
    """
    if not s:
        return 0
    sum = 0
    for c in s:
        sum = 31 * sum + ord(c)  # ord returns the value of char c
    return sum


def hash_path(G: nx.Graph, path: list):
    # path of types
    return hash_string("/".join([G.nodes[id]["type"] for id in path]))
    # return hash_string("/".join(path[1:-1]))
    # TODO: better path embedding


def get_tree_stack(G: nx.DiGraph, node):
    stack = []
    current = [node]
    while current:
        stack.append(current[0])
        current = list(G.pred[current[0]].keys())  # TODO more ellegant
    return stack


def find_common_prefix(stack_u: list, stack_v: list):
    common_prefix = 0
    current_ancestor_index_u = len(stack_u) - 1
    current_ancestor_index_v = len(stack_v) - 1
    while (
        current_ancestor_index_u >= 0
        and current_ancestor_index_v >= 0
        and stack_u[current_ancestor_index_u] == stack_v[current_ancestor_index_v]
    ):
        common_prefix += 1
        current_ancestor_index_u -= 1
        current_ancestor_index_v -= 1
    return common_prefix, current_ancestor_index_u, current_ancestor_index_v


def path_length(stack_u: list, stack_v: list, common_prefix: int):
    return len(stack_u) + len(stack_v) - 2 * common_prefix


def path_width(
    stack_u: list,
    stack_v: list,
    current_ancestor_index_u: int,
    current_ancestor_index_v: int,
):
    return stack_u[current_ancestor_index_u] - stack_v[current_ancestor_index_v]


def generate_path(G: nx.DiGraph, s: int, t: int, max_length=None, max_width=None):
    start_symbol = "("
    end_symbol = ")"
    up_symbol = "^"
    down_symbol = "_"

    stack_s = get_tree_stack(G, s)
    stack_t = get_tree_stack(G, t)
    common_prefix, i_s, i_t = find_common_prefix(stack_s, stack_t)

    if path_length(stack_s, stack_t, common_prefix) > max_length:
        return None
    if path_width(stack_s, stack_t, i_s, i_t) > max_width:
        return None

    path = []
    for i in range(1, len(stack_s) - common_prefix):
        current_id = stack_s[i]
        child_id = stack_s[i - 1]
        # parent_id = stack_s[i + 1]  # TODO FeatureExtractor line 158
        node_type = G.nodes[current_id]["type"]
        path.append(start_symbol + node_type + str(child_id) + end_symbol + up_symbol)

    for i in [len(stack_s) - common_prefix]:  # TODO this is ugly - for symetry
        current_id = stack_s[i]
        child_id = stack_s[i - 1]
        node_type = G.nodes[current_id]["type"]
        path.append(start_symbol + node_type + str(child_id) + end_symbol)

    for i in range(len(stack_t) - common_prefix - 1, 0, -1):
        current_id = stack_t[i]
        child_id = stack_t[i - 1]
        node_type = G.nodes[current_id]["type"]
        path.append(down_symbol + start_symbol + node_type + str(child_id) + end_symbol)

    return "".join(path)  # Join is faster than +, because str is immutable


def generate_path_features_for_function(
    G: nx.DiGraph, leaves: list[str], max_path_length, max_path_width
):
    features = []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            s = G.nodes[leaves[i]]
            t = G.nodes[leaves[j]]
            path = generate_path(
                G, leaves[i], leaves[j], max_path_length, max_path_width
            )
            if not path:
                continue
            features.append(
                s["desc"] + "," + str(hash_string(path)) + "," + t["desc"]
            )  # TODO here we define the path separator
            # TODO should be source, path, sink
    return features


def format_as_label(s: str):
    new_s = ""
    for c in s:
        if c == "." or c == " " or c == "_":
            new_s += "|"
        elif c.isupper():
            new_s += "|" + c.lower()
        else:
            new_s += c
    new_s = new_s.replace("||", "|")
    # TODO remove "|" preceding the string
    return new_s


def extract_graph(file_path):
    children = {}
    G = nx.DiGraph()
    leaves = []
    name = None
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
        G.add_node(
            node_id,
            type=node_type.replace(":", ""),
            desc=format_as_label(node_description),
        )
        if node_type == ":name" and not node_children:
            leaves.append(node_id)
        if name:
            continue
        # TODO make sure the first :name node is the name of the function
        if node_type == ":name":
            name = format_as_label(node_description)
    file.close()
    for node_id, children_ids in children.items():
        for child_id in children_ids:
            G.add_edge(node_id, child_id)
    return G, leaves, name


# mirror original extractSingleFile
def extract_single_entry_file(file_path, args):
    graph, leaves, name = extract_graph(file_path)
    separator = " "
    # if args.pretty_print:
    #     separator = "\n\t"
    features = generate_path_features_for_function(
        graph, leaves, int(args.max_path_length), int(args.max_path_width)
    )
    # TODO separators as args
    if features:
        return name + separator + separator.join(features)
    return ""  # TODO should log how many were skipped


def extract_file_features(file, args):
    """
    This only works with our Entries, which are each a sepparate file
    """
    if not file.endswith(".dag"):
        return
    entry_file = os.path.join(args.dir, file)
    tmp_file = os.path.join(args.tmpdir, str(os.getpid()))
    LOG_FILE = os.path.join(args.logdir, str(os.getpid()))
    helpers.write_log("Extracting " + file.replace(".dag", ""), LOG_FILE)
    start_time = time.time()
    with open(tmp_file, "a", encoding="utf-8") as tmp:
        to_print = extract_single_entry_file(entry_file, args)
        if not to_print:
            helpers.write_log("\tNo appropriate paths", LOG_FILE)
            # TODO should log how many were skipped
        print(
            to_print,
            file=tmp,
        )
    elapsed = time.time() - start_time
    helpers.write_log(f"\tDone in {round(elapsed,4)} s", LOG_FILE)
    # TODO print to file


def extract_dir(args):
    start = time.time()
    helpers.write_log("Extracting " + args.dir + f" with total files {len(os.listdir(args.dir))}")
    try:
        with multiprocessing.get_context("spawn").Pool(int(args.num_threads)) as p:
            result = p.starmap_async(
                extract_file_features,
                zip(
                    os.listdir(args.dir),
                    itertools.repeat(args),
                    # itertools.repeat(tmp_dir),
                ),
            )
            result.get(timeout=None)
    except Exception as e:
        print(
            f"Exception while extracting dir {args.dir}: ", e
        )  # TODO file parameter for logs
    stop = time.time()
    helpers.write_log("Done extracting " + args.dir + f" in {round((stop - start)/60)} min")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-maxlen",
        "--max_path_length",
        dest="max_path_length",
        required=False,
        default=16,
    )
    parser.add_argument(
        "-maxwidth",
        "--max_path_width",
        dest="max_path_width",
        required=False,
        default=2,
    )
    parser.add_argument(
        "-threads", "--num_threads", dest="num_threads", required=False, default=64
    )
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)
    parser.add_argument("-tmpdir", "--tmpdir", dest="tmpdir", required=False)
    parser.add_argument("-logdir", "--logdir", dest="logdir", required=False)

    args = parser.parse_args()

    if args.file:
        pass  # TODO
    elif args.dir:
        extract_dir(args)
    else:
        # Debug
        args.dir = ".\\stdlib\\code2vec\\train\\"
        args.tmpdir = "D:\\Nik\\Projects\\mlfmf-poskusi\\tmp\\debug\\"
        extract_dir(args)
