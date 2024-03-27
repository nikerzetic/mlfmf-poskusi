import multiprocessing
import itertools
import os
import sys
import tqdm
import networkx as nx
from argparse import ArgumentParser


def hash_string(s: str):
    """
    Mirrors Java String.hashCode() method. To be consistent with original code.
    """
    if not s:
        return 0
    sum = 0
    for c in s:
        sum = 31 * sum + ord(c) # ord returns the value of char c
    return sum


def hash_path(G: nx.Graph, path: list): 
    return hash_string("/".join([G.nodes[id]["type"] for id in path])) # path of types
    # return hash_string("/".join(path[1:-1]))
    #TODO: better path embedding


def generate_path_features_for_function(G: nx.Graph, leaves: list[str]):
    features = []
    for i in range(len(leaves)):
        for j in range(i+1, len(leaves)):
            s = G.nodes[leaves[i]]
            t = G.nodes[leaves[j]]
            # print(
            #     "Source: ", s,
            #       "Sink: ", t,
            #       )
            path = nx.shortest_path(G, source=leaves[i], target=leaves[j])
            features.append(
                s["desc"] + "," + str(hash_path(G, path)) + "," + t["desc"]
                ) #TODO here we define the path separator
            #TODO should be source, path, sink
    return features


def format_as_label(s: str):
    # s.encode("unicode-escape").decode("unicode-escape")
    new_s = ""
    for c in s:
        if c == "." or c == " " or c == "_":
            new_s += "|"
        elif c.isupper():
            new_s += "|" + c.lower()
        else:
            new_s += c
    new_s = new_s.replace("||", "|")
    #TODO remove "|" preceding the string
    return new_s


def extract_graph(file_path):
    # print("Extracting graph: ", file_path)
    children = {}
    G = nx.Graph()
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
                    desc=format_as_label(node_description)
                    )
        if not node_children:
            leaves.append(node_id)
        if name:
            continue
        #TODO make sure the first :name node is the name of the function
        if node_type == ":name":
            name = format_as_label(node_description)
    file.close()
    for node_id, children_ids in children.items():
        for child_id in children_ids:
            G.add_edge(node_id, child_id)
    return G, leaves, name


def extract_single_entry_file(file_path, args): # mirror original extractSingleFile
    # print("Extracting file: ", file_path)
    graph, leaves, name = extract_graph(file_path)
    # print("Extracted graph: ", graph)
    separator = " "
    # if args.pretty_print:
    #     separator = "\n\t"
    # print("Generating path features for: ", name)
    features = generate_path_features_for_function(graph, leaves)
    return name + separator + separator.join(features) #TODO separators as args


def extract_file_features(file, args):
    """
    This only works with our Entries, which are each a sepparate file
    """
    if not file.endswith(".dag"):
        return
    entry_file = os.path.join(args.dir, file)
    # LOGS = open(os.path.abspath("D:\\Nik\\Projects\\mlfmf-poskusi\\LOGS.txt"), "a", encoding="utf-8")
    print(
        extract_single_entry_file(entry_file, args),
        # file=LOGS
        )
    # LOGS.close()
    #TODO print to file


def extract_dir(args):
    try:
        with multiprocessing.Pool(4) as p:
            result = p.starmap_async(extract_file_features, zip(os.listdir(args.dir),
                itertools.repeat(args), 
                # itertools.repeat(tmp_dir), 
                ))
            result.get(timeout=None)
    except Exception as e:
        print(e) #TODO file parameter for logs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-maxlen", "--max_path_length",
                        dest="max_path_length", required=False, default=8)
    parser.add_argument("-maxwidth", "--max_path_width",
                        dest="max_path_width", required=False, default=2)
    parser.add_argument("-threads", "--num_threads",
                        dest="num_threads", required=False, default=64)
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)

    args = parser.parse_args()

    if args.file:
        pass  # TODO
    elif args.dir:
        extract_dir(args)
    else:
        # Debug
        args.dir = ".\\stdlib\\code2vec\\train\\"
        extract_dir(args)
        
