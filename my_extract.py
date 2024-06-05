"""
Reimplementation of *code2vec* `extract.py` function.
"""

import os
import argparse
import helpers
import shutil
import time
import json
import tqdm
import entries_extractor as ee
import multiprocessing as mp


def single_thread_extract_dir(dir, args):
    with open(os.path.join("stdlib", "dictionaries", "type2id.json"), "r", encoding="utf-8") as f:
        type2id = json.load(f)
    
    to_print = []
    for file in tqdm.tqdm(os.listdir(args.dir)):
        file_path = os.path.join(args.dir, file)
        result = ee.extract_entry_file(file_path, args, type2id)
        to_print.append(result)

    print("Writing output to file...")
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.writelines(to_print)


def new_extract_dir(dir, args):
    with open(os.path.join(dir, "dictionaries", "type2id.json"), "r", encoding="utf-8") as f:
        type2id = json.load(f)

    #TODO: batch load entries if neccessary
    entries_dir = os.path.join(dir, "entries")
    entries = helpers.load_entries(entries_dir)

    manager = mp.Manager()
    q = manager.Queue()
    type_dict = manager.dict(type2id)

    pool = mp.get_context("spawn").Pool(
        int(args.num_threads)
    )  # TODO: number of processes
    writer = pool.apply_async(write_queue_to_file, (args.out_file, q))

    jobs = []
    for file in os.listdir(args.dir):
        job = pool.apply_async(extract_entry_file_to_queue, (file, q, args, type_dict))
        jobs.append(job)

    for job in tqdm.tqdm(jobs):
        job.get()

    q.put("\t\t\tkill")
    pool.close()
    pool.join()


def get_immediate_subdirectories(a_dir):
    return [
        (os.path.join(a_dir, name))
        for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


def concatenate_dir_files(
    dir, out_file=None
):  # TODO move to helpers so both extract and entries_extractor can access
    to_path = ""
    if out_file:
        to_path = f" > {out_file}"
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(os.path.abspath(dir), file)
        os.system(
            f"type {path}{to_path}"
        )  # TODO this only writes the last file to out_file
        os.remove(path)
    shutil.rmtree(dir)


def write_queue_to_file(output_file: str, q: mp.Queue):
    """
    Parallel output file writer
    """
    f = open(output_file, "w", encoding="utf-8")
    while True:
        s = q.get()
        if s == "\t\t\tkill":
            break
        f.write(s)
        # f.flush()
    f.close()


def extract_entry_file_to_queue(file: str, q: mp.Queue, args, type_dict: dict):
    if not file.endswith(".dag"):
        return
    file_path = os.path.join(args.dir, file)
    result = ee.extract_entry_file(file_path, args, type_dict)
    q.put(result)


def extract_dir(dir, args):  # TODO recursive extract dir
    # TODO: dictionary
    with open(os.path.join(dir, "dictionaries", "type2id.json"), "r", encoding="utf-8") as f:
        type2id = json.load(f)
    subdirs = get_immediate_subdirectories(dir)
    to_extract = (
        subdirs if subdirs else [args.dir.rstrip("/")]
    )  # TODO: preprocessing subdirs

    manager = mp.Manager()
    q = manager.Queue()
    type_dict = manager.dict(type2id)

    pool = mp.get_context("spawn").Pool(
        int(args.num_threads)
    )  # TODO: number of processes
    writer = pool.apply_async(write_queue_to_file, (args.out_file, q))

    jobs = []
    for file in os.listdir(args.dir):
        job = pool.apply_async(extract_entry_file_to_queue, (file, q, args, type_dict))
        jobs.append(job)

    for job in tqdm.tqdm(jobs):
        job.get()

    q.put("\t\t\tkill")
    pool.close()
    pool.join()


def main(args):
    start = time.time()  # TODO replace with logger
    helpers.write_log(
        "Extracting " + args.dir + f" with total files {len(os.listdir(args.dir))}"
    )

    single_thread_extract_dir(args.dir, args)

    stop = time.time()
    helpers.write_log(f"Done extracting {args.dir} in {round((stop - start)/60)} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-maxlen",
        "--max_path_length",
        dest="max_path_length",
        required=False,
        default=8,
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
    # TODO: change to something more appropriate
    parser.add_argument("-j", "--jar", dest="jar", required=False)
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        dest="batch_size",
        required=False,
        default=3,
        type=int,
    )
    parser.add_argument("-out_file", "--out_file", dest="out_file", required=True)

    args = parser.parse_args()
    helpers.write_log(
        "*******************************New Run*********************************"
    )

    if args.file is not None:
        command = (
            "python "
            + " entries_extractor.py --max_path_length "
            + str(args.max_path_length)
            + " --max_path_width "
            + str(args.max_path_width)
            + " --file "
            + args.file
        )
        os.system(command)
    elif args.dir is not None:
        main(args)
