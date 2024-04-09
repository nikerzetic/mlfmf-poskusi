#!/usr/bin/python

import itertools
import multiprocessing
import os
import shutil
import subprocess
import datetime
import helpers
from threading import Timer
from argparse import ArgumentParser


def get_immediate_subdirectories(a_dir):
    return [
        (os.path.join(a_dir, name))
        for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


def ParallelExtractDir(args, tmpdir, logdir, dir_):
    ExtractFeaturesForDir(args, tmpdir, logdir, dir_, "")


def concatenate_dir_files(
    dir, out_file=None
):  # TODO move to helpers so both extract and entries_extractor can access
    to_path = ""
    if out_file:
        to_path = f" > {out_file}"
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(os.path.abspath(dir), file)
        os.system(f"type {path}{to_path}")
        os.remove(path)
    shutil.rmtree(dir)


def ExtractFeaturesForDir(args, tmpdir, logdir, dir_, prefix):
    """Recursively extract features from Entry files in the directory."""

    def kill(process):
        return process.kill()

    outputFileName = tmpdir + prefix + os.path.basename(os.path.abspath(dir_))
    out_dir = tmpdir + prefix + os.path.basename(os.path.abspath(dir_)) + "_files"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    command = [
        "python",
        "entries_extractor.py",
        "--max_path_length",
        str(args.max_path_length),
        "--max_path_width",
        str(args.max_path_width),
        "--dir",
        dir_,
        "--num_threads",
        str(args.num_threads),
        "--tmpdir",
        out_dir,
        "--logdir",
        logdir,
    ]

    failed = False
    with open(outputFileName, "a", encoding="utf-8") as outputFile:
        sleeper = subprocess.Popen(
            command, stdout=outputFile, stderr=subprocess.PIPE, encoding="utf-8"
        )
        timer = Timer(600000, kill, [sleeper])
        try:
            timer.start()
            stdout, stderr = sleeper.communicate()
        finally:
            timer.cancel()

        if sleeper.poll() != 0:
            failed = True
            subdirs = get_immediate_subdirectories(dir_)
            for subdir in subdirs:
                ExtractFeaturesForDir(args, subdir, prefix + dir_.split("/")[-1] + "_")

        helpers.write_log("Concatenating out_dir" + out_dir)
        concatenate_dir_files(out_dir)

    if failed and os.path.exists(outputFileName):
        os.remove(outputFileName)


def ExtractFeaturesForDirsList(args, dirs):
    tmp_dir = f"./tmp/feature_extractor{os.getpid()}/"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    logdir = os.path.abspath(
        "./logs/extract_"
        + os.path.basename(os.path.abspath(args.dir))
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        + "/"
    )
    if os.path.exists(logdir):
        shutil.rmtree(logdir, ignore_errors=True)
    os.makedirs(logdir)

    helpers.write_log(f"Total dirs: {len(dirs)} (batches of {args.batch_size})")
    for i in range(0, len(dirs), args.batch_size):
        helpers.write_log(f"Iteration {i}")
        batch_dirs = dirs[i : i + args.batch_size]
        files_num = sum([len(os.listdir(d)) for d in batch_dirs])
        timeout_seconds = 3 * files_num  # timeout setting
        #TODO this is so ugly; a fixed timeot that always happens in our case + working through console commands that leaves the process running
        helpers.write_log(f"Will get results in {round(timeout_seconds / 60)} min.")
        try:
            with multiprocessing.get_context("spawn").Pool(4) as p:
                result = p.starmap_async(
                    ParallelExtractDir,
                    zip(itertools.repeat(args), itertools.repeat(tmp_dir), itertools.repeat(logdir), batch_dirs),
                )
                result.get(timeout=timeout_seconds)
        except multiprocessing.TimeoutError:
            helpers.write_log("Timeout error")
            continue

        helpers.write_log("Concatenating temp_dir " + tmp_dir)
        concatenate_dir_files(tmp_dir)
        # concatenate_dir_files(logdir, f"{os.path.dirname(logdir)}.txt")


if __name__ == "__main__":
    parser = ArgumentParser()
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

    args = parser.parse_args()
    helpers.write_log("*******************************New Run*********************************")

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
        subdirs = get_immediate_subdirectories(args.dir)
        to_extract = subdirs if subdirs else [args.dir.rstrip("/")]
        ExtractFeaturesForDirsList(args, to_extract)
