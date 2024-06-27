from typing import List

from tqdm.auto import tqdm


def count_lines_in_file(file_path: str, buffer_size: int = 1024 * 1024) -> int:
    """Count the number of lines in the given file.

    :param file_path: path to the file
    :param buffer_size: size of temporary buffer during reading the file
    :return: number of lines
    """
    n_lines = 0
    with open(file_path, "rb") as file:
        file_reader = file.read
        buffer = file_reader(buffer_size)
        while buffer:
            n_lines += buffer.count(b"\n")
            buffer = file_reader(buffer_size)
    return n_lines


def get_lines_offsets(file_path: str, show_progress_bar: bool = True) -> List[int]:
    """Calculate cumulative offsets for all lines in the given file.

    :param file_path: path to the file
    :param show_progress_bar: if True then tqdm progress bar will be display
    :return: list of ints with cumulative offsets
    """
    line_offsets: List[int] = []
    cumulative_offset = 0
    # HACK: we must read the file as bytes to avoid incorrectly reading files that were produced on Windows
    with open(file_path, "rb") as file:
        file_iter = tqdm(file, total=count_lines_in_file(file_path)) if show_progress_bar else file
        for line in file_iter:
            line_offsets.append(cumulative_offset)
            cumulative_offset += len(line) #BUG: Windows encodes new line with two characters
    return line_offsets


def get_line_by_offset(file_path: str, offset: int) -> str:
    """Get line by byte offset from the given file.

    :param file_path: path to the file
    :param offset: byte offset
    :return: read line
    """
    with open(file_path, "r") as data_file:
        data_file.seek(offset)
        line = data_file.readline().strip("\n")
    return line
