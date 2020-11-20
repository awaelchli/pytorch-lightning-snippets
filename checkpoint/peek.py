import code
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


class COLORS:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


PRIMITIVE_TYPES = (int, float, bool, str)


def pretty_print(contents: dict):
    """ Prints a nice summary of the top-level contens in a checkpoint dictionary. """
    col_size = max(len(str(k)) for k in contents)
    for k, v in sorted(contents.items()):
        key_length = len(str(k))
        line = " " * (col_size - key_length)
        line += f"{k}: {COLORS.BLUE}{type(v).__name__}{COLORS.END}"
        if isinstance(v, PRIMITIVE_TYPES):
            line += f" = "
            line += f"{COLORS.CYAN}{repr(v)}{COLORS.END}"
        elif isinstance(v, Sequence):
            line += ", "
            line += f"{COLORS.CYAN}len={len(v)}{COLORS.END}"
        elif isinstance(v, torch.Tensor):
            line += ", "
            line += f"{COLORS.CYAN}shape={list(v.shape)}{COLORS.END}"
            line += ", "
            line += f"{COLORS.CYAN}dtype={v.dtype}{COLORS.END}"
        print(line)


def get_attribute(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return obj[name]
    if isinstance(obj, Namespace):
        return obj.name
    return getattr(object, name)


def peek(args):
    file = Path(args.file).absolute()
    ckpt = torch.load(file, map_location=torch.device("cpu"))
    selection = dict()
    attribute_names = args.attributes or list(ckpt.keys())
    for name in attribute_names:
        parts = name.split("/")
        current = ckpt
        for part in parts:
            current = get_attribute(current, part)
        selection.update({name: current})
    pretty_print(selection)


def main():
    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("attributes", nargs="*")
    args = parser.parse_args()
    peek(args)


if __name__ == "__main__":
    main()
