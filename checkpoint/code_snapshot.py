import zipfile
from pathlib import Path
from typing import Union, List, Optional

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_info, rank_zero_only


class CodeSnapshot(Callback):

    DEFAULT_FILENAME = "code.zip"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = ".",
        output_file: Optional[Union[str, Path]] = "./someplace/hello.zip",
        filetype: Union[str, List[str]] = ".py",
    ):
        """
        Callback that takes a snapshot of all source files and saves them to a ZIP file.
        By default, the file is saved to the folder where checkpoints are saved, i.e., the dirpath
        of ModelCheckpoint.

        Arguments:
            root: the root folder containing the files for collection
            output_file: path to zip file, e.g., "path/to/code.zip"
            filetype: list of file types, e.g., ".py", ".txt", etc.
        """
        self._root = root
        self._output_file = output_file
        self._filetype = filetype

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if not self._output_file and not trainer.checkpoint_callback:
            rank_zero_warn(
                "Trainer has no checkpoint callback and output file where to save snapshot was not specified."
                " Code snapshot not saved!"
            )
            return
        self._output_file = self._output_file or Path(
            trainer.checkpoint_callback.dirpath, CodeSnapshot.DEFAULT_FILENAME
        )
        self._output_file = Path(self._output_file).absolute()
        snapshot_files(
            root=self._root, output_file=self._output_file, filetype=self._filetype
        )
        rank_zero_info(
            f"Code snapshot saved to {self._output_file.relative_to(Path.cwd())}"
        )


def snapshot_files(
    root: Union[str, Path] = ".",
    output_file: Union[str, Path] = "code.zip",
    filetype: Union[str, List[str]] = ".py",
):
    """
    Collects all source files in a folder and saves them to a ZIP file.

    Arguments:
        root: the root folder containing the files for collection
        output_file: path to zip file, e.g., "path/to/code.zip"
        filetype: list of file types, e.g., ".py", ".txt", etc.
    """
    root = Path(root).absolute()
    output_file = Path(output_file).absolute()
    output_file.parent.mkdir(exist_ok=True, parents=True)
    suffixes = [filetype] if isinstance(filetype, str) else filetype

    zip_file = zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED)

    for p in root.rglob("*"):
        if p.suffix in suffixes:
            zip_file.write(p.relative_to(root))

    zip_file.close()
