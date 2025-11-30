import tempfile
import time
from pathlib import Path
from typing import ClassVar

from typing_extensions import Buffer


class ContentSnapshotStore:
    """Content snapshot store.

    :param base_path: The base path where the snapshots are stored.
    :param content_snapshots_num: The number of snapshots to keep.
    :param temporary: If temporary is True, the snapshots are deleted when the store is deleted.
    """

    DEFAULT_SNAPSHOTS_NUM: ClassVar[int] = 2

    def __init__(
        self,
        base_path: str = None,
        content_snapshots_num: int = DEFAULT_SNAPSHOTS_NUM,
        temporary: bool = True,
    ) -> None:
        # Use tmp file lib if not base_path is provided
        self._base_path = Path(base_path) if base_path else Path(tempfile.gettempdir())
        self._content_snapshots_num = content_snapshots_num
        self._snapshots: dict[str, list[Path]] = {}
        self._temporary = temporary

    def __del__(self):
        if not self._temporary:
            return
        try:
            for content_id, snapshots in self._snapshots.items():
                for snapshot in snapshots:
                    snapshot.unlink()
        except AttributeError:
            pass

    def load_snapshot(
        self, content_id: str, timestamp: str | None = None
    ) -> str | None:
        """Load a content snapshot from the store.

        :param content_id: The content ID.
        :param timestamp: The timestamp of the snapshot.
        :return: The content snapshot.
        """
        if content_id not in self._snapshots:
            return None

        content_path: Path | None = None
        if timestamp:
            for snapshot in self._snapshots[content_id]:
                if snapshot.name == timestamp:
                    content_path = snapshot
        else:
            content_path = self._snapshots[content_id][-1]

        if content_path:
            return content_path.read_text()
        return None

    def store_snapshot(
        self,
        content_id: str,
        content: str | Buffer,
        file_ext: str,
        timestamp: str | None = None,
    ) -> Path:
        """Save a content snapshot and register it in the store.

        :param content_id: The content ID.
        :param content: The content to save.
        :param file_ext: The file extension of the content.
        :param timestamp: The timestamp of the snapshot (ns)
        """
        timestamp = timestamp or str(int(time.time_ns()))
        content_path = self._generate_content_path(content_id, file_ext, timestamp)
        self._store_content_snapshot(content, content_path)
        self._register_snapshot(content_id, content_path)
        return content_path

    def _register_snapshot(self, content_id: str, content_path: Path):
        """Register a snapshot in the store ensuring order and that the number of snapshots does
        not exceed the limit.
        """
        self._snapshots.setdefault(content_id, []).append(content_path)
        if len(self._snapshots[content_id]) > self._content_snapshots_num:
            self._snapshots[content_id].pop(0)

    @staticmethod
    def _store_content_snapshot(content: str | Buffer, content_path: Path) -> None:
        # Create parent directories if they don't exist
        content_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content_path.write_text(content)
        else:
            content_path.write_bytes(content)

    def _generate_content_path(
        self, content_id: str, file_ext: str, timestamp: str | None
    ) -> Path:
        content_path = (
            self._base_path / content_id / f"{content_id}-{timestamp}.{file_ext}"
        )
        return content_path
