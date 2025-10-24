import collections
import enum
from pathlib import Path

# Data for one entry in the git index (.git/index)
IndexEntry = collections.namedtuple('IndexEntry', [
    'ctime_s', 'ctime_n', 'mtime_s', 'mtime_n', 'dev', 'ino', 'mode', 'uid',
    'gid', 'size', 'sha1', 'flags', 'path',
])

class ObjectType(enum.Enum):
    commit = 1
    tree = 2
    blob = 3

# Return file contents from `path` as raw bytes.
def read_file(path):
    return Path(path).read_bytes()

# Write `data` bytes to `path`.
def write_file(path, data):
    Path(path).write_bytes(data)

# Create a new repository skeleton rooted at `repo`.
def init(repo):
    def make_dir(path):
        # Mirror Git's directory layout under repo/.git.
        path.mkdir()
        return path

    repo_path = make_dir(Path(repo))
    git_dir = make_dir(repo_path / '.git')
    for name in ['objects', 'refs', 'refs/heads']:
        make_dir(git_dir / name)
    write_file(git_dir / 'HEAD', b'ref: refs/heads/master')
    print(f'initialized empty repository: {repo_path}')