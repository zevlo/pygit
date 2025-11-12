import argparse, collections, difflib, enum, hashlib, operator, os, shutil, stat
import struct, sys, time, urllib.request, zlib


# Data for one entry in the git index (.git/index)
IndexEntry = collections.namedtuple('IndexEntry', [
    'ctime_s', 'ctime_n', 'mtime_s', 'mtime_n', 'dev', 'ino', 'mode', 'uid',
    'gid', 'size', 'sha1', 'flags', 'path',
])


class ObjectType(enum.Enum):
    """Object type enum. There are other types too, but we don't need them.
    See "enum object_type" in git's source (git/cache.h).
    """
    commit = 1
    tree = 2
    blob = 3


def read_file(path):
    """Read contents of file at given path as bytes."""
    with open(path, 'rb') as f:
        return f.read()


def write_file(path, data):
    """Write data bytes to file at given path."""
    with open(path, 'wb') as f:
        f.write(data)


def ensure_clean_working_tree():
    """Raise ValueError if working tree or index have pending changes."""
    changed, new, deleted = get_status()
    if changed or new or deleted:
        raise ValueError(
                'working tree has unstaged changes; please commit or stash them first')


def init(repo):
    """Create directory for repo and initialize .git directory."""
    os.mkdir(repo)
    os.mkdir(os.path.join(repo, '.git'))
    for name in ['objects', 'refs', 'refs/heads']:
        os.mkdir(os.path.join(repo, '.git', name))
    write_file(os.path.join(repo, '.git', 'refs/heads/master'), b'')
    set_head('refs/heads/master', git_dir=os.path.join(repo, '.git'))
    print('initialized empty repository: {}'.format(repo))


def hash_object(data, obj_type, write=True):
    """Compute hash of object data of given type and write to object store if
    "write" is True. Return SHA-1 object hash as hex string.
    """
    header = '{} {}'.format(obj_type, len(data)).encode()
    full_data = header + b'\x00' + data
    sha1 = hashlib.sha1(full_data).hexdigest()
    if write:
        path = os.path.join('.git', 'objects', sha1[:2], sha1[2:])
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_file(path, zlib.compress(full_data))
    return sha1


def find_object(sha1_prefix):
    """Find object with given SHA-1 prefix and return path to object in object
    store, or raise ValueError if there are no objects or multiple objects
    with this prefix.
    """
    if len(sha1_prefix) < 2:
        raise ValueError('hash prefix must be 2 or more characters')
    obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
    rest = sha1_prefix[2:]
    objects = [name for name in os.listdir(obj_dir) if name.startswith(rest)]
    if not objects:
        raise ValueError('object {!r} not found'.format(sha1_prefix))
    if len(objects) >= 2:
        raise ValueError('multiple objects ({}) with prefix {!r}'.format(
                len(objects), sha1_prefix))
    return os.path.join(obj_dir, objects[0])


def read_object(sha1_prefix):
    """Read object with given SHA-1 prefix and return tuple of
    (object_type, data_bytes), or raise ValueError if not found.
    """
    path = find_object(sha1_prefix)
    full_data = zlib.decompress(read_file(path))
    nul_index = full_data.index(b'\x00')
    header = full_data[:nul_index]
    obj_type, size_str = header.decode().split()
    size = int(size_str)
    data = full_data[nul_index + 1:]
    assert size == len(data), 'expected size {}, got {} bytes'.format(
            size, len(data))
    return (obj_type, data)


def cat_file(mode, sha1_prefix):
    """Write the contents of (or info about) object with given SHA-1 prefix to
    stdout. If mode is 'commit', 'tree', or 'blob', print raw data bytes of
    object. If mode is 'size', print the size of the object. If mode is
    'type', print the type of the object. If mode is 'pretty', print a
    prettified version of the object.
    """
    obj_type, data = read_object(sha1_prefix)
    if mode in ['commit', 'tree', 'blob']:
        if obj_type != mode:
            raise ValueError('expected object type {}, got {}'.format(
                    mode, obj_type))
        sys.stdout.buffer.write(data)
    elif mode == 'size':
        print(len(data))
    elif mode == 'type':
        print(obj_type)
    elif mode == 'pretty':
        if obj_type in ['commit', 'blob']:
            sys.stdout.buffer.write(data)
        elif obj_type == 'tree':
            for mode, path, sha1 in read_tree(data=data):
                type_str = 'tree' if stat.S_ISDIR(mode) else 'blob'
                print('{:06o} {} {}\t{}'.format(mode, type_str, sha1, path))
        else:
            assert False, 'unhandled object type {!r}'.format(obj_type)
    else:
        raise ValueError('unexpected mode {!r}'.format(mode))


def read_index():
    """Read git index file and return list of IndexEntry objects."""
    try:
        data = read_file(os.path.join('.git', 'index'))
    except FileNotFoundError:
        return []
    digest = hashlib.sha1(data[:-20]).digest()
    assert digest == data[-20:], 'invalid index checksum'
    signature, version, num_entries = struct.unpack('!4sLL', data[:12])
    assert signature == b'DIRC', \
            'invalid index signature {}'.format(signature)
    assert version == 2, 'unknown index version {}'.format(version)
    entry_data = data[12:-20]
    entries = []
    i = 0
    while i + 62 <= len(entry_data):
        fields_end = i + 62
        fields = struct.unpack('!LLLLLLLLLL20sH', entry_data[i:fields_end])
        path_end = entry_data.index(b'\x00', fields_end)
        path = entry_data[fields_end:path_end]
        entry = IndexEntry(*(fields + (path.decode(),)))
        entries.append(entry)
        entry_len = ((62 + len(path) + 8) // 8) * 8
        i += entry_len
    assert len(entries) == num_entries
    return entries


def ls_files(details=False):
    """Print list of files in index (including mode, SHA-1, and stage number
    if "details" is True).
    """
    for entry in read_index():
        if details:
            stage = (entry.flags >> 12) & 3
            print('{:6o} {} {:}\t{}'.format(
                    entry.mode, entry.sha1.hex(), stage, entry.path))
        else:
            print(entry.path)


def get_status():
    """Get status of working copy, return tuple of (changed_paths, new_paths,
    deleted_paths).
    """
    paths = set()
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d != '.git']
        for file in files:
            path = os.path.join(root, file)
            path = path.replace('\\', '/')
            if path.startswith('./'):
                path = path[2:]
            paths.add(path)
    entries_by_path = {e.path: e for e in read_index()}
    entry_paths = set(entries_by_path)
    changed = {p for p in (paths & entry_paths)
               if hash_object(read_file(p), 'blob', write=False) !=
                  entries_by_path[p].sha1.hex()}
    new = paths - entry_paths
    deleted = entry_paths - paths
    return (sorted(changed), sorted(new), sorted(deleted))


def status():
    """Show status of working copy."""
    changed, new, deleted = get_status()
    if changed:
        print('changed files:')
        for path in changed:
            print('   ', path)
    if new:
        print('new files:')
        for path in new:
            print('   ', path)
    if deleted:
        print('deleted files:')
        for path in deleted:
            print('   ', path)


def diff():
    """Show diff of files changed (between index and working copy)."""
    changed, _, _ = get_status()
    entries_by_path = {e.path: e for e in read_index()}
    for i, path in enumerate(changed):
        sha1 = entries_by_path[path].sha1.hex()
        obj_type, data = read_object(sha1)
        assert obj_type == 'blob'
        index_lines = data.decode().splitlines()
        working_lines = read_file(path).decode().splitlines()
        diff_lines = difflib.unified_diff(
                index_lines, working_lines,
                '{} (index)'.format(path),
                '{} (working copy)'.format(path),
                lineterm='')
        for line in diff_lines:
            print(line)
        if i < len(changed) - 1:
            print('-' * 70)


def write_index(entries):
    """Write list of IndexEntry objects to git index file."""
    packed_entries = []
    for entry in entries:
        entry_head = struct.pack('!LLLLLLLLLL20sH',
                entry.ctime_s, entry.ctime_n, entry.mtime_s, entry.mtime_n,
                entry.dev, entry.ino, entry.mode, entry.uid, entry.gid,
                entry.size, entry.sha1, entry.flags)
        path = entry.path.encode()
        length = ((62 + len(path) + 8) // 8) * 8
        packed_entry = entry_head + path + b'\x00' * (length - 62 - len(path))
        packed_entries.append(packed_entry)
    header = struct.pack('!4sLL', b'DIRC', 2, len(entries))
    all_data = header + b''.join(packed_entries)
    digest = hashlib.sha1(all_data).digest()
    write_file(os.path.join('.git', 'index'), all_data + digest)


def add(paths):
    """Add all file paths to git index."""
    paths = [p.replace('\\', '/') for p in paths]
    all_entries = read_index()
    entries = [e for e in all_entries if e.path not in paths]
    for path in paths:
        sha1 = hash_object(read_file(path), 'blob')
        st = os.stat(path)
        flags = len(path.encode())
        assert flags < (1 << 12)
        entry = IndexEntry(
                int(st.st_ctime), 0, int(st.st_mtime), 0, st.st_dev,
                st.st_ino, st.st_mode, st.st_uid, st.st_gid, st.st_size,
                bytes.fromhex(sha1), flags, path)
        entries.append(entry)
    entries.sort(key=operator.attrgetter('path'))
    write_index(entries)


def write_tree():
    """Write a tree object from the current index entries."""
    tree_entries = []
    for entry in read_index():
        assert '/' not in entry.path, \
                'currently only supports a single, top-level directory'
        mode_path = '{:o} {}'.format(entry.mode, entry.path).encode()
        tree_entry = mode_path + b'\x00' + entry.sha1
        tree_entries.append(tree_entry)
    return hash_object(b''.join(tree_entries), 'tree')


def resolve_head():
    """Return tuple of (ref_path, sha1) for current HEAD reference."""
    head_path = os.path.join('.git', 'HEAD')
    try:
        head_contents = read_file(head_path).decode().strip()
    except FileNotFoundError:
        return (None, None)
    if head_contents.startswith('ref: '):
        ref_path = head_contents[5:]
        try:
            sha1 = read_file(os.path.join('.git', ref_path)).decode().strip()
        except FileNotFoundError:
            sha1 = None
        if not sha1:
            sha1 = None
        return (ref_path, sha1)
    return (None, head_contents or None)


def update_ref(ref_path, sha1):
    """Update the given ref path to point to the specified SHA-1."""
    path = os.path.join('.git', ref_path)
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    write_file(path, (sha1 + '\n').encode())


def set_head(ref_path, git_dir='.git'):
    """Point HEAD at the given ref path."""
    write_file(os.path.join(git_dir, 'HEAD'),
               ('ref: {}\n'.format(ref_path)).encode())


def create_branch(name, start_sha1=None):
    """Create a new branch ref pointing at the given (or current HEAD) SHA-1."""
    if start_sha1 is None:
        _, start_sha1 = resolve_head()
        if not start_sha1:
            raise ValueError('cannot determine start commit for new branch')

    ref_path = os.path.join('refs', 'heads', name)
    full_path = os.path.join('.git', ref_path)
    if os.path.exists(full_path):
        raise ValueError('branch {!r} already exists'.format(name))

    update_ref(ref_path, start_sha1)


def checkout_tree(tree_sha1):
    """Replace working tree and index with the contents of tree."""
    for name in os.listdir('.'):
        if name == '.git':
            continue
        full_path = os.path.join('.', name)
        if os.path.islink(full_path) or not os.path.isdir(full_path):
            try:
                os.remove(full_path)
            except FileNotFoundError:
                pass
        else:
            shutil.rmtree(full_path)

    files = read_tree_files(tree_sha1)
    entries = []
    for mode, path, sha1 in files:
        if not stat.S_ISREG(mode):
            raise ValueError('cannot checkout non-regular file {!r}'.format(path))
        fs_path = path.replace('/', os.sep)
        dir_name = os.path.dirname(fs_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        obj_type, data = read_object(sha1)
        if obj_type != 'blob':
            raise ValueError('expected blob object for {!r}, got {}'.format(path, obj_type))
        write_file(fs_path, data)
        os.chmod(fs_path, stat.S_IMODE(mode))
        st = os.stat(fs_path)
        flags = len(path.encode())
        assert flags < (1 << 12)
        entry = IndexEntry(
                int(st.st_ctime), 0, int(st.st_mtime), 0, st.st_dev,
                st.st_ino, st.st_mode, st.st_uid, st.st_gid, st.st_size,
                bytes.fromhex(sha1), flags, path)
        entries.append(entry)

    entries.sort(key=operator.attrgetter('path'))
    write_index(entries)


def checkout(branch_name):
    """Check out the given branch by updating HEAD, index, and working tree."""
    ref_path = os.path.join('refs', 'heads', branch_name)
    full_ref_path = os.path.join('.git', ref_path)
    if not os.path.exists(full_ref_path):
        raise ValueError('branch {!r} does not exist'.format(branch_name))

    changed, new, deleted = get_status()
    if changed or new or deleted:
        raise ValueError('working tree has unstaged changes; please commit or stash them before checkout')

    commit_sha1 = read_file(full_ref_path).decode().strip()
    if not commit_sha1:
        raise ValueError('branch {!r} has no commits'.format(branch_name))

    obj_type, commit_data = read_object(commit_sha1)
    if obj_type != 'commit':
        raise ValueError('expected commit object, got {}'.format(obj_type))

    tree_sha1 = None
    for line in commit_data.decode().splitlines():
        if line.startswith('tree '):
            parts = line.split()
            if len(parts) >= 2:
                tree_sha1 = parts[1]
            break
    if tree_sha1 is None:
        raise ValueError('commit {} does not have a tree'.format(commit_sha1))

    checkout_tree(tree_sha1)

    set_head(ref_path)
    print('Switched to branch {}'.format(branch_name))


def get_ref(ref_path):
    """Return the SHA-1 string stored at the given ref path."""
    path = os.path.join('.git', ref_path)
    return read_file(path).decode().strip()


def get_local_master_hash():
    """Get current commit hash (SHA-1 string) of current HEAD ref."""
    _, sha1 = resolve_head()
    return sha1


def read_commit(sha1):
    """Return tuple of (tree_sha1, parent_list, metadata, message) for commit."""
    obj_type, data = read_object(sha1)
    if obj_type != 'commit':
        raise ValueError('expected commit object, got {}'.format(obj_type))

    text = data.decode()
    header, _, message = text.partition('\n\n')
    metadata = {}
    parents = []
    tree = None
    for line in header.splitlines():
        key, value = line.split(' ', 1)
        if key == 'parent':
            parents.append(value)
        elif key == 'tree':
            tree = value
        else:
            metadata[key] = value
    return tree, parents, metadata, message.rstrip('\n')


def get_commit_tree(sha1):
    """Return tree SHA-1 for the given commit SHA-1."""
    if sha1 is None:
        return None
    tree, _, _, _ = read_commit(sha1)
    if tree is None:
        raise ValueError('commit {} does not have a tree'.format(sha1))
    return tree


def find_merge_base(left_sha1, right_sha1):
    """Find merge base SHA-1 between the two commits."""
    if left_sha1 is None or right_sha1 is None:
        return left_sha1 or right_sha1

    ancestors = {}
    queue = [(left_sha1, 0)]
    while queue:
        sha1, depth = queue.pop()
        if sha1 in ancestors and depth >= ancestors[sha1]:
            continue
        ancestors[sha1] = depth
        _, parents, _, _ = read_commit(sha1)
        for parent in parents:
            queue.append((parent, depth + 1))

    queue = collections.deque([(right_sha1, 0)])
    best = None
    best_score = None
    seen = set()
    while queue:
        sha1, depth = queue.popleft()
        if sha1 in seen:
            continue
        seen.add(sha1)
        if sha1 in ancestors:
            score = depth + ancestors[sha1]
            if best is None or score < best_score:
                best = sha1
                best_score = score
            continue
        _, parents, _, _ = read_commit(sha1)
        for parent in parents:
            queue.append((parent, depth + 1))
    return best


def create_commit(tree_sha1, parents, message, author=None):
    """Create commit object with given tree, parents, and message."""
    if author is None:
        author = '{} <{}>'.format(
                os.environ['GIT_AUTHOR_NAME'], os.environ['GIT_AUTHOR_EMAIL'])
    timestamp = int(time.mktime(time.localtime()))
    utc_offset = -time.timezone
    author_time = '{} {}{:02}{:02}'.format(
            timestamp,
            '+' if utc_offset > 0 else '-',
            abs(utc_offset) // 3600,
            (abs(utc_offset) // 60) % 60)
    lines = ['tree ' + tree_sha1]
    for parent in parents:
        lines.append('parent ' + parent)
    lines.append('author {} {}'.format(author, author_time))
    lines.append('committer {} {}'.format(author, author_time))
    lines.append('')
    lines.append(message)
    lines.append('')
    data = '\n'.join(lines).encode()
    sha1 = hash_object(data, 'commit')
    return sha1


def commit(message, author=None):
    """Commit the current state of the index to master with given message.
    Return hash of commit object.
    """
    tree = write_tree()
    parent = get_local_master_hash()
    sha1 = create_commit(tree, [parent] if parent else [], message, author=author)
    ref_path, _ = resolve_head()
    if ref_path:
        update_ref(ref_path, sha1)
    else:
        update_ref('HEAD', sha1)
    target = ref_path or 'HEAD'
    print('committed to {}: {:7}'.format(target, sha1))
    return sha1


def iterate_commits(start_sha1):
    """Yield commit metadata and parent hash for commits starting at SHA-1."""
    current = start_sha1
    while current:
        obj_type, data = read_object(current)
        if obj_type != 'commit':
            raise ValueError('expected commit object, got {}'.format(obj_type))
        text = data.decode()
        header, _, message = text.partition('\n\n')
        metadata = {}
        parent = None
        for line in header.splitlines():
            key, value = line.split(' ', 1)
            if key == 'parent':
                parent = value
            else:
                metadata[key] = value
        metadata['message'] = message.rstrip('\n')
        yield current, metadata, parent
        current = parent


def log():
    """Print commit log starting from current HEAD."""
    _, head_sha1 = resolve_head()
    if not head_sha1:
        print('no commits yet')
        return

    for sha1, metadata, _ in iterate_commits(head_sha1):
        author_line = metadata.get('author', '')
        author_name = author_line
        date_str = ''
        if author_line:
            parts = author_line.rsplit(' ', 2)
            if len(parts) == 3:
                author_name = parts[0]
                timestamp_str, tz = parts[1], parts[2]
                try:
                    timestamp = int(timestamp_str)
                    formatted = time.strftime(
                            '%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                    date_str = '{} {}'.format(formatted, tz)
                except ValueError:
                    date_str = '{} {}'.format(timestamp_str, tz)
            else:
                author_name = author_line

        message = metadata.get('message', '')

        print('commit {}'.format(sha1))
        if author_name:
            print('Author: {}'.format(author_name))
        if date_str:
            print('Date:   {}'.format(date_str))
        print()
        for line in message.splitlines():
            print('    ' + line)
        print()


def extract_lines(data):
    """Extract list of lines from given server data."""
    lines = []
    i = 0
    for _ in range(1000):
        line_length = int(data[i:i + 4], 16)
        line = data[i + 4:i + line_length]
        lines.append(line)
        if line_length == 0:
            i += 4
        else:
            i += line_length
        if i >= len(data):
            break
    return lines


def build_lines_data(lines):
    """Build byte string from given lines to send to server."""
    result = []
    for line in lines:
        result.append('{:04x}'.format(len(line) + 5).encode())
        result.append(line)
        result.append(b'\n')
    result.append(b'0000')
    return b''.join(result)


def http_request(url, username, password, data=None):
    """Make an authenticated HTTP request to given URL (GET by default, POST
    if "data" is not None).
    """
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, url, username, password)
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_manager)
    opener = urllib.request.build_opener(auth_handler)
    f = opener.open(url, data=data)
    return f.read()


def get_remote_master_hash(git_url, username, password):
    """Get commit hash of remote master branch, return SHA-1 hex string or
    None if no remote commits.
    """
    url = git_url + '/info/refs?service=git-receive-pack'
    response = http_request(url, username, password)
    lines = extract_lines(response)
    assert lines[0] == b'# service=git-receive-pack\n'
    assert lines[1] == b''
    if lines[2][:40] == b'0' * 40:
        return None
    master_sha1, master_ref = lines[2].split(b'\x00')[0].split()
    assert master_ref == b'refs/heads/master'
    assert len(master_sha1) == 40
    return master_sha1.decode()


def read_tree(sha1=None, data=None):
    """Read tree object with given SHA-1 (hex string) or data, and return list
    of (mode, path, sha1) tuples.
    """
    if sha1 is not None:
        obj_type, data = read_object(sha1)
        assert obj_type == 'tree'
    elif data is None:
        raise TypeError('must specify "sha1" or "data"')
    i = 0
    entries = []
    for _ in range(1000):
        end = data.find(b'\x00', i)
        if end == -1:
            break
        mode_str, path = data[i:end].decode().split()
        mode = int(mode_str, 8)
        digest = data[end + 1:end + 21]
        entries.append((mode, path, digest.hex()))
        i = end + 1 + 20
    return entries


def read_tree_files(tree_sha1, prefix=""):
    """Return list of blob entries from tree (recursively).

    The result is a list of (mode, path, sha1) tuples where each tuple
    represents a blob entry and the path is relative to the repository root.
    """
    files = []
    for mode, path, sha1 in read_tree(sha1=tree_sha1):
        full_path = os.path.join(prefix, path) if prefix else path
        full_path = full_path.replace('\\', '/')
        if stat.S_ISDIR(mode):
            files.extend(read_tree_files(sha1, prefix=full_path))
        else:
            files.append((mode, full_path, sha1))
    return files


def tree_to_dict(tree_sha1):
    """Return mapping of path -> (mode, sha1) for files in tree."""
    if tree_sha1 is None:
        return {}
    return {path: (mode, sha1) for mode, path, sha1 in read_tree_files(tree_sha1)}


def read_blob(sha1):
    """Return blob data bytes for given SHA-1 or None if SHA-1 is None."""
    if sha1 is None:
        return None
    obj_type, data = read_object(sha1)
    if obj_type != 'blob':
        raise ValueError('expected blob object, got {}'.format(obj_type))
    return data


def blob_to_lines(data):
    """Convert blob bytes to list of text lines and trailing newline flag."""
    if data is None:
        return None, False
    text = data.decode('utf-8', errors='surrogateescape')
    return text.splitlines(), text.endswith('\n')


def build_change_list(base_lines, new_lines):
    """Return list of change dicts describing transformation from base."""
    matcher = difflib.SequenceMatcher(None, base_lines, new_lines)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        changes.append({'start': i1, 'end': i2, 'lines': new_lines[j1:j2]})
    return changes


def build_conflict_lines(current_lines, target_lines, current_label, target_label):
    """Return list of lines containing merge conflict markers."""
    lines = ['<<<<<<< ' + current_label]
    lines.extend(current_lines)
    lines.append('=======')
    lines.extend(target_lines)
    lines.append('>>>>>>> ' + target_label)
    return lines


def merge_lines(base_lines, current_lines, target_lines,
                current_label, target_label):
    """Perform a 3-way merge of the given line sets."""
    if current_lines == target_lines:
        return current_lines, False
    if base_lines == current_lines:
        return target_lines, False
    if base_lines == target_lines:
        return current_lines, False

    cur_changes = build_change_list(base_lines, current_lines)
    tgt_changes = build_change_list(base_lines, target_lines)
    result = []
    ci = 0
    ti = 0
    pos = 0
    conflict = False

    def get_change(changes, index):
        return changes[index] if index < len(changes) else None

    def collect_inserts(changes, index, pos):
        collected = []
        while index < len(changes):
            change = changes[index]
            if change['start'] == pos and change['start'] == change['end']:
                collected.append(change['lines'])
                index += 1
            else:
                break
        return collected, index

    while pos < len(base_lines):
        cur_change = get_change(cur_changes, ci)
        tgt_change = get_change(tgt_changes, ti)

        cur_inserts, ci = collect_inserts(cur_changes, ci, pos)
        tgt_inserts, ti = collect_inserts(tgt_changes, ti, pos)
        if cur_inserts or tgt_inserts:
            cur_lines = [line for block in cur_inserts for line in block]
            tgt_lines = [line for block in tgt_inserts for line in block]
            if cur_lines and tgt_lines:
                if cur_lines == tgt_lines:
                    result.extend(cur_lines)
                else:
                    conflict = True
                    result.extend(build_conflict_lines(
                            cur_lines, tgt_lines, current_label, target_label))
            elif cur_lines:
                result.extend(cur_lines)
            elif tgt_lines:
                result.extend(tgt_lines)

        cur_change = get_change(cur_changes, ci)
        tgt_change = get_change(tgt_changes, ti)

        cur_block = (cur_change if cur_change and cur_change['start'] == pos
                     and cur_change['end'] > pos else None)
        tgt_block = (tgt_change if tgt_change and tgt_change['start'] == pos
                     and tgt_change['end'] > pos else None)

        if cur_block or tgt_block:
            if cur_block and tgt_block:
                cur_lines = cur_block['lines']
                tgt_lines = tgt_block['lines']
                if cur_lines == tgt_lines:
                    result.extend(cur_lines)
                else:
                    conflict = True
                    result.extend(build_conflict_lines(
                            cur_lines, tgt_lines, current_label, target_label))
                pos = max(cur_block['end'], tgt_block['end'])
                ci += 1
                ti += 1
            elif cur_block:
                result.extend(cur_block['lines'])
                pos = cur_block['end']
                ci += 1
            else:
                result.extend(tgt_block['lines'])
                pos = tgt_block['end']
                ti += 1
        else:
            result.append(base_lines[pos])
            pos += 1

    cur_change = get_change(cur_changes, ci)
    tgt_change = get_change(tgt_changes, ti)
    cur_inserts, ci = collect_inserts(cur_changes, ci, len(base_lines))
    tgt_inserts, ti = collect_inserts(tgt_changes, ti, len(base_lines))
    cur_lines = [line for block in cur_inserts for line in block]
    tgt_lines = [line for block in tgt_inserts for line in block]
    if cur_lines and tgt_lines:
        if cur_lines == tgt_lines:
            result.extend(cur_lines)
        else:
            conflict = True
            result.extend(build_conflict_lines(
                    cur_lines, tgt_lines, current_label, target_label))
    elif cur_lines:
        result.extend(cur_lines)
    elif tgt_lines:
        result.extend(tgt_lines)

    return result, conflict


def merge_trees(current_tree_sha1, target_tree_sha1, base_tree_sha1, target_label):
    """Merge current and target trees using base tree as ancestor."""
    base_files = tree_to_dict(base_tree_sha1)
    current_files = tree_to_dict(current_tree_sha1)
    target_files = tree_to_dict(target_tree_sha1)
    all_paths = sorted(set(base_files) | set(current_files) | set(target_files))

    results = {}
    conflicts = []
    for path in all_paths:
        base_entry = base_files.get(path)
        current_entry = current_files.get(path)
        target_entry = target_files.get(path)

        base_data = read_blob(base_entry[1]) if base_entry else None
        current_data = read_blob(current_entry[1]) if current_entry else None
        target_data = read_blob(target_entry[1]) if target_entry else None

        base_lines, base_nl = blob_to_lines(base_data)
        current_lines, current_nl = blob_to_lines(current_data)
        target_lines, target_nl = blob_to_lines(target_data)

        final_data = None
        conflict = False

        if current_data == target_data:
            final_data = current_data
        elif base_data == current_data:
            final_data = target_data
        elif base_data == target_data:
            final_data = current_data
        elif target_data is None and current_data is None:
            final_data = None
        elif target_data is None:
            if base_data is None:
                final_data = current_data
            else:
                conflict = True
        elif current_data is None:
            if base_data is None:
                final_data = target_data
            else:
                conflict = True
        else:
            merged_lines, merge_conflict = merge_lines(
                    base_lines or [], current_lines or [], target_lines or [],
                    'HEAD', target_label)
            conflict = merge_conflict
            if not conflict:
                newline = current_nl or target_nl or base_nl
                text = '\n'.join(merged_lines)
                if newline:
                    text += '\n'
                final_data = text.encode('utf-8', errors='surrogateescape')
            else:
                conflict_lines = merged_lines
                text = '\n'.join(conflict_lines) + '\n'
                final_data = text.encode('utf-8', errors='surrogateescape')

        if conflict:
            conflicts.append(path)
            if final_data is None:
                text = '\n'.join(build_conflict_lines(
                        current_lines or [], target_lines or [], 'HEAD', target_label)) + '\n'
                final_data = text.encode('utf-8', errors='surrogateescape')

        if final_data is None:
            results[path] = {'mode': None, 'data': None, 'conflict': conflict}
        else:
            if conflict and current_data is None and target_data is None:
                mode = base_entry[0] if base_entry else 0o100644
            else:
                mode = None
                for entry in (current_entry, target_entry, base_entry):
                    if entry is not None:
                        mode = entry[0]
                        break
                if mode is None:
                    mode = 0o100644
            results[path] = {'mode': mode, 'data': final_data, 'conflict': conflict}

    return results, conflicts


def apply_merge_results(results):
    """Write merge results to working tree (including deletions)."""
    for path in sorted(results):
        info = results[path]
        fs_path = path.replace('/', os.sep)
        if info['data'] is None:
            try:
                os.remove(fs_path)
            except FileNotFoundError:
                pass
            dir_name = os.path.dirname(fs_path)
            while dir_name:
                try:
                    os.rmdir(dir_name)
                except OSError:
                    break
                dir_name = os.path.dirname(dir_name)
        else:
            dir_name = os.path.dirname(fs_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            write_file(fs_path, info['data'])
            if info.get('mode') is not None:
                os.chmod(fs_path, stat.S_IMODE(info['mode']))


def find_tree_objects(tree_sha1):
    """Return set of SHA-1 hashes of all objects in this tree (recursively),
    including the hash of the tree itself.
    """
    objects = {tree_sha1}
    for mode, path, sha1 in read_tree(sha1=tree_sha1):
        if stat.S_ISDIR(mode):
            objects.update(find_tree_objects(sha1))
        else:
            objects.add(sha1)
    return objects


def find_commit_objects(commit_sha1):
    """Return set of SHA-1 hashes of all objects in this commit (recursively),
    its tree, its parents, and the hash of the commit itself.
    """
    objects = {commit_sha1}
    obj_type, commit = read_object(commit_sha1)
    assert obj_type == 'commit'
    lines = commit.decode().splitlines()
    tree = next(l[5:45] for l in lines if l.startswith('tree '))
    objects.update(find_tree_objects(tree))
    parents = (l[7:47] for l in lines if l.startswith('parent '))
    for parent in parents:
        objects.update(find_commit_objects(parent))
    return objects


def find_missing_objects(local_sha1, remote_sha1):
    """Return set of SHA-1 hashes of objects in local commit that are missing
    at the remote (based on the given remote commit hash).
    """
    local_objects = find_commit_objects(local_sha1)
    if remote_sha1 is None:
        return local_objects
    remote_objects = find_commit_objects(remote_sha1)
    return local_objects - remote_objects


def encode_pack_object(obj):
    """Encode a single object for a pack file and return bytes (variable-
    length header followed by compressed data bytes).
    """
    obj_type, data = read_object(obj)
    type_num = ObjectType[obj_type].value
    size = len(data)
    byte = (type_num << 4) | (size & 0x0f)
    size >>= 4
    header = []
    while size:
        header.append(byte | 0x80)
        byte = size & 0x7f
        size >>= 7
    header.append(byte)
    return bytes(header) + zlib.compress(data)


def create_pack(objects):
    """Create pack file containing all objects in given given set of SHA-1
    hashes, return data bytes of full pack file.
    """
    header = struct.pack('!4sLL', b'PACK', 2, len(objects))
    body = b''.join(encode_pack_object(o) for o in sorted(objects))
    contents = header + body
    sha1 = hashlib.sha1(contents).digest()
    data = contents + sha1
    return data


def push(git_url, username=None, password=None):
    """Push master branch to given git repo URL."""
    if username is None:
        username = os.environ['GIT_USERNAME']
    if password is None:
        password = os.environ['GIT_PASSWORD']
    remote_sha1 = get_remote_master_hash(git_url, username, password)
    local_sha1 = get_local_master_hash()
    missing = find_missing_objects(local_sha1, remote_sha1)
    print('updating remote master from {} to {} ({} object{})'.format(
            remote_sha1 or 'no commits', local_sha1, len(missing),
            '' if len(missing) == 1 else 's'))
    lines = ['{} {} refs/heads/master\x00 report-status'.format(
            remote_sha1 or ('0' * 40), local_sha1).encode()]
    data = build_lines_data(lines) + create_pack(missing)
    url = git_url + '/git-receive-pack'
    response = http_request(url, username, password, data=data)
    lines = extract_lines(response)
    assert len(lines) >= 2, \
        'expected at least 2 lines, got {}'.format(len(lines))
    assert lines[0] == b'unpack ok\n', \
        "expected line 1 b'unpack ok', got: {}".format(lines[0])
    assert lines[1] == b'ok refs/heads/master\n', \
        "expected line 2 b'ok refs/heads/master\n', got: {}".format(lines[1])
    return (remote_sha1, missing)


def merge(branch_name):
    """Merge the given branch into the current HEAD."""
    head_ref, current_sha1 = resolve_head()
    if head_ref is None:
        raise ValueError('cannot merge in detached HEAD state')

    target_ref = os.path.join('refs', 'heads', branch_name)
    full_target = os.path.join('.git', target_ref)
    if not os.path.exists(full_target):
        raise ValueError('branch {!r} does not exist'.format(branch_name))

    target_sha1 = read_file(full_target).decode().strip()
    if not target_sha1:
        raise ValueError('branch {!r} has no commits'.format(branch_name))

    ensure_clean_working_tree()

    if current_sha1 == target_sha1:
        print('Already up to date.')
        return

    base_sha1 = find_merge_base(current_sha1, target_sha1)
    if current_sha1 is None or base_sha1 == current_sha1:
        tree_sha1 = get_commit_tree(target_sha1)
        checkout_tree(tree_sha1)
        update_ref(head_ref, target_sha1)
        print('Fast-forward merge to {}.'.format(branch_name))
        return

    if base_sha1 == target_sha1:
        print('Already up to date.')
        return

    base_tree_sha1 = get_commit_tree(base_sha1) if base_sha1 else None
    current_tree_sha1 = get_commit_tree(current_sha1)
    target_tree_sha1 = get_commit_tree(target_sha1)

    results, conflicts = merge_trees(current_tree_sha1, target_tree_sha1,
            base_tree_sha1, branch_name)
    apply_merge_results(results)

    if conflicts:
        print('Merge conflict{} in:'.format('s' if len(conflicts) != 1 else ''))
        for path in conflicts:
            print('   ', path)
        print('Fix conflicts and commit the result.')
        return

    merged_paths = [path for path, info in results.items() if info['data'] is not None]
    if merged_paths:
        add(merged_paths)
    else:
        write_index([])

    tree_sha1 = write_tree()
    merge_message = 'Merge branch {}'.format(branch_name)
    merge_sha1 = create_commit(tree_sha1, [current_sha1, target_sha1], merge_message)
    update_ref(head_ref, merge_sha1)
    print('Merge made commit {}.'.format(merge_sha1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='command', metavar='command')
    sub_parsers.required = True

    sub_parser = sub_parsers.add_parser('add',
            help='add file(s) to index')
    sub_parser.add_argument('paths', nargs='+', metavar='path',
            help='path(s) of files to add')

    sub_parser = sub_parsers.add_parser('branch',
            help='create a new branch pointing at a commit (defaults to HEAD)')
    sub_parser.add_argument('name', help='name of branch to create')
    sub_parser.add_argument('--start', dest='start', metavar='sha1',
            help='SHA-1 of commit where the new branch should start (defaults '
                 'to current HEAD)')

    sub_parser = sub_parsers.add_parser('checkout',
            help='switch to an existing branch')
    sub_parser.add_argument('branch_name', help='name of branch to switch to')

    sub_parser = sub_parsers.add_parser('cat-file',
            help='display contents of object')
    valid_modes = ['commit', 'tree', 'blob', 'size', 'type', 'pretty']
    sub_parser.add_argument('mode', choices=valid_modes,
            help='object type (commit, tree, blob) or display mode (size, '
                 'type, pretty)')
    sub_parser.add_argument('hash_prefix',
            help='SHA-1 hash (or hash prefix) of object to display')

    sub_parser = sub_parsers.add_parser('commit',
            help='commit current state of index to master branch')
    sub_parser.add_argument('-a', '--author',
            help='commit author in format "A U Thor <author@example.com>" '
                 '(uses GIT_AUTHOR_NAME and GIT_AUTHOR_EMAIL environment '
                 'variables by default)')
    sub_parser.add_argument('-m', '--message', required=True,
            help='text of commit message')

    sub_parser = sub_parsers.add_parser('diff',
            help='show diff of files changed (between index and working '
                 'copy)')

    sub_parser = sub_parsers.add_parser('hash-object',
            help='hash contents of given path (and optionally write to '
                 'object store)')
    sub_parser.add_argument('path',
            help='path of file to hash')
    sub_parser.add_argument('-t', choices=['commit', 'tree', 'blob'],
            default='blob', dest='type',
            help='type of object (default %(default)r)')
    sub_parser.add_argument('-w', action='store_true', dest='write',
            help='write object to object store (as well as printing hash)')

    sub_parser = sub_parsers.add_parser('init',
            help='initialize a new repo')
    sub_parser.add_argument('repo',
            help='directory name for new repo')

    sub_parser = sub_parsers.add_parser('log',
            help='show commit log')

    sub_parser = sub_parsers.add_parser('ls-files',
            help='list files in index')
    sub_parser.add_argument('-s', '--stage', action='store_true',
            help='show object details (mode, hash, and stage number) in '
                 'addition to path')

    sub_parser = sub_parsers.add_parser('push',
            help='push master branch to given git server URL')
    sub_parser.add_argument('git_url',
            help='URL of git repo, eg: https://github.com/benhoyt/pygit.git')
    sub_parser.add_argument('-p', '--password',
            help='password to use for authentication (uses GIT_PASSWORD '
                 'environment variable by default)')
    sub_parser.add_argument('-u', '--username',
            help='username to use for authentication (uses GIT_USERNAME '
                 'environment variable by default)')

    sub_parser = sub_parsers.add_parser('merge',
            help='merge given branch into current branch')
    sub_parser.add_argument('branch', help='name of branch to merge')

    sub_parser = sub_parsers.add_parser('status',
            help='show status of working copy')

    args = parser.parse_args()
    if args.command == 'add':
        add(args.paths)
    elif args.command == 'branch':
        try:
            create_branch(args.name, start_sha1=args.start)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'checkout':
        try:
            checkout(args.branch_name)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'cat-file':
        try:
            cat_file(args.mode, args.hash_prefix)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'commit':
        commit(args.message, author=args.author)
    elif args.command == 'diff':
        diff()
    elif args.command == 'hash-object':
        sha1 = hash_object(read_file(args.path), args.type, write=args.write)
        print(sha1)
    elif args.command == 'init':
        init(args.repo)
    elif args.command == 'log':
        log()
    elif args.command == 'ls-files':
        ls_files(details=args.stage)
    elif args.command == 'push':
        push(args.git_url, username=args.username, password=args.password)
    elif args.command == 'merge':
        try:
            merge(args.branch)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'status':
        status()
    else:
        assert False, 'unexpected command {!r}'.format(args.command)
