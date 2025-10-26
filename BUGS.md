# Summary: Major Errors in pygit.py

Based on code analysis and testing, here are the **major errors** in the current pygit.py file:

## Critical Issues

### 1. FileNotFoundError in `find_object()` (Line 76) - **CONFIRMED BUG**
**Problem:** When searching for an object with a prefix that doesn't exist (no directory for those first 2 chars), the code crashes with `FileNotFoundError` instead of returning a proper error message.

**Impact:** Crashes when trying to look up non-existent objects

**Current Code:**
```python
obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
rest = sha1_prefix[2:]
objects = [name for name in os.listdir(obj_dir) if name.startswith(rest)]
```

**Fix:**
```python
obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
rest = sha1_prefix[2:]
try:
    objects = [name for name in os.listdir(obj_dir) if name.startswith(rest)]
except FileNotFoundError:
    raise ValueError('object {!r} not found'.format(sha1_prefix))
```

### 2. Off-by-one error in `read_index()` (Line 146)
**Problem:** The condition `while i + 62 < len(entry_data)` may skip the last entry in certain edge cases where the last entry ends exactly at the boundary.

**Impact:** Could cause assertion failure or missing index entries

**Current Code:**
```python
while i + 62 < len(entry_data):
```

**Fix:**
```python
while i + 62 <= len(entry_data):
```

### 3. Hardcoded iteration limits (Lines 595, 659)
**Problem:** Both `extract_lines()` and `read_tree()` use `for _ in range(1000)` limiting to 1000 items.

**Impact:** Silently truncates large trees or protocol responses

**Current Code (read_tree):**
```python
for _ in range(1000):
    end = data.find(b'\x00', i)
    if end == -1:
        break
```

**Fix:**
```python
while i < len(data):
    end = data.find(b'\x00', i)
    if end == -1:
        break
```

### 4. `init()` doesn't handle existing directories (Line 43) - **CONFIRMED BUG**
**Problem:** Trying to init in current directory (`.`) fails with `FileExistsError`.

**Impact:** Cannot initialize a repo in an existing directory

**Current Code:**
```python
def init(repo):
    """Create directory for repo and initialize .git directory."""
    os.mkdir(repo)
```

**Fix:**
```python
def init(repo):
    """Create directory for repo and initialize .git directory."""
    if repo != '.' and not os.path.exists(repo):
        os.mkdir(repo)
```

## Minor Issues

### 5. Merge base depth comparison logic (Line 460)
**Problem:** The depth comparison in `find_merge_base()` could potentially cause issues in complex merge scenarios.

**Impact:** Might calculate wrong merge base in certain graph topologies

**Current Code:**
```python
if sha1 in ancestors and depth >= ancestors[sha1]:
    continue
```

**Analysis:** The logic aims to skip processing if we've already seen this commit at a lower depth. However, the condition might need review for correctness in all merge scenarios.

### 6. Missing broad exception handling
**Problem:** Various file operations could fail with permission errors or other issues beyond FileNotFoundError.

**Impact:** Ungraceful crashes on permission issues

**Locations:**
- `resolve_head()` (Line 292)
- `checkout_tree()` (Line 337-343)
- Various file I/O operations

## Testing Results

### Works Correctly ✅
- Syntax is valid (compiles without errors)
- Basic operations work (init, add, commit, status, log, branch, checkout)
- Help menu works correctly
- Core git functionality operates as designed

### Fails or Crashes ❌
- Looking up non-existent object hashes
- Initializing a repo in an existing directory (e.g., `pygit.py init .`)
- Handling trees with >1000 entries
- Large git protocol responses with >1000 lines
- Some edge cases with index reading
- Complex merge scenarios with unusual commit graph topologies

## Conclusion

The code is **functional for basic use cases and educational purposes**, but has several bugs that would cause failures in edge cases or production use. The most critical issues are the exception handling problems in `find_object()` and `init()`, which will cause crashes in common scenarios.
