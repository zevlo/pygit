# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyGit is a Python implementation of a subset of Git's core functionality. It's a single-file (~1200 lines) educational implementation that demonstrates Git's internal object model, index management, and basic version control operations.

## Development Commands

```bash
# Run the tool directly (no build step needed)
python pygit.py <command> [args]

# Common commands
python pygit.py init <repo>              # Initialize a new repository
python pygit.py add <path>               # Stage files
python pygit.py commit -m "message"      # Commit staged changes
python pygit.py status                   # Show working tree status
python pygit.py log                      # Show commit history
python pygit.py branch <name>            # Create a new branch
python pygit.py checkout <branch>        # Switch branches
python pygit.py merge <branch>           # Merge a branch into current HEAD
```

## Architecture

### Single-File Design
The entire implementation is in `pygit.py`. Functions are organized by functionality rather than classes, making the code flow procedural and straightforward to follow.

### Git Object Model
- **Objects stored in `.git/objects/`**: Uses SHA-1 hashing with the same format as real Git (zlib-compressed)
- **Object types**: `commit`, `tree`, `blob` (represented by the `ObjectType` enum)
- **Object reading/writing**:
  - `hash_object()` creates and stores objects
  - `read_object()` retrieves and decompresses objects
  - `find_object()` handles SHA-1 prefix matching

### Index Management
- **Index format**: Binary format compatible with Git's index v2
- **IndexEntry**: Named tuple containing file metadata (timestamps, mode, SHA-1, path)
- **Key functions**:
  - `read_index()` parses the binary index file
  - `write_index()` serializes entries with proper padding and checksums
  - Index entries must be sorted by path

### Tree Operations
- **Tree traversal**: `read_tree()` parses tree objects into (mode, path, sha1) tuples
- **Recursive reading**: `read_tree_files()` flattens nested trees into a flat list of blobs
- **Tree writing**: `write_tree()` creates tree objects from index entries
- **Current limitation**: Only supports flat directory structures (no subdirectories in working tree, but can read nested trees from commits)

### Merge Implementation
- **Three-way merge**: Uses `find_merge_base()` to find common ancestor
- **Line-based merging**: `merge_lines()` performs text-based 3-way merge with conflict detection
- **Tree merging**: `merge_trees()` handles file additions, deletions, and modifications
- **Conflict markers**: Standard Git-style `<<<<<<<`, `=======`, `>>>>>>>` markers
- **Fast-forward detection**: Automatically fast-forwards when possible

### Branch and Reference Management
- **HEAD resolution**: `resolve_head()` returns tuple of (ref_path, sha1)
  - Symbolic refs like `ref: refs/heads/main` vs detached HEAD
- **Branch operations**:
  - `create_branch()` creates refs in `.git/refs/heads/`
  - `checkout()` updates HEAD, index, and working tree
  - `update_ref()` updates ref files with new commit SHAs
- **Default branch**: Uses `master` (not `main`) for compatibility with original Git conventions

### Remote Operations (Push)
- **HTTP-based**: Uses Git's "smart HTTP" protocol
- **Authentication**: HTTP Basic Auth via environment variables `GIT_USERNAME` and `GIT_PASSWORD`
- **Pack protocol**:
  - `find_missing_objects()` computes objects to send
  - `create_pack()` generates Git pack format
  - `encode_pack_object()` handles variable-length headers and compression
- **Server communication**: `http_request()` with urllib for GET/POST

### Status and Diff
- **Status calculation**: `get_status()` compares working tree SHA-1s against index
  - Returns tuple of (changed, new, deleted) path lists
  - Uses `hash_object()` with `write=False` for comparison
- **Diff output**: `diff()` uses Python's `difflib.unified_diff()` for text comparison

### Working Tree Operations
- **checkout_tree()**: Destructively replaces working tree with tree contents
  - Removes all non-.git files first
  - Recreates files from blob objects
  - Rebuilds index to match new working tree
  - Preserves file modes (executable bits)

## Code Patterns

### Error Handling
- Functions raise `ValueError` with descriptive messages for user errors
- Assertions used for internal consistency checks (e.g., object type validation)
- Main block catches ValueError and prints to stderr with exit code 1

### File Operations
- `read_file()` and `write_file()` are thin wrappers for binary I/O
- All internal operations use forward slashes for paths
- Converts OS-specific separators when interacting with filesystem

### Object References
- SHA-1 hashes stored as lowercase hex strings
- Prefix matching supported (minimum 2 characters)
- Binary SHA-1 digests (20 bytes) used in index and tree objects

### Environment Variables
- `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL`: Required for commits
- `GIT_USERNAME` and `GIT_PASSWORD`: Required for push operations

## Implementation Notes

- No staging area concept beyond the index (no separate "staging" vs "unstaged" changes)
- Commits always include all changes in the index
- No support for `.gitignore`, submodules, or advanced Git features
- Pack file generation is self-contained (no delta compression)
- Merge conflicts must be manually resolved before committing
