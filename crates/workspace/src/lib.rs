//! Workspace indexing, file access, ignore rules, and search boundaries.

use seekcode_common::{SeekCodeResult, WorkspaceId};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Open workspace root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceRoot {
    /// Workspace identifier.
    pub id: WorkspaceId,
    /// Absolute root path.
    pub path: PathBuf,
}

/// File or directory entry in a workspace tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    /// Path relative to the workspace root.
    pub relative_path: PathBuf,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// File size in bytes, if known.
    pub size: Option<u64>,
}

/// Snapshot of a file at read time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileSnapshot {
    /// Path relative to the workspace root.
    pub relative_path: PathBuf,
    /// UTF-8 content.
    pub content: String,
    /// Optional content hash.
    pub hash: Option<String>,
}

/// Options for listing a workspace tree.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ListOptions {
    /// Maximum traversal depth.
    pub max_depth: Option<usize>,
    /// Whether ignored files should be included.
    pub include_ignored: bool,
}

/// Text search query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Search pattern.
    pub pattern: String,
    /// Optional glob filter.
    pub glob: Option<String>,
    /// Maximum number of matches.
    pub limit: Option<usize>,
}

/// One text search match.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Path relative to the workspace root.
    pub relative_path: PathBuf,
    /// One-based line number.
    pub line: usize,
    /// Matching line text.
    pub text: String,
}

/// Workspace service boundary.
#[derive(Default)]
pub struct WorkspaceService;

impl WorkspaceService {
    /// Creates a workspace service.
    pub fn new() -> Self {
        Self
    }

    /// Opens a workspace root.
    pub async fn open(&self, _path: PathBuf) -> SeekCodeResult<WorkspaceRoot> {
        todo!("open and validate workspace root")
    }

    /// Lists files under a workspace root.
    pub async fn list_tree(
        &self,
        _root: &WorkspaceRoot,
        _options: ListOptions,
    ) -> SeekCodeResult<Vec<FileEntry>> {
        todo!("list workspace tree with ignore rules")
    }

    /// Reads a UTF-8 file from the workspace.
    pub async fn read_file(
        &self,
        _root: &WorkspaceRoot,
        _path: PathBuf,
    ) -> SeekCodeResult<FileSnapshot> {
        todo!("read workspace file")
    }

    /// Writes a UTF-8 file within the workspace.
    pub async fn write_file(
        &self,
        _root: &WorkspaceRoot,
        _path: PathBuf,
        _content: String,
    ) -> SeekCodeResult<FileSnapshot> {
        todo!("write workspace file")
    }

    /// Searches text within a workspace.
    pub async fn search_text(
        &self,
        _root: &WorkspaceRoot,
        _query: SearchQuery,
    ) -> SeekCodeResult<Vec<SearchResult>> {
        todo!("search text with ignore and glob rules")
    }

    /// Returns whether a path is inside the workspace root.
    pub fn is_inside_workspace(&self, _root: &WorkspaceRoot, _path: &Path) -> bool {
        todo!("check canonical workspace path boundary")
    }
}
