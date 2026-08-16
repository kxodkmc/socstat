//! Shared, stateful session storage for the MCP server.
//!
//! MCP tool calls are otherwise stateless; this shared store lets the AI host
//! `load_dataset` once and then reference the dataset by name in every analysis
//! tool, mirroring a real analysis workflow.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use socstat::data::Dataset;

/// A thread-safe map of named datasets held for the lifetime of the server.
#[derive(Default)]
pub struct SharedState {
    datasets: Mutex<BTreeMap<String, Dataset>>,
}

impl SharedState {
    /// Create an empty shared store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap a store in an `Arc` for sharing into the server handler.
    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Insert (or replace) a dataset under a name.
    pub fn load(&self, name: String, ds: Dataset) {
        self.lock().insert(name, ds);
    }

    /// Clone a dataset by name, or `None` if absent.
    pub fn get(&self, name: &str) -> Option<Dataset> {
        self.lock().get(name).cloned()
    }

    /// Remove a dataset by name; returns whether it existed.
    pub fn remove(&self, name: &str) -> bool {
        self.lock().remove(name).is_some()
    }

    /// All registered dataset names.
    pub fn names(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    /// Overwrite a dataset by name, erroring if it does not exist.
    pub fn replace(&self, name: &str, ds: Dataset) -> Result<(), String> {
        let mut map = self.lock();
        if map.contains_key(name) {
            map.insert(name.to_string(), ds);
            Ok(())
        } else {
            Err(format!("dataset '{name}' not found"))
        }
    }

    /// Look up a dataset, returning a user-friendly error when missing.
    pub fn require(&self, name: &str) -> Result<Dataset, String> {
        self.get(name).ok_or_else(|| format!("dataset '{name}' not found; load it first with `load_dataset`"))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Dataset>> {
        // A poisoned mutex means a previous panicked thread; recover the data
        // instead of propagating the (irrelevant) panic.
        self.datasets.lock().unwrap_or_else(|e| e.into_inner())
    }
}