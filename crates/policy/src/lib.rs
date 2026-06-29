//! Permission policy, approval strategy, and action risk assessment.

use seekcode_common::{SeekCodeResult, WorkspaceId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Risk level assigned to a requested action.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// Safe workspace-scoped action.
    Low,
    /// Action deserves audit attention.
    Medium,
    /// Action can mutate or destroy important state.
    High,
}

/// Kind of action being evaluated.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum ActionKind {
    /// Read a workspace file.
    ReadFile { path: PathBuf },
    /// Write a workspace file.
    WriteFile { path: PathBuf },
    /// Run a shell command.
    RunCommand { program: String, args: Vec<String> },
    /// Call a local tool.
    ToolCall { name: String },
}

/// Context supplied to policy evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyContext {
    /// Workspace associated with the action.
    pub workspace_id: Option<WorkspaceId>,
    /// Requested action.
    pub action: ActionKind,
}

/// Decision returned by policy evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyDecision {
    /// Whether execution is allowed.
    pub allowed: bool,
    /// Assigned risk level.
    pub risk: RiskLevel,
    /// Human-readable reason.
    pub reason: String,
}

/// Policy engine interface.
pub trait PolicyEngine: Send + Sync {
    /// Evaluates a requested action.
    fn evaluate(&self, context: &PolicyContext) -> SeekCodeResult<PolicyDecision>;
}

/// Fully autonomous policy with workspace-boundary hooks.
#[derive(Default)]
pub struct AutonomousPolicy;

impl PolicyEngine for AutonomousPolicy {
    fn evaluate(&self, context: &PolicyContext) -> SeekCodeResult<PolicyDecision> {
        Ok(PolicyDecision {
            allowed: true,
            risk: classify_risk(&context.action),
            reason: "autonomous mode allows workspace-scoped action".to_string(),
        })
    }
}

/// Classifies an action before detailed policy checks.
pub fn classify_risk(action: &ActionKind) -> RiskLevel {
    match action {
        ActionKind::ReadFile { .. } => RiskLevel::Low,
        ActionKind::ToolCall { .. } => RiskLevel::Low,
        ActionKind::WriteFile { .. } => RiskLevel::Medium,
        ActionKind::RunCommand { .. } => RiskLevel::High,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autonomous_policy_allows_workspace_action() {
        let policy = AutonomousPolicy;
        let decision = policy
            .evaluate(&PolicyContext {
                workspace_id: Some(WorkspaceId::new()),
                action: ActionKind::ReadFile {
                    path: "src/lib.rs".into(),
                },
            })
            .expect("policy evaluates");

        assert!(decision.allowed);
        assert_eq!(decision.risk, RiskLevel::Low);
    }
}
