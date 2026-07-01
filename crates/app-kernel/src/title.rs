use crate::kernel::SessionService;
use crate::SessionTitleChanged;
use seekcode_common::{
    ChatMessage, ChatRole, ModelCallLogId, SeekCodeResult, SessionId, TokenUsage,
};
use seekcode_deepseek_client::{ChatRequest, ModelProvider};
use seekcode_storage::{local_now_text, NewModelCallLog};
use std::sync::Arc;
use std::time::Instant;

pub(crate) async fn generate_session_title(
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    title_model: String,
    prompt: String,
) -> SeekCodeResult<Option<SessionTitleChanged>> {
    let session = sessions.get_session(session_id).await?;
    if !session.name.trim().is_empty() {
        return Ok(None);
    }

    let called_at = local_now_text();
    let started_at = Instant::now();
    let model = title_model.clone();
    let response = provider
        .complete_chat(ChatRequest {
            model: title_model.clone(),
            messages: vec![
                ChatMessage::new(
                    ChatRole::System,
                    "Generate a concise chat title from the user's first message. Return only the title, with no quotes, punctuation wrapper, or explanation. Use the same language as the user. Keep it under 18 Chinese characters or 8 English words.",
                ),
                ChatMessage::new(ChatRole::User, prompt),
            ],
            tools: Vec::new(),
            thinking: false,
            reasoning_effort: None,
            strict_tools: false,
        })
        .await;
    let elapsed_ms = started_at.elapsed().as_millis().min(i64::MAX as u128) as i64;
    match &response {
        Ok(response) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    session.model_provider.clone(),
                    model,
                    session_id,
                    response.usage.clone(),
                    elapsed_ms,
                    true,
                    called_at,
                ))
                .await;
        }
        Err(_) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    session.model_provider,
                    model,
                    session_id,
                    None,
                    elapsed_ms,
                    false,
                    called_at,
                ))
                .await;
        }
    }

    let response = response?;

    let title = normalize_generated_session_title(&response.content);
    if title.is_empty() {
        return Ok(None);
    }

    let session = sessions.get_session(session_id).await?;
    if session.name.trim().is_empty() {
        let session = sessions.rename_session(session_id, title).await?;
        return Ok(Some(SessionTitleChanged {
            session_id,
            title: session.name,
        }));
    }

    Ok(None)
}

fn new_model_call_log(
    model_provider: String,
    model: String,
    session_id: SessionId,
    usage: Option<TokenUsage>,
    elapsed_ms: i64,
    success: bool,
    called_at: String,
) -> NewModelCallLog {
    let usage = usage.unwrap_or_default();
    NewModelCallLog {
        id: ModelCallLogId::new(),
        model_provider,
        model,
        session_id,
        input_tokens: i64::from(usage.prompt_tokens),
        output_tokens: i64::from(usage.completion_tokens),
        cache_hit_tokens: i64::from(usage.cached_tokens),
        elapsed_ms,
        success,
        called_at,
    }
}

pub(crate) fn normalize_generated_session_title(value: &str) -> String {
    let mut title = value
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .trim_start_matches(|ch: char| {
            ch.is_ascii_digit() || matches!(ch, '-' | '*' | '.' | ')' | ']' | '、' | ' ' | '\t')
        })
        .trim()
        .trim_matches(|ch| {
            matches!(
                ch,
                '"' | '\'' | '`' | '“' | '”' | '‘' | '’' | '「' | '」' | '『' | '』'
            )
        })
        .trim()
        .to_string();

    if title.chars().count() > 48 {
        title = title.chars().take(48).collect();
    }

    title
}
