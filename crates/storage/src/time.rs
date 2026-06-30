use chrono::{DateTime, Local, Utc};

pub const LOCAL_TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

pub fn local_now_text() -> String {
    Local::now().format(LOCAL_TIME_FORMAT).to_string()
}

pub fn utc_to_local_text(value: DateTime<Utc>) -> String {
    value
        .with_timezone(&Local)
        .format(LOCAL_TIME_FORMAT)
        .to_string()
}
