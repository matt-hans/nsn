// ICN Viewer Client - Tauri Commands (IPC handlers)
// Frontend calls these via invoke() from @tauri-apps/api

use serde::{Deserialize, Serialize};
use crate::storage;

/// Relay node information returned from DHT or fallback
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RelayInfo {
    pub peer_id: String,
    pub multiaddr: String,
    pub region: String,
    pub latency_ms: Option<u32>,
    pub is_fallback: bool,
}

/// User settings persisted across app restarts
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ViewerSettings {
    pub volume: u8,
    pub quality: String,
    pub seeding_enabled: bool,
    pub last_slot: Option<u64>,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        Self {
            volume: 80,
            quality: "auto".to_string(),
            seeding_enabled: false,
            last_slot: None,
        }
    }
}

/// Get list of available relays (mocked for now, will integrate libp2p-js DHT)
#[tauri::command]
pub async fn get_relays() -> Result<Vec<RelayInfo>, String> {
    storage::load_relays().map_err(|e| format!("Failed to load relays: {}", e))
}

/// Save user settings to local storage
#[tauri::command]
pub async fn save_settings(settings: ViewerSettings) -> Result<(), String> {
    storage::save_settings(&settings)
        .map_err(|e| format!("Failed to save settings: {}", e))
}

/// Load user settings from local storage
#[tauri::command]
pub async fn load_settings() -> Result<ViewerSettings, String> {
    storage::load_settings()
        .map_err(|e| format!("Failed to load settings: {}", e))
}

/// Get application version
#[tauri::command]
pub fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_relays() {
        let relays = get_relays().await.unwrap();
        assert!(relays.is_empty() || relays.iter().all(|relay| !relay.peer_id.is_empty()));
    }

    #[test]
    fn test_default_settings() {
        let settings = ViewerSettings::default();
        assert_eq!(settings.volume, 80);
        assert_eq!(settings.quality, "auto");
        assert!(!settings.seeding_enabled);
    }
}
