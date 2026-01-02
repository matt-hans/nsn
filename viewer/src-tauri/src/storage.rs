// ICN Viewer Client - Local Storage Management
// Persists user settings and app state using JSON files

use crate::commands::{RelayInfo, ViewerSettings};
use std::fs;
use std::io;
use std::path::PathBuf;

/// Get path to settings file in app data directory
fn get_settings_path() -> Result<PathBuf, io::Error> {
    let mut path = dirs::config_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Config dir not found"))?;

    path.push("icn-viewer");
    fs::create_dir_all(&path)?;
    path.push("settings.json");

    Ok(path)
}

/// Get path to relay config file in app data directory
fn get_relays_path() -> Result<PathBuf, io::Error> {
    let mut path = dirs::config_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Config dir not found"))?;

    path.push("icn-viewer");
    fs::create_dir_all(&path)?;
    path.push("relays.json");

    Ok(path)
}

/// Save settings to JSON file
pub fn save_settings(settings: &ViewerSettings) -> Result<(), io::Error> {
    let path = get_settings_path()?;
    let json = serde_json::to_string_pretty(settings)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load settings from JSON file, or return default if not exists
pub fn load_settings() -> Result<ViewerSettings, io::Error> {
    let path = get_settings_path()?;

    if !path.exists() {
        return Ok(ViewerSettings::default());
    }

    let json = fs::read_to_string(path)?;
    let settings = serde_json::from_str(&json)?;
    Ok(settings)
}

/// Save relays to JSON file
pub fn save_relays(relays: &[RelayInfo]) -> Result<(), io::Error> {
    let path = get_relays_path()?;
    let json = serde_json::to_string_pretty(relays)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load relays from JSON file, or return empty if not exists
pub fn load_relays() -> Result<Vec<RelayInfo>, io::Error> {
    let path = get_relays_path()?;

    if !path.exists() {
        return Ok(Vec::new());
    }

    let json = fs::read_to_string(path)?;
    let relays = serde_json::from_str(&json)?;
    Ok(relays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_load_settings() {
        let settings = ViewerSettings {
            volume: 50,
            quality: "720p".to_string(),
            seeding_enabled: true,
            last_slot: Some(12345),
        };

        save_settings(&settings).unwrap();
        let loaded = load_settings().unwrap();

        assert_eq!(loaded.volume, 50);
        assert_eq!(loaded.quality, "720p");
        assert!(loaded.seeding_enabled);
        assert_eq!(loaded.last_slot, Some(12345));
    }

    #[test]
    fn test_load_nonexistent_returns_default() {
        // Clear settings file if exists
        if let Ok(path) = get_settings_path() {
            let _ = fs::remove_file(path);
        }

        let settings = load_settings().unwrap();
        assert_eq!(settings.volume, 80);
        assert_eq!(settings.quality, "auto");
    }

    #[test]
    fn test_load_relays_empty_when_missing() {
        if let Ok(path) = get_relays_path() {
            let _ = fs::remove_file(path);
        }
        let relays = load_relays().unwrap();
        assert!(relays.is_empty());
    }
}
