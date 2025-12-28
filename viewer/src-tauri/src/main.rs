// ICN Viewer Client - Tauri Main Entry Point
// Handles window creation, IPC setup, and app lifecycle

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod storage;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_relays,
            commands::save_settings,
            commands::load_settings,
            commands::get_app_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
