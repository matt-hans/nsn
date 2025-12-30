//! Build script for compiling Protocol Buffer definitions.
//!
//! This script uses tonic-build to generate Rust code from the sidecar.proto file.
//! The generated code is placed in OUT_DIR and included at compile time.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell cargo to rerun this script if the proto file changes
    println!("cargo:rerun-if-changed=proto/sidecar.proto");

    // Configure tonic-build
    tonic_build::configure()
        // Generate server code for implementing the service
        .build_server(true)
        // Generate client code for connecting to the service
        .build_client(true)
        // Generate transport-agnostic code
        .build_transport(true)
        // Compile the proto file
        .compile(&["proto/sidecar.proto"], &["proto/"])?;

    Ok(())
}
