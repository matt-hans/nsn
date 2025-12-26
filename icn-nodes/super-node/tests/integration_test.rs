//! Integration tests for Super-Node
//!
//! Test Case 1: Video chunk reception and erasure encoding
//! Test Case 2: Shard reconstruction (data loss simulation)
//! Test Case 3: Audit challenge response

use icn_super_node::{
    audit_monitor::{generate_audit_proof, AuditChallenge},
    erasure::ErasureCoder,
    storage::Storage,
};
use tempfile::tempdir;

/// Test Case 1: End-to-end video chunk encoding and storage
/// Purpose: Verify complete flow from raw video to stored shards
/// Contract: 50MB video → 14 shards → stored to disk → retrievable
#[tokio::test]
async fn test_video_chunk_encoding_and_storage() {
    let tmp_dir = tempdir().unwrap();
    let storage_path = tmp_dir.path().join("storage");
    std::fs::create_dir(&storage_path).unwrap();

    let storage = Storage::new(storage_path.clone());
    let coder = ErasureCoder::new().unwrap();

    // Simulate 50MB video chunk
    let video_data = vec![42u8; 50 * 1024 * 1024];
    let _original_size = video_data.len();

    // Encode to shards
    let shards = coder.encode(&video_data).expect("Encoding failed");
    assert_eq!(shards.len(), 14, "Should produce 14 shards");

    // Store shards
    let cid = storage
        .store_shards(&video_data, shards.clone())
        .await
        .expect("Storage failed");

    assert!(!cid.is_empty(), "CID should be generated");

    // Retrieve shard
    let shard_0 = storage.get_shard(&cid, 0).await.expect("Retrieval failed");
    assert_eq!(shard_0, shards[0], "Retrieved shard should match original");
}

/// Test Case 2: Shard reconstruction with data loss
/// Purpose: Verify erasure coding recovery with missing shards
/// Contract: Lose 4 shards → reconstruct from remaining 10 → matches original
#[tokio::test]
async fn test_shard_reconstruction_data_loss() {
    let coder = ErasureCoder::new().unwrap();

    // Test data
    let original_data = vec![123u8; 100_000];
    let original_size = original_data.len();

    // Encode
    let shards = coder.encode(&original_data).unwrap();

    // Simulate data loss: remove shards 2, 5, 11, 13
    let mut shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
    shards_opt[2] = None;
    shards_opt[5] = None;
    shards_opt[11] = None;
    shards_opt[13] = None;

    // Reconstruct
    let reconstructed = coder.decode(shards_opt, original_size).unwrap();

    // Verify
    assert_eq!(
        reconstructed, original_data,
        "Reconstructed data must match original"
    );
}

/// Test Case 3: Audit proof generation
/// Purpose: Verify audit challenge response mechanism
/// Contract: Read challenged bytes → hash with nonce → submit proof
#[tokio::test]
async fn test_audit_challenge_response() {
    use tokio::fs;

    let tmp_dir = tempdir().unwrap();
    let shard_path = tmp_dir.path().join("shard_05.bin");

    // Create test shard
    let shard_data =
        b"This is shard data for audit testing. Contains enough bytes for offset tests.";
    fs::write(&shard_path, shard_data).await.unwrap();

    let challenge = AuditChallenge {
        audit_id: 999,
        cid: "bafytest".to_string(),
        shard_index: 5,
        byte_offset: 8,
        byte_length: 10,
        nonce: vec![1, 2, 3, 4, 5],
    };

    // Generate proof
    let proof = generate_audit_proof(&shard_path, &challenge)
        .await
        .expect("Proof generation failed");

    // Verify proof format (SHA256 hash = 32 bytes)
    assert_eq!(proof.len(), 32, "Proof should be SHA256 hash");

    // Verify determinism (same inputs → same proof)
    let proof2 = generate_audit_proof(&shard_path, &challenge).await.unwrap();
    assert_eq!(proof, proof2, "Proofs should be deterministic");
}

/// Test Case 4: Storage cleanup
/// Purpose: Verify expired content deletion
/// Contract: Delete shards → files removed → get_shard returns error
#[tokio::test]
async fn test_storage_cleanup() {
    let tmp_dir = tempdir().unwrap();
    let storage = Storage::new(tmp_dir.path().to_path_buf());

    let data = b"Expired content";
    let shards = vec![vec![1, 2, 3], vec![4, 5, 6]];

    // Store
    let cid = storage.store_shards(data, shards).await.unwrap();

    // Verify exists
    assert!(storage.get_shard(&cid, 0).await.is_ok());

    // Cleanup
    storage.delete_shards(&cid).await.unwrap();

    // Verify deleted
    assert!(storage.get_shard(&cid, 0).await.is_err());
}
