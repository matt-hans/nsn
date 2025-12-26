use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info};

use crate::chain_client::ChainClient;
use crate::config::ChallengeConfig;
use crate::error::Result;

/// Monitor for on-chain challenges that require validator attestation
pub struct ChallengeMonitor {
    config: ChallengeConfig,
    chain_client: ChainClient,
}

impl ChallengeMonitor {
    pub fn new(config: ChallengeConfig, chain_client: ChainClient) -> Self {
        Self {
            config,
            chain_client,
        }
    }

    /// Start monitoring for challenges
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Challenge participation disabled");
            return Ok(());
        }

        info!(
            "Starting challenge monitor (poll interval: {}s)",
            self.config.poll_interval_secs
        );

        let mut ticker = interval(Duration::from_secs(self.config.poll_interval_secs));

        loop {
            ticker.tick().await;

            match self.check_pending_challenges().await {
                Ok(challenges) => {
                    if !challenges.is_empty() {
                        debug!("Found {} pending challenges", challenges.len());
                        // Process each challenge
                        for slot in challenges {
                            if let Err(e) = self.handle_challenge(slot).await {
                                tracing::error!(
                                    "Failed to handle challenge for slot {}: {}",
                                    slot,
                                    e
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Error checking challenges: {}", e);
                }
            }
        }
    }

    async fn check_pending_challenges(&self) -> Result<Vec<u64>> {
        self.chain_client.get_pending_challenges().await
    }

    async fn handle_challenge(&self, slot: u64) -> Result<()> {
        use tracing::warn;

        debug!("Handling challenge for slot {}", slot);

        // Step 1: Retrieve video chunk from DHT
        // In real implementation, this would query libp2p Kademlia DHT
        // For now, return error since DHT integration is pending
        warn!(
            "DHT retrieval not yet implemented - challenge response requires P2P DHT integration"
        );

        // When DHT is available, the flow would be:
        // let video_data = self.dht_client.get_video_chunk(slot).await?;
        // let prompt = self.dht_client.get_recipe_prompt(slot).await?;
        //
        // Step 2: Re-run CLIP verification on retrieved data
        // let clip_result = self.clip_engine.compute_score(&frames, &prompt).await?;
        //
        // Step 3: Generate attestation with verification result
        // let attestation = Attestation::new(slot, validator_id, clip_result, threshold)?
        //     .sign(&signing_key)?;
        //
        // Step 4: Submit challenge attestation to chain
        // self.chain_client.submit_challenge_attestation(slot, attestation).await?;

        // For now, just log that we detected the challenge
        info!(
            "Challenge detected for slot {} - awaiting DHT integration for response",
            slot
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_chain_client() -> ChainClient {
        ChainClient::new("ws://localhost:9944".to_string())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_challenge_monitor_creation() {
        let config = ChallengeConfig {
            enabled: true,
            response_buffer_blocks: 40,
            poll_interval_secs: 6,
        };
        let chain_client = test_chain_client().await;

        let monitor = ChallengeMonitor::new(config, chain_client);
        // Just verify it constructs without error
        assert!(monitor.config.enabled);
    }

    #[tokio::test]
    async fn test_challenge_monitor_disabled() {
        let config = ChallengeConfig {
            enabled: false,
            response_buffer_blocks: 40,
            poll_interval_secs: 6,
        };
        let chain_client = test_chain_client().await;

        let monitor = ChallengeMonitor::new(config, chain_client);
        let result = monitor.start().await;

        // Should return immediately when disabled
        assert!(result.is_ok());
    }
}
