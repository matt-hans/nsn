//! Chain client utilities for NSN nodes.
//!
//! Provides a chain-backed executor registry sourced from on-chain stake and
//! reputation data.

use nsn_scheduler::{
    AttestationBundle, AttestationError, AttestationSubmitter, ExecutorInfo, ExecutorRegistry,
};
use nsn_types::{Lane, NodeCapability};
use parity_scale_codec::Decode;
use sp_core::crypto::AccountId32;
use sp_core::sr25519;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use subxt::{dynamic::storage, dynamic::tx, dynamic::Value, OnlineClient, PolkadotConfig};
use thiserror::Error;
use tracing::{info, warn};

/// Errors returned by the chain client registry.
#[derive(Debug, Error)]
pub enum ChainClientError {
    #[error("Subxt error: {0}")]
    Subxt(#[from] subxt::Error),
    #[error("Storage decode error: {0}")]
    Decode(String),
    #[error("Invalid storage key bytes")]
    InvalidKey,
}

#[derive(Debug, Clone, Decode)]
enum NodeRole {
    None,
    Relay,
    Validator,
    SuperNode,
    Director,
    ActiveDirector,
    Reserve,
}

#[derive(Debug, Clone, Decode)]
enum Region {
    NaWest,
    NaEast,
    EuWest,
    EuEast,
    Apac,
    Latam,
    Mena,
}

#[derive(Debug, Clone, Decode)]
#[allow(dead_code)]
enum NodeMode {
    Lane1Active,
    Draining { epoch_start: u32 },
    Lane0Active { epoch_end: u32 },
    Offline,
}

#[derive(Debug, Clone, Decode)]
struct StakeInfo {
    #[allow(dead_code)]
    amount: u128,
    #[allow(dead_code)]
    locked_until: u32,
    role: NodeRole,
    #[allow(dead_code)]
    region: Region,
    #[allow(dead_code)]
    delegated_to_me: u128,
}

#[derive(Debug, Clone, Decode)]
struct ReputationScore {
    director_score: u64,
    validator_score: u64,
    seeder_score: u64,
    #[allow(dead_code)]
    last_activity: u64,
}

impl ReputationScore {
    fn total(&self) -> u64 {
        let director_weighted = self.director_score.saturating_mul(50);
        let validator_weighted = self.validator_score.saturating_mul(30);
        let seeder_weighted = self.seeder_score.saturating_mul(20);
        director_weighted
            .saturating_add(validator_weighted)
            .saturating_add(seeder_weighted)
            .saturating_div(100)
    }
}

#[derive(Debug, Clone)]
struct ExecutorCandidate {
    id: String,
    capability: NodeCapability,
    reputation: u64,
    mode: NodeMode,
}

/// Chain-backed executor registry with periodic refresh.
#[derive(Debug, Clone)]
pub struct ChainExecutorRegistry {
    client: OnlineClient<PolkadotConfig>,
    refresh_interval: Duration,
    cache: Arc<RwLock<Vec<ExecutorCandidate>>>,
}

impl ChainExecutorRegistry {
    /// Create a new registry with the given chain RPC URL.
    pub async fn new(
        rpc_url: String,
        refresh_interval: Duration,
    ) -> Result<Self, ChainClientError> {
        let client = OnlineClient::<PolkadotConfig>::from_url(rpc_url).await?;
        Ok(Self {
            client,
            refresh_interval,
            cache: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start a background refresh loop.
    pub fn start_refresh(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                if let Err(err) = self.refresh().await {
                    warn!(error = %err, "Failed to refresh executor registry");
                }
                tokio::time::sleep(self.refresh_interval).await;
            }
        })
    }

    /// Refresh executor registry from chain storage.
    pub async fn refresh(&self) -> Result<(), ChainClientError> {
        let stakes = self.fetch_stakes().await?;
        let modes = self.fetch_node_modes().await?;
        let reputations = self.fetch_reputations().await?;

        let mut candidates = Vec::new();
        for (account, stake_info) in stakes {
            let capability = match stake_info.role {
                NodeRole::SuperNode => Some(NodeCapability::SuperNode),
                NodeRole::Director | NodeRole::ActiveDirector | NodeRole::Reserve => {
                    Some(NodeCapability::DirectorOnly)
                }
                NodeRole::Validator => Some(NodeCapability::ValidatorOnly),
                NodeRole::Relay | NodeRole::None => None,
            };

            let Some(capability) = capability else {
                continue;
            };
            let mode = modes
                .get(&account)
                .cloned()
                .unwrap_or(NodeMode::Lane1Active);
            let reputation = reputations.get(&account).copied().unwrap_or_default();

            candidates.push(ExecutorCandidate {
                id: account.to_string(),
                capability,
                reputation,
                mode,
            });
        }

        let count = candidates.len();
        *self.cache.write().expect("executor cache lock poisoned") = candidates;
        info!(count, "Executor registry refreshed from chain");
        Ok(())
    }

    async fn fetch_stakes(&self) -> Result<HashMap<AccountId32, StakeInfo>, ChainClientError> {
        let query = storage("NsnStake", "Stakes", Vec::<subxt::dynamic::Value>::new());
        let mut iter = self.client.storage().at_latest().await?.iter(query).await?;
        let mut stakes = HashMap::new();

        while let Some(Ok((key_bytes, value))) = iter.next().await {
            let account = account_from_key(&key_bytes)?;
            let stake_info = decode_value::<StakeInfo>(value.encoded())?;
            stakes.insert(account, stake_info);
        }

        Ok(stakes)
    }

    async fn fetch_node_modes(&self) -> Result<HashMap<AccountId32, NodeMode>, ChainClientError> {
        let query = storage("NsnStake", "NodeModes", Vec::<subxt::dynamic::Value>::new());
        let mut iter = self.client.storage().at_latest().await?.iter(query).await?;
        let mut modes = HashMap::new();

        while let Some(Ok((key_bytes, value))) = iter.next().await {
            let account = account_from_key(&key_bytes)?;
            let mode = decode_value::<NodeMode>(value.encoded())?;
            modes.insert(account, mode);
        }

        Ok(modes)
    }

    async fn fetch_reputations(&self) -> Result<HashMap<AccountId32, u64>, ChainClientError> {
        let query = storage(
            "NsnReputation",
            "ReputationScores",
            Vec::<subxt::dynamic::Value>::new(),
        );
        let mut iter = self.client.storage().at_latest().await?.iter(query).await?;
        let mut reputations = HashMap::new();

        while let Some(Ok((key_bytes, value))) = iter.next().await {
            let account = account_from_key(&key_bytes)?;
            let score = decode_value::<ReputationScore>(value.encoded())?;
            reputations.insert(account, score.total());
        }

        Ok(reputations)
    }
}

impl ExecutorRegistry for ChainExecutorRegistry {
    fn eligible_executors(&self, lane: Lane) -> Vec<ExecutorInfo> {
        let cache = self.cache.read().expect("executor cache lock poisoned");
        cache
            .iter()
            .filter(|candidate| mode_allows_lane(&candidate.mode, lane))
            .map(|candidate| ExecutorInfo {
                id: candidate.id.clone(),
                capability: candidate.capability,
                reputation: candidate.reputation,
            })
            .collect()
    }
}

/// On-chain attestation submitter using subxt.
pub struct ChainAttestationSubmitter {
    client: OnlineClient<PolkadotConfig>,
    signer: subxt::tx::PairSigner<PolkadotConfig, sr25519::Pair>,
}

impl ChainAttestationSubmitter {
    pub async fn new(rpc_url: String, signer: sr25519::Pair) -> Result<Self, ChainClientError> {
        let client = OnlineClient::<PolkadotConfig>::from_url(rpc_url).await?;
        let signer = subxt::tx::PairSigner::new(signer);
        Ok(Self { client, signer })
    }
}

impl AttestationSubmitter for ChainAttestationSubmitter {
    fn submit_attestation(
        &mut self,
        attestation: AttestationBundle,
    ) -> Result<(), AttestationError> {
        let attestation_cid = option_bytes(attestation.attestation_cid);
        let call = tx(
            "NsnTaskMarket",
            "submit_attestation",
            vec![
                Value::u128(attestation.task_id.0 as u128),
                Value::u128(attestation.score as u128),
                attestation_cid,
            ],
        );

        let tx_client = self.client.tx();
        let submit = tx_client.sign_and_submit_default(&call, &self.signer);

        let result = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(submit),
            Err(_) => {
                let runtime = tokio::runtime::Runtime::new()
                    .map_err(|err| AttestationError::SubmissionFailed(err.to_string()))?;
                runtime.block_on(submit)
            }
        };

        result
            .map(|_| ())
            .map_err(|err| AttestationError::SubmissionFailed(err.to_string()))
    }
}

fn mode_allows_lane(mode: &NodeMode, lane: Lane) -> bool {
    match (mode, lane) {
        (NodeMode::Lane1Active, Lane::Lane1) => true,
        (NodeMode::Lane0Active { .. }, Lane::Lane0) => true,
        (NodeMode::Draining { .. }, Lane::Lane0) => true,
        _ => false,
    }
}

fn account_from_key(key_bytes: &[u8]) -> Result<AccountId32, ChainClientError> {
    if key_bytes.len() < 32 {
        return Err(ChainClientError::InvalidKey);
    }
    let start = key_bytes.len() - 32;
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&key_bytes[start..]);
    Ok(AccountId32::from(bytes))
}

fn decode_value<T: Decode>(bytes: &[u8]) -> Result<T, ChainClientError> {
    Decode::decode(&mut &bytes[..]).map_err(|err| ChainClientError::Decode(err.to_string()))
}

fn option_bytes(value: Option<String>) -> Value {
    match value {
        Some(cid) => Value::unnamed_variant("Some", vec![Value::from_bytes(cid.as_bytes())]),
        None => Value::unnamed_variant("None", Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_allows_lane_filters_correctly() {
        assert!(mode_allows_lane(&NodeMode::Lane1Active, Lane::Lane1));
        assert!(!mode_allows_lane(&NodeMode::Lane1Active, Lane::Lane0));
        assert!(mode_allows_lane(
            &NodeMode::Lane0Active { epoch_end: 1 },
            Lane::Lane0
        ));
        assert!(mode_allows_lane(
            &NodeMode::Draining { epoch_start: 1 },
            Lane::Lane0
        ));
        assert!(!mode_allows_lane(&NodeMode::Offline, Lane::Lane1));
    }
}
