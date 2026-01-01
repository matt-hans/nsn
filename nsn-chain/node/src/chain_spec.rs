use polkadot_sdk::*;

use nsn_runtime as runtime;
use sc_chain_spec::{ChainSpecExtension, ChainSpecGroup};
use sc_service::ChainType;
use serde::{Deserialize, Serialize};

/// Specialized `ChainSpec` for the NSN runtime.
pub type ChainSpec = sc_service::GenericChainSpec<Extensions>;
/// The relay chain that you want to configure this parachain to connect to (for future parachain migration).
pub const RELAY_CHAIN: &str = "rococo-local";

/// The extensions for the [`ChainSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ChainSpecGroup, ChainSpecExtension)]
pub struct Extensions {
    /// The relay chain of the Parachain.
    #[serde(alias = "relayChain", alias = "RelayChain")]
    pub relay_chain: String,
    /// The id of the Parachain.
    #[serde(alias = "paraId", alias = "ParaId")]
    pub para_id: u32,
}

impl Extensions {
    /// Try to get the extension from the given `ChainSpec`.
    pub fn try_get(chain_spec: &dyn sc_service::ChainSpec) -> Option<&Self> {
        sc_chain_spec::get_extension(chain_spec.extensions())
    }
}

/// Helper to create NSN chain properties
fn nsn_properties() -> sc_chain_spec::Properties {
    let mut properties = sc_chain_spec::Properties::new();
    properties.insert("tokenSymbol".into(), "NSN".into());
    properties.insert("tokenDecimals".into(), 18.into());
    properties.insert("ss58Format".into(), 42.into()); // Generic SS58 format (can be updated to NSN-specific later)
    properties
}

pub fn development_chain_spec() -> ChainSpec {
    ChainSpec::builder(
        runtime::WASM_BINARY.expect("WASM binary was not built, please build it!"),
        Extensions {
            relay_chain: RELAY_CHAIN.into(),
            para_id: runtime::PARACHAIN_ID,
        },
    )
    .with_name("NSN Development")
    .with_id("nsn-dev")
    .with_chain_type(ChainType::Development)
    .with_genesis_config_preset_name(sp_genesis_builder::DEV_RUNTIME_PRESET)
    .with_properties(nsn_properties())
    .with_protocol_id("nsn")
    .build()
}

pub fn local_chain_spec() -> ChainSpec {
    #[allow(deprecated)]
    ChainSpec::builder(
        runtime::WASM_BINARY.expect("WASM binary was not built, please build it!"),
        Extensions {
            relay_chain: RELAY_CHAIN.into(),
            para_id: runtime::PARACHAIN_ID,
        },
    )
    .with_name("NSN Local Testnet")
    .with_id("nsn-local")
    .with_chain_type(ChainType::Local)
    .with_genesis_config_preset_name(sc_chain_spec::LOCAL_TESTNET_RUNTIME_PRESET)
    .with_protocol_id("nsn-local")
    .with_properties(nsn_properties())
    .build()
}

/// NSN Testnet chain spec
/// Public testnet for integration testing before mainnet launch
pub fn nsn_testnet_chain_spec() -> ChainSpec {
    ChainSpec::builder(
        runtime::WASM_BINARY.expect("WASM binary was not built, please build it!"),
        Extensions {
            relay_chain: RELAY_CHAIN.into(),
            para_id: runtime::PARACHAIN_ID,
        },
    )
    .with_name("NSN Testnet")
    .with_id("nsn-testnet")
    .with_chain_type(ChainType::Live)
    .with_genesis_config_preset_name(runtime::genesis_config_presets::NSN_TESTNET_PRESET)
    .with_protocol_id("nsn")
    .with_properties(nsn_properties())
    // TODO: Add bootnode addresses when infrastructure is ready
    // .with_boot_nodes(vec![
    //     "/dns/boot1.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    //     "/dns/boot2.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    // ])
    .build()
}

/// NSN Mainnet chain spec (TEMPLATE - DO NOT USE IN PRODUCTION)
/// WARNING: Replace genesis accounts and validator keys before mainnet launch
pub fn nsn_mainnet_chain_spec() -> ChainSpec {
    ChainSpec::builder(
        runtime::WASM_BINARY.expect("WASM binary was not built, please build it!"),
        Extensions {
            relay_chain: RELAY_CHAIN.into(),
            para_id: runtime::PARACHAIN_ID,
        },
    )
    .with_name("NSN Mainnet")
    .with_id("nsn-mainnet")
    .with_chain_type(ChainType::Live)
    .with_genesis_config_preset_name(runtime::genesis_config_presets::NSN_MAINNET_PRESET)
    .with_protocol_id("nsn")
    .with_properties(nsn_properties())
    // TODO: Add production bootnode addresses
    // .with_boot_nodes(vec![
    //     "/dns/boot1.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    //     "/dns/boot2.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    //     "/dns/boot3.nsn.network/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    // ])
    .build()
}
