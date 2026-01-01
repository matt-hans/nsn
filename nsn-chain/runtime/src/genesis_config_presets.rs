use crate::{
    AccountId, BalancesConfig, CollatorSelectionConfig, ParachainInfoConfig, PolkadotXcmConfig,
    RuntimeGenesisConfig, SessionConfig, SessionKeys, SudoConfig, EXISTENTIAL_DEPOSIT, NSN,
};

use alloc::{vec, vec::Vec};

use polkadot_sdk::{staging_xcm as xcm, *};

use cumulus_primitives_core::ParaId;
use frame_support::build_struct_json_patch;
use parachains_common::AuraId;
use serde_json::Value;
use sp_genesis_builder::PresetId;
use sp_keyring::Sr25519Keyring;

/// The default XCM version to set in genesis config.
const SAFE_XCM_VERSION: u32 = xcm::prelude::XCM_VERSION;
/// Parachain id used for genesis config presets of NSN parachain.
/// Note: NSN Chain starts as a solochain and may optionally migrate to parachain later.
#[docify::export_content]
pub const PARACHAIN_ID: u32 = 2000;

/// Generate the session keys from individual elements.
///
/// The input must be a tuple of individual keys (a single arg for now since we have just one key).
pub fn template_session_keys(keys: AuraId) -> SessionKeys {
    SessionKeys { aura: keys }
}

fn testnet_genesis(
    invulnerables: Vec<(AccountId, AuraId)>,
    endowed_accounts: Vec<AccountId>,
    root: AccountId,
    id: ParaId,
) -> Value {
    build_struct_json_patch!(RuntimeGenesisConfig {
        balances: BalancesConfig {
            balances: endowed_accounts
                .iter()
                .cloned()
                .map(|k| (k, 1u128 << 60))
                .collect::<Vec<_>>(),
        },
        parachain_info: ParachainInfoConfig { parachain_id: id },
        collator_selection: CollatorSelectionConfig {
            invulnerables: invulnerables
                .iter()
                .cloned()
                .map(|(acc, _)| acc)
                .collect::<Vec<_>>(),
            candidacy_bond: EXISTENTIAL_DEPOSIT * 16,
        },
        session: SessionConfig {
            keys: invulnerables
                .into_iter()
                .map(|(acc, aura)| {
                    (
                        acc.clone(),                 // account id
                        acc,                         // validator id
                        template_session_keys(aura), // session keys
                    )
                })
                .collect::<Vec<_>>(),
        },
        polkadot_xcm: PolkadotXcmConfig {
            safe_xcm_version: Some(SAFE_XCM_VERSION)
        },
        sudo: SudoConfig { key: Some(root) },
    })
}

fn local_testnet_genesis() -> Value {
    testnet_genesis(
        // initial collators.
        vec![
            (
                Sr25519Keyring::Alice.to_account_id(),
                Sr25519Keyring::Alice.public().into(),
            ),
            (
                Sr25519Keyring::Bob.to_account_id(),
                Sr25519Keyring::Bob.public().into(),
            ),
        ],
        Sr25519Keyring::well_known()
            .map(|k| k.to_account_id())
            .collect(),
        Sr25519Keyring::Alice.to_account_id(),
        PARACHAIN_ID.into(),
    )
}

fn development_config_genesis() -> Value {
    testnet_genesis(
        // initial collators.
        vec![
            (
                Sr25519Keyring::Alice.to_account_id(),
                Sr25519Keyring::Alice.public().into(),
            ),
            (
                Sr25519Keyring::Bob.to_account_id(),
                Sr25519Keyring::Bob.public().into(),
            ),
        ],
        Sr25519Keyring::well_known()
            .map(|k| k.to_account_id())
            .collect(),
        Sr25519Keyring::Alice.to_account_id(),
        PARACHAIN_ID.into(),
    )
}

/// NSN Testnet genesis configuration
/// - 3 initial validators (Alice, Bob, Charlie)
/// - Generous token allocations for testing
/// - Faucet account for distribution
fn nsn_testnet_genesis() -> Value {
    testnet_genesis(
        // initial validators (Alice, Bob, Charlie)
        vec![
            (
                Sr25519Keyring::Alice.to_account_id(),
                Sr25519Keyring::Alice.public().into(),
            ),
            (
                Sr25519Keyring::Bob.to_account_id(),
                Sr25519Keyring::Bob.public().into(),
            ),
            (
                Sr25519Keyring::Charlie.to_account_id(),
                Sr25519Keyring::Charlie.public().into(),
            ),
        ],
        // Endowed accounts for testnet
        vec![
            Sr25519Keyring::Alice.to_account_id(),
            Sr25519Keyring::Bob.to_account_id(),
            Sr25519Keyring::Charlie.to_account_id(),
            Sr25519Keyring::Dave.to_account_id(),
            Sr25519Keyring::Eve.to_account_id(),
            Sr25519Keyring::Ferdie.to_account_id(),
        ],
        // Sudo key (Alice for testnet)
        Sr25519Keyring::Alice.to_account_id(),
        PARACHAIN_ID.into(),
    )
}

/// NSN Mainnet genesis configuration template
/// WARNING: This is a TEMPLATE. Replace with actual production keys before mainnet launch.
/// Total supply: 1B NSN (exactly)
/// Allocations:
/// - Treasury: 39.9% (399M NSN) - includes 1M operational budget
/// - Development Fund: 20% (200M NSN)
/// - Ecosystem Growth: 15% (150M NSN)
/// - Team & Advisors: 15% (150M NSN) - with vesting
/// - Initial Liquidity: 10% (100M NSN)
/// - Operational: 0.1% (1M NSN) - sudo account for chain operations
fn nsn_mainnet_genesis_template() -> Value {
    const TOTAL_SUPPLY: Balance = 1_000_000_000 * NSN; // 1 billion NSN

    // WARNING: Replace these with actual production accounts before mainnet
    let treasury_account = Sr25519Keyring::Alice.to_account_id(); // REPLACE
    let dev_fund_account = Sr25519Keyring::Bob.to_account_id(); // REPLACE
    let ecosystem_account = Sr25519Keyring::Charlie.to_account_id(); // REPLACE
    let team_account = Sr25519Keyring::Dave.to_account_id(); // REPLACE
    let liquidity_account = Sr25519Keyring::Eve.to_account_id(); // REPLACE
    let sudo_account = Sr25519Keyring::Ferdie.to_account_id(); // REPLACE with multisig

    // Initial validators (replace with actual production validator keys)
    let validators = vec![
        (
            Sr25519Keyring::Alice.to_account_id(),
            Sr25519Keyring::Alice.public().into(),
        ),
        (
            Sr25519Keyring::Bob.to_account_id(),
            Sr25519Keyring::Bob.public().into(),
        ),
        (
            Sr25519Keyring::Charlie.to_account_id(),
            Sr25519Keyring::Charlie.public().into(),
        ),
    ];

    // Mainnet allocations
    let endowed_accounts = vec![
        treasury_account.clone(),
        dev_fund_account.clone(),
        ecosystem_account.clone(),
        team_account.clone(),
        liquidity_account.clone(),
        sudo_account.clone(),
    ];

    // Custom balances for mainnet
    // Total Supply: 1B NSN exactly
    // Treasury allocation adjusted to include 1M operational budget
    build_struct_json_patch!(RuntimeGenesisConfig {
        balances: BalancesConfig {
            balances: vec![
                (treasury_account, TOTAL_SUPPLY * 40 / 100 - 1_000_000 * NSN), // 399M NSN (40% - operational)
                (dev_fund_account, TOTAL_SUPPLY * 20 / 100),                   // 200M NSN
                (ecosystem_account, TOTAL_SUPPLY * 15 / 100),                  // 150M NSN
                (team_account, TOTAL_SUPPLY * 15 / 100), // 150M NSN (vesting TBD)
                (liquidity_account, TOTAL_SUPPLY * 10 / 100), // 100M NSN
                (sudo_account, 1_000_000 * NSN),         // 1M NSN for operational expenses
            ],
        },
        parachain_info: ParachainInfoConfig {
            parachain_id: PARACHAIN_ID.into()
        },
        collator_selection: CollatorSelectionConfig {
            invulnerables: validators
                .iter()
                .cloned()
                .map(|(acc, _)| acc)
                .collect::<Vec<_>>(),
            candidacy_bond: 100 * NSN, // 100 NSN minimum bond for validators
        },
        session: SessionConfig {
            keys: validators
                .into_iter()
                .map(|(acc, aura)| {
                    (
                        acc.clone(),                 // account id
                        acc,                         // validator id
                        template_session_keys(aura), // session keys
                    )
                })
                .collect::<Vec<_>>(),
        },
        polkadot_xcm: PolkadotXcmConfig {
            safe_xcm_version: Some(SAFE_XCM_VERSION)
        },
        sudo: SudoConfig {
            key: Some(sudo_account)
        },
    })
}

/// NSN-specific preset IDs
pub const NSN_TESTNET_PRESET: &str = "nsn-testnet";
pub const NSN_MAINNET_PRESET: &str = "nsn-mainnet";

/// Provides the JSON representation of predefined genesis config for given `id`.
pub fn get_preset(id: &PresetId) -> Option<vec::Vec<u8>> {
    let patch = match id.as_ref() {
        sp_genesis_builder::LOCAL_TESTNET_RUNTIME_PRESET => local_testnet_genesis(),
        sp_genesis_builder::DEV_RUNTIME_PRESET => development_config_genesis(),
        NSN_TESTNET_PRESET => nsn_testnet_genesis(),
        NSN_MAINNET_PRESET => nsn_mainnet_genesis_template(),
        _ => return None,
    };
    Some(
        serde_json::to_string(&patch)
            .expect("serialization to json is expected to work. qed.")
            .into_bytes(),
    )
}

/// List of supported presets.
pub fn preset_names() -> Vec<PresetId> {
    vec![
        PresetId::from(sp_genesis_builder::DEV_RUNTIME_PRESET),
        PresetId::from(sp_genesis_builder::LOCAL_TESTNET_RUNTIME_PRESET),
        PresetId::from(NSN_TESTNET_PRESET),
        PresetId::from(NSN_MAINNET_PRESET),
    ]
}
