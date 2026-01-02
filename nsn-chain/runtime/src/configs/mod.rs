// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org>

mod xcm_config;

use polkadot_sdk::{staging_parachain_info as parachain_info, staging_xcm as xcm, *};
#[cfg(not(feature = "runtime-benchmarks"))]
use polkadot_sdk::{staging_xcm_builder as xcm_builder, staging_xcm_executor as xcm_executor};

// Substrate and Polkadot dependencies
use cumulus_pallet_parachain_system::RelayNumberMonotonicallyIncreases;
use cumulus_primitives_core::{AggregateMessageOrigin, ParaId};
use frame_support::{
    derive_impl,
    dispatch::DispatchClass,
    parameter_types,
    traits::{
        ConstBool, ConstU32, ConstU64, ConstU8, EitherOfDiverse, TransformOrigin, VariantCountOf,
    },
    weights::{ConstantMultiplier, Weight},
    PalletId,
};
use frame_system::{
    limits::{BlockLength, BlockWeights},
    EnsureRoot,
};
use pallet_xcm::{EnsureXcm, IsVoiceOfBody};
use parachains_common::message_queue::{NarrowOriginToSibling, ParaIdToSibling};
use polkadot_runtime_common::{
    xcm_sender::NoPriceForMessageDelivery, BlockHashCount, SlowAdjustingFeeUpdate,
};
use sp_consensus_aura::sr25519::AuthorityId as AuraId;
use sp_runtime::{traits::AccountIdConversion, Perbill};
use sp_std::vec::Vec;
use sp_version::RuntimeVersion;
use xcm::latest::prelude::BodyId;

// Local module imports
use super::{
    weights::{BlockExecutionWeight, ExtrinsicBaseWeight, RocksDbWeight},
    AccountId, Aura, Balance, Balances, Block, BlockNumber, CollatorSelection, ConsensusHook, Hash,
    MessageQueue, Nonce, PalletInfo, ParachainSystem, RandomnessCollectiveFlip, Runtime,
    RuntimeCall, RuntimeEvent, RuntimeFreezeReason, RuntimeHoldReason, RuntimeOrigin, RuntimeTask,
    Session, SessionKeys, System, WeightToFee, XcmpQueue, AVERAGE_ON_INITIALIZE_RATIO,
    EXISTENTIAL_DEPOSIT, HOURS, MAXIMUM_BLOCK_WEIGHT, MICRO_UNIT, NORMAL_DISPATCH_RATIO,
    SLOT_DURATION, UNIT, VERSION,
};
use xcm_config::{RelayLocation, XcmOriginToTransactDispatchOrigin};

parameter_types! {
    pub const Version: RuntimeVersion = VERSION;

    // This part is copied from Substrate's `bin/node/runtime/src/lib.rs`.
    //  The `RuntimeBlockLength` and `RuntimeBlockWeights` exist here because the
    // `DeletionWeightLimit` and `DeletionQueueDepth` depend on those to parameterize
    // the lazy contract deletion.
    pub RuntimeBlockLength: BlockLength =
        BlockLength::max_with_normal_ratio(5 * 1024 * 1024, NORMAL_DISPATCH_RATIO);
    pub RuntimeBlockWeights: BlockWeights = BlockWeights::builder()
        .base_block(BlockExecutionWeight::get())
        .for_class(DispatchClass::all(), |weights| {
            weights.base_extrinsic = ExtrinsicBaseWeight::get();
        })
        .for_class(DispatchClass::Normal, |weights| {
            weights.max_total = Some(NORMAL_DISPATCH_RATIO * MAXIMUM_BLOCK_WEIGHT);
        })
        .for_class(DispatchClass::Operational, |weights| {
            weights.max_total = Some(MAXIMUM_BLOCK_WEIGHT);
            // Operational transactions have some extra reserved space, so that they
            // are included even if block reached `MAXIMUM_BLOCK_WEIGHT`.
            weights.reserved = Some(
                MAXIMUM_BLOCK_WEIGHT - NORMAL_DISPATCH_RATIO * MAXIMUM_BLOCK_WEIGHT
            );
        })
        .avg_block_initialization(AVERAGE_ON_INITIALIZE_RATIO)
        .build_or_panic();
    pub const SS58Prefix: u16 = 42;
}

/// The default types are being injected by [`derive_impl`](`frame_support::derive_impl`) from
/// [`ParaChainDefaultConfig`](`struct@frame_system::config_preludes::ParaChainDefaultConfig`),
/// but overridden as needed.
#[derive_impl(frame_system::config_preludes::ParaChainDefaultConfig)]
impl frame_system::Config for Runtime {
    /// The identifier used to distinguish between accounts.
    type AccountId = AccountId;
    /// The index type for storing how many extrinsics an account has signed.
    type Nonce = Nonce;
    /// The type for hashing blocks and tries.
    type Hash = Hash;
    /// The block type.
    type Block = Block;
    /// Maximum number of block number to block hash mappings to keep (oldest pruned first).
    type BlockHashCount = BlockHashCount;
    /// Runtime version.
    type Version = Version;
    /// The data to be stored in an account.
    type AccountData = pallet_balances::AccountData<Balance>;
    /// The weight of database operations that the runtime can invoke.
    type DbWeight = RocksDbWeight;
    /// Block & extrinsics weights: base values and limits.
    type BlockWeights = RuntimeBlockWeights;
    /// The maximum length of a block (in bytes).
    type BlockLength = RuntimeBlockLength;
    /// This is used as an identifier of the chain. 42 is the generic substrate prefix.
    type SS58Prefix = SS58Prefix;
    /// The action to take on a Runtime Upgrade
    type OnSetCode = cumulus_pallet_parachain_system::ParachainSetCode<Self>;
    type MaxConsumers = frame_support::traits::ConstU32<16>;
}

/// Configure the palelt weight reclaim tx.
impl cumulus_pallet_weight_reclaim::Config for Runtime {
    type WeightInfo = ();
}

impl pallet_timestamp::Config for Runtime {
    /// A timestamp: milliseconds since the unix epoch.
    type Moment = u64;
    type OnTimestampSet = Aura;
    type MinimumPeriod = ConstU64<0>;
    type WeightInfo = ();
}

impl pallet_authorship::Config for Runtime {
    type FindAuthor = pallet_session::FindAccountFromAuthorIndex<Self, Aura>;
    type EventHandler = (CollatorSelection,);
}

parameter_types! {
    pub const ExistentialDeposit: Balance = EXISTENTIAL_DEPOSIT;
}

impl pallet_balances::Config for Runtime {
    type MaxLocks = ConstU32<50>;
    /// The type for recording an account's balance.
    type Balance = Balance;
    /// The ubiquitous event type.
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = pallet_balances::weights::SubstrateWeight<Runtime>;
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];
    type RuntimeHoldReason = RuntimeHoldReason;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type FreezeIdentifier = RuntimeFreezeReason;
    type MaxFreezes = VariantCountOf<RuntimeFreezeReason>;
    type DoneSlashHandler = ();
}

parameter_types! {
    /// Relay Chain `TransactionByteFee` / 10
    pub const TransactionByteFee: Balance = 10 * MICRO_UNIT;
}

impl pallet_transaction_payment::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type OnChargeTransaction = pallet_transaction_payment::FungibleAdapter<Balances, ()>;
    type WeightToFee = WeightToFee;
    type LengthToFee = ConstantMultiplier<Balance, TransactionByteFee>;
    type FeeMultiplierUpdate = SlowAdjustingFeeUpdate<Self>;
    type OperationalFeeMultiplier = ConstU8<5>;
    type WeightInfo = ();
}

impl pallet_sudo::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type RuntimeCall = RuntimeCall;
    type WeightInfo = ();
}

parameter_types! {
    pub const ReservedXcmpWeight: Weight = MAXIMUM_BLOCK_WEIGHT.saturating_div(4);
    pub const ReservedDmpWeight: Weight = MAXIMUM_BLOCK_WEIGHT.saturating_div(4);
    pub const RelayOrigin: AggregateMessageOrigin = AggregateMessageOrigin::Parent;
}

impl cumulus_pallet_parachain_system::Config for Runtime {
    type WeightInfo = ();
    type RuntimeEvent = RuntimeEvent;
    type OnSystemEvent = ();
    type SelfParaId = parachain_info::Pallet<Runtime>;
    type OutboundXcmpMessageSource = XcmpQueue;
    type DmpQueue = frame_support::traits::EnqueueWithOrigin<MessageQueue, RelayOrigin>;
    type ReservedDmpWeight = ReservedDmpWeight;
    type XcmpMessageHandler = XcmpQueue;
    type ReservedXcmpWeight = ReservedXcmpWeight;
    type CheckAssociatedRelayNumber = RelayNumberMonotonicallyIncreases;
    type ConsensusHook = ConsensusHook;
    type RelayParentOffset = ConstU32<1>;
}

impl parachain_info::Config for Runtime {}

parameter_types! {
    pub MessageQueueServiceWeight: Weight = Perbill::from_percent(35) * RuntimeBlockWeights::get().max_block;
}

impl pallet_message_queue::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type WeightInfo = ();
    #[cfg(feature = "runtime-benchmarks")]
    type MessageProcessor = pallet_message_queue::mock_helpers::NoopMessageProcessor<
        cumulus_primitives_core::AggregateMessageOrigin,
    >;
    #[cfg(not(feature = "runtime-benchmarks"))]
    type MessageProcessor = xcm_builder::ProcessXcmMessage<
        AggregateMessageOrigin,
        xcm_executor::XcmExecutor<xcm_config::XcmConfig>,
        RuntimeCall,
    >;
    type Size = u32;
    // The XCMP queue pallet is only ever able to handle the `Sibling(ParaId)` origin:
    type QueueChangeHandler = NarrowOriginToSibling<XcmpQueue>;
    type QueuePausedQuery = NarrowOriginToSibling<XcmpQueue>;
    type HeapSize = sp_core::ConstU32<{ 103 * 1024 }>;
    type MaxStale = sp_core::ConstU32<8>;
    type ServiceWeight = MessageQueueServiceWeight;
    type IdleMaxServiceWeight = ();
}

impl cumulus_pallet_aura_ext::Config for Runtime {}

impl cumulus_pallet_xcmp_queue::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type ChannelInfo = ParachainSystem;
    type VersionWrapper = ();
    // Enqueue XCMP messages from siblings for later processing.
    type XcmpQueue = TransformOrigin<MessageQueue, AggregateMessageOrigin, ParaId, ParaIdToSibling>;
    type MaxInboundSuspended = sp_core::ConstU32<1_000>;
    type MaxActiveOutboundChannels = ConstU32<128>;
    type MaxPageSize = ConstU32<{ 1 << 16 }>;
    type ControllerOrigin = EnsureRoot<AccountId>;
    type ControllerOriginConverter = XcmOriginToTransactDispatchOrigin;
    type WeightInfo = ();
    type PriceForSiblingDelivery = NoPriceForMessageDelivery<ParaId>;
}

parameter_types! {
    pub const Period: u32 = 6 * HOURS;
    pub const Offset: u32 = 0;
}

impl pallet_session::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type ValidatorId = <Self as frame_system::Config>::AccountId;
    // we don't have stash and controller, thus we don't need the convert as well.
    type ValidatorIdOf = pallet_collator_selection::IdentityCollator;
    type ShouldEndSession = pallet_session::PeriodicSessions<Period, Offset>;
    type NextSessionRotation = pallet_session::PeriodicSessions<Period, Offset>;
    type SessionManager = CollatorSelection;
    // Essentially just Aura, but let's be pedantic.
    type SessionHandler = <SessionKeys as sp_runtime::traits::OpaqueKeys>::KeyTypeIdProviders;
    type Keys = SessionKeys;
    type DisablingStrategy = ();
    type WeightInfo = ();
    type Currency = Balances;
    type KeyDeposit = ExistentialDeposit;
}

#[docify::export(aura_config)]
impl pallet_aura::Config for Runtime {
    type AuthorityId = AuraId;
    type DisabledValidators = ();
    type MaxAuthorities = ConstU32<100_000>;
    type AllowMultipleBlocksPerSlot = ConstBool<true>;
    type SlotDuration = ConstU64<SLOT_DURATION>;
}

parameter_types! {
    pub const PotId: PalletId = PalletId(*b"PotStake");
    pub const SessionLength: BlockNumber = 6 * HOURS;
    // StakingAdmin pluralistic body.
    pub const StakingAdminBodyId: BodyId = BodyId::Defense;
}

/// We allow root and the StakingAdmin to execute privileged collator selection operations.
pub type CollatorSelectionUpdateOrigin = EitherOfDiverse<
    EnsureRoot<AccountId>,
    EnsureXcm<IsVoiceOfBody<RelayLocation, StakingAdminBodyId>>,
>;

impl pallet_collator_selection::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    type UpdateOrigin = CollatorSelectionUpdateOrigin;
    type PotId = PotId;
    type MaxCandidates = ConstU32<100>;
    type MinEligibleCollators = ConstU32<4>;
    type MaxInvulnerables = ConstU32<20>;
    // should be a multiple of session or things will get inconsistent
    type KickThreshold = Period;
    type ValidatorId = <Self as frame_system::Config>::AccountId;
    type ValidatorIdOf = pallet_collator_selection::IdentityCollator;
    type ValidatorRegistration = Session;
    type WeightInfo = ();
}

// NSN Custom Pallet Configurations

parameter_types! {
    // ICN Stake parameters (from PRD)
    pub const MinStakeDirector: Balance = 100 * UNIT;  // 100 ICN
    pub const MinStakeSuperNode: Balance = 50 * UNIT;   // 50 ICN
    pub const MinStakeValidator: Balance = 10 * UNIT;   // 10 ICN
    pub const MinStakeRelay: Balance = 5 * UNIT;        // 5 ICN
    pub const MaxStakePerNode: Balance = 1_000 * UNIT;  // 1000 ICN (anti-centralization)
    pub const MaxRegionPercentage: u32 = 20;            // 20% max per region
    pub const DelegationMultiplier: u32 = 5;            // 5× validator stake
    pub const RegionCapBootstrapStake: Balance = 1_000 * UNIT; // Enforce caps after 1000 ICN total
    pub const MaxDelegationsPerDelegator: u32 = 10;     // L0 constraint: bounded
    pub const MaxDelegatorsPerValidator: u32 = 100;     // L0 constraint: bounded
}

parameter_types! {
    // NSN Director parameters (epoch-based election)
    pub const ChallengeBond: Balance = 25 * UNIT;
    pub const DirectorSlashAmount: Balance = 100 * UNIT;
    pub const ChallengerReward: Balance = 10 * UNIT;
    pub const MaxDirectorsPerSlot: u32 = 5;
    pub const MaxPendingSlots: u32 = 100;
    pub const EpochDuration: BlockNumber = 600; // 1 hour at 6s/block
    pub const EpochLookahead: BlockNumber = 20; // 2 minutes at 6s/block
    pub const MaxDirectorsPerEpoch: u32 = 5;
}

parameter_types! {
    // ICN Reputation parameters (from PRD)
    pub const ReputationMaxEventsPerBlock: u32 = 50;
    pub const ReputationDefaultRetentionPeriod: BlockNumber = 2_592_000;
    pub const ReputationCheckpointInterval: BlockNumber = 1_000;
    pub const ReputationDecayRatePerWeek: u64 = 5;
    pub const ReputationMaxCheckpointAccounts: u32 = 10_000;
    pub const ReputationMaxPrunePerBlock: u32 = 10_000;
}

parameter_types! {
    // Task Market parameters (Lane 1 compute marketplace)
    pub const TaskMarketMaxPendingTasks: u32 = 1_000;
    pub const TaskMarketMaxAssignedLane1Tasks: u32 = 1_000;
    pub const TaskMarketMaxAssignmentCandidates: u32 = 100;
    pub const TaskMarketMaxExpiredPerBlock: u32 = 100;
    pub const TaskMarketMaxPreemptionsPerBlock: u32 = 10;
    pub const TaskMarketMaxModelIdLen: u32 = 64;
    pub const TaskMarketMaxCidLen: u32 = 128;
    pub const TaskMarketMaxRegisteredRenderers: u32 = 256;
    pub const TaskMarketMaxLane0LatencyMs: u32 = 15_000;
    pub const TaskMarketMaxLane1LatencyMs: u32 = 120_000;
    pub const TaskMarketMaxRendererVramMb: u32 = 11_500;
    pub const TaskMarketMinEscrow: Balance = UNIT / 10; // 0.1 NSN minimum
    pub const TaskMarketMaxAttestations: u32 = 5;
    pub const TaskMarketMaxPendingVerifications: u32 = 1_000;
    pub const TaskMarketVerificationQuorum: u32 = 2;
    pub const TaskMarketVerificationPeriod: BlockNumber = 100; // ~10 minutes at 6s/block
    pub const TaskMarketMinAttestationScore: u8 = 70;
    pub const TaskMarketVerificationFailureSlash: Balance = 5 * UNIT;
    pub const TaskMarketTaskAbandonmentSlash: Balance = 5 * UNIT;
}

parameter_types! {
    // BFT parameters
    pub const BftDefaultRetentionPeriod: BlockNumber = 2_592_000; // 6 months at 6s/block
}

parameter_types! {
    // Bootstrap signer registry parameters
    pub const BootstrapMaxSigners: u32 = 8;
    pub const BootstrapMaxSignerBytes: u32 = 256;
}

parameter_types! {
    // Storage parameters
    pub const StorageAuditSlashAmount: Balance = 10 * UNIT;
    pub const StorageMaxShardsPerDeal: u32 = 20;
    pub const StorageMaxPinnersPerShard: u32 = 10;
    pub const StorageMaxActiveDeals: u32 = 100;
    pub const StorageMaxPendingAudits: u32 = 100;
    pub const StorageMaxSelectableCandidates: u32 = 50;
    pub const StoragePalletId: PalletId = PalletId(*b"nsn/stor");
}

parameter_types! {
    // Treasury parameters
    pub const TreasuryPalletId: PalletId = PalletId(*b"nsn/trea");
    pub const TreasuryDistributionFrequency: BlockNumber = 14_400; // ~1 day at 6s/block
}

pub struct TreasuryAccount;
impl frame_support::traits::Get<AccountId> for TreasuryAccount {
    fn get() -> AccountId {
        TreasuryPalletId::get().into_account_truncating()
    }
}

parameter_types! {
    // Model Registry parameters
    pub const ModelRegistryMaxModelIdLen: u32 = 64;
    pub const ModelRegistryMaxCidLen: u32 = 128;
    pub const ModelRegistryMaxHotModels: u32 = 10;
    pub const ModelRegistryMaxWarmModels: u32 = 20;
}

impl pallet_nsn_stake::Config for Runtime {
    type Currency = Balances;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type MinStakeDirector = MinStakeDirector;
    type MinStakeSuperNode = MinStakeSuperNode;
    type MinStakeValidator = MinStakeValidator;
    type MinStakeRelay = MinStakeRelay;
    type MaxStakePerNode = MaxStakePerNode;
    type MaxRegionPercentage = MaxRegionPercentage;
    type RegionCapBootstrapStake = RegionCapBootstrapStake;
    type DelegationMultiplier = DelegationMultiplier;
    type MaxDelegationsPerDelegator = MaxDelegationsPerDelegator;
    type MaxDelegatorsPerValidator = MaxDelegatorsPerValidator;
    type WeightInfo = pallet_nsn_stake::weights::SubstrateWeight<Runtime>;
}

impl pallet_nsn_reputation::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type MaxEventsPerBlock = ReputationMaxEventsPerBlock;
    type DefaultRetentionPeriod = ReputationDefaultRetentionPeriod;
    type CheckpointInterval = ReputationCheckpointInterval;
    type DecayRatePerWeek = ReputationDecayRatePerWeek;
    type MaxCheckpointAccounts = ReputationMaxCheckpointAccounts;
    type MaxPrunePerBlock = ReputationMaxPrunePerBlock;
    type WeightInfo = pallet_nsn_reputation::weights::SubstrateWeight<Runtime>;
}

/// Wrapper struct for LaneNodeProvider implementation (avoids orphan rules)
pub struct StakeLaneNodeProvider;

impl pallet_nsn_task_market::LaneNodeProvider<AccountId, Balance> for StakeLaneNodeProvider {
    fn eligible_nodes(
        lane: pallet_nsn_task_market::TaskLane,
        max: u32,
    ) -> Vec<(AccountId, Balance)> {
        use pallet_nsn_stake::{NodeMode, NodeModes, NodeRole, Stakes};
        use sp_runtime::traits::Zero;

        let limit = max as usize;
        if limit == 0 {
            return Vec::new();
        }

        Stakes::<Runtime>::iter()
            .filter(|(_, stake)| !stake.amount.is_zero())
            .filter(|(account, stake)| {
                let mode = NodeModes::<Runtime>::get(account);
                match lane {
                    pallet_nsn_task_market::TaskLane::Lane0 => {
                        matches!(mode, NodeMode::Lane0Active { .. })
                            && matches!(stake.role, NodeRole::ActiveDirector)
                    }
                    pallet_nsn_task_market::TaskLane::Lane1 => {
                        matches!(mode, NodeMode::Lane1Active)
                            && matches!(stake.role, NodeRole::Reserve | NodeRole::Director)
                    }
                }
            })
            .take(limit)
            .map(|(account, stake)| (account, stake.amount))
            .collect()
    }

    fn is_eligible(account: &AccountId, lane: pallet_nsn_task_market::TaskLane) -> bool {
        use pallet_nsn_stake::{NodeMode, NodeModes, NodeRole, Stakes};
        use sp_runtime::traits::Zero;

        let stake = Stakes::<Runtime>::get(account);
        if stake.amount.is_zero() {
            return false;
        }

        let mode = NodeModes::<Runtime>::get(account);
        match lane {
            pallet_nsn_task_market::TaskLane::Lane0 => {
                matches!(mode, NodeMode::Lane0Active { .. })
                    && matches!(stake.role, NodeRole::ActiveDirector)
            }
            pallet_nsn_task_market::TaskLane::Lane1 => {
                matches!(mode, NodeMode::Lane1Active)
                    && matches!(stake.role, NodeRole::Reserve | NodeRole::Director)
            }
        }
    }
}

/// Wrapper struct for ReputationUpdater implementation (avoids orphan rules)
pub struct ReputationUpdaterWrapper;

impl pallet_nsn_task_market::ReputationUpdater<AccountId> for ReputationUpdaterWrapper {
    fn record_task_result(account: &AccountId, success: bool) {
        let _ = pallet_nsn_reputation::Pallet::<Runtime>::record_task_outcome(account, success);
    }
}

/// Wrapper struct for TaskSlashHandler implementation (avoids orphan rules)
pub struct StakeSlashHandler;

impl pallet_nsn_task_market::TaskSlashHandler<AccountId, Balance> for StakeSlashHandler {
    fn slash_for_abandonment(
        account: &AccountId,
        amount: Balance,
    ) -> frame_support::dispatch::DispatchResult {
        pallet_nsn_stake::Pallet::<Runtime>::slash_for_abandonment(account, amount)
    }
}

/// Wrapper for validator eligibility checks.
pub struct StakeValidatorProvider;

impl pallet_nsn_task_market::ValidatorProvider<AccountId> for StakeValidatorProvider {
    fn is_validator(account: &AccountId) -> bool {
        use pallet_nsn_stake::{NodeRole, Stakes};
        use sp_runtime::traits::Zero;

        let stake = Stakes::<Runtime>::get(account);
        if stake.amount.is_zero() {
            return false;
        }
        matches!(
            stake.role,
            NodeRole::Validator
                | NodeRole::SuperNode
                | NodeRole::Director
                | NodeRole::ActiveDirector
                | NodeRole::Reserve
        )
    }
}

impl pallet_insecure_randomness_collective_flip::Config for Runtime {}

impl pallet_nsn_director::Config for Runtime {
    type Currency = Balances;
    type RuntimeHoldReason = RuntimeHoldReason;
    type Randomness = RandomnessCollectiveFlip;
    type ChallengeBond = ChallengeBond;
    type DirectorSlashAmount = DirectorSlashAmount;
    type ChallengerReward = ChallengerReward;
    type MaxDirectorsPerSlot = MaxDirectorsPerSlot;
    type MaxPendingSlots = MaxPendingSlots;
    type EpochDuration = EpochDuration;
    type EpochLookahead = EpochLookahead;
    type MaxDirectorsPerEpoch = MaxDirectorsPerEpoch;
    type NodeModeUpdater = pallet_nsn_stake::Pallet<Runtime>;
    type NodeRoleUpdater = pallet_nsn_stake::Pallet<Runtime>;
    type WeightInfo = pallet_nsn_director::weights::SubstrateWeight<Runtime>;
}

impl pallet_nsn_bft::Config for Runtime {
    type DefaultRetentionPeriod = BftDefaultRetentionPeriod;
    type WeightInfo = pallet_nsn_bft::weights::SubstrateWeight<Runtime>;
}

impl pallet_nsn_bootstrap::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type MaxSigners = BootstrapMaxSigners;
    type MaxSignerBytes = BootstrapMaxSignerBytes;
    type WeightInfo = ();
}

impl pallet_nsn_storage::Config for Runtime {
    type Currency = Balances;
    type RuntimeHoldReason = RuntimeHoldReason;
    type Randomness = RandomnessCollectiveFlip;
    type AuditSlashAmount = StorageAuditSlashAmount;
    type MaxShardsPerDeal = StorageMaxShardsPerDeal;
    type MaxPinnersPerShard = StorageMaxPinnersPerShard;
    type MaxActiveDeals = StorageMaxActiveDeals;
    type MaxPendingAudits = StorageMaxPendingAudits;
    type MaxSelectableCandidates = StorageMaxSelectableCandidates;
    type PalletId = StoragePalletId;
    type WeightInfo = pallet_nsn_storage::weights::SubstrateWeight<Runtime>;
}

impl pallet_nsn_treasury::Config for Runtime {
    type Currency = Balances;
    type PalletId = TreasuryPalletId;
    type DistributionFrequency = TreasuryDistributionFrequency;
    type WeightInfo = pallet_nsn_treasury::weights::SubstrateWeight<Runtime>;
}

impl pallet_nsn_task_market::Config for Runtime {
    type Currency = Balances;
    type MaxPendingTasks = TaskMarketMaxPendingTasks;
    type MaxAssignedLane1Tasks = TaskMarketMaxAssignedLane1Tasks;
    type MaxAssignmentCandidates = TaskMarketMaxAssignmentCandidates;
    type MaxExpiredPerBlock = TaskMarketMaxExpiredPerBlock;
    type MaxPreemptionsPerBlock = TaskMarketMaxPreemptionsPerBlock;
    type MaxModelIdLen = TaskMarketMaxModelIdLen;
    type MaxCidLen = TaskMarketMaxCidLen;
    type MaxRegisteredRenderers = TaskMarketMaxRegisteredRenderers;
    type MaxLane0LatencyMs = TaskMarketMaxLane0LatencyMs;
    type MaxLane1LatencyMs = TaskMarketMaxLane1LatencyMs;
    type MaxRendererVramMb = TaskMarketMaxRendererVramMb;
    type MinEscrow = TaskMarketMinEscrow;
    type MaxAttestations = TaskMarketMaxAttestations;
    type MaxPendingVerifications = TaskMarketMaxPendingVerifications;
    type VerificationQuorum = TaskMarketVerificationQuorum;
    type VerificationPeriod = TaskMarketVerificationPeriod;
    type MinAttestationScore = TaskMarketMinAttestationScore;
    type VerificationFailureSlash = TaskMarketVerificationFailureSlash;
    type LaneNodeProvider = StakeLaneNodeProvider;
    type ReputationUpdater = ReputationUpdaterWrapper;
    type TaskSlashHandler = StakeSlashHandler;
    type ValidatorProvider = StakeValidatorProvider;
    type TaskAbandonmentSlash = TaskMarketTaskAbandonmentSlash;
    type TreasuryAccount = TreasuryAccount;
    type WeightInfo = pallet_nsn_task_market::weights::SubstrateWeight<Runtime>;
    type RendererRegistrarOrigin = frame_system::EnsureRoot<Self::AccountId>;
    type Randomness = RandomnessCollectiveFlip;
}

impl pallet_nsn_model_registry::Config for Runtime {
    type MaxModelIdLen = ModelRegistryMaxModelIdLen;
    type MaxCidLen = ModelRegistryMaxCidLen;
    type MaxHotModels = ModelRegistryMaxHotModels;
    type MaxWarmModels = ModelRegistryMaxWarmModels;
    type WeightInfo = pallet_nsn_model_registry::weights::SubstrateWeight<Runtime>;
}
