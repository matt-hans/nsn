// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration test mock runtime with all NSN pallets configured together.
// Unlike unit test mocks, this uses REAL trait implementations for inter-pallet calls.

use frame_support::{
    construct_runtime, derive_impl, parameter_types,
    traits::{ConstU32, ConstU64, Hooks},
    PalletId,
};
use pallet_nsn_stake::Region;
use sp_core::H256;
use sp_runtime::{
    traits::{BlakeTwo256, IdentityLookup},
    BuildStorage, Perbill,
};
use sp_std::vec::Vec;

pub type AccountId = u64;
pub type Balance = u128;
pub type BlockNumber = u32;

type Block = frame_system::mocking::MockBlockU32<Test>;

// Configure mock runtime with ALL NSN pallets
construct_runtime!(
    pub enum Test
    {
        System: frame_system,
        Balances: pallet_balances,
        NsnStake: pallet_nsn_stake,
        NsnReputation: pallet_nsn_reputation,
        NsnDirector: pallet_nsn_director,
        NsnBft: pallet_nsn_bft,
        NsnTreasury: pallet_nsn_treasury,
        NsnTaskMarket: pallet_nsn_task_market,
    }
);

parameter_types! {
    pub const BlockHashCount: u32 = 250;
}

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
    type BaseCallFilter = frame_support::traits::Everything;
    type BlockWeights = ();
    type BlockLength = ();
    type DbWeight = ();
    type RuntimeOrigin = RuntimeOrigin;
    type RuntimeCall = RuntimeCall;
    type RuntimeTask = RuntimeTask;
    type Nonce = u64;
    type Hash = H256;
    type Hashing = BlakeTwo256;
    type AccountId = AccountId;
    type Lookup = IdentityLookup<Self::AccountId>;
    type Block = Block;
    type RuntimeEvent = RuntimeEvent;
    type BlockHashCount = BlockHashCount;
    type Version = ();
    type PalletInfo = PalletInfo;
    type AccountData = pallet_balances::AccountData<Balance>;
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
    type SS58Prefix = ();
    type OnSetCode = ();
    type MaxConsumers = ConstU32<16>;
}

parameter_types! {
    pub const ExistentialDeposit: Balance = 1;
}

impl pallet_balances::Config for Test {
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ConstU32<50>;
    type ReserveIdentifier = [u8; 8];
    type Balance = Balance;
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ExistentialDeposit;
    type AccountStore = System;
    type WeightInfo = ();
    type FreezeIdentifier = RuntimeFreezeReason;
    type MaxFreezes = ConstU32<50>;
    type RuntimeHoldReason = RuntimeHoldReason;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type DoneSlashHandler = ();
}

// ============================================================================
// NSN Stake Pallet Configuration
// ============================================================================

parameter_types! {
    pub const MinStakeDirector: Balance = 100_000_000_000_000_000_000; // 100 NSN
    pub const MinStakeSuperNode: Balance = 50_000_000_000_000_000_000; // 50 NSN
    pub const MinStakeValidator: Balance = 10_000_000_000_000_000_000; // 10 NSN
    pub const MinStakeRelay: Balance = 5_000_000_000_000_000_000; // 5 NSN
    pub const MaxStakePerNode: Balance = 1_000_000_000_000_000_000_000; // 1000 NSN
    pub const MaxRegionPercentage: u32 = 20;
    pub const DelegationMultiplier: u32 = 5;
    pub const MaxDelegationsPerDelegator: u32 = 10;
    pub const MaxDelegatorsPerValidator: u32 = 100;
    pub const RegionCapBootstrapStake: Balance = 1_000_000_000_000_000_000_000; // 1000 NSN
}

impl pallet_nsn_stake::Config for Test {
    type Currency = Balances;
    type RuntimeFreezeReason = RuntimeFreezeReason;
    type MinStakeDirector = MinStakeDirector;
    type MinStakeSuperNode = MinStakeSuperNode;
    type MinStakeValidator = MinStakeValidator;
    type MinStakeRelay = MinStakeRelay;
    type MaxStakePerNode = MaxStakePerNode;
    type MaxRegionPercentage = MaxRegionPercentage;
    type DelegationMultiplier = DelegationMultiplier;
    type MaxDelegationsPerDelegator = MaxDelegationsPerDelegator;
    type MaxDelegatorsPerValidator = MaxDelegatorsPerValidator;
    type RegionCapBootstrapStake = RegionCapBootstrapStake;
    type WeightInfo = ();
}

// ============================================================================
// NSN Reputation Pallet Configuration
// ============================================================================

parameter_types! {
    pub const MaxEventsPerBlock: u32 = 50;
    pub const DefaultRetentionPeriod: BlockNumber = 2592000; // ~6 months
    pub const CheckpointInterval: BlockNumber = 1000;
    pub const DecayRatePerWeek: u64 = 5;
    pub const MaxCheckpointAccounts: u32 = 10_000;
    pub const MaxPrunePerBlock: u32 = 10_000;
}

impl pallet_nsn_reputation::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type MaxEventsPerBlock = MaxEventsPerBlock;
    type DefaultRetentionPeriod = DefaultRetentionPeriod;
    type CheckpointInterval = CheckpointInterval;
    type DecayRatePerWeek = DecayRatePerWeek;
    type MaxCheckpointAccounts = MaxCheckpointAccounts;
    type MaxPrunePerBlock = MaxPrunePerBlock;
    type WeightInfo = ();
}

// ============================================================================
// NSN Director Pallet Configuration
// ============================================================================

parameter_types! {
    pub const ChallengeBond: Balance = 25_000_000_000_000_000_000; // 25 NSN
    pub const DirectorSlashAmount: Balance = 100_000_000_000_000_000_000; // 100 NSN
    pub const ChallengerReward: Balance = 10_000_000_000_000_000_000; // 10 NSN
    pub const MaxDirectorsPerSlot: u32 = 5;
    pub const MaxPendingSlots: u32 = 100;
    pub const EpochDuration: BlockNumber = 100; // Shorter for testing (100 blocks)
    pub const EpochLookahead: BlockNumber = 10; // 10 blocks for testing
    pub const MaxDirectorsPerEpoch: u32 = 5;
}

/// Simple deterministic randomness for testing
pub struct TestRandomness;

impl frame_support::traits::Randomness<H256, BlockNumber> for TestRandomness {
    fn random(subject: &[u8]) -> (H256, BlockNumber) {
        let hash = BlakeTwo256::hash(subject);
        (hash, 0)
    }
}

impl pallet_nsn_director::Config for Test {
    type Currency = Balances;
    type RuntimeHoldReason = RuntimeHoldReason;
    type Randomness = TestRandomness;
    type ChallengeBond = ChallengeBond;
    type DirectorSlashAmount = DirectorSlashAmount;
    type ChallengerReward = ChallengerReward;
    type MaxDirectorsPerSlot = MaxDirectorsPerSlot;
    type MaxPendingSlots = MaxPendingSlots;
    type EpochDuration = EpochDuration;
    type EpochLookahead = EpochLookahead;
    type MaxDirectorsPerEpoch = MaxDirectorsPerEpoch;
    // REAL trait implementations - not mocks!
    type NodeModeUpdater = NsnStake;
    type NodeRoleUpdater = NsnStake;
    type WeightInfo = ();
}

// ============================================================================
// NSN BFT Pallet Configuration
// ============================================================================

parameter_types! {
    pub const BftDefaultRetentionPeriod: BlockNumber = 2592000; // ~6 months
}

impl pallet_nsn_bft::Config for Test {
    type DefaultRetentionPeriod = BftDefaultRetentionPeriod;
    type WeightInfo = ();
}

// ============================================================================
// NSN Treasury Pallet Configuration
// ============================================================================

parameter_types! {
    pub const TreasuryPalletId: PalletId = PalletId(*b"nsn/trea");
    pub const DistributionFrequency: BlockNumber = 100; // Shorter for testing
}

impl pallet_nsn_treasury::Config for Test {
    type Currency = Balances;
    type PalletId = TreasuryPalletId;
    type DistributionFrequency = DistributionFrequency;
    type WeightInfo = ();
}

// ============================================================================
// NSN Task Market Pallet Configuration
// ============================================================================

parameter_types! {
    pub const MaxPendingTasks: u32 = 100;
    pub const MaxAssignedLane1Tasks: u32 = 100;
    pub const MaxAssignmentCandidates: u32 = 10;
    pub const MaxExpiredPerBlock: u32 = 25;
    pub const MaxPreemptionsPerBlock: u32 = 10;
    pub const MaxModelIdLen: u32 = 64;
    pub const MaxCidLen: u32 = 64;
    pub const MaxRegisteredRenderers: u32 = 32;
    pub const MaxLane0LatencyMs: u32 = 15_000;
    pub const MaxLane1LatencyMs: u32 = 120_000;
    pub const MaxRendererVramMb: u32 = 11_500;
    pub const MinEscrow: Balance = 10_000_000_000_000_000_000; // 10 NSN
    pub const MaxAttestations: u32 = 5;
    pub const MaxPendingVerifications: u32 = 100;
    pub const VerificationQuorum: u32 = 2;
    pub const VerificationPeriod: BlockNumber = 10;
    pub const MinAttestationScore: u8 = 70;
    pub const VerificationFailureSlash: Balance = 5_000_000_000_000_000_000; // 5 NSN
    pub const TaskAbandonmentSlash: Balance = 5_000_000_000_000_000_000; // 5 NSN
}

/// REAL LaneNodeProvider that queries NsnStake pallet
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

        Stakes::<Test>::iter()
            .filter(|(_, stake)| !stake.amount.is_zero())
            .filter(|(account, stake)| {
                let mode = NodeModes::<Test>::get(account);
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

        let stake = Stakes::<Test>::get(account);
        if stake.amount.is_zero() {
            return false;
        }

        let mode = NodeModes::<Test>::get(account);
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

/// REAL ReputationUpdater that calls NsnReputation pallet
pub struct ReputationUpdaterImpl;

impl pallet_nsn_task_market::ReputationUpdater<AccountId> for ReputationUpdaterImpl {
    fn record_task_result(account: &AccountId, success: bool) {
        let _ = pallet_nsn_reputation::Pallet::<Test>::record_task_outcome(account, success);
    }
}

/// REAL TaskSlashHandler that calls NsnStake pallet
pub struct StakeSlashHandler;

impl pallet_nsn_task_market::TaskSlashHandler<AccountId, Balance> for StakeSlashHandler {
    fn slash_for_abandonment(
        account: &AccountId,
        amount: Balance,
    ) -> frame_support::dispatch::DispatchResult {
        pallet_nsn_stake::Pallet::<Test>::slash_for_abandonment(account, amount)
    }
}

/// REAL ValidatorProvider that queries NsnStake pallet
pub struct StakeValidatorProvider;

impl pallet_nsn_task_market::ValidatorProvider<AccountId> for StakeValidatorProvider {
    fn is_validator(account: &AccountId) -> bool {
        use pallet_nsn_stake::{NodeRole, Stakes};
        use sp_runtime::traits::Zero;

        let stake = Stakes::<Test>::get(account);
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

pub struct TreasuryAccount;
impl frame_support::traits::Get<AccountId> for TreasuryAccount {
    fn get() -> AccountId {
        // Treasury pallet account
        use sp_runtime::traits::AccountIdConversion;
        TreasuryPalletId::get().into_account_truncating()
    }
}

impl pallet_nsn_task_market::Config for Test {
    type Currency = Balances;
    type MaxPendingTasks = MaxPendingTasks;
    type MaxAssignedLane1Tasks = MaxAssignedLane1Tasks;
    type MaxAssignmentCandidates = MaxAssignmentCandidates;
    type MaxExpiredPerBlock = MaxExpiredPerBlock;
    type MaxPreemptionsPerBlock = MaxPreemptionsPerBlock;
    type MaxModelIdLen = MaxModelIdLen;
    type MaxCidLen = MaxCidLen;
    type MaxRegisteredRenderers = MaxRegisteredRenderers;
    type MaxLane0LatencyMs = MaxLane0LatencyMs;
    type MaxLane1LatencyMs = MaxLane1LatencyMs;
    type MaxRendererVramMb = MaxRendererVramMb;
    type MinEscrow = MinEscrow;
    type MaxAttestations = MaxAttestations;
    type MaxPendingVerifications = MaxPendingVerifications;
    type VerificationQuorum = VerificationQuorum;
    type VerificationPeriod = VerificationPeriod;
    type MinAttestationScore = MinAttestationScore;
    type VerificationFailureSlash = VerificationFailureSlash;
    // REAL trait implementations - not mocks!
    type LaneNodeProvider = StakeLaneNodeProvider;
    type ReputationUpdater = ReputationUpdaterImpl;
    type TaskSlashHandler = StakeSlashHandler;
    type ValidatorProvider = StakeValidatorProvider;
    type TaskAbandonmentSlash = TaskAbandonmentSlash;
    type TreasuryAccount = TreasuryAccount;
    type WeightInfo = ();
    type RendererRegistrarOrigin = frame_system::EnsureRoot<AccountId>;
    type Randomness = TestRandomness;
}

// ============================================================================
// Test Accounts and Constants
// ============================================================================

pub const ALICE: AccountId = 1;
pub const BOB: AccountId = 2;
pub const CHARLIE: AccountId = 3;
pub const DAVE: AccountId = 4;
pub const EVE: AccountId = 5;
pub const FRANK: AccountId = 6;
pub const GRACE: AccountId = 7;
pub const HENRY: AccountId = 8;
pub const IVAN: AccountId = 9;
pub const JULIA: AccountId = 10;

/// NSN token unit (18 decimals)
pub const NSN: Balance = 1_000_000_000_000_000_000;

// ============================================================================
// Test Externalities Builder
// ============================================================================

pub struct ExtBuilder {
    balances: Vec<(AccountId, Balance)>,
}

impl Default for ExtBuilder {
    fn default() -> Self {
        Self {
            balances: vec![
                (ALICE, 10_000 * NSN),
                (BOB, 10_000 * NSN),
                (CHARLIE, 10_000 * NSN),
                (DAVE, 10_000 * NSN),
                (EVE, 10_000 * NSN),
                (FRANK, 10_000 * NSN),
                (GRACE, 10_000 * NSN),
                (HENRY, 10_000 * NSN),
                (IVAN, 10_000 * NSN),
                (JULIA, 10_000 * NSN),
            ],
        }
    }
}

impl ExtBuilder {
    pub fn with_balances(mut self, balances: Vec<(AccountId, Balance)>) -> Self {
        self.balances = balances;
        self
    }

    pub fn build(self) -> sp_io::TestExternalities {
        let mut storage = frame_system::GenesisConfig::<Test>::default()
            .build_storage()
            .unwrap();

        pallet_balances::GenesisConfig::<Test> {
            balances: self.balances,
            dev_accounts: None,
        }
        .assimilate_storage(&mut storage)
        .unwrap();

        // Initialize treasury emission schedule
        pallet_nsn_treasury::GenesisConfig::<Test> {
            emission_schedule: pallet_nsn_treasury::EmissionSchedule {
                base_emission: 100_000_000 * NSN, // 100M NSN year 1
                decay_rate: Perbill::from_percent(15),
                current_year: 1,
                launch_block: 1,
            },
            reward_distribution: pallet_nsn_treasury::RewardDistribution {
                director_percent: Perbill::from_percent(40),
                validator_percent: Perbill::from_percent(25),
                pinner_percent: Perbill::from_percent(20),
                treasury_percent: Perbill::from_percent(15),
            },
        }
        .assimilate_storage(&mut storage)
        .unwrap();

        let mut ext = sp_io::TestExternalities::new(storage);
        ext.execute_with(|| {
            System::set_block_number(1);
        });
        ext
    }
}

/// Convenience function to create test externalities with default config
pub fn new_test_ext() -> sp_io::TestExternalities {
    ExtBuilder::default().build()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Advance blocks, calling on_finalize and on_initialize hooks
pub fn roll_to(n: BlockNumber) {
    while System::block_number() < n {
        let current = System::block_number();

        // Call on_finalize for all pallets
        <NsnDirector as Hooks<BlockNumber>>::on_finalize(current);
        <NsnReputation as Hooks<BlockNumber>>::on_finalize(current);
        <NsnBft as Hooks<BlockNumber>>::on_finalize(current);
        <NsnTreasury as Hooks<BlockNumber>>::on_finalize(current);

        System::set_block_number(current + 1);

        // Call on_initialize for all pallets
        <NsnDirector as Hooks<BlockNumber>>::on_initialize(current + 1);
        <NsnReputation as Hooks<BlockNumber>>::on_initialize(current + 1);
        <NsnBft as Hooks<BlockNumber>>::on_initialize(current + 1);
        <NsnTreasury as Hooks<BlockNumber>>::on_initialize(current + 1);
    }
}

/// Roll forward by a specific number of blocks
pub fn roll_forward(blocks: BlockNumber) {
    let target = System::block_number() + blocks;
    roll_to(target);
}

/// Stake tokens as a director
pub fn stake_as_director(who: AccountId, amount: Balance, region: Region) {
    assert!(Balances::free_balance(who) >= amount, "Insufficient balance for staking");
    pallet_nsn_stake::Pallet::<Test>::deposit_stake(
        RuntimeOrigin::signed(who),
        amount,
        100, // lock blocks
        region,
    )
    .expect("Staking should succeed");
}

/// Record a reputation event via root origin
pub fn record_reputation(
    who: AccountId,
    event_type: pallet_nsn_reputation::ReputationEventType,
    slot: u64,
) {
    pallet_nsn_reputation::Pallet::<Test>::record_event(
        RuntimeOrigin::root(),
        who,
        event_type,
        slot,
    )
    .expect("Recording reputation should succeed");
}

/// Store BFT consensus result via root origin
pub fn store_bft_result(
    slot: u64,
    embeddings_hash: H256,
    directors: Vec<AccountId>,
    success: bool,
) {
    pallet_nsn_bft::Pallet::<Test>::store_embeddings_hash(
        RuntimeOrigin::root(),
        slot,
        embeddings_hash,
        directors,
        success,
    )
    .expect("Storing BFT result should succeed");
}

/// Record director work for treasury rewards
pub fn record_director_work(who: AccountId, slots: u64) {
    pallet_nsn_treasury::Pallet::<Test>::record_director_work(
        RuntimeOrigin::root(),
        who,
        slots,
    )
    .expect("Recording director work should succeed");
}

/// Record validator work for treasury rewards
pub fn record_validator_work(who: AccountId, votes: u64) {
    pallet_nsn_treasury::Pallet::<Test>::record_validator_work(
        RuntimeOrigin::root(),
        who,
        votes,
    )
    .expect("Recording validator work should succeed");
}

/// Create a test hash from data
pub fn test_hash(data: &[u8]) -> H256 {
    use sp_runtime::traits::Hash;
    BlakeTwo256::hash(data)
}

/// Get the last emitted event
pub fn last_event() -> RuntimeEvent {
    System::events().pop().expect("Event expected").event
}

/// Get all events
pub fn events() -> Vec<RuntimeEvent> {
    System::events().into_iter().map(|r| r.event).collect()
}

/// Get reputation score for an account
pub fn get_reputation(who: AccountId) -> pallet_nsn_reputation::ReputationScore {
    pallet_nsn_reputation::Pallet::<Test>::get_reputation(&who)
}

/// Get stake info for an account
pub fn get_stake(who: AccountId) -> pallet_nsn_stake::StakeInfo<Balance, BlockNumber> {
    pallet_nsn_stake::Stakes::<Test>::get(&who)
}

/// Get node mode for an account
pub fn get_node_mode(who: AccountId) -> pallet_nsn_stake::NodeMode {
    pallet_nsn_stake::NodeModes::<Test>::get(&who)
}

/// Get BFT consensus stats
pub fn get_bft_stats() -> pallet_nsn_bft::ConsensusStats {
    pallet_nsn_bft::Pallet::<Test>::get_stats()
}

/// Get treasury balance
pub fn get_treasury_balance() -> Balance {
    pallet_nsn_treasury::Pallet::<Test>::treasury_balance()
}

/// Get accumulated contributions for an account
pub fn get_accumulated_contributions(who: AccountId) -> pallet_nsn_treasury::AccumulatedContributions {
    pallet_nsn_treasury::Pallet::<Test>::accumulated_contributions(&who)
}
