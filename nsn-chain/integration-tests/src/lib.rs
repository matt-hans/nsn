// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Integration Tests
//!
//! This crate contains integration tests that validate cross-pallet interactions
//! in the NSN chain runtime. Unlike unit tests which mock inter-pallet dependencies,
//! these tests use REAL trait implementations to verify pallets work correctly together.
//!
//! ## Test Modules
//!
//! - `stake_director_tests` - Stake → Director election integration
//! - `director_reputation_tests` - Director → Reputation scoring integration
//! - `director_bft_tests` - Director → BFT storage integration
//! - `task_market_stake_tests` - Task Market → Stake eligibility integration
//! - `treasury_tests` - Treasury work recording and reward distribution
//! - `epoch_lifecycle_test` - End-to-end epoch lifecycle test
//!
//! ## Critical Integration Chains
//!
//! 1. **Stake → Director Election Chain:**
//!    - Stakes::stakes() queried by Director::elect_directors()
//!    - NodeMode and NodeRole updated via trait impls
//!
//! 2. **Director → Reputation Chain:**
//!    - Director finalization records reputation events
//!    - Challenge resolution affects reputation scores
//!
//! 3. **Director → BFT Chain:**
//!    - BFT stores consensus results from finalized director slots
//!
//! 4. **Task Market → Stake/Reputation Chain:**
//!    - LaneNodeProvider queries eligible nodes from Stakes
//!    - ReputationUpdater records task outcomes
//!
//! 5. **Treasury → Work Recording Chain:**
//!    - Treasury records director and validator work for rewards

#![cfg(test)]

mod mock;

mod stake_director_tests;
mod director_reputation_tests;
mod director_bft_tests;
mod task_market_stake_tests;
mod treasury_tests;
mod epoch_lifecycle_test;
