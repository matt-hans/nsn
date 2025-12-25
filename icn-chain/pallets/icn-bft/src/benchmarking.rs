// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Benchmarking setup for pallet-icn-bft

use super::*;

#[allow(unused)]
use crate::Pallet as IcnBft;
use frame_benchmarking::v2::*;
use frame_system::RawOrigin;
use sp_std::vec;

#[benchmarks]
mod benchmarks {
    use super::*;

    #[benchmark]
    fn store_embeddings_hash() {
        let slot = 1u64;
        let embeddings_hash = T::Hash::default();
        let directors = vec![
            account("director", 0, 0),
            account("director", 1, 0),
            account("director", 2, 0),
            account("director", 3, 0),
            account("director", 4, 0),
        ];

        #[extrinsic_call]
        store_embeddings_hash(RawOrigin::Root, slot, embeddings_hash, directors, true);

        assert!(EmbeddingsHashes::<T>::contains_key(slot));
    }

    #[benchmark]
    fn prune_old_consensus() {
        // Setup: Store 10 old consensus rounds
        for i in 0..10u64 {
            let _ = IcnBft::<T>::store_embeddings_hash(
                RawOrigin::Root.into(),
                i,
                T::Hash::default(),
                vec![],
                true,
            );
        }

        let before_slot = 100u64;

        #[extrinsic_call]
        prune_old_consensus(RawOrigin::Root, before_slot);

        // Verify all pruned
        assert!(!EmbeddingsHashes::<T>::contains_key(5));
    }
}
