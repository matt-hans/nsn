// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Bootstrap Pallet
//!
//! On-chain registry for trusted bootstrap manifest signers with
//! rotation, revocation, and quorum enforcement.

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;
pub use weights::WeightInfo;

pub mod weights;

#[frame_support::pallet]
pub mod pallet {
    use super::WeightInfo;
    use frame_support::pallet_prelude::*;
    use frame_support::traits::GenesisBuild;
    use frame_system::pallet_prelude::*;
    use sp_std::collections::btree_set::BTreeSet;
    use sp_std::vec::Vec;

    /// Bounded signer key bytes (libp2p PublicKey protobuf bytes).
    pub type SignerKey<T> = BoundedVec<u8, <T as Config>::MaxSignerBytes>;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        #[allow(deprecated)]
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        /// Maximum number of trusted signers.
        #[pallet::constant]
        type MaxSigners: Get<u32>;

        /// Maximum byte length of a signer public key.
        #[pallet::constant]
        type MaxSignerBytes: Get<u32>;

        /// Weight info for extrinsics.
        type WeightInfo: WeightInfo;
    }

    /// Current trusted signer public keys.
    #[pallet::storage]
    #[pallet::getter(fn trusted_signers)]
    pub type TrustedSigners<T: Config> =
        StorageValue<_, BoundedVec<SignerKey<T>, T::MaxSigners>, ValueQuery>;

    /// Revoked signer public keys.
    #[pallet::storage]
    #[pallet::getter(fn revoked_signers)]
    pub type RevokedSigners<T: Config> =
        StorageValue<_, BoundedVec<SignerKey<T>, T::MaxSigners>, ValueQuery>;

    /// Required signature quorum.
    #[pallet::storage]
    #[pallet::getter(fn signer_quorum)]
    pub type SignerQuorum<T: Config> = StorageValue<_, u32, ValueQuery>;

    /// Signer set epoch for rotation tracking.
    #[pallet::storage]
    #[pallet::getter(fn signer_epoch)]
    pub type SignerEpoch<T: Config> = StorageValue<_, u64, ValueQuery>;

    #[pallet::genesis_config]
    pub struct GenesisConfig<T: Config> {
        pub trusted_signers: Vec<Vec<u8>>,
        pub signer_quorum: u32,
        pub _marker: core::marker::PhantomData<T>,
    }

    #[cfg(feature = "std")]
    impl<T: Config> Default for GenesisConfig<T> {
        fn default() -> Self {
            Self {
                trusted_signers: Vec::new(),
                signer_quorum: 0,
                _marker: core::marker::PhantomData,
            }
        }
    }

    #[pallet::genesis_build]
    impl<T: Config> GenesisBuild<T> for GenesisConfig<T> {
        fn build(&self) {
            let signers = normalize_signers::<T>(&self.trusted_signers)
                .expect("invalid bootstrap signers in genesis");
            if !signers.is_empty() {
                assert!(
                    self.signer_quorum > 0,
                    "signer quorum must be > 0 when signers are configured"
                );
                assert!(
                    (self.signer_quorum as usize) <= signers.len(),
                    "signer quorum exceeds signer count"
                );
            } else {
                assert!(
                    self.signer_quorum == 0,
                    "signer quorum must be 0 when no signers configured"
                );
            }
            TrustedSigners::<T>::put(signers);
            RevokedSigners::<T>::put(BoundedVec::default());
            SignerQuorum::<T>::put(self.signer_quorum);
            SignerEpoch::<T>::put(0);
        }
    }

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Trusted signers replaced (epoch incremented).
        SignersUpdated { epoch: u64, quorum: u32 },
        /// Signer revoked.
        SignerRevoked { signer: Vec<u8>, epoch: u64 },
        /// Signer restored.
        SignerRestored { signer: Vec<u8>, epoch: u64 },
        /// Quorum updated.
        QuorumUpdated { quorum: u32, epoch: u64 },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Too many signers provided.
        TooManySigners,
        /// Signer public key is too large.
        SignerKeyTooLong,
        /// Signer public key is empty.
        EmptySignerKey,
        /// Invalid quorum value.
        InvalidQuorum,
        /// Signer is not in trusted set.
        SignerNotTrusted,
        /// Signer already revoked.
        SignerAlreadyRevoked,
        /// Signer not revoked.
        SignerNotRevoked,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Replace the trusted signer set and quorum (root-only).
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::set_signers())]
        pub fn set_signers(
            origin: OriginFor<T>,
            signers: Vec<Vec<u8>>,
            quorum: u32,
        ) -> DispatchResult {
            ensure_root(origin)?;

            let signers = normalize_signers::<T>(&signers)?;
            ensure!(
                !signers.is_empty(),
                Error::<T>::InvalidQuorum
            );
            ensure!(quorum > 0, Error::<T>::InvalidQuorum);
            ensure!(
                (quorum as usize) <= signers.len(),
                Error::<T>::InvalidQuorum
            );

            TrustedSigners::<T>::put(signers);
            RevokedSigners::<T>::put(BoundedVec::default());
            SignerQuorum::<T>::put(quorum);

            let epoch = SignerEpoch::<T>::get().saturating_add(1);
            SignerEpoch::<T>::put(epoch);

            Self::deposit_event(Event::SignersUpdated { epoch, quorum });
            Ok(())
        }

        /// Revoke a trusted signer (root-only).
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::revoke_signer())]
        pub fn revoke_signer(origin: OriginFor<T>, signer: Vec<u8>) -> DispatchResult {
            ensure_root(origin)?;
            let signer_key = to_signer_key::<T>(&signer)?;

            let trusted = TrustedSigners::<T>::get();
            ensure!(trusted.contains(&signer_key), Error::<T>::SignerNotTrusted);

            let mut revoked = RevokedSigners::<T>::get();
            ensure!(
                !revoked.contains(&signer_key),
                Error::<T>::SignerAlreadyRevoked
            );
            revoked
                .try_push(signer_key.clone())
                .map_err(|_| Error::<T>::TooManySigners)?;
            RevokedSigners::<T>::put(revoked);

            let epoch = SignerEpoch::<T>::get().saturating_add(1);
            SignerEpoch::<T>::put(epoch);
            Self::deposit_event(Event::SignerRevoked {
                signer: signer_key.to_vec(),
                epoch,
            });
            Ok(())
        }

        /// Restore a revoked signer (root-only).
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::restore_signer())]
        pub fn restore_signer(origin: OriginFor<T>, signer: Vec<u8>) -> DispatchResult {
            ensure_root(origin)?;
            let signer_key = to_signer_key::<T>(&signer)?;

            let mut revoked = RevokedSigners::<T>::get();
            let idx = revoked
                .iter()
                .position(|item| item == &signer_key)
                .ok_or(Error::<T>::SignerNotRevoked)?;
            revoked.remove(idx);
            RevokedSigners::<T>::put(revoked);

            let epoch = SignerEpoch::<T>::get().saturating_add(1);
            SignerEpoch::<T>::put(epoch);
            Self::deposit_event(Event::SignerRestored {
                signer: signer_key.to_vec(),
                epoch,
            });
            Ok(())
        }

        /// Update the signer quorum (root-only).
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::set_quorum())]
        pub fn set_quorum(origin: OriginFor<T>, quorum: u32) -> DispatchResult {
            ensure_root(origin)?;
            let active = active_signer_count::<T>();
            ensure!(quorum > 0, Error::<T>::InvalidQuorum);
            ensure!((quorum as usize) <= active, Error::<T>::InvalidQuorum);

            SignerQuorum::<T>::put(quorum);
            let epoch = SignerEpoch::<T>::get().saturating_add(1);
            SignerEpoch::<T>::put(epoch);
            Self::deposit_event(Event::QuorumUpdated { quorum, epoch });
            Ok(())
        }
    }

    fn normalize_signers<T: Config>(
        signers: &[Vec<u8>],
    ) -> Result<BoundedVec<SignerKey<T>, T::MaxSigners>, Error<T>> {
        let mut unique = BTreeSet::new();
        let mut bounded: BoundedVec<SignerKey<T>, T::MaxSigners> = BoundedVec::default();

        for signer in signers {
            ensure!(!signer.is_empty(), Error::<T>::EmptySignerKey);
            if unique.insert(signer.clone()) {
                let key = to_signer_key::<T>(signer)?;
                bounded
                    .try_push(key)
                    .map_err(|_| Error::<T>::TooManySigners)?;
            }
        }

        Ok(bounded)
    }

    fn to_signer_key<T: Config>(signer: &[u8]) -> Result<SignerKey<T>, Error<T>> {
        ensure!(!signer.is_empty(), Error::<T>::EmptySignerKey);
        SignerKey::<T>::try_from(signer.to_vec()).map_err(|_| Error::<T>::SignerKeyTooLong)
    }

    fn active_signer_count<T: Config>() -> usize {
        let trusted = TrustedSigners::<T>::get();
        let revoked = RevokedSigners::<T>::get();
        trusted
            .iter()
            .filter(|signer| !revoked.contains(signer))
            .count()
    }
}
