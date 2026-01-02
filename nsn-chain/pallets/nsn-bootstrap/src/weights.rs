//! Weight definitions for pallet-nsn-bootstrap.

use frame_support::weights::Weight;

pub trait WeightInfo {
    fn set_signers() -> Weight;
    fn revoke_signer() -> Weight;
    fn restore_signer() -> Weight;
    fn set_quorum() -> Weight;
}

impl WeightInfo for () {
    fn set_signers() -> Weight {
        Weight::zero()
    }

    fn revoke_signer() -> Weight {
        Weight::zero()
    }

    fn restore_signer() -> Weight {
        Weight::zero()
    }

    fn set_quorum() -> Weight {
        Weight::zero()
    }
}
