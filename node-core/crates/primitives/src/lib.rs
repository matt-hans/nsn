#![cfg_attr(not(feature = "std"), no_std)]

use core::convert::TryFrom;

use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Canonical lane designation shared across runtime and off-chain nodes.
#[derive(
    Clone,
    Copy,
    Encode,
    Decode,
    DecodeWithMemTracking,
    Eq,
    PartialEq,
    RuntimeDebug,
    TypeInfo,
    MaxEncodedLen,
)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum Lane {
    /// Lane 0: Time-triggered video generation.
    Lane0 = 0,
    /// Lane 1: Demand-triggered general compute.
    Lane1 = 1,
}

impl Lane {
    /// Convert lane to numeric representation.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for Lane {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Lane::Lane0),
            1 => Ok(Lane::Lane1),
            _ => Err("invalid lane"),
        }
    }
}
