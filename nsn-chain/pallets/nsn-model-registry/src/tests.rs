// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Unit tests for pallet-nsn-model-registry

use crate::{mock::*, Error, Event, ModelCapabilities, ModelState};
use frame_support::{assert_noop, assert_ok, BoundedVec};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a default model_id for testing
fn default_model_id() -> BoundedVec<u8, MaxModelIdLen> {
    BoundedVec::try_from(b"flux-schnell-nf4".to_vec()).unwrap()
}

/// Create a second model_id for testing
fn model_id_2() -> BoundedVec<u8, MaxModelIdLen> {
    BoundedVec::try_from(b"kokoro-82m".to_vec()).unwrap()
}

/// Create a third model_id for testing
fn model_id_3() -> BoundedVec<u8, MaxModelIdLen> {
    BoundedVec::try_from(b"clip-vit-l-14".to_vec()).unwrap()
}

/// Create a default container CID for testing
fn default_container_cid() -> BoundedVec<u8, MaxCidLen> {
    BoundedVec::try_from(b"bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".to_vec())
        .unwrap()
}

/// Create default capabilities for video generation model
fn video_capabilities() -> ModelCapabilities {
    ModelCapabilities {
        video_generation: true,
        image_generation: true,
        ..Default::default()
    }
}

/// Create capabilities for a text model
fn text_capabilities() -> ModelCapabilities {
    ModelCapabilities {
        text_generation: true,
        ..Default::default()
    }
}

/// Create capabilities for an embedding model
fn embedding_capabilities() -> ModelCapabilities {
    ModelCapabilities {
        embedding: true,
        ..Default::default()
    }
}

/// Create capabilities for a speech synthesis model
fn speech_capabilities() -> ModelCapabilities {
    ModelCapabilities {
        speech_synthesis: true,
        ..Default::default()
    }
}

/// Default VRAM requirement for Flux model
const DEFAULT_VRAM_MB: u32 = 6000;

// ============================================================================
// Required Tests (per task spec)
// ============================================================================

#[test]
fn test_register_model() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: No models in the catalog
        assert!(!NsnModelRegistry::model_exists(&default_model_id()));

        // WHEN: Alice registers a model
        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(ALICE),
            default_model_id(),
            default_container_cid(),
            DEFAULT_VRAM_MB,
            video_capabilities(),
        ));

        // THEN: Model is in the catalog with correct metadata
        assert!(NsnModelRegistry::model_exists(&default_model_id()));

        let model = NsnModelRegistry::get_model(&default_model_id()).expect("Model should exist");
        assert_eq!(model.container_cid, default_container_cid());
        assert_eq!(model.vram_required_mb, DEFAULT_VRAM_MB);
        assert_eq!(model.capabilities, video_capabilities());
        assert_eq!(model.registered_by, ALICE);
        assert_eq!(model.registered_at, 1);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnModelRegistry(Event::ModelRegistered {
                model_id: _,
                container_cid: _,
                vram_required_mb: 6000,
                registered_by: ALICE,
            })
        ));
    });
}

#[test]
fn test_model_already_registered() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has already registered a model
        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(ALICE),
            default_model_id(),
            default_container_cid(),
            DEFAULT_VRAM_MB,
            video_capabilities(),
        ));

        // WHEN: Bob tries to register the same model ID
        assert_noop!(
            NsnModelRegistry::register_model(
                RuntimeOrigin::signed(BOB),
                default_model_id(),
                default_container_cid(),
                5000,
                text_capabilities(),
            ),
            Error::<Test>::ModelAlreadyRegistered
        );

        // THEN: Original model is unchanged
        let model = NsnModelRegistry::get_model(&default_model_id()).expect("Model should exist");
        assert_eq!(model.registered_by, ALICE);
        assert_eq!(model.vram_required_mb, DEFAULT_VRAM_MB);
    });
}

#[test]
fn test_update_capabilities() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: No node capabilities registered
        assert!(NsnModelRegistry::get_node_capabilities(&ALICE).is_none());

        // Create hot and warm model lists
        let hot_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();
        let warm_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::try_from(vec![model_id_2(), model_id_3()]).unwrap();

        // WHEN: Alice updates her node capabilities
        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            8000, // available VRAM in MB
            hot_models.clone(),
            warm_models.clone(),
        ));

        // THEN: Node capabilities are stored correctly
        let caps = NsnModelRegistry::get_node_capabilities(&ALICE).expect("Caps should exist");
        assert_eq!(caps.available_vram_mb, 8000);
        assert_eq!(caps.hot_models.len(), 1);
        assert_eq!(caps.warm_models.len(), 2);
        assert_eq!(caps.last_updated, 1);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnModelRegistry(Event::NodeCapabilityUpdated {
                node: ALICE,
                available_vram_mb: 8000,
                hot_model_count: 1,
                warm_model_count: 2,
            })
        ));
    });
}

#[test]
fn test_multiple_capability_updates() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has registered capabilities
        let hot_models_1: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();
        let warm_models_1: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::default();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            10000,
            hot_models_1,
            warm_models_1,
        ));

        let caps_1 = NsnModelRegistry::get_node_capabilities(&ALICE).expect("Caps should exist");
        assert_eq!(caps_1.available_vram_mb, 10000);
        assert_eq!(caps_1.hot_models.len(), 1);
        assert_eq!(caps_1.warm_models.len(), 0);

        // Advance to a new block
        System::set_block_number(5);

        // WHEN: Alice updates capabilities again
        let hot_models_2: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id(), model_id_2()]).unwrap();
        let warm_models_2: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::try_from(vec![model_id_3()]).unwrap();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            5000, // Less VRAM available after loading more models
            hot_models_2,
            warm_models_2,
        ));

        // THEN: Capabilities are updated
        let caps_2 = NsnModelRegistry::get_node_capabilities(&ALICE).expect("Caps should exist");
        assert_eq!(caps_2.available_vram_mb, 5000);
        assert_eq!(caps_2.hot_models.len(), 2);
        assert_eq!(caps_2.warm_models.len(), 1);
        assert_eq!(caps_2.last_updated, 5);
    });
}

#[test]
fn test_model_capabilities_flags() {
    ExtBuilder::default().build().execute_with(|| {
        // Test 1: Video generation model (Lane 0)
        let video_caps = ModelCapabilities {
            video_generation: true,
            image_generation: true,
            text_generation: false,
            code_generation: false,
            embedding: false,
            speech_synthesis: false,
        };

        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(ALICE),
            default_model_id(),
            default_container_cid(),
            DEFAULT_VRAM_MB,
            video_caps,
        ));

        let model = NsnModelRegistry::get_model(&default_model_id()).expect("Model should exist");
        assert!(model.capabilities.video_generation);
        assert!(model.capabilities.image_generation);
        assert!(!model.capabilities.text_generation);
        assert!(!model.capabilities.code_generation);
        assert!(!model.capabilities.embedding);
        assert!(!model.capabilities.speech_synthesis);
        assert!(model.capabilities.is_lane0()); // Video generation = Lane 0
        assert!(model.capabilities.has_any());

        // Test 2: Text generation model (Lane 1)
        let text_caps = ModelCapabilities {
            text_generation: true,
            code_generation: true,
            video_generation: false,
            image_generation: false,
            embedding: false,
            speech_synthesis: false,
        };

        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(BOB),
            model_id_2(),
            default_container_cid(),
            400,
            text_caps,
        ));

        let model2 = NsnModelRegistry::get_model(&model_id_2()).expect("Model should exist");
        assert!(model2.capabilities.text_generation);
        assert!(model2.capabilities.code_generation);
        assert!(!model2.capabilities.video_generation);
        assert!(!model2.capabilities.is_lane0()); // Not video generation

        // Test 3: Embedding model
        let embed_caps = ModelCapabilities {
            embedding: true,
            video_generation: false,
            image_generation: false,
            text_generation: false,
            code_generation: false,
            speech_synthesis: false,
        };

        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(CHARLIE),
            model_id_3(),
            default_container_cid(),
            600,
            embed_caps,
        ));

        let model3 = NsnModelRegistry::get_model(&model_id_3()).expect("Model should exist");
        assert!(model3.capabilities.embedding);
        assert!(!model3.capabilities.is_lane0());
        assert!(model3.capabilities.has_any());
    });
}

// ============================================================================
// Additional Tests
// ============================================================================

#[test]
fn test_empty_model_id_fails() {
    ExtBuilder::default().build().execute_with(|| {
        let empty_model_id: BoundedVec<u8, MaxModelIdLen> = BoundedVec::default();

        assert_noop!(
            NsnModelRegistry::register_model(
                RuntimeOrigin::signed(ALICE),
                empty_model_id,
                default_container_cid(),
                DEFAULT_VRAM_MB,
                video_capabilities(),
            ),
            Error::<Test>::InvalidModelId
        );
    });
}

#[test]
fn test_empty_container_cid_fails() {
    ExtBuilder::default().build().execute_with(|| {
        let empty_cid: BoundedVec<u8, MaxCidLen> = BoundedVec::default();

        assert_noop!(
            NsnModelRegistry::register_model(
                RuntimeOrigin::signed(ALICE),
                default_model_id(),
                empty_cid,
                DEFAULT_VRAM_MB,
                video_capabilities(),
            ),
            Error::<Test>::InvalidContainerCid
        );
    });
}

#[test]
fn test_model_state_helper() {
    ExtBuilder::default().build().execute_with(|| {
        // Set up: Alice has model 1 hot, model 2 warm
        let hot_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();
        let warm_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::try_from(vec![model_id_2()]).unwrap();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            8000,
            hot_models,
            warm_models,
        ));

        // Test get_model_state helper
        assert_eq!(
            NsnModelRegistry::get_model_state(&ALICE, &default_model_id()),
            ModelState::Hot
        );
        assert_eq!(
            NsnModelRegistry::get_model_state(&ALICE, &model_id_2()),
            ModelState::Warm
        );
        assert_eq!(
            NsnModelRegistry::get_model_state(&ALICE, &model_id_3()),
            ModelState::Cold
        );

        // Bob has no capabilities registered
        assert_eq!(
            NsnModelRegistry::get_model_state(&BOB, &default_model_id()),
            ModelState::Cold
        );
    });
}

#[test]
fn test_node_has_model_helpers() {
    ExtBuilder::default().build().execute_with(|| {
        // Set up: Alice has models loaded
        let hot_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();
        let warm_models: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::try_from(vec![model_id_2()]).unwrap();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            8000,
            hot_models,
            warm_models,
        ));

        // Test helper functions
        assert!(NsnModelRegistry::node_has_model_hot(&ALICE, &default_model_id()));
        assert!(!NsnModelRegistry::node_has_model_hot(&ALICE, &model_id_2()));
        assert!(!NsnModelRegistry::node_has_model_hot(&ALICE, &model_id_3()));

        assert!(!NsnModelRegistry::node_has_model_warm(&ALICE, &default_model_id()));
        assert!(NsnModelRegistry::node_has_model_warm(&ALICE, &model_id_2()));
        assert!(!NsnModelRegistry::node_has_model_warm(&ALICE, &model_id_3()));
    });
}

#[test]
fn test_multiple_nodes_capabilities() {
    ExtBuilder::default().build().execute_with(|| {
        // Alice registers her capabilities
        let alice_hot: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();
        let alice_warm: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::default();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            12000,
            alice_hot,
            alice_warm,
        ));

        // Bob registers different capabilities
        let bob_hot: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::try_from(vec![model_id_2(), model_id_3()]).unwrap();
        let bob_warm: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::try_from(vec![default_model_id()]).unwrap();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(BOB),
            6000,
            bob_hot,
            bob_warm,
        ));

        // Verify Alice's state
        let alice_caps = NsnModelRegistry::get_node_capabilities(&ALICE).expect("Should exist");
        assert_eq!(alice_caps.available_vram_mb, 12000);
        assert_eq!(alice_caps.hot_models.len(), 1);
        assert!(NsnModelRegistry::node_has_model_hot(&ALICE, &default_model_id()));

        // Verify Bob's state
        let bob_caps = NsnModelRegistry::get_node_capabilities(&BOB).expect("Should exist");
        assert_eq!(bob_caps.available_vram_mb, 6000);
        assert_eq!(bob_caps.hot_models.len(), 2);
        assert!(NsnModelRegistry::node_has_model_hot(&BOB, &model_id_2()));
        assert!(NsnModelRegistry::node_has_model_warm(&BOB, &default_model_id()));
    });
}

#[test]
fn test_zero_vram_model() {
    ExtBuilder::default().build().execute_with(|| {
        // Embedding models can have very low VRAM requirements
        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(ALICE),
            default_model_id(),
            default_container_cid(),
            0, // Zero VRAM (edge case)
            embedding_capabilities(),
        ));

        let model = NsnModelRegistry::get_model(&default_model_id()).expect("Model should exist");
        assert_eq!(model.vram_required_mb, 0);
    });
}

#[test]
fn test_capability_helper_methods() {
    // Test ModelCapabilities helper methods
    let empty_caps = ModelCapabilities::new();
    assert!(!empty_caps.has_any());
    assert!(!empty_caps.is_lane0());

    let video_caps = ModelCapabilities::video_model();
    assert!(video_caps.has_any());
    assert!(video_caps.is_lane0());
    assert!(video_caps.video_generation);
    assert!(!video_caps.text_generation);

    let text_caps = ModelCapabilities::text_model();
    assert!(text_caps.has_any());
    assert!(!text_caps.is_lane0());
    assert!(text_caps.text_generation);
    assert!(!text_caps.video_generation);
}

#[test]
fn test_empty_capability_update() {
    ExtBuilder::default().build().execute_with(|| {
        // Node can advertise no models (clearing capabilities)
        let empty_hot: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxHotModels> =
            BoundedVec::default();
        let empty_warm: BoundedVec<BoundedVec<u8, MaxModelIdLen>, MaxWarmModels> =
            BoundedVec::default();

        assert_ok!(NsnModelRegistry::update_capabilities(
            RuntimeOrigin::signed(ALICE),
            12000,
            empty_hot,
            empty_warm,
        ));

        let caps = NsnModelRegistry::get_node_capabilities(&ALICE).expect("Should exist");
        assert_eq!(caps.hot_models.len(), 0);
        assert_eq!(caps.warm_models.len(), 0);
        assert_eq!(caps.available_vram_mb, 12000);
    });
}

#[test]
fn test_multiple_models_registered() {
    ExtBuilder::default().build().execute_with(|| {
        // Register multiple models from different users
        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(ALICE),
            default_model_id(),
            default_container_cid(),
            6000,
            video_capabilities(),
        ));

        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(BOB),
            model_id_2(),
            default_container_cid(),
            400,
            speech_capabilities(),
        ));

        assert_ok!(NsnModelRegistry::register_model(
            RuntimeOrigin::signed(CHARLIE),
            model_id_3(),
            default_container_cid(),
            600,
            embedding_capabilities(),
        ));

        // Verify all models exist
        assert!(NsnModelRegistry::model_exists(&default_model_id()));
        assert!(NsnModelRegistry::model_exists(&model_id_2()));
        assert!(NsnModelRegistry::model_exists(&model_id_3()));

        // Verify each model has correct registrant
        assert_eq!(
            NsnModelRegistry::get_model(&default_model_id())
                .unwrap()
                .registered_by,
            ALICE
        );
        assert_eq!(
            NsnModelRegistry::get_model(&model_id_2())
                .unwrap()
                .registered_by,
            BOB
        );
        assert_eq!(
            NsnModelRegistry::get_model(&model_id_3())
                .unwrap()
                .registered_by,
            CHARLIE
        );
    });
}

#[test]
fn test_model_state_default() {
    // Test that ModelState::default() is Cold
    let state = ModelState::default();
    assert_eq!(state, ModelState::Cold);
}

#[test]
fn test_capabilities_default() {
    // Test that ModelCapabilities::default() has all flags false
    let caps = ModelCapabilities::default();
    assert!(!caps.text_generation);
    assert!(!caps.image_generation);
    assert!(!caps.code_generation);
    assert!(!caps.embedding);
    assert!(!caps.speech_synthesis);
    assert!(!caps.video_generation);
    assert!(!caps.has_any());
}
