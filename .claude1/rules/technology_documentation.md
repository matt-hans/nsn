# ICN Project Technology Documentation

**Generated from Context7 on December 25, 2025**

This document contains detailed documentation and code examples for all key technologies and libraries used in the Interdimensional Cable Network (ICN) project. All documentation is sourced from Context7 and represents the latest available information.

---

## Table of Contents

1. [Blockchain & On-Chain Technologies](#1-blockchain--on-chain-technologies)
   - [Polkadot SDK / Substrate](#11-polkadot-sdk--substrate)
   - [subxt (Substrate Client)](#12-subxt-substrate-client)
2. [P2P Networking](#2-p2p-networking)
   - [libp2p](#21-libp2p)
3. [AI/ML Technologies](#3-aiml-technologies)
   - [PyTorch](#31-pytorch)
   - [OpenCLIP](#32-openclip)
   - [Hugging Face Transformers](#33-hugging-face-transformers)
4. [Frontend Technologies](#4-frontend-technologies)
   - [Tauri](#41-tauri)
   - [React](#42-react)
   - [Zustand](#43-zustand)
5. [Backend & Infrastructure](#5-backend--infrastructure)
   - [Tokio](#51-tokio)
   - [PyO3](#52-pyo3)
6. [DevOps & Observability](#6-devops--observability)
   - [Prometheus](#61-prometheus)
   - [Grafana](#62-grafana)
   - [Docker](#63-docker)
   - [Kubernetes](#64-kubernetes)

---

## 1. Blockchain & On-Chain Technologies

### 1.1 Polkadot SDK / Substrate

**Library ID:** `/websites/paritytech_github_io_polkadot-sdk_master`

The Polkadot SDK provides all components for building on the Polkadot network, enabling interoperability and secure information sharing between blockchains. ICN uses Polkadot SDK to build its own Substrate-based blockchain with custom FRAME pallets.

#### FRAME Runtime Construction

The FRAME framework is Substrate's modular runtime framework. Use the `construct_runtime!` macro to define your runtime with pallets:

```rust
use super::pallet_with_custom_origin;
use frame::{runtime::prelude::*, testing_prelude::*};

construct_runtime!(
    pub struct Runtime {
        System: frame_system,
        PalletWithCustomOrigin: pallet_with_custom_origin,
    }
);

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Runtime {
    type Block = MockBlock<Self>;
}

impl pallet_with_custom_origin::Config for Runtime {
    // Custom pallet configuration
}
```

#### Building a Complete Runtime with Multiple Pallets

```rust
#[frame_support::runtime]
mod runtime {
    #[runtime::runtime]
    #[runtime::derive(
        RuntimeCall,
        RuntimeEvent,
        RuntimeError,
        RuntimeOrigin,
        RuntimeFreezeReason,
        RuntimeHoldReason,
        RuntimeSlashReason,
        RuntimeLockId,
        RuntimeTask,
        RuntimeViewFunction
    )]
    pub struct Runtime;

    #[runtime::pallet_index(0)]
    pub type System = frame_system;

    #[runtime::pallet_index(1)]
    pub type Timestamp = pallet_timestamp;

    #[runtime::pallet_index(2)]
    pub type Aura = pallet_aura;

    #[runtime::pallet_index(3)]
    pub type Grandpa = pallet_grandpa;

    #[runtime::pallet_index(4)]
    pub type Balances = pallet_balances;

    #[runtime::pallet_index(5)]
    pub type TransactionPayment = pallet_transaction_payment;

    #[runtime::pallet_index(6)]
    pub type Sudo = pallet_sudo;

    // Include custom logic from pallet-template
    #[runtime::pallet_index(7)]
    pub type Template = pallet_template;
}
```

#### Frame System Pallet Configuration

```rust
#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::pallet::Config for Runtime {
    type BlockWeights = RuntimeBlockWeights;
    type Nonce = Nonce;
    type AccountId = AccountId;
    type Lookup = sp_runtime::traits::IdentityLookup<Self::AccountId>;
    type Block = Block;
    type AccountData = pallet_balances::AccountData<Balance>;
}
```

#### Key Concepts for ICN Pallets

| Concept | Description |
|---------|-------------|
| **Pallet** | Modular runtime component (e.g., `pallet-icn-stake`) |
| **Storage** | On-chain state stored in Merkle-Patricia trie |
| **Extrinsic** | Transaction or on-chain call |
| **Event** | Runtime log for off-chain consumption |
| **Error** | Typed errors for pallet operations |
| **Origin** | Authorization context for calls |

---

### 1.2 subxt (Substrate Client)

**Library ID:** `/paritytech/subxt`

subxt is the primary library for interacting with Substrate-based nodes from Rust or WebAssembly. ICN off-chain nodes use subxt to:
- Subscribe to finalized blocks
- Listen for `DirectorsElected` events
- Submit `submit_bft_result()` extrinsics
- Query reputation scores

#### Basic Usage

```rust
use subxt::{OnlineClient, PolkadotConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to a Substrate node
    let api = OnlineClient::<PolkadotConfig>::from_url("ws://127.0.0.1:9944").await?;
    
    // Subscribe to finalized blocks
    let mut blocks_sub = api.blocks().subscribe_finalized().await?;
    
    while let Some(block) = blocks_sub.next().await {
        let block = block?;
        println!("Block number: {}", block.header().number);
        
        // Process events in the block
        let events = block.events().await?;
        for event in events.iter() {
            let event = event?;
            // Handle specific events
        }
    }
    
    Ok(())
}
```

#### Submitting Extrinsics

```rust
use subxt::{tx::Signer, OnlineClient, PolkadotConfig};
use subxt_signer::sr25519::dev;

async fn submit_transaction() -> Result<(), Box<dyn std::error::Error>> {
    let api = OnlineClient::<PolkadotConfig>::from_url("ws://127.0.0.1:9944").await?;
    let signer = dev::alice();
    
    // Create and submit a transaction
    let tx = api.tx().sign_and_submit_then_watch_default(
        &your_extrinsic_call,
        &signer
    ).await?;
    
    // Wait for transaction to be finalized
    let events = tx.wait_for_finalized_success().await?;
    
    Ok(())
}
```

---

## 2. P2P Networking

### 2.1 libp2p

**Library ID:** `/libp2p/specs`

libp2p is a modular framework and suite of peer-to-peer networking protocols. ICN uses rust-libp2p 0.53.0 with GossipSub, Kademlia DHT, and QUIC transport.

#### GossipSub Configuration Parameters

| Parameter | Purpose | ICN Default |
|-----------|---------|-------------|
| `D` | Desired outbound degree | 6 |
| `D_low` | Lower bound for outbound degree | 4 |
| `D_high` | Upper bound for outbound degree | 12 |
| `D_lazy` | Outbound degree for gossip emission | 6 |
| `heartbeat_interval` | Time between heartbeats | 1 second |
| `fanout_ttl` | Time-to-live for topic's fanout state | 60 seconds |
| `mcache_len` | Number of history windows in message cache | 5 |
| `mcache_gossip` | History windows for gossip emission | 3 |
| `seen_ttl` | Expiry time for seen message IDs | 2 minutes |

#### Mesh Maintenance Algorithm

```pseudocode
for each topic in mesh:
  if |mesh[topic]| < D_low:
    select D - |mesh[topic]| peers from peers.gossipsub[topic]
    add selected peers to mesh[topic]
    emit GRAFT to selected peers
  else if |mesh[topic]| > D_high:
    select |mesh[topic]| - D peers from mesh[topic]
    remove selected peers from mesh[topic]
    emit PRUNE to selected peers
```

#### ICN GossipSub Topics

| Topic | Purpose | Priority Weight |
|-------|---------|-----------------|
| `/icn/recipes/1.0.0` | Recipe JSON broadcast | 1.0 |
| `/icn/video/1.0.0` | Video chunks (16MB max) | 2.0 |
| `/icn/bft/1.0.0` | BFT signals (critical) | 3.0 |
| `/icn/attestations/1.0.0` | Validator attestations | 1.5 |
| `/icn/challenges/1.0.0` | Challenges | 1.5 |

#### Protobuf Definition for Control Messages

```protobuf
syntax = "proto2";

message ControlPrune {
    optional string topicID = 1;
    repeated PeerInfo peers = 2; // Gossipsub v1.1 PX
    optional uint64 backoff = 3; // Backoff time (seconds)
}

message PeerInfo {
    optional bytes peerID = 1;
    optional bytes signedPeerRecord = 2;
}
```

#### Fanout Maintenance

```pseudocode
for each topic in fanout:
  if time since last published > fanout_ttl:
    remove topic from fanout
  else if |fanout[topic]| < D:
    select D - |fanout[topic]| peers from peers.gossipsub[topic] - fanout[topic]
    add the peers to fanout[topic]
```

#### ICN Peer Scoring Thresholds

| Threshold | Value | Effect |
|-----------|-------|--------|
| `gossip_threshold` | -10 | Below this, no IHAVE/IWANT exchange |
| `publish_threshold` | -50 | Below this, no message publishing |
| `graylist_threshold` | -100 | Below this, all messages rejected |

---

## 3. AI/ML Technologies

### 3.1 PyTorch

**Library ID:** `/pytorch/pytorch`

PyTorch is the ML runtime for ICN's Vortex engine. All models must remain resident in VRAM (RTX 3060 12GB minimum).

#### GPU Tensor Operations

```python
import torch

# Create tensors on GPU
cuda = torch.device('cuda')
cuda0 = torch.device('cuda:0')

# Different ways to create GPU tensors
x = torch.tensor([1., 2.], device=cuda)
y = torch.tensor([1., 2.]).cuda()
z = torch.tensor([1., 2.]).to(cuda)

# Check if tensor is on CUDA
print(x.is_cuda)  # True
```

#### Managing Multiple GPUs

```python
cuda = torch.device('cuda')      # Default CUDA device
cuda0 = torch.device('cuda:0')   # GPU 0
cuda1 = torch.device('cuda:1')   # GPU 1

# Create tensor on specific GPU
x = torch.tensor([1., 2.], device=cuda1)

# Move between devices
x = x.to(cuda0)
x = x.cpu()  # Move to CPU
```

#### C++ Tensor Device Management

```cpp
// Move tensor to default GPU
torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);

// Move to specific GPU by index
torch::Tensor gpu_two_tensor = float_tensor.to(torch::Device(torch::kCUDA, 1));

// Asynchronous CPU/GPU copy
torch::Tensor async_cpu_tensor = gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);
```

#### ICN VRAM Budget

| Component | Model | Precision | VRAM |
|-----------|-------|-----------|------|
| Actor Generation | Flux-Schnell | NF4 (4-bit) | ~6.0 GB |
| Video Warping | LivePortrait | FP16 | ~3.5 GB |
| Text-to-Speech | Kokoro-82M | FP32 | ~0.4 GB |
| Semantic Verify (Primary) | CLIP-ViT-B-32 | INT8 | ~0.3 GB |
| Semantic Verify (Secondary) | CLIP-ViT-L-14 | INT8 | ~0.6 GB |
| System Overhead | PyTorch/CUDA | - | ~1.0 GB |
| **TOTAL** | | | **~11.8 GB** |

---

### 3.2 OpenCLIP

**Library ID:** `/mlfoundations/open_clip`

OpenCLIP is an open-source implementation of OpenAI's CLIP model, used for semantic verification in ICN.

#### Basic Usage

```python
import torch
from PIL import Image
import open_clip

# Create model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k'
)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Process image and text
image = preprocess(Image.open("image.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

# Compute embeddings
with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

#### ICN Dual CLIP Ensemble

ICN uses a dual CLIP ensemble for semantic verification:

```python
import open_clip
import torch

# Load both models
model_b32, _, preprocess_b32 = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model_l14, _, preprocess_l14 = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s34b_b88k'
)

model_b32.eval()
model_l14.eval()

# Weights for ensemble
WEIGHT_B32 = 0.4
WEIGHT_L14 = 0.6

# Thresholds
THRESHOLD_B32 = 0.70
THRESHOLD_L14 = 0.72

def verify_semantic(image, text_prompt):
    """Dual CLIP verification for ICN"""
    with torch.no_grad():
        # Encode with both models
        img_b32 = model_b32.encode_image(preprocess_b32(image).unsqueeze(0))
        img_l14 = model_l14.encode_image(preprocess_l14(image).unsqueeze(0))
        
        text_b32 = model_b32.encode_text(tokenizer_b32([text_prompt]))
        text_l14 = model_l14.encode_text(tokenizer_l14([text_prompt]))
        
        # Compute similarities
        sim_b32 = (img_b32 @ text_b32.T).item()
        sim_l14 = (img_l14 @ text_l14.T).item()
        
        # Check thresholds
        passes_b32 = sim_b32 >= THRESHOLD_B32
        passes_l14 = sim_l14 >= THRESHOLD_L14
        
        return passes_b32 and passes_l14
```

#### Generate Image Captions with CoCa

```python
import open_clip
import torch
from PIL import Image

# Load CoCa model
model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model.eval()

# Generate caption
image = Image.open("image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
    generated_tokens = model.generate(image_tensor, seq_len=20)

caption = open_clip.decode(generated_tokens[0])
caption = caption.split("<end_of_text>")[0].replace("<start_of_text>", "")
print(f"Generated caption: {caption}")
```

---

### 3.3 Hugging Face Transformers

**Library ID:** `/websites/huggingface_co_transformers_v4_56_2_en`

Transformers provides pre-trained models for NLP, vision, and audio tasks.

#### Using Pipelines for Inference

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# Image classification
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("image.jpg")

# Fill-mask
mask_filler = pipeline("fill-mask", "bert-base-uncased")
result = mask_filler("The Milky Way is a [MASK] galaxy.", top_k=3)
```

#### Enable FP16 Inference

```python
import torch
from transformers import pipeline

# FP16 for faster GPU inference
pipe = pipeline(
    "text-generation",
    model="gpt2-large",
    dtype=torch.float16,
    device="cuda"
)
```

#### Manual Inference with Models

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image

# Load processor and model
processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")
model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

# Prepare inputs
image = Image.open("image.jpg")
question = "What is in this image?"
inputs = processor(image, question, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```

#### Clean Up Resources

```python
# Always clean up after inference
del model
del pipe
torch.cuda.empty_cache()
```

---

## 4. Frontend Technologies

### 4.1 Tauri

**Library ID:** `/websites/rs_tauri_2_9_5`

Tauri is a framework for building desktop applications with Rust backend and web frontend. ICN uses Tauri 2.0 for the Viewer Client.

#### App-Wide Menu Management

```rust
/// Show the app-wide menu for all windows
pub fn show_menu(&self) -> crate::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let is_app_menu_set = self.manager.menu.menu_lock().is_some();
        if is_app_menu_set {
            for window in self.manager.windows().values() {
                if window.has_app_wide_menu() {
                    window.show_menu()?;
                }
            }
        }
    }
    Ok(())
}

/// Hide the app-wide menu
pub fn hide_menu(&self) -> crate::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let is_app_menu_set = self.manager.menu.menu_lock().is_some();
        if is_app_menu_set {
            for window in self.manager.windows().values() {
                if window.has_app_wide_menu() {
                    window.hide_menu()?;
                }
            }
        }
    }
    Ok(())
}
```

#### Setting System Tray Icon

```rust
/// Set the icon for the system tray
#[cfg(all(desktop, feature = "tray-icon"))]
#[inline(always)]
pub fn set_tray_icon(&mut self, icon: Option<image::Image<'static>>) {
    self.tray_icon = icon;
}
```

#### Retrieving Application Assets

```rust
pub fn get(&self, path: String) -> Option<Asset> {
    let use_https_scheme = /* infer URL scheme */;
    // Falls back to reading from 'distDir' if necessary
    // Requires frontend assets to be built first
}
```

---

### 4.2 React

**Library ID:** `/websites/react_dev`

React is used as the frontend framework for the Tauri-based Viewer Client.

#### useState Hook

```javascript
import { useState } from 'react';

function MyButton() {
    const [count, setCount] = useState(0);
    
    function handleClick() {
        setCount(count + 1);
    }
    
    return (
        <button onClick={handleClick}>
            Clicked {count} times
        </button>
    );
}
```

#### Lifting State Up

```javascript
export default function MyApp() {
    const [count, setCount] = useState(0);
    
    function handleClick() {
        setCount(count + 1);
    }
    
    return (
        <div>
            <h1>Counters that update together</h1>
            <MyButton count={count} onClick={handleClick} />
            <MyButton count={count} onClick={handleClick} />
        </div>
    );
}
```

#### useReducer for Complex State

```javascript
import { useReducer } from 'react';

function tasksReducer(tasks, action) {
    switch (action.type) {
        case 'added':
            return [...tasks, { id: action.id, text: action.text, done: false }];
        case 'changed':
            return tasks.map(t => t.id === action.task.id ? action.task : t);
        case 'deleted':
            return tasks.filter(t => t.id !== action.id);
        default:
            throw Error('Unknown action: ' + action.type);
    }
}

function TaskApp() {
    const [tasks, dispatch] = useReducer(tasksReducer, initialTasks);
    
    function handleAddTask(text) {
        dispatch({ type: 'added', id: nextId++, text });
    }
    // ...
}
```

---

### 4.3 Zustand

**Library ID:** `/pmndrs/zustand`

Zustand is a small, fast state-management solution used in the ICN Viewer Client.

#### Creating a Store

```jsx
import { create } from 'zustand';

const useBearStore = create((set) => ({
    bears: 0,
    increasePopulation: () => set((state) => ({ bears: state.bears + 1 })),
    removeAllBears: () => set({ bears: 0 }),
}));
```

#### Using the Store in Components

```jsx
function BearCounter() {
    const bears = useBearStore((state) => state.bears);
    return <h1>{bears} around here...</h1>;
}

function Controls() {
    const increasePopulation = useBearStore((state) => state.increasePopulation);
    return <button onClick={increasePopulation}>one up</button>;
}
```

#### State Persistence with Migration

```jsx
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useStore = create(
    persist(
        (set) => ({
            // state
            version: 1,
        }),
        {
            name: 'icn-storage',
            version: 1,
            migrate: (persistedState, version) => {
                if (version === 0) {
                    // Migration from v0 to v1
                    return { ...persistedState, newField: 'default' };
                }
                return persistedState;
            },
        }
    )
);
```

#### Overwriting State

```jsx
const useFishStore = create((set) => ({
    salmon: 1,
    tuna: 2,
    deleteEverything: () => set({}, true), // Clears entire store
    deleteTuna: () => set(({ tuna, ...rest }) => rest, true),
}));
```

---

## 5. Backend & Infrastructure

### 5.1 Tokio

**Library ID:** `/websites/rs_tokio_tokio`

Tokio is the async runtime for Rust, powering all ICN off-chain nodes.

#### Basic Async TCP Server

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    
    loop {
        let (mut socket, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buf = [0; 1024];
            
            loop {
                let n = match socket.read(&mut buf).await {
                    Ok(0) => return, // Connection closed
                    Ok(n) => n,
                    Err(e) => {
                        println!("Read error: {:?}", e);
                        return;
                    }
                };
                
                if let Err(e) = socket.write_all(&buf[0..n]).await {
                    println!("Write error: {:?}", e);
                    return;
                }
            }
        });
    }
}
```

#### Manual Runtime Management

```rust
use tokio::runtime::Runtime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = Runtime::new()?;
    
    rt.block_on(async {
        // Your async code here
        let listener = TcpListener::bind("127.0.0.1:8080").await?;
        // ...
        Ok(())
    })
}
```

#### Spawning Tasks

```rust
use tokio::runtime::Runtime;

let rt = Runtime::new().unwrap();

// Spawn a future onto the runtime
rt.spawn(async {
    println!("Running on a worker thread");
});
```

#### Using Handle for Cross-Thread Execution

```rust
use tokio::runtime::Handle;

#[tokio::main]
async fn main() {
    let handle = Handle::current();
    
    std::thread::spawn(move || {
        handle.block_on(async {
            println!("Running async code from std thread");
        });
    });
}
```

---

### 5.2 PyO3

**Library ID:** `/pyo3/pyo3`

PyO3 provides Rust bindings for Python, enabling the Vortex engine to interface between Rust and Python.

#### Embedding Python in Rust

```rust
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::ffi::c_str;

fn main() -> PyResult<()> {
    Python::attach(|py| {
        let sys = py.import("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        println!("Python version: {}", version);
        Ok(())
    })
}
```

#### Creating Python Extension Modules

```rust
use pyo3::prelude::*;

/// A simple function that adds two numbers
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// The Python module
#[pymodule]
fn string_sum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
```

#### Wrapping Python Objects in Rust Traits

```rust
use pyo3::prelude::*;
use pyo3::types::PyList;

pub trait Model {
    fn set_variables(&mut self, inputs: &Vec<f64>);
    fn compute(&mut self);
    fn get_results(&self) -> Vec<f64>;
}

struct UserModel {
    model: Py<PyAny>,
}

impl Model for UserModel {
    fn set_variables(&mut self, var: &Vec<f64>) {
        Python::attach(|py| {
            self.model
                .bind(py)
                .call_method("set_variables", (PyList::new(py, var).unwrap(),), None)
                .unwrap();
        })
    }
    
    fn get_results(&self) -> Vec<f64> {
        Python::attach(|py| {
            self.model
                .bind(py)
                .call_method("get_results", (), None)
                .unwrap()
                .extract()
                .unwrap()
        })
    }
    
    fn compute(&mut self) {
        Python::attach(|py| {
            self.model
                .bind(py)
                .call_method("compute", (), None)
                .unwrap();
        })
    }
}
```

#### Building with Maturin

```bash
# Create new project
mkdir string_sum && cd string_sum
python -m venv .env
source .env/bin/activate
pip install maturin

# Initialize with PyO3 bindings
maturin init
# Select: pyo3

# Build and install
maturin develop
```

---

## 6. DevOps & Observability

### 6.1 Prometheus

**Library ID:** `/prometheus/docs`

Prometheus is the monitoring system for ICN, collecting metrics from all nodes.

#### Basic Configuration

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 10s

rule_files:
  - rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - localhost:9093

scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
  
  - job_name: icn_director
    static_configs:
      - targets: ["localhost:9100"]
  
  - job_name: icn_validator
    static_configs:
      - targets: ["localhost:9101"]
```

#### Defining Alert Rules

```yaml
groups:
  - name: icn_alerts
    rules:
      - alert: DirectorSlotMissed
        expr: icn_director_slot_missed_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Director missed slot"
      
      - alert: BftLatencyHigh
        expr: icn_bft_round_duration_seconds > 10
        for: 2m
        labels:
          severity: warning
```

#### Defining Counter Metrics (Go)

```go
var pingCounter = prometheus.NewCounter(
    prometheus.CounterOpts{
        Name: "ping_request_count",
        Help: "Number of requests handled by Ping handler",
    },
)

func init() {
    prometheus.MustRegister(pingCounter)
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
    pingCounter.Inc()
    w.Write([]byte("pong"))
}
```

#### Node Exporter Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: node
    static_configs:
      - targets: ['localhost:9100']
```

#### ICN Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `icn_vortex_generation_time_seconds` | Histogram | Video generation latency (P99 < 15s) |
| `icn_bft_round_duration_seconds` | Histogram | BFT consensus round time (P99 < 10s) |
| `icn_p2p_connected_peers` | Gauge | Number of connected peers (> 10) |
| `icn_total_staked_tokens` | Gauge | Total ICN tokens staked |
| `icn_slashing_events_total` | Counter | Total slashing events |
| `icn_chain_block_height` | Gauge | Current chain block height |

---

### 6.2 Grafana

**Library ID:** `/grafana/grafana`

Grafana is the visualization platform for ICN dashboards.

#### Dashboard API - Update Permissions

```json
POST /api/dashboards/id/:dashboardId/permissions
{
    "items": [
        {
            "userId": 1,
            "role": "Viewer",
            "permission": 1
        },
        {
            "teamId": 2,
            "role": "Editor",
            "permission": 2
        }
    ]
}
```

#### KQL Query for VM Performance

```kusto
Perf
| where $__timeFilter(TimeGenerated)
| where CounterName == "% Processor Time"
| summarize avg(CounterValue) by bin(TimeGenerated, 5m), Computer
| order by TimeGenerated asc
```

#### Performance Tracking with TypeScript

```typescript
import { performanceUtils } from '@grafana/scenes';

interface ScenePerformanceObserver {
    onDashboardInteractionStart?(data: performanceUtils.DashboardInteractionStartData): void;
    onDashboardInteractionComplete?(data: performanceUtils.DashboardInteractionCompleteData): void;
    onPanelOperationStart?(data: performanceUtils.PanelPerformanceData): void;
    onPanelOperationComplete?(data: performanceUtils.PanelPerformanceData): void;
    onQueryStart?(data: performanceUtils.QueryPerformanceData): void;
    onQueryComplete?(data: performanceUtils.QueryPerformanceData): void;
}

const tracker = performanceUtils.getScenePerformanceTracker();
tracker.addObserver(myObserver);
```

---

### 6.3 Docker

**Library ID:** `/websites/docs_docker_com`

Docker is used for containerizing ICN components.

#### Basic Container Commands

```bash
# List all containers
docker ps --all

# Run container interactively
docker run --name=app-container -ti node-base

# Run with GPU support
docker run --gpus all -d icn-director

# Start container and bind ports
docker run -p 8080:80 -d my-app
```

#### Container Image for Kubernetes

```yaml
# In Kubernetes Deployment spec
containers:
  - name: icn-director
    image: ghcr.io/icn-network/icn-director:latest
    resources:
      limits:
        nvidia.com/gpu: 1
```

#### Docker Desktop Kubernetes Images

```bash
# List Kubernetes images
docker desktop kubernetes images

# JSON format
docker desktop kubernetes images --format json
```

---

### 6.4 Kubernetes

**Library ID:** `/websites/kubernetes_io`

Kubernetes orchestrates ICN Super-Nodes and Regional Relays.

#### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: icn-supernode
spec:
  selector:
    matchLabels:
      app: icn-supernode
  replicas: 2
  template:
    metadata:
      labels:
        app: icn-supernode
    spec:
      containers:
        - name: icn-supernode
          image: ghcr.io/icn-network/supernode:latest
          ports:
            - containerPort: 8080
```

#### Exposing as a Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: icn-supernode-service
spec:
  selector:
    app: icn-supernode
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

#### kubectl Commands

```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Get pods
kubectl get pods -l app=icn-supernode -o wide

# Scale deployment
kubectl scale deployment icn-supernode --replicas=4

# Expose deployment
kubectl expose deployment icn-supernode --port=80 --target-port=8080
```

#### ConfigMap for Application Config

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: icn-config
data:
  CHAIN_WS_URL: "ws://icn-chain:9944"
  P2P_PORT: "30333"
  REGION: "us-east-1"
```

---

## Quick Reference

### Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **On-Chain** | Polkadot SDK, FRAME, Custom Pallets |
| **Chain Client** | subxt |
| **P2P** | rust-libp2p, GossipSub, Kademlia, QUIC |
| **AI/ML** | PyTorch, OpenCLIP, HF Transformers |
| **Frontend** | Tauri, React, Zustand |
| **Backend** | Rust, Tokio, PyO3 |
| **DevOps** | Docker, Kubernetes, Prometheus, Grafana |

### Key Dependencies (from PRD)

| Dependency | Version | Purpose |
|------------|---------|---------|
| Polkadot SDK | polkadot-stable2409 | Runtime framework |
| libp2p | 0.53.0 | P2P networking |
| Flux-Schnell | NF4 | Image generation |
| LivePortrait | v1.0 | Video warping |
| Kokoro-82M | v1.0 | Text-to-speech |
| CLIP-ViT | B-32, L-14 | Semantic verification |
| PyTorch | 2.1+ | ML runtime |

---

*Document generated from Context7 - Last updated: December 25, 2025*
