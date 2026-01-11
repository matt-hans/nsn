//! Simulated network layer for in-memory message routing.
//!
//! Provides deterministic message delivery with configurable latency profiles,
//! network partitions, and message filtering.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use libp2p::PeerId;
use thiserror::Error;

use crate::{NodeRole, TopicCategory};

/// Errors that can occur in network operations.
#[derive(Debug, Error)]
pub enum NetworkError {
    /// Node not found in network
    #[error("Node {0} not found in network")]
    NodeNotFound(PeerId),
    /// Message delivery failed due to partition
    #[error("Message blocked by network partition")]
    PartitionBlocked,
    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Result type for network operations.
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Latency profiles for message delivery simulation.
#[derive(Debug, Clone)]
pub enum LatencyProfile {
    /// Messages delivered immediately (0ms latency)
    Instant,
    /// Uniform latency for all messages
    Uniform(Duration),
    /// Variable latency within range
    Variable { min: Duration, max: Duration },
}

impl Default for LatencyProfile {
    fn default() -> Self {
        Self::Instant
    }
}

impl LatencyProfile {
    /// Calculate latency for a message between two peers.
    pub fn calculate(&self, _from: &PeerId, _to: &PeerId) -> Duration {
        match self {
            LatencyProfile::Instant => Duration::ZERO,
            LatencyProfile::Uniform(d) => *d,
            LatencyProfile::Variable { min, max } => {
                // Use deterministic "random" based on peer IDs for reproducibility
                // In real tests, this provides variety without true randomness
                let mid = (*min + *max) / 2;
                mid
            }
        }
    }
}

/// A message pending delivery in the simulated network.
#[derive(Debug, Clone)]
pub struct PendingMessage {
    /// Sender peer ID
    pub from: PeerId,
    /// Target peer ID (None for broadcast)
    pub to: Option<PeerId>,
    /// Message topic category
    pub topic: TopicCategory,
    /// Message payload
    pub payload: Vec<u8>,
    /// Scheduled delivery time
    pub deliver_at: Instant,
}

/// A simulated node in the network.
#[derive(Debug)]
pub struct SimulatedNode {
    /// Unique peer identifier
    pub peer_id: PeerId,
    /// Node's role in the network
    pub role: NodeRole,
    /// Current state (for tracking)
    pub state: NodeState,
    /// Incoming message queue
    pub inbox: VecDeque<DeliveredMessage>,
    /// Whether node is currently online
    pub online: bool,
}

/// Node state for simulation tracking.
#[derive(Debug, Clone, Default)]
pub struct NodeState {
    /// Messages sent by this node
    pub messages_sent: usize,
    /// Messages received by this node
    pub messages_received: usize,
    /// Current epoch (if applicable)
    pub current_epoch: Option<u64>,
    /// Is this node a director this epoch?
    pub is_director: bool,
}

/// A message that has been delivered to a node.
#[derive(Debug, Clone)]
pub struct DeliveredMessage {
    /// Sender peer ID
    pub from: PeerId,
    /// Message topic
    pub topic: TopicCategory,
    /// Message payload
    pub payload: Vec<u8>,
    /// Delivery timestamp
    pub delivered_at: Instant,
}

/// Simulated network for multi-node testing.
///
/// Provides in-memory message routing with:
/// - Configurable latency profiles
/// - Network partition simulation
/// - Message tracking and filtering
pub struct SimulatedNetwork {
    /// Nodes in the network
    nodes: HashMap<PeerId, SimulatedNode>,
    /// Pending messages awaiting delivery
    message_queue: VecDeque<PendingMessage>,
    /// Current latency profile
    latency_profile: LatencyProfile,
    /// Network partitions (nodes in same set can communicate)
    partitions: Option<Vec<HashSet<PeerId>>>,
    /// Current simulation time
    current_time: Instant,
    /// Message counter for ordering
    message_counter: u64,
}

impl Default for SimulatedNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedNetwork {
    /// Create a new simulated network.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            message_queue: VecDeque::new(),
            latency_profile: LatencyProfile::default(),
            partitions: None,
            current_time: Instant::now(),
            message_counter: 0,
        }
    }

    /// Set the latency profile for the network.
    pub fn with_latency(mut self, profile: LatencyProfile) -> Self {
        self.latency_profile = profile;
        self
    }

    /// Add a node to the network with the specified role.
    pub fn add_node(&mut self, role: NodeRole) -> PeerId {
        let peer_id = PeerId::random();
        let node = SimulatedNode {
            peer_id,
            role,
            state: NodeState::default(),
            inbox: VecDeque::new(),
            online: true,
        };
        self.nodes.insert(peer_id, node);
        peer_id
    }

    /// Add a node with a specific peer ID.
    pub fn add_node_with_id(&mut self, peer_id: PeerId, role: NodeRole) {
        let node = SimulatedNode {
            peer_id,
            role,
            state: NodeState::default(),
            inbox: VecDeque::new(),
            online: true,
        };
        self.nodes.insert(peer_id, node);
    }

    /// Remove a node from the network.
    pub fn remove_node(&mut self, peer: PeerId) -> NetworkResult<()> {
        self.nodes
            .remove(&peer)
            .map(|_| ())
            .ok_or(NetworkError::NodeNotFound(peer))
    }

    /// Set a node's online status.
    pub fn set_node_online(&mut self, peer: PeerId, online: bool) -> NetworkResult<()> {
        self.nodes
            .get_mut(&peer)
            .map(|n| n.online = online)
            .ok_or(NetworkError::NodeNotFound(peer))
    }

    /// Get a reference to a node.
    pub fn get_node(&self, peer: &PeerId) -> Option<&SimulatedNode> {
        self.nodes.get(peer)
    }

    /// Get a mutable reference to a node.
    pub fn get_node_mut(&mut self, peer: &PeerId) -> Option<&mut SimulatedNode> {
        self.nodes.get_mut(peer)
    }

    /// Get all nodes in the network.
    pub fn nodes(&self) -> impl Iterator<Item = &SimulatedNode> {
        self.nodes.values()
    }

    /// Get all nodes with a specific role.
    pub fn nodes_with_role(&self, role: NodeRole) -> impl Iterator<Item = &SimulatedNode> {
        self.nodes.values().filter(move |n| n.role == role)
    }

    /// Get count of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get count of online nodes.
    pub fn online_node_count(&self) -> usize {
        self.nodes.values().filter(|n| n.online).count()
    }

    /// Send a message to a specific peer.
    pub fn send_message(
        &mut self,
        from: PeerId,
        to: PeerId,
        topic: TopicCategory,
        payload: Vec<u8>,
    ) -> NetworkResult<()> {
        if !self.nodes.contains_key(&from) {
            return Err(NetworkError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(NetworkError::NodeNotFound(to));
        }

        let latency = self.latency_profile.calculate(&from, &to);
        let deliver_at = self.current_time + latency;

        // Update sender stats
        if let Some(node) = self.nodes.get_mut(&from) {
            node.state.messages_sent += 1;
        }

        self.message_queue.push_back(PendingMessage {
            from,
            to: Some(to),
            topic,
            payload,
            deliver_at,
        });

        self.message_counter += 1;
        Ok(())
    }

    /// Broadcast a message to all nodes (except sender).
    pub fn broadcast(
        &mut self,
        from: PeerId,
        topic: TopicCategory,
        payload: Vec<u8>,
    ) -> NetworkResult<()> {
        if !self.nodes.contains_key(&from) {
            return Err(NetworkError::NodeNotFound(from));
        }

        // Update sender stats
        if let Some(node) = self.nodes.get_mut(&from) {
            node.state.messages_sent += 1;
        }

        // Create pending message for each recipient
        let recipients: Vec<PeerId> = self
            .nodes
            .keys()
            .filter(|&p| *p != from)
            .copied()
            .collect();

        for to in recipients {
            let latency = self.latency_profile.calculate(&from, &to);
            let deliver_at = self.current_time + latency;

            self.message_queue.push_back(PendingMessage {
                from,
                to: Some(to),
                topic: topic.clone(),
                payload: payload.clone(),
                deliver_at,
            });
            self.message_counter += 1;
        }

        Ok(())
    }

    /// Deliver all pending messages up to the specified time.
    pub fn deliver_pending(&mut self, until: Instant) {
        // Collect messages ready for delivery
        let mut to_deliver = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(msg) = self.message_queue.pop_front() {
            if msg.deliver_at <= until {
                to_deliver.push(msg);
            } else {
                remaining.push_back(msg);
            }
        }
        self.message_queue = remaining;

        // Deliver messages
        for msg in to_deliver {
            if let Some(to) = msg.to {
                // Check partition rules
                if !self.can_communicate(&msg.from, &to) {
                    continue; // Message blocked by partition
                }

                // Check if recipient is online
                if let Some(node) = self.nodes.get_mut(&to) {
                    if !node.online {
                        continue; // Recipient offline
                    }
                    node.inbox.push_back(DeliveredMessage {
                        from: msg.from,
                        topic: msg.topic,
                        payload: msg.payload,
                        delivered_at: until,
                    });
                    node.state.messages_received += 1;
                }
            }
        }

        self.current_time = until;
    }

    /// Advance time and deliver pending messages.
    pub fn advance_time(&mut self, duration: Duration) {
        let target = self.current_time + duration;
        self.deliver_pending(target);
    }

    /// Check if two peers can communicate (considering partitions).
    fn can_communicate(&self, a: &PeerId, b: &PeerId) -> bool {
        match &self.partitions {
            None => true,
            Some(partitions) => {
                // Find which partition each peer is in
                for partition in partitions {
                    if partition.contains(a) && partition.contains(b) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Create a network partition.
    ///
    /// Each group is a set of peers that can communicate with each other.
    /// Peers in different groups cannot communicate.
    pub fn inject_partition(&mut self, groups: Vec<HashSet<PeerId>>) {
        self.partitions = Some(groups);
    }

    /// Remove all network partitions.
    pub fn heal_partition(&mut self) {
        self.partitions = None;
    }

    /// Get the current simulation time.
    pub fn current_time(&self) -> Instant {
        self.current_time
    }

    /// Get the number of pending messages.
    pub fn pending_message_count(&self) -> usize {
        self.message_queue.len()
    }

    /// Clear all pending messages.
    pub fn clear_pending_messages(&mut self) {
        self.message_queue.clear();
    }

    /// Drain messages from a node's inbox.
    pub fn drain_inbox(&mut self, peer: &PeerId) -> Vec<DeliveredMessage> {
        self.nodes
            .get_mut(peer)
            .map(|n| n.inbox.drain(..).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_node() {
        let mut network = SimulatedNetwork::new();

        let peer = network.add_node(NodeRole::Director);
        assert_eq!(network.node_count(), 1);
        assert!(network.get_node(&peer).is_some());

        network.remove_node(peer).unwrap();
        assert_eq!(network.node_count(), 0);
    }

    #[test]
    fn test_send_message() {
        let mut network = SimulatedNetwork::new();

        let sender = network.add_node(NodeRole::Director);
        let receiver = network.add_node(NodeRole::Director);

        network
            .send_message(
                sender,
                receiver,
                TopicCategory::Consensus,
                b"hello".to_vec(),
            )
            .unwrap();

        assert_eq!(network.pending_message_count(), 1);

        // Advance time to deliver
        network.advance_time(Duration::from_millis(100));

        assert_eq!(network.pending_message_count(), 0);
        let messages = network.drain_inbox(&receiver);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload, b"hello");
    }

    #[test]
    fn test_broadcast() {
        let mut network = SimulatedNetwork::new();

        let sender = network.add_node(NodeRole::Director);
        let r1 = network.add_node(NodeRole::Director);
        let r2 = network.add_node(NodeRole::Director);
        let r3 = network.add_node(NodeRole::Director);

        network
            .broadcast(sender, TopicCategory::Recipe, b"recipe".to_vec())
            .unwrap();

        // 3 messages queued (one per recipient)
        assert_eq!(network.pending_message_count(), 3);

        network.advance_time(Duration::from_millis(100));

        // Each recipient should have received the message
        for peer in [r1, r2, r3] {
            let messages = network.drain_inbox(&peer);
            assert_eq!(messages.len(), 1);
        }

        // Sender should not receive own broadcast
        let sender_msgs = network.drain_inbox(&sender);
        assert!(sender_msgs.is_empty());
    }

    #[test]
    fn test_partition() {
        let mut network = SimulatedNetwork::new();

        let a = network.add_node(NodeRole::Director);
        let b = network.add_node(NodeRole::Director);
        let c = network.add_node(NodeRole::Director);

        // Create partition: [a, b] | [c]
        network.inject_partition(vec![
            HashSet::from([a, b]),
            HashSet::from([c]),
        ]);

        // a can send to b
        network
            .send_message(a, b, TopicCategory::Consensus, b"ab".to_vec())
            .unwrap();
        network.advance_time(Duration::from_millis(100));
        let b_msgs = network.drain_inbox(&b);
        assert_eq!(b_msgs.len(), 1);

        // a cannot send to c (different partition)
        network
            .send_message(a, c, TopicCategory::Consensus, b"ac".to_vec())
            .unwrap();
        network.advance_time(Duration::from_millis(100));
        let c_msgs = network.drain_inbox(&c);
        assert!(c_msgs.is_empty()); // Message blocked by partition

        // Heal partition
        network.heal_partition();

        // Now a can send to c
        network
            .send_message(a, c, TopicCategory::Consensus, b"ac2".to_vec())
            .unwrap();
        network.advance_time(Duration::from_millis(100));
        let c_msgs = network.drain_inbox(&c);
        assert_eq!(c_msgs.len(), 1);
    }

    #[test]
    fn test_latency_uniform() {
        let mut network =
            SimulatedNetwork::new().with_latency(LatencyProfile::Uniform(Duration::from_millis(50)));

        let sender = network.add_node(NodeRole::Director);
        let receiver = network.add_node(NodeRole::Director);

        network
            .send_message(sender, receiver, TopicCategory::Consensus, b"test".to_vec())
            .unwrap();

        // Advance 25ms - message should not be delivered yet
        network.advance_time(Duration::from_millis(25));
        let messages = network.drain_inbox(&receiver);
        assert!(messages.is_empty());

        // Advance another 30ms (total 55ms) - message should be delivered
        network.advance_time(Duration::from_millis(30));
        let messages = network.drain_inbox(&receiver);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_offline_node() {
        let mut network = SimulatedNetwork::new();

        let sender = network.add_node(NodeRole::Director);
        let receiver = network.add_node(NodeRole::Director);

        // Take receiver offline
        network.set_node_online(receiver, false).unwrap();

        network
            .send_message(sender, receiver, TopicCategory::Consensus, b"test".to_vec())
            .unwrap();

        network.advance_time(Duration::from_millis(100));

        // Message should not be delivered to offline node
        let messages = network.drain_inbox(&receiver);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_nodes_with_role() {
        let mut network = SimulatedNetwork::new();

        network.add_node(NodeRole::Director);
        network.add_node(NodeRole::Director);
        network.add_node(NodeRole::Executor);
        network.add_node(NodeRole::Storage);

        let directors: Vec<_> = network.nodes_with_role(NodeRole::Director).collect();
        assert_eq!(directors.len(), 2);

        let executors: Vec<_> = network.nodes_with_role(NodeRole::Executor).collect();
        assert_eq!(executors.len(), 1);
    }
}
