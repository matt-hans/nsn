# T049: Implement Docker Container Manager

## Priority: P1 (Critical Path)
## Complexity: 1 week
## Status: Pending
## Depends On: None

---

## Objective

Replace the stubbed container manager with a real Docker-based implementation that provides process isolation for AI workloads on Lane 1 (general compute).

## Background

Current implementation in `node-core/crates/container_manager/` is a stub:

```rust
pub async fn spawn_container(&self, _task: &Task) -> Result<ContainerId, Error> {
    // TODO: Implement container spawning
    Ok(ContainerId::default())
}
```

This means Lane 1 tasks execute without isolation, creating security vulnerabilities.

## Implementation

### Step 1: Docker Client Integration

```rust
use bollard::Docker;
use bollard::container::{Config, CreateContainerOptions, StartContainerOptions};
use bollard::models::{HostConfig, DeviceRequest};

pub struct DockerContainerManager {
    docker: Docker,
    config: ContainerConfig,
}

impl DockerContainerManager {
    pub async fn new() -> Result<Self, ContainerError> {
        let docker = Docker::connect_with_local_defaults()?;

        // Verify Docker is running
        docker.ping().await?;

        Ok(Self {
            docker,
            config: ContainerConfig::default(),
        })
    }
}
```

### Step 2: Container Creation with GPU Passthrough

```rust
impl DockerContainerManager {
    pub async fn spawn_container(
        &self,
        task: &Task,
        image: &str,
    ) -> Result<ContainerId, ContainerError> {
        let container_name = format!("nsn-task-{}", task.id);

        // Configure GPU access
        let device_requests = if task.requires_gpu {
            Some(vec![DeviceRequest {
                driver: Some("nvidia".to_string()),
                count: Some(-1), // All GPUs
                capabilities: Some(vec![vec!["gpu".to_string()]]),
                ..Default::default()
            }])
        } else {
            None
        };

        let host_config = HostConfig {
            memory: Some(task.memory_limit_mb * 1024 * 1024),
            cpu_quota: Some(task.cpu_limit_percent * 1000),
            device_requests,
            network_mode: Some("none".to_string()), // Network isolation
            readonly_rootfs: Some(true),
            security_opt: Some(vec!["no-new-privileges".to_string()]),
            ..Default::default()
        };

        let config = Config {
            image: Some(image.to_string()),
            env: Some(self.build_env(task)),
            host_config: Some(host_config),
            cmd: Some(task.command.clone()),
            ..Default::default()
        };

        let container = self.docker
            .create_container(
                Some(CreateContainerOptions { name: &container_name, .. }),
                config,
            )
            .await?;

        // Start the container
        self.docker
            .start_container(&container.id, None::<StartContainerOptions<String>>)
            .await?;

        Ok(ContainerId(container.id))
    }
}
```

### Step 3: Resource Limits and Monitoring

```rust
pub struct ResourceLimits {
    pub memory_mb: u64,
    pub cpu_percent: u32,
    pub gpu_memory_mb: Option<u64>,
    pub timeout_secs: u64,
}

impl DockerContainerManager {
    pub async fn monitor_container(
        &self,
        container_id: &ContainerId,
    ) -> Result<ContainerStats, ContainerError> {
        let stats = self.docker
            .stats(&container_id.0, None)
            .await?;

        Ok(ContainerStats {
            memory_usage_mb: stats.memory_stats.usage / (1024 * 1024),
            cpu_percent: self.calculate_cpu_percent(&stats),
            running: stats.state == "running",
        })
    }

    pub async fn enforce_timeout(
        &self,
        container_id: &ContainerId,
        timeout: Duration,
    ) -> Result<(), ContainerError> {
        tokio::select! {
            _ = tokio::time::sleep(timeout) => {
                tracing::warn!("Container {} timed out, killing", container_id.0);
                self.kill_container(container_id).await?;
                Err(ContainerError::Timeout)
            }
            result = self.wait_container(container_id) => {
                result
            }
        }
    }
}
```

### Step 4: Output Collection

```rust
impl DockerContainerManager {
    pub async fn collect_output(
        &self,
        container_id: &ContainerId,
    ) -> Result<TaskOutput, ContainerError> {
        // Get container logs
        let logs = self.docker
            .logs(
                &container_id.0,
                Some(LogsOptions {
                    stdout: true,
                    stderr: true,
                    ..Default::default()
                }),
            )
            .try_collect::<Vec<_>>()
            .await?;

        // Get exit code
        let inspect = self.docker.inspect_container(&container_id.0, None).await?;
        let exit_code = inspect.state.and_then(|s| s.exit_code).unwrap_or(-1);

        // Collect output files if specified
        let output_files = self.extract_output_files(container_id).await?;

        Ok(TaskOutput {
            exit_code,
            stdout: self.parse_logs(&logs, LogType::Stdout),
            stderr: self.parse_logs(&logs, LogType::Stderr),
            output_files,
        })
    }
}
```

### Step 5: Cleanup and Resource Reclamation

```rust
impl DockerContainerManager {
    pub async fn cleanup_container(
        &self,
        container_id: &ContainerId,
    ) -> Result<(), ContainerError> {
        // Stop container if running
        let _ = self.docker
            .stop_container(&container_id.0, None)
            .await;

        // Remove container
        self.docker
            .remove_container(
                &container_id.0,
                Some(RemoveContainerOptions {
                    force: true,
                    v: true, // Remove volumes
                    ..Default::default()
                }),
            )
            .await?;

        Ok(())
    }

    /// Cleanup all orphaned NSN containers
    pub async fn cleanup_orphans(&self) -> Result<u32, ContainerError> {
        let containers = self.docker
            .list_containers(Some(ListContainersOptions {
                all: true,
                filters: HashMap::from([
                    ("name".to_string(), vec!["nsn-task-*".to_string()]),
                ]),
                ..Default::default()
            }))
            .await?;

        let mut cleaned = 0;
        for container in containers {
            if let Some(id) = container.id {
                self.cleanup_container(&ContainerId(id)).await?;
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }
}
```

## Security Considerations

1. **No network access by default** - Containers run with `network_mode: none`
2. **Read-only root filesystem** - Prevents container modifications
3. **No privilege escalation** - `no-new-privileges` security option
4. **Resource limits** - Memory and CPU caps prevent DoS
5. **Timeout enforcement** - Tasks cannot run indefinitely
6. **Isolated GPU access** - NVIDIA runtime with memory limits

## Acceptance Criteria

- [ ] Docker containers spawn and execute tasks
- [ ] GPU passthrough works for AI workloads
- [ ] Resource limits (memory, CPU) are enforced
- [ ] Timeouts kill runaway containers
- [ ] Output collection retrieves logs and files
- [ ] Cleanup removes containers and volumes
- [ ] Integration tests with real Docker
- [ ] Documentation for container security model

## Testing

```rust
#[tokio::test]
async fn test_container_spawn() {
    let manager = DockerContainerManager::new().await.unwrap();

    let task = Task {
        id: TaskId::new(),
        command: vec!["echo".to_string(), "hello".to_string()],
        requires_gpu: false,
        memory_limit_mb: 512,
        ..Default::default()
    };

    let container_id = manager.spawn_container(&task, "alpine:latest").await.unwrap();
    let output = manager.collect_output(&container_id).await.unwrap();

    assert_eq!(output.exit_code, 0);
    assert!(output.stdout.contains("hello"));

    manager.cleanup_container(&container_id).await.unwrap();
}
```

## Deliverables

1. `node-core/crates/container_manager/src/docker.rs` - Docker implementation
2. `node-core/crates/container_manager/src/limits.rs` - Resource limiting
3. `node-core/crates/container_manager/src/output.rs` - Output collection
4. Integration tests
5. Documentation

---

**This task is critical for Lane 1 security.**
