use crate::{Cid, StorageBackend, StorageError};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone)]
pub struct LocalBackend {
    root: PathBuf,
}

impl LocalBackend {
    pub fn new(root: PathBuf) -> Result<Self, StorageError> {
        if root.as_os_str().is_empty() {
            return Err(StorageError::Backend("storage root is empty".to_string()));
        }
        Ok(Self { root })
    }

    fn validate_cid(&self, cid: &Cid) -> Result<(), StorageError> {
        if cid.trim().is_empty() || cid.contains('/') || cid.contains('\\') || cid.contains("..") {
            return Err(StorageError::InvalidCid(cid.clone()));
        }
        Ok(())
    }

    fn cid_dir(&self, cid: &Cid) -> PathBuf {
        self.root.join(cid)
    }

    fn data_path(&self, cid: &Cid) -> PathBuf {
        self.cid_dir(cid).join("data.bin")
    }

    fn pin_path(&self, cid: &Cid) -> PathBuf {
        self.cid_dir(cid).join(".pin")
    }

    async fn ensure_parent(path: &Path) -> Result<(), StorageError> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for LocalBackend {
    async fn put(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
        self.validate_cid(cid)?;
        let target = self.data_path(cid);
        let temp = target.with_extension("tmp");

        Self::ensure_parent(&target).await?;

        let mut file = tokio::fs::File::create(&temp).await?;
        file.write_all(data).await?;
        file.flush().await?;
        drop(file);

        tokio::fs::rename(&temp, &target).await?;
        Ok(())
    }

    async fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError> {
        self.validate_cid(cid)?;
        let path = self.data_path(cid);
        let data = tokio::fs::read(path).await?;
        Ok(data)
    }

    async fn pin(&self, cid: &Cid) -> Result<(), StorageError> {
        self.validate_cid(cid)?;
        let path = self.pin_path(cid);
        Self::ensure_parent(&path).await?;
        tokio::fs::write(path, b"pinned").await?;
        Ok(())
    }

    async fn unpin(&self, cid: &Cid) -> Result<(), StorageError> {
        self.validate_cid(cid)?;
        let path = self.pin_path(cid);
        if tokio::fs::metadata(&path).await.is_ok() {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }

    async fn pin_status(&self, cid: &Cid) -> Result<crate::PinStatus, StorageError> {
        self.validate_cid(cid)?;
        let path = self.pin_path(cid);
        if tokio::fs::metadata(&path).await.is_ok() {
            Ok(crate::PinStatus::Pinned)
        } else {
            Ok(crate::PinStatus::NotPinned)
        }
    }
}
