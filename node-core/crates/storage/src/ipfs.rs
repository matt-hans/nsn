use crate::{Cid, StorageBackend, StorageError};
use async_trait::async_trait;
use reqwest::multipart;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct IpfsBackend {
    api_url: String,
    client: Client,
}

#[derive(Debug, Deserialize)]
struct IpfsAddResponse {
    #[serde(rename = "Hash")]
    hash: String,
}

impl IpfsBackend {
    pub fn new(api_url: String) -> Result<Self, StorageError> {
        if api_url.trim().is_empty() {
            return Err(StorageError::Backend("IPFS API URL is empty".to_string()));
        }

        let client = Client::builder().no_proxy().build()?;

        Ok(Self {
            api_url: api_url.trim_end_matches('/').to_string(),
            client,
        })
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/api/v0/{}", self.api_url, path.trim_start_matches('/'))
    }
}

#[async_trait]
impl StorageBackend for IpfsBackend {
    async fn put(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError> {
        let form = multipart::Form::new().part(
            "file",
            multipart::Part::bytes(data.to_vec()).file_name("payload.bin"),
        );

        let response = self
            .client
            .post(self.endpoint("add?pin=false"))
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;

        let body: IpfsAddResponse = response.json().await?;
        if body.hash != *cid {
            return Err(StorageError::CidMismatch {
                expected: cid.clone(),
                actual: body.hash,
            });
        }

        Ok(())
    }

    async fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError> {
        let response = self
            .client
            .post(self.endpoint(&format!("cat?arg={cid}")))
            .send()
            .await?
            .error_for_status()?;

        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    }

    async fn pin(&self, cid: &Cid) -> Result<(), StorageError> {
        self.client
            .post(self.endpoint(&format!("pin/add?arg={cid}")))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    async fn unpin(&self, cid: &Cid) -> Result<(), StorageError> {
        self.client
            .post(self.endpoint(&format!("pin/rm?arg={cid}")))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipfs_backend_url_normalization() {
        let backend = IpfsBackend::new("http://127.0.0.1:5001/".to_string()).unwrap();
        assert_eq!(backend.endpoint("add"), "http://127.0.0.1:5001/api/v0/add");
    }
}
