//! Content-addressed artifact resolver backed directly by the journal blob CAS.

use std::{fmt, fs, io, ops::Range};

use omp_core::{CowBytes, Str};
use omp_journal::blob::{BlobRef, BlobStore};
use omp_tool::ArtifactLifetime;
use omp_tools::read::{
	Fault,
	resolver::{
		ArtifactCatalog, ArtifactRecord, ArtifactResolver, BlobAuthority, BlobStat, Resolve,
		ResourceCompletion, ResourceList,
	},
	selector::ParsedSelector,
};
use url::Url;

const MAX_INLINE_BYTES: u64 = 8 * 1024 * 1024;

/// The stores an `artifact://` digest may live in, in lookup order: the
/// environment's own blob store, then the verdict store that retains every
/// tool outcome delivered to a session. A job settlement the kernel spills is
/// the tool's own verdict, so an address the kernel hands the model resolves
/// here even though the kernel keeps its copy in the session journal's CAS.
#[derive(Clone, Debug)]
struct BlobStoreAuthority {
	stores: Vec<BlobStore>,
}

impl BlobStoreAuthority {
	/// The store holding `digest` and the reference into it, or the I/O fault
	/// of the last store tried.
	fn locate(&self, digest: &str) -> Result<(&BlobStore, BlobRef), Fault> {
		let probe = BlobRef::parse_hex(digest, 0).map_err(storage_fault)?;
		let mut missing = None;
		for store in &self.stores {
			match fs::metadata(store.path(&probe)) {
				Ok(metadata) => {
					let reference = BlobRef::parse_hex(digest, metadata.len()).map_err(storage_fault)?;
					return Ok((store, reference));
				},
				Err(source) => missing = Some(source),
			}
		}
		Err(io_fault(missing.unwrap_or_else(|| {
			io::Error::new(io::ErrorKind::NotFound, "no artifact store is configured")
		})))
	}

	fn reference(&self, digest: &str) -> Result<BlobRef, Fault> {
		self.locate(digest).map(|(_, reference)| reference)
	}
}

#[derive(Clone, Debug)]
struct DigestCatalog {
	blobs: BlobStoreAuthority,
}

impl ArtifactCatalog for DigestCatalog {
	async fn by_ordinal(&self, _ordinal: u64) -> Result<Option<ArtifactRecord>, Fault> {
		Ok(None)
	}

	async fn by_digest<'a>(&'a self, digest: &'a str) -> Result<Option<ArtifactRecord>, Fault> {
		match self.blobs.reference(digest) {
			Ok(_) => Ok(Some(ArtifactRecord {
				digest:   Str::new(digest),
				lifetime: ArtifactLifetime::Durable,
			})),
			Err(Fault::Source { .. }) => Ok(None),
			Err(error) => Err(error),
		}
	}
}

impl BlobAuthority for BlobStoreAuthority {
	async fn stat<'a>(&'a self, digest: &'a str) -> Result<BlobStat, Fault> {
		Ok(BlobStat { byte_len: self.reference(digest)?.size })
	}

	async fn read_range<'a>(
		&'a self,
		digest: &'a str,
		range: Range<u64>,
	) -> Result<CowBytes<'static>, Fault> {
		let (store, reference) = self.locate(digest)?;
		let bytes = store.get(&reference).map_err(storage_fault)?;
		let start = usize::try_from(range.start).map_err(|_| Fault::Invalid {
			message: Str::new_static("Artifact range exceeds host address limits."),
		})?;
		let end = usize::try_from(range.end).map_err(|_| Fault::Invalid {
			message: Str::new_static("Artifact range exceeds host address limits."),
		})?;
		if start > end || end > bytes.len() {
			return Err(Fault::Invalid {
				message: Str::new_static("Artifact range exceeds stored content."),
			});
		}
		Ok(CowBytes::from(bytes.slice(start..end)))
	}
}

/// Production artifact resolver for durable `artifact://sha256/<digest>` data.
pub(crate) struct ArtifactUrlResolver {
	inner: ArtifactResolver<DigestCatalog, BlobStoreAuthority>,
	blobs: BlobStoreAuthority,
}

impl ArtifactUrlResolver {
	/// Opens the resolver over `store`, falling back to `verdicts` for a
	/// digest the environment store does not hold.
	pub(super) fn open(
		store: BlobStore,
		verdicts: Option<BlobStore>,
		_session: &str,
	) -> Result<Self, io::Error> {
		let mut stores = vec![store];
		stores.extend(verdicts);
		let blobs = BlobStoreAuthority { stores };
		let catalog = DigestCatalog { blobs: blobs.clone() };
		Ok(Self { inner: ArtifactResolver::new(catalog, blobs.clone()), blobs })
	}

	fn digest<'a>(&self, resource: &'a str) -> Result<&'a str, Fault> {
		let digest = resource.strip_prefix("sha256/").unwrap_or(resource);
		if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
			Ok(digest)
		} else {
			Err(Fault::Invalid {
				message: Str::new_static(
					"Artifact addresses must be artifact://sha256/<64-hex-digest>.",
				),
			})
		}
	}
}

impl Resolve for ArtifactUrlResolver {
	async fn read<'a>(
		&'a self,
		resource: &'a str,
		selector: &'a ParsedSelector,
	) -> Result<CowBytes<'static>, Fault> {
		let digest = self.digest(resource)?;
		let reference = self.blobs.reference(digest)?;
		if reference.size > MAX_INLINE_BYTES
			&& matches!(selector, ParsedSelector::None | ParsedSelector::Raw)
		{
			return Err(Fault::Invalid {
				message: Str::new(format!(
					"Artifact {digest} is {} bytes; use a line selector or path-only mode.",
					reference.size
				)),
			});
		}
		self.inner.read(digest, selector).await
	}

	async fn list(
		&self,
		resource: &str,
		_max_entries: usize,
		_max_bytes: usize,
	) -> Result<ResourceList, Fault> {
		if resource.trim_matches('/').is_empty() {
			Ok(ResourceList { entries: Vec::new(), truncated: false })
		} else {
			Err(Fault::Invalid {
				message: Str::new_static(
					"Content-addressed artifacts do not expose a directory listing.",
				),
			})
		}
	}

	async fn path(&self, resource: &str) -> Result<Option<Str>, Fault> {
		let (store, reference) = self.blobs.locate(self.digest(resource)?)?;
		let url = Url::from_file_path(store.path(&reference)).map_err(|()| Fault::Invalid {
			message: Str::new_static("Artifact path cannot be represented as a file URI."),
		})?;
		Ok(Some(Str::new(url.as_str())))
	}

	async fn complete(
		&self,
		_query: &str,
		_max_results: usize,
	) -> Result<Vec<ResourceCompletion>, Fault> {
		Ok(Vec::new())
	}
}

impl fmt::Debug for ArtifactUrlResolver {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("ArtifactUrlResolver(..)")
	}
}

fn storage_fault(error: impl fmt::Display) -> Fault {
	Fault::Source { message: Str::new(format!("Artifact storage failed: {error}")) }
}

fn io_fault(source: io::Error) -> Fault {
	Fault::Source { message: Str::new(format!("Artifact storage I/O failed: {source}")) }
}

#[cfg(test)]
mod tests {
	use omp_journal::blob::BlobStore;
	use omp_tools::read::{resolver::Resolve, selector::ParsedSelector};

	use super::ArtifactUrlResolver;

	#[tokio::test]
	async fn a_digest_only_the_verdict_store_holds_still_resolves() {
		// A job settlement the kernel hands the model as artifact://sha256/…
		// is the tool's own verdict, retained in the verdict namespace rather
		// than the environment's main store.
		let root = tempfile::tempdir().expect("temporary directory");
		let store = BlobStore::open(root.path().join("blobs")).expect("main store");
		let verdicts = BlobStore::open(root.path().join("verdicts")).expect("verdict store");
		let reference = verdicts.put(b"{\"kind\":\"ok\"}").expect("verdict blob");
		let digest = format!("sha256/{}", reference.to_hex());

		let alone = ArtifactUrlResolver::open(store.clone(), None, "session").expect("resolver");
		assert!(alone.read(&digest, &ParsedSelector::None).await.is_err());

		let resolver = ArtifactUrlResolver::open(store, Some(verdicts), "session").expect("resolver");
		let bytes = resolver
			.read(&digest, &ParsedSelector::None)
			.await
			.expect("verdict resolves");
		assert_eq!(bytes.as_ref(), b"{\"kind\":\"ok\"}");
		let path = resolver
			.path(&digest)
			.await
			.expect("path resolves")
			.expect("a file URI");
		assert!(path.contains("verdicts"), "{path}");
	}
}
