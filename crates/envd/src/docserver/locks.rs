//! Owner-private process locks shared across isolated document authorities.

use std::{
	fs, io,
	path::{Path, PathBuf},
};

use omp_core::Hash32;

/// Physical directory identity, unaffected by path aliases or a root rename.
pub(super) fn project_key(root: &fs::Metadata, path: &Path) -> io::Result<Hash32> {
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt;
		let mut digest = Hash32::hasher();
		digest.update(root.dev().to_le_bytes());
		digest.update(root.ino().to_le_bytes());
		let _ = path;
		Ok(digest.finalize())
	}
	#[cfg(windows)]
	{
		use std::os::windows::fs::MetadataExt;
		let volume = root
			.volume_serial_number()
			.ok_or_else(|| io::Error::other("missing root volume identity"))?;
		let index = root
			.file_index()
			.ok_or_else(|| io::Error::other("missing root file identity"))?;
		let mut digest = Hash32::hasher();
		digest.update(volume.to_le_bytes());
		digest.update(index.to_le_bytes());
		let _ = path;
		Ok(digest.finalize())
	}
	#[cfg(not(any(unix, windows)))]
	{
		let _ = root;
		Ok(Hash32::sum(path.as_os_str().as_encoded_bytes()))
	}
}

/// Canonical owner namespace; data-directory aliases identify the same owner.
pub(super) fn namespace_key(path: &Path) -> io::Result<Hash32> {
	let path = match fs::canonicalize(path) {
		Ok(path) => path,
		Err(error) if error.kind() == io::ErrorKind::NotFound => std::path::absolute(path)?,
		Err(error) => return Err(error),
	};
	Ok(Hash32::sum(path.as_os_str().as_encoded_bytes()))
}

/// Opens a stable lock inode outside the tool-writable project. Never unlink
/// lock files: replacement would let concurrent processes lock different
/// inodes.
pub(super) fn open(kind: &str, key: &Hash32) -> io::Result<fs::File> {
	let directory = directory()?;
	let path = directory.join(format!("{kind}-{}.lock", key.to_hex()));
	let mut options = fs::OpenOptions::new();
	options.read(true).write(true).create(true).truncate(false);
	#[cfg(unix)]
	{
		use std::os::unix::fs::OpenOptionsExt;
		options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
	}
	let file = options.open(path)?;
	let metadata = file.metadata()?;
	if !metadata.is_file() {
		return Err(io::Error::new(
			io::ErrorKind::InvalidData,
			"document lock is not a regular file",
		));
	}
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt;
		if metadata.uid() != rustix::process::geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
			return Err(io::Error::new(
				io::ErrorKind::PermissionDenied,
				"document lock must be owner-only",
			));
		}
	}
	Ok(file)
}

fn directory() -> io::Result<PathBuf> {
	// Unix socket paths already use /tmp. Do not let app-specific TMPDIR or
	// OMP_* state overrides split the physical-project mutation lock domain.
	#[cfg(unix)]
	let directory = PathBuf::from("/tmp")
		.join(format!("omp-envd-authorities-{}", rustix::process::geteuid().as_raw()));
	#[cfg(not(unix))]
	let directory = std::env::temp_dir().join("omp-envd-authorities");
	let mut builder = fs::DirBuilder::new();
	#[cfg(unix)]
	{
		use std::os::unix::fs::DirBuilderExt;
		builder.mode(0o700);
	}
	match builder.create(&directory) {
		Ok(()) => {},
		Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {},
		Err(error) => return Err(error),
	}
	let metadata = fs::symlink_metadata(&directory)?;
	if !metadata.is_dir() {
		return Err(io::Error::new(
			io::ErrorKind::InvalidData,
			"document lock root is not a directory",
		));
	}
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt;
		if metadata.uid() != rustix::process::geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
			return Err(io::Error::new(
				io::ErrorKind::PermissionDenied,
				"document lock root must be owner-only",
			));
		}
	}
	Ok(directory)
}
