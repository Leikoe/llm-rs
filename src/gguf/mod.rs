pub mod metadata;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt};
use crate::tensor::{DType, TensorView};
use metadata::{MetadataArray, MetadataValue};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn size_bytes(&self) -> usize {
        let n: usize = self.shape.iter().map(|&d| d as usize).product();
        self.dtype.storage_size(n)
    }
}

/// Page-aligned, page-padded heap buffer backed by an anonymous mmap.
/// The base pointer is 16KB-aligned and the allocation length is rounded up
/// to a page multiple, so Metal's `newBufferWithBytesNoCopy` can wrap any
/// page-aligned subregion safely (the trailing pages are valid memory).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,        // logical length (file size)
    mmap_len: usize,   // physical mapping length (multiple of PAGE_SIZE)
}

const PAGE_SIZE: usize = 16384;

impl AlignedBuf {
    fn alloc(len: usize) -> io::Result<Self> {
        let mmap_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(AlignedBuf { ptr: ptr as *mut u8, len, mmap_len })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr }

    /// Total mapped length (page-padded). The region [ptr, ptr+mmap_len) is
    /// all valid memory — needed for Metal to safely wrap with newBufferWithBytesNoCopy.
    pub fn mmap_len(&self) -> usize { self.mmap_len }
    pub fn base_ptr(&self) -> *const u8 { self.ptr }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut _, self.mmap_len); }
    }
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    data: AlignedBuf,
    data_offset: usize,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    UnknownDType(u32),
    UnknownMetadataType(u32),
    InvalidUtf8,
    TensorNotFound(String),
}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "IO error: {e}"),
            GgufError::InvalidMagic(m) => write!(f, "invalid GGUF magic: 0x{m:08x}"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::UnknownDType(t) => write!(f, "unknown dtype: {t}"),
            GgufError::UnknownMetadataType(t) => write!(f, "unknown metadata type: {t}"),
            GgufError::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            GgufError::TensorNotFound(n) => write!(f, "tensor not found: {n}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let start = Instant::now();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        let data = read_file_fast(&file, file_size)?;

        let gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("Read {:.2} GB in {:.3}s ({:.1} GB/s)", gb, elapsed, gb / elapsed);

        let mut cursor = Cursor::new(data.as_slice());

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            let dtype_id = cursor.read_u32::<LittleEndian>()?;
            let dtype = DType::from_gguf_type(dtype_id)
                .ok_or(GgufError::UnknownDType(dtype_id))?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            tensors.push(TensorInfo { name, shape, dtype, offset });
        }

        let header_end = cursor.position() as usize;
        let data_offset = (header_end + GGUF_DEFAULT_ALIGNMENT - 1) & !(GGUF_DEFAULT_ALIGNMENT - 1);

        Ok(GgufFile { version, metadata, tensors, data, data_offset })
    }

    /// Page-aligned base pointer of the underlying allocation. Length is page-padded.
    /// Used by the Metal backend for zero-copy tensor uploads via newBufferWithBytesNoCopy.
    pub fn aligned_base(&self) -> (*const u8, usize) {
        (self.data.base_ptr(), self.data.mmap_len())
    }

    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>, GgufError> {
        let info = self.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let start = self.data_offset + info.offset as usize;
        let size = info.size_bytes();
        Ok(TensorView {
            name: info.name.clone(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            data: &self.data.as_slice()[start..start + size],
        })
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
}

fn read_metadata_value(cursor: &mut Cursor<&[u8]>) -> Result<MetadataValue, GgufError> {
    let value_type = cursor.read_u32::<LittleEndian>()?;
    match value_type {
        0 => Ok(MetadataValue::U8(cursor.read_u8()?)),
        1 => Ok(MetadataValue::I8(cursor.read_i8()?)),
        2 => Ok(MetadataValue::U16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::I16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::U32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::I32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::F32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_string(cursor)?)),
        9 => read_metadata_array(cursor),
        10 => Ok(MetadataValue::U64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::I64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::F64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(GgufError::UnknownMetadataType(value_type)),
    }
}

fn read_metadata_array(cursor: &mut Cursor<&[u8]>) -> Result<MetadataValue, GgufError> {
    let element_type = cursor.read_u32::<LittleEndian>()?;
    let count = cursor.read_u64::<LittleEndian>()? as usize;

    macro_rules! read_array {
        ($variant:ident, $read:expr) => {{
            let mut v = Vec::with_capacity(count);
            for _ in 0..count { v.push($read); }
            Ok(MetadataValue::Array(MetadataArray::$variant(v)))
        }};
    }

    match element_type {
        0 => read_array!(U8, cursor.read_u8()?),
        1 => read_array!(I8, cursor.read_i8()?),
        2 => read_array!(U16, cursor.read_u16::<LittleEndian>()?),
        3 => read_array!(I16, cursor.read_i16::<LittleEndian>()?),
        4 => read_array!(U32, cursor.read_u32::<LittleEndian>()?),
        5 => read_array!(I32, cursor.read_i32::<LittleEndian>()?),
        6 => read_array!(F32, cursor.read_f32::<LittleEndian>()?),
        7 => read_array!(Bool, cursor.read_u8()? != 0),
        8 => read_array!(String, read_string(cursor)?),
        10 => read_array!(U64, cursor.read_u64::<LittleEndian>()?),
        11 => read_array!(I64, cursor.read_i64::<LittleEndian>()?),
        12 => read_array!(F64, cursor.read_f64::<LittleEndian>()?),
        _ => Err(GgufError::UnknownMetadataType(element_type)),
    }
}

/// Parallel pread to saturate NVMe queue depth.
/// ~5 GB/s cold (SSD-bound), ~18 GB/s warm (4x over single-threaded read_exact).
fn read_file_fast(file: &File, size: usize) -> io::Result<AlignedBuf> {
    let fd = file.as_raw_fd();
    let mut data = AlignedBuf::alloc(size)?;

    const N_THREADS: usize = 8;
    const CHUNK: usize = 4 * 1024 * 1024;
    let base = data.as_mut_ptr() as usize;
    let stripe = (size + N_THREADS - 1) / N_THREADS;

    std::thread::scope(|s| {
        for t in 0..N_THREADS {
            let start = t * stripe;
            let end = (start + stripe).min(size);
            s.spawn(move || {
                let mut off = start;
                while off < end {
                    let len = CHUNK.min(end - off);
                    let buf = unsafe {
                        std::slice::from_raw_parts_mut((base + off) as *mut u8, len)
                    };
                    let mut read = 0;
                    while read < len {
                        let n = unsafe {
                            libc::pread(fd, buf[read..].as_mut_ptr().cast(), len - read, (off + read) as libc::off_t)
                        };
                        if n <= 0 { return; }
                        read += n as usize;
                    }
                    off += len;
                }
            });
        }
    });

    Ok(data)
}

