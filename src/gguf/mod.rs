pub mod metadata;

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Instant;

use byteorder::{ByteOrder, LittleEndian};
use crate::tensor::{DType, TensorView};
use metadata::{MetadataArray, MetadataValue};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug)]
pub struct TensorInfo {
    /// Byte range of the name within the GGUF buffer.
    name_start: usize,
    name_len: usize,
    /// Byte offset of the shape u64s within the GGUF buffer.
    shape_start: usize,
    n_dims: usize,
    pub dtype: DType,
    pub offset: u64,
}

/// Page-aligned, page-padded heap buffer backed by an anonymous mmap.
/// The base pointer is 16KB-aligned and the allocation length is rounded up
/// to a page multiple, so Metal's `newBufferWithBytesNoCopy` can wrap any
/// page-aligned subregion safely (the trailing pages are valid memory).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    mmap_len: usize,
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
    fn from(e: io::Error) -> Self { GgufError::Io(e) }
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

/// Zero-copy cursor into a byte slice. Unlike std::io::Cursor, records byte
/// positions so we can borrow &str and &[u64] directly from the buffer.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self { Reader { buf, pos: 0 } }

    fn u8(&mut self) -> u8 { let v = self.buf[self.pos]; self.pos += 1; v }
    fn i8(&mut self) -> i8 { self.u8() as i8 }
    fn u16(&mut self) -> u16 { let v = LittleEndian::read_u16(&self.buf[self.pos..]); self.pos += 2; v }
    fn i16(&mut self) -> i16 { self.u16() as i16 }
    fn u32(&mut self) -> u32 { let v = LittleEndian::read_u32(&self.buf[self.pos..]); self.pos += 4; v }
    fn i32(&mut self) -> i32 { self.u32() as i32 }
    fn f32(&mut self) -> f32 { let v = LittleEndian::read_f32(&self.buf[self.pos..]); self.pos += 4; v }
    fn u64(&mut self) -> u64 { let v = LittleEndian::read_u64(&self.buf[self.pos..]); self.pos += 8; v }
    fn i64(&mut self) -> i64 { self.u64() as i64 }
    fn f64(&mut self) -> f64 { let v = LittleEndian::read_f64(&self.buf[self.pos..]); self.pos += 8; v }

    /// Read a GGUF string, returning a borrowed &str into the underlying buffer.
    fn str(&mut self) -> Result<&'a str, GgufError> {
        let len = self.u64() as usize;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|_| GgufError::InvalidUtf8)?;
        self.pos += len;
        Ok(s)
    }

    /// Record the byte position and length of a string without copying.
    fn str_range(&mut self) -> Result<(usize, usize), GgufError> {
        let len = self.u64() as usize;
        let start = self.pos;
        // Validate UTF-8
        std::str::from_utf8(&self.buf[start..start + len])
            .map_err(|_| GgufError::InvalidUtf8)?;
        self.pos += len;
        Ok((start, len))
    }

    /// Record the byte position of n_dims contiguous u64 shape values.
    fn shape_range(&mut self, n_dims: usize) -> (usize, usize) {
        let start = self.pos;
        self.pos += n_dims * 8;
        (start, n_dims)
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let start = Instant::now();
        let file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        let data = read_file_fast(&file, file_size)?;

        let gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("Read model weights into RAM: {:.2} GB in {:.3}s ({:.1} GB/s)", gb, elapsed, gb / elapsed);

        let mut r = Reader::new(data.as_slice());

        let magic = r.u32();
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = r.u32();
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = r.u64() as usize;
        let metadata_kv_count = r.u64() as usize;

        // Metadata keys must be owned Strings because they're HashMap keys.
        // Values with strings also need owned copies (tokenizer vocab etc.).
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = r.str()?.to_string();
            let value = read_metadata_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Tensor info: names and shapes borrow directly from the buffer.
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (name_start, name_len) = r.str_range()?;
            let n_dims = r.u32() as usize;
            let (shape_start, _) = r.shape_range(n_dims);
            let dtype_id = r.u32();
            let dtype = DType::from_gguf_type(dtype_id)
                .ok_or(GgufError::UnknownDType(dtype_id))?;
            let offset = r.u64();
            tensors.push(TensorInfo { name_start, name_len, shape_start, n_dims, dtype, offset });
        }

        let data_offset = (r.pos + GGUF_DEFAULT_ALIGNMENT - 1) & !(GGUF_DEFAULT_ALIGNMENT - 1);

        Ok(GgufFile { version, metadata, tensors, data, data_offset })
    }

    pub fn aligned_base(&self) -> (*const u8, usize) {
        (self.data.base_ptr(), self.data.mmap_len())
    }

    pub fn file_size(&self) -> usize {
        self.data.as_slice().len()
    }

    fn tensor_name(&self, info: &TensorInfo) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data.as_slice()[info.name_start..info.name_start + info.name_len]) }
    }

    fn tensor_shape(&self, info: &TensorInfo) -> &[u64] {
        let bytes = &self.data.as_slice()[info.shape_start..info.shape_start + info.n_dims * 8];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, info.n_dims) }
    }

    fn tensor_size_bytes(info: &TensorInfo, shape: &[u64]) -> usize {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        info.dtype.storage_size(n)
    }

    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>, GgufError> {
        let info = self.tensors.iter().find(|t| self.tensor_name(t) == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let shape = self.tensor_shape(info);
        let start = self.data_offset + info.offset as usize;
        let size = Self::tensor_size_bytes(info, shape);
        Ok(TensorView {
            name: self.tensor_name(info),
            dtype: info.dtype,
            shape,
            data: &self.data.as_slice()[start..start + size],
        })
    }
}

fn read_metadata_value(r: &mut Reader) -> Result<MetadataValue, GgufError> {
    let vtype = r.u32();
    match vtype {
        0 => Ok(MetadataValue::U8(r.u8())),
        1 => Ok(MetadataValue::I8(r.i8())),
        2 => Ok(MetadataValue::U16(r.u16())),
        3 => Ok(MetadataValue::I16(r.i16())),
        4 => Ok(MetadataValue::U32(r.u32())),
        5 => Ok(MetadataValue::I32(r.i32())),
        6 => Ok(MetadataValue::F32(r.f32())),
        7 => Ok(MetadataValue::Bool(r.u8() != 0)),
        8 => Ok(MetadataValue::String(r.str()?.to_string())),
        9 => read_metadata_array(r),
        10 => Ok(MetadataValue::U64(r.u64())),
        11 => Ok(MetadataValue::I64(r.i64())),
        12 => Ok(MetadataValue::F64(r.f64())),
        _ => Err(GgufError::UnknownMetadataType(vtype)),
    }
}

fn read_metadata_array(r: &mut Reader) -> Result<MetadataValue, GgufError> {
    let etype = r.u32();
    let count = r.u64() as usize;

    macro_rules! read_array {
        ($variant:ident, $read:expr) => {{
            let mut v = Vec::with_capacity(count);
            for _ in 0..count { v.push($read); }
            Ok(MetadataValue::Array(MetadataArray::$variant(v)))
        }};
    }

    match etype {
        0 => read_array!(U8, r.u8()),
        1 => read_array!(I8, r.i8()),
        2 => read_array!(U16, r.u16()),
        3 => read_array!(I16, r.i16()),
        4 => read_array!(U32, r.u32()),
        5 => read_array!(I32, r.i32()),
        6 => read_array!(F32, r.f32()),
        7 => read_array!(Bool, r.u8() != 0),
        8 => read_array!(String, r.str()?.to_string()),
        10 => read_array!(U64, r.u64()),
        11 => read_array!(I64, r.i64()),
        12 => read_array!(F64, r.f64()),
        _ => Err(GgufError::UnknownMetadataType(etype)),
    }
}

/// Parallel pread to saturate NVMe queue depth.
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
