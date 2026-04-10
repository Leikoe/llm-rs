#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(MetadataArray),
}

#[derive(Debug, Clone)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::U32(n) => Some(*n),
            MetadataValue::I32(n) => Some(*n as u32),
            MetadataValue::U64(n) => Some(*n as u32),
            MetadataValue::I64(n) => Some(*n as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::F32(n) => Some(*n),
            MetadataValue::F64(n) => Some(*n as f32),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<&[String]> {
        match self {
            MetadataValue::Array(MetadataArray::String(v)) => Some(v),
            _ => None,
        }
    }
}
