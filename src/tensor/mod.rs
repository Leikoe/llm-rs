#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl DType {
    /// Number of elements per quantization block.
    pub const fn block_elems(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 | DType::F64 => 1,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 => 32,
            DType::Q8_0 | DType::Q8_1 => 32,
            DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K | DType::Q8K => 256,
            DType::IQ2XXS | DType::IQ2XS | DType::IQ2S => 256,
            DType::IQ3XXS | DType::IQ3S => 256,
            DType::IQ1S | DType::IQ1M => 256,
            DType::IQ4NL | DType::IQ4XS => 32,
        }
    }

    /// Size in bytes of one quantization block.
    pub const fn block_size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Q4_0 => 18,   // 2 (scale) + 16 (32 x 4-bit)
            DType::Q4_1 => 20,   // 2 (scale) + 2 (min) + 16
            DType::Q5_0 => 22,   // 2 + 4 (high bits) + 16
            DType::Q5_1 => 24,   // 2 + 2 + 4 + 16
            DType::Q8_0 => 34,   // 2 (scale) + 32 (32 x 8-bit)
            DType::Q8_1 => 40,   // 4 (scale f32) + 4 (min f32) + 32
            // K-quant block sizes (256 elements per block)
            DType::Q2K => 84,
            DType::Q3K => 110,
            DType::Q4K => 144,
            DType::Q5K => 176,
            DType::Q6K => 210,
            DType::Q8K => 292,
            // IQ types
            DType::IQ2XXS => 66,
            DType::IQ2XS => 74,
            DType::IQ2S => 82,
            DType::IQ3XXS => 98,
            DType::IQ3S => 110,
            DType::IQ1S => 50,
            DType::IQ1M => 56,
            DType::IQ4NL => 18,
            DType::IQ4XS => 36,
        }
    }

    /// Total bytes needed for `n_elements` stored in this dtype.
    pub fn storage_size(self, n_elements: usize) -> usize {
        let be = self.block_elems();
        assert!(n_elements % be == 0, "element count {n_elements} not divisible by block size {be}");
        (n_elements / be) * self.block_size_bytes()
    }

    /// Try to parse from the GGUF type id.
    pub fn from_gguf_type(t: u32) -> Option<Self> {
        match t {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            3 => Some(DType::Q4_1),
            6 => Some(DType::Q5_0),
            7 => Some(DType::Q5_1),
            8 => Some(DType::Q8_0),
            9 => Some(DType::Q8_1),
            10 => Some(DType::Q2K),
            11 => Some(DType::Q3K),
            12 => Some(DType::Q4K),
            13 => Some(DType::Q5K),
            14 => Some(DType::Q6K),
            15 => Some(DType::Q8K),
            16 => Some(DType::IQ2XXS),
            17 => Some(DType::IQ2XS),
            18 => Some(DType::IQ3XXS),
            19 => Some(DType::IQ1S),
            20 => Some(DType::IQ4NL),
            21 => Some(DType::IQ3S),
            22 => Some(DType::IQ2S),
            23 => Some(DType::IQ4XS),
            24 => Some(DType::I8),
            25 => Some(DType::I16),
            26 => Some(DType::I32),
            27 => Some(DType::I64),
            28 => Some(DType::F64),
            29 => Some(DType::IQ1M),
            30 => Some(DType::BF16),
            _ => None,
        }
    }
}

/// RoPE pair-selection strategy.
#[derive(Debug, Clone, Copy)]
pub enum RopeLayout {
    /// LLaMA: rotate `(x[2i], x[2i+1])`.
    Interleaved,
    /// Qwen/HF: rotate `(x[i], x[i+head_dim/2])`.
    SplitHalf,
}

#[derive(Debug)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [u64],
    pub data: &'a [u8],
}

impl TensorView<'_> {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }
}
