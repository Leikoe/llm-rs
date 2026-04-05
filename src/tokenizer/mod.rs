mod bpe;

use std::collections::HashMap;

/// GPT-2 bytes_to_unicode: bytes that don't map to themselves (0-32, 127-160, 173).
/// Shared between encoder (bpe.rs) and decoder (this file).
pub(crate) static OTHER_BYTES: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
    155, 156, 157, 158, 159, 160, 173,
];

use crate::gguf::metadata::MetadataValue;

pub struct Tokenizer {
    /// Token ID -> token bytes
    vocab: Vec<Vec<u8>>,
    /// Token bytes -> token ID
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Merge pairs: (pair_bytes) -> merge rank (lower = higher priority)
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    /// Whether this tokenizer uses GPT-2 byte-level encoding (LLaMA 3)
    /// vs sentencepiece (LLaMA 2).
    is_byte_level: bool,
}

impl Tokenizer {
    /// Build tokenizer from GGUF metadata.
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        // Extract token strings
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .expect("missing tokenizer.ggml.tokens");

        // Build vocab: decode token strings to raw bytes
        // GGUF tokens may contain raw byte escapes like <0x0A>
        let vocab: Vec<Vec<u8>> = tokens.iter().map(|t| decode_token_str(t)).collect();

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token_bytes) in vocab.iter().enumerate() {
            token_to_id.insert(token_bytes.clone(), id as u32);
        }

        // Extract merge rules
        let mut merges = HashMap::new();
        if let Some(merge_strs) = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
        {
            for (rank, merge_str) in merge_strs.iter().enumerate() {
                // Each merge is "token_a token_b"
                if let Some((a, b)) = merge_str.split_once(' ') {
                    merges.insert(
                        (decode_token_str(a), decode_token_str(b)),
                        rank as u32,
                    );
                }
            }
        }

        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);

        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        // Detect GPT-2 byte-level tokenizer: check if vocab contains Ġ (U+0120)
        // which is the GPT-2 encoding of space (0x20).
        let tokenizer_model = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let is_byte_level = tokenizer_model == "gpt2";

        Tokenizer {
            vocab,
            token_to_id,
            merges,
            bos_id,
            eos_id,
            is_byte_level,
        }
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }
        bpe::bpe_encode(text, &self.token_to_id, &self.merges, self.is_byte_level)
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            bytes.extend_from_slice(&self.decode_token(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token to a displayable string.
    pub fn decode_token(&self, id: u32) -> Vec<u8> {
        let bytes = self.vocab.get(id as usize).map(|v| v.as_slice()).unwrap_or(b"");
        if self.is_byte_level {
            gpt2_bytes_decode(bytes)
        } else {
            // Sentencepiece: replace ▁ with space
            let s = String::from_utf8_lossy(bytes);
            s.replace('\u{2581}', " ").into_bytes()
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Decode a GGUF token string to raw bytes.
/// Handles byte-level tokens like `<0x0A>` and sentencepiece `\u2581` (▁ = space).
fn decode_token_str(s: &str) -> Vec<u8> {
    // Check for byte tokens like <0x0A>
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }
    // Store token string as raw UTF-8 bytes.
    // We handle space markers during decode, not during vocab construction,
    // since different models use different conventions.
    s.as_bytes().to_vec()
}

/// GPT-2 byte-level BPE maps each byte 0x00-0xFF to a Unicode codepoint.
/// This function reverses that mapping: given token bytes (UTF-8 encoded Unicode),
/// decode each character back to its original byte value.
///
/// The GPT-2 mapping:
/// - Printable ASCII bytes (!, ", #, ..., ~) map to themselves
/// - Other bytes map to Unicode codepoints 0x100-0x143
/// - Specifically: space (0x20) -> Ġ (U+0120), newline (0x0A) -> Ċ (U+010A), etc.
fn gpt2_bytes_decode(token_utf8: &[u8]) -> Vec<u8> {
    let s = match std::str::from_utf8(token_utf8) {
        Ok(s) => s,
        Err(_) => return token_utf8.to_vec(),
    };

    let mut result = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let cp = ch as u32;
        let byte = gpt2_char_to_byte(cp);
        result.push(byte);
    }
    result
}

/// Map a GPT-2 Unicode codepoint back to the original byte.
fn gpt2_char_to_byte(cp: u32) -> u8 {
    // The GPT-2 bytes_to_unicode mapping:
    // Printable bytes that map to themselves:
    //   '!' (33) to '~' (126), '¡' (161) to '¬' (172), '®' (174) to 'ÿ' (255)
    // All other bytes (0-32, 127-160, 173) are shifted to 256+n where n is their
    // position in the "remaining" list.
    match cp {
        // Direct mappings: these codepoints equal the byte value
        33..=126 | 161..=172 | 174..=255 => cp as u8,
        // Shifted mappings: codepoints 256+ map back to the "other" bytes
        256.. => {
            let idx = (cp - 256) as usize;
            if idx < OTHER_BYTES.len() {
                OTHER_BYTES[idx]
            } else {
                b'?' // shouldn't happen
            }
        }
        // Shouldn't happen for valid GPT-2 tokens
        _ => cp as u8,
    }
}
