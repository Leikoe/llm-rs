use std::collections::HashMap;

/// BPE encode a text string into token IDs.
/// If `byte_level` is true, input bytes are first mapped through GPT-2's bytes_to_unicode
/// table before matching against the vocabulary.
pub fn bpe_encode(
    text: &str,
    token_to_id: &HashMap<Vec<u8>, u32>,
    merges: &HashMap<(Vec<u8>, Vec<u8>), u32>,
    byte_level: bool,
) -> Vec<u32> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return vec![];
    }

    // Start with individual bytes, mapped through GPT-2 encoding if needed
    let mut tokens: Vec<Vec<u8>> = if byte_level {
        bytes
            .iter()
            .map(|&b| {
                let ch = gpt2_byte_to_char(b);
                let mut buf = [0u8; 4];
                ch.encode_utf8(&mut buf).as_bytes().to_vec()
            })
            .collect()
    } else {
        bytes.iter().map(|&b| vec![b]).collect()
    };

    // Repeatedly merge the highest-priority (lowest rank) pair
    loop {
        if tokens.len() < 2 {
            break;
        }

        let mut best_rank = u32::MAX;
        let mut best_idx = usize::MAX;

        for i in 0..tokens.len() - 1 {
            if let Some(&rank) = merges.get(&(tokens[i].clone(), tokens[i + 1].clone())) {
                if rank < best_rank {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if best_idx == usize::MAX {
            break;
        }

        let mut merged = tokens[best_idx].clone();
        merged.extend_from_slice(&tokens[best_idx + 1]);
        tokens[best_idx] = merged;
        tokens.remove(best_idx + 1);
    }

    tokens
        .iter()
        .map(|t| {
            token_to_id
                .get(t)
                .copied()
                .unwrap_or(0)
        })
        .collect()
}

/// GPT-2 bytes_to_unicode mapping: map a raw byte to its Unicode codepoint.
fn gpt2_byte_to_char(b: u8) -> char {
    let cp: u32 = match b {
        // These bytes map to themselves as Unicode codepoints
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => b as u32,
        // Other bytes are shifted to 256+
        _ => {
            let idx = super::OTHER_BYTES.iter().position(|&x| x == b).unwrap();
            256 + idx as u32
        }
    };
    char::from_u32(cp).unwrap()
}
