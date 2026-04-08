fn main() {
    println!("cargo::rustc-check-cfg=cfg(metal_precompiled)");
    #[cfg(target_os = "macos")]
    compile_metal_shaders();
}

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    use std::path::Path;
    use std::process::Command;

    let shader_dir = Path::new("shaders");
    if !shader_dir.exists() {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();

    let metal_files: Vec<_> = std::fs::read_dir(shader_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "metal")
        })
        .collect();

    if metal_files.is_empty() {
        return;
    }

    // Compile .metal -> .air -> .metallib ahead of time.
    // The .metallib is embedded in the binary and loaded at runtime,
    // avoiding any source-level JIT compilation.
    let mut air_files = Vec::new();
    for entry in &metal_files {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let air_path = format!("{out_dir}/{stem}.air");

        // `-frecord-sources` + `-gline-tables-only` embed the shader source
        // and line tables into the .metallib so Xcode's GPU frame capture can
        // show per-source-line counters in the shader profiler.
        let ok = Command::new("xcrun")
            .args([
                "metal",
                "-c",
                path.to_str().unwrap(),
                "-o",
                &air_path,
                "-std=metal4.0",
                "-O2",
                "-frecord-sources",
                "-gline-tables-only",
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !ok {
            println!("cargo:warning=Metal shader compilation failed — will JIT from source at runtime");
            return;
        }

        air_files.push(air_path);
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let metallib_path = format!("{out_dir}/default.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["metallib", "-o", &metallib_path]);
    for air in &air_files {
        cmd.arg(air);
    }

    if !cmd.status().map(|s| s.success()).unwrap_or(false) {
        println!("cargo:warning=Metal library linking failed — will JIT from source at runtime");
        return;
    }

    // Signal to the runtime that we have a pre-compiled metallib
    println!("cargo:rustc-cfg=metal_precompiled");
}
