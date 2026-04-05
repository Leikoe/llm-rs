fn main() {
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

    let mut air_files = Vec::new();
    for entry in &metal_files {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let air_path = format!("{out_dir}/{stem}.air");

        let status = Command::new("xcrun")
            .args([
                "metal",
                "-c",
                path.to_str().unwrap(),
                "-o",
                &air_path,
                "-std=metal3.1",
            ])
            .status()
            .expect("failed to compile Metal shader");

        if !status.success() {
            panic!("Metal shader compilation failed for {}", path.display());
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

    let status = cmd.status().expect("failed to link Metal library");
    if !status.success() {
        panic!("Metal library linking failed");
    }
}
