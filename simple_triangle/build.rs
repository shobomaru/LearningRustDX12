fn main() {
    // delayload
    println!("cargo:rustc-link-arg=delayimp.lib");
    println!("cargo:rustc-link-arg=/DELAYLOAD:dxcompiler.dll");
}
