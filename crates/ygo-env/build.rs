fn main() {
    // The engine keeps each duel in an arena by replacing the global C++
    // `operator new` and `operator delete`.  Python loads this library with
    // `dlopen`, after other libraries that carry their own: bind the engine's
    // calls to the ones compiled in here, or a restored snapshot goes wrong
    // (see the engine's `docs/snapshot-restore-prototype.md`).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-Bsymbolic-functions");
    }
}
