use std::vec::Vec;

#[unsafe(no_mangle)]
pub extern "C" fn std_surface_len() -> i32 {
    let data = Vec::from([1i32, 2, 3]);
    data.len() as i32
}
