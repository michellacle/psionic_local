#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

unsafe extern "C" {
    fn imported_increment(value: i32) -> i32;
}

#[unsafe(no_mangle)]
pub extern "C" fn host_import_bridge(value: i32) -> i32 {
    unsafe { imported_increment(value) }
}
