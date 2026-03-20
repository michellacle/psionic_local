#![no_std]
#![no_main]

use core::hint::unreachable_unchecked;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn ub_guard(value: i32) -> i32 {
    if value == 0 {
        unsafe {
            unreachable_unchecked();
        }
    }
    value
}
