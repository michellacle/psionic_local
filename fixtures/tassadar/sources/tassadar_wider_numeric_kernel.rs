#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn pair_add_i64(left: i64, right: i64) -> i64 {
    left + right
}

#[unsafe(no_mangle)]
pub fn pair_sum_and_diff(left: i32, right: i32) -> (i32, i32) {
    (left + right, left - right)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sum_and_max_i64_into_buffer(
    values: *const i64,
    len: i32,
    out: *mut i64,
    out_len: i32,
) -> i32 {
    if out_len < 2 {
        return 1;
    }

    let mut total: i64 = 0;
    let mut maximum: i64 = 0;
    let mut index = 0;
    while index < len {
        let value = unsafe { *values.add(index as usize) };
        total += value;
        if index == 0 || value > maximum {
            maximum = value;
        }
        index += 1;
    }

    unsafe {
        *out.add(0) = total;
        *out.add(1) = maximum;
    }
    0
}
