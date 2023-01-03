pub mod simple_triangle;

const WINDOW_WIDTH: u32 = 640;
const WINDOW_HEIGHT: u32 = 360;

use windows::core::Result;

fn main() -> Result<()> {
    use crate::simple_triangle::*;
    let r = default_main(WINDOW_WIDTH, WINDOW_HEIGHT);
    r
}
