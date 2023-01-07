pub mod simple_texture;

const WINDOW_WIDTH: u32 = 640;
const WINDOW_HEIGHT: u32 = 360;

use windows::core::Result;

fn main() -> Result<()> {
    use crate::simple_texture::*;
    let r = default_main(WINDOW_WIDTH, WINDOW_HEIGHT);
    r
}
