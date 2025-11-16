mod vad;

use vad::VAD;

fn main() {
    println!("init");

    let mut vad = VAD::new();
    vad.set_path("out.wav".to_string());
    vad.start_with_vad(None);
}
