mod vad;

use vad::VAD;

fn main() {
    println!("init");

    let mut vad = VAD::new("/home/heaust/whisper/ggml/ggml-distil-large-v3.bin");
    vad.set_path("out.wav".to_string());
    vad.start_with_vad(None);
}
