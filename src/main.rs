mod vad;

use vad::VAD;

fn main() {
    println!("init");

    let mut vad = VAD::new();
    vad.set_path("out.wav".to_string());
    vad.start();

    println!("recording. press enter to stop");
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();

    vad.save();
}
