mod vad;

use std::ffi::c_void;

use vad::VAD;

fn main() {
    unsafe {
        // silence whisper logs
        unsafe extern "C" fn silent_cb(_level: u32, _msg: *const i8, _user: *mut c_void) {
            // do nothing => fully silent
        }
        whisper_rs::set_log_callback(Some(silent_cb), std::ptr::null_mut());
    }

    let mut vad = VAD::new("/home/heaust/whisper/ggml/ggml-distil-large-v3.bin");
    vad.set_path("out.wav".to_string());
    vad.start_with_vad(None);
}
