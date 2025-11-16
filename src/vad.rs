use std::io::{self, Write};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use cpal::traits::StreamTrait;
use cpal::traits::{DeviceTrait, HostTrait};
use hound;
use voice_activity_detector::VoiceActivityDetector;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

fn f32_to_i16(sample: f32) -> i16 {
    let s = sample.clamp(-1.0, 1.0);
    (s * i16::MAX as f32) as i16
}

pub struct VAD {
    output_path: String,
    accumulator: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    //
    sample_rate: u32,
    channels: u16,
    //
    whisper_state: WhisperState,
}

impl VAD {
    pub fn new(path_to_whisper: &str) -> Self {
        let output_path = "speechSamplerOut.wav".to_string();
        let accumulator = Arc::new(Mutex::new(Vec::new()));

        let whisper_context =
            WhisperContext::new_with_params(path_to_whisper, WhisperContextParameters::default())
                .expect("failed to load whisper context");

        let whisper_state = whisper_context
            .create_state()
            .expect("failed to create state");

        VAD {
            output_path,
            accumulator,
            stream: None,
            sample_rate: 44_100,
            channels: 1,
            whisper_state,
        }
    }

    pub fn set_path(&mut self, path: String) {
        self.output_path = path;
    }

    pub fn stop(&mut self) {
        self.stream = None;

        let mut guard = self.accumulator.lock().unwrap();
        guard.clear();
    }

    pub fn save(&mut self) {
        let sample_rate = self.sample_rate;
        let channels = self.channels;

        let samples: Vec<f32> = {
            let mut gaurd = self.accumulator.lock().expect("error getting samples");
            let mut taken = Vec::with_capacity(gaurd.len());
            std::mem::swap(&mut *gaurd, &mut taken);

            taken
        };

        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let writer = hound::WavWriter::create(&self.output_path, spec);
        match writer {
            Ok(mut w) => {
                for &s in &samples {
                    let res = w.write_sample(f32_to_i16(s));
                    if res.is_err() {
                        eprintln!("error writing sample");
                    }
                }

                let res = w.finalize();
                if res.is_err() {
                    eprintln!("error finalizing write");
                }
            }

            Err(err) => {
                eprint!("could not make writer {}", err);
            }
        }

        self.stop();

        self.transcribe(samples);
    }

    pub fn transcribe(&mut self, samples: Vec<f32>) {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.whisper_state
            .full(params, &samples[..])
            .expect("failed to run model");

        // fetch the results
        let num_segments = self.whisper_state.full_n_segments();

        let mut t = "".to_string();

        for i in 0..num_segments {
            // fetch the results
            let segment = self
                .whisper_state
                .get_segment(i)
                .expect("failed to get segment");

            t += &segment.to_string();
        }

        println!("transciption: {}", t);
    }

    pub fn start_with_vad(&mut self, vad_threshold: Option<f32>) {
        let vad_threshold = vad_threshold.unwrap_or(0.75);

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_sample_rate(cpal::SampleRate(16_000));

        let acc = Arc::clone(&self.accumulator);

        self.sample_rate = supported_config.sample_rate().0;
        self.channels = supported_config.channels() as u16;

        let acc_vad = Arc::clone(&self.accumulator);

        let sr = self.sample_rate;
        let (vad_thread_sender, vad_thread_receiver) = mpsc::channel::<i32>();

        // vad
        thread::spawn(move || {
            let mut vad = VoiceActivityDetector::builder()
                .sample_rate(sr)
                .chunk_size(512usize)
                .build()
                .expect("failed making vad");

            let mut vad_level = 0; // 0 = not started talking, 1 = talking, 2 = stopped talking
            // maybe?, 3 yeah definitely stopped talking

            loop {
                let l: usize;

                let data: Vec<f32> = {
                    let gaurd = acc_vad.lock().expect("unlock failed");
                    let len = gaurd.len();

                    l = len;

                    // only get the last 512 'cuz that's what this package supports
                    gaurd[len.saturating_sub(512)..].to_vec()
                };

                let probability = vad.predict(data);

                if probability > vad_threshold {
                    match vad_level {
                        0 => vad_level = 1,
                        2 => vad_level = 1,
                        _ => {}
                    }
                }

                if probability < vad_threshold {
                    match vad_level {
                        0 => {
                            // clear buffer before some past once detection starts to minimize data
                            // and only have the speech parts in the recording
                            let keep_past_seconds = 5;

                            let mut gaurd = acc_vad.lock().unwrap();
                            let len = gaurd.len();

                            if len > (sr as usize) * keep_past_seconds {
                                gaurd.drain(0..(len - (sr as usize) * keep_past_seconds));
                            }
                        }
                        1 => vad_level = 2,
                        2 => vad_level = 3,
                        _ => {}
                    }
                }

                if vad_level == 3 {
                    println!();
                    vad_thread_sender.send(1).expect("error sending vad stop");
                } else {
                    print!(
                        "\r\x1b[2Kprobability: {}, vad_level: {}, buffer: {}s",
                        probability,
                        vad_level,
                        l / (sr as usize)
                    );
                    io::stdout().flush().expect("error flushing log line");
                }

                thread::sleep(Duration::from_secs(1));
            }
        });

        let stream = device.build_input_stream(
            &supported_config.config(),
            move |data: &[f32], _| {
                if let Ok(mut gaurd) = acc.lock() {
                    gaurd.extend_from_slice(data);
                }
            },
            move |err| {
                eprintln!("an error occurred on the output audio stream: {}", err);
            },
            None,
        );

        match stream {
            Ok(value) => {
                let is_play = value.play();

                if is_play.is_err() {
                    eprintln!("error recording");
                }

                self.stream = Some(value);
            }

            Err(err) => {
                eprintln!("error making stream: {}", err);
            }
        }

        vad_thread_receiver.recv().unwrap();
        self.save();
    }

    pub fn start(&mut self) {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();

        let acc = Arc::clone(&self.accumulator);

        self.sample_rate = supported_config.sample_rate().0;
        self.channels = supported_config.channels() as u16;

        let stream = device.build_input_stream(
            &supported_config.config(),
            move |data: &[f32], _| {
                if let Ok(mut gaurd) = acc.lock() {
                    gaurd.extend_from_slice(data);
                }
            },
            move |err| {
                eprintln!("an error occurred on the output audio stream: {}", err);
            },
            None,
        );

        match stream {
            Ok(value) => {
                let is_play = value.play();

                if is_play.is_err() {
                    eprintln!("error recording");
                }

                self.stream = Some(value);
            }

            Err(err) => {
                eprintln!("error making stream: {}", err);
            }
        }
    }
}
