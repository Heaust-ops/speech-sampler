use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::StreamTrait;
use cpal::traits::{DeviceTrait, HostTrait};
use hound;
use voice_activity_detector::VoiceActivityDetector;

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
}

impl VAD {
    pub fn new() -> Self {
        let output_path = "speechSamplerOut.wav".to_string();
        let accumulator = Arc::new(Mutex::new(Vec::new()));

        VAD {
            output_path,
            accumulator,
            stream: None,
            sample_rate: 44_100,
            channels: 1,
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
    }

    pub fn start_with_vad(&mut self) {
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
        thread::spawn(move || {
            let mut vad = VoiceActivityDetector::builder()
                .sample_rate(sr)
                .chunk_size(512usize)
                .build()
                .expect("failed making vad");

            loop {
                let data: Vec<f32> = {
                    let gaurd = acc_vad.lock().expect("unlock failed");
                    let len = gaurd.len();
                    gaurd[len.saturating_sub(512)..].to_vec()
                };

                let l = data.len();
                let probability = vad.predict(data);

                println!("probability: {}, len: {}", probability, l);

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
