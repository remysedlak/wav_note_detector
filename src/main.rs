use hound;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::env;

// Custom Hann window function
fn hann(i: usize, size: usize) -> f32 {
    let pi = std::f32::consts::PI;
    (pi * i as f32 / (size as f32)).sin().powi(2)
}

fn freq_to_midi_note(freq: f32) -> Option<u8> {
    if freq <= 0.0 {
        return None;
    }
    let note_num = 69.0 + 12.0 * (freq / 440.0).log2();
    if note_num < 0.0 || note_num > 127.0 {
        None
    } else {
        Some(note_num.round() as u8)
    }
}

fn midi_note_to_name(note: u8) -> String {
    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let octave = (note / 12).saturating_sub(1);
    let note_index = (note % 12) as usize;
    format!("{}{}", note_names[note_index], octave)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input.wav>", args[0]);
        std::process::exit(1);
    }
    let filename = &args[1];

    let mut reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f32;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    let duration_seconds = samples.len() as f32 / sample_rate;
    println!("File: {}", filename);
    println!("Sample rate: {} Hz", sample_rate);
    println!("Duration: {:.2} seconds", duration_seconds);

    let fft_size = 4096;
    let hop_size = fft_size / 2;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let window: Vec<f32> = (0..fft_size).map(|i| hann(i, fft_size)).collect();
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];

    // Frequency count map: MIDI note -> count
    let mut note_counts: HashMap<u8, usize> = HashMap::new();

    println!("Processing...");

    for start in (0..samples.len().saturating_sub(fft_size)).step_by(hop_size) {
        for i in 0..fft_size {
            buffer[i].re = samples[start + i] * window[i];
            buffer[i].im = 0.0;
        }

        fft.process(&mut buffer);

        let mut max_mag = 0.0;
        let mut max_bin = 0;
        for i in 1..(fft_size / 2) {
            let mag = buffer[i].norm();
            if mag > max_mag {
                max_mag = mag;
                max_bin = i;
            }
        }

        let freq = max_bin as f32 * sample_rate / fft_size as f32;

        // Filter out very low frequencies and low magnitude noise
        if freq < 20.0 || max_mag < 0.01 {
            continue;
        }

        if let Some(midi_note) = freq_to_midi_note(freq) {
            // Count the note
            *note_counts.entry(midi_note).or_insert(0) += 1;
        }
    }

    // Sort notes by frequency descending
    let mut counts_vec: Vec<(u8, usize)> = note_counts.into_iter().collect();
    counts_vec.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nMost common notes detected:");
    for (note, count) in counts_vec.iter().take(10) {
        let name = midi_note_to_name(*note);
        println!("{}: {} occurrences", name, count);
    }

    Ok(())
}
