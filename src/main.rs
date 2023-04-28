extern crate ffmpeg_next as ffmpeg;

use byteorder::{LittleEndian, ByteOrder};
use ffmpeg::{ChannelLayout, decoder, codec, Packet};
use ffmpeg::format::Sample;
use ffmpeg::frame::Audio;
use ffmpeg::software::resampling::{context::Context as ResampleContext};

use std::{io::{BufRead, BufReader}, process::{Command, Stdio}, time::{Instant, Duration}};
use wait_timeout::ChildExt;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy, WhisperState};

fn main() -> std::io::Result<()> {
    run_yt().unwrap();
    Ok(())
}

fn run_yt() -> std::io::Result<()> {
    ffmpeg::init().unwrap();

    let mut cmd = Command::new("yt-dlp");
    cmd.arg("https://www.youtube.com/watch?v=RXJjd1KIC7k")
        .args(["-f", "w"])
        .args(["--quiet"])
        .args(["-o", "-"]);

    let mut child = cmd.stdout(Stdio::piped())
        .spawn()
        .expect("failed to execute yt-dlp");

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut buffer: Vec<u8> = vec![];

    let mut last_processed = Instant::now();
    let target_time = Duration::from_secs(5);

    // load a context and model
    let ctx = WhisperContext::new("./ggml-base.bin").expect("failed to load model");
    // make a state
    let state = ctx.create_state().expect("failed to create state");

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        buffer.extend_from_slice(buf);

        if last_processed.elapsed() >= target_time {
            let audio_data = buffer.to_vec();
            buffer.clear();
            process(&ctx, &state, audio_data).expect("failed to process");
            last_processed = Instant::now();
        }

        reader.consume(len);
    }

    if !buffer.is_empty() {
        process(&ctx, &state, buffer.to_vec()).expect("failed to process");
    }

    child.wait_timeout(Duration::from_secs(5)).expect("failed to wait on yt-dlp");
    Ok(())
}


fn get_params<'a, 'b>() -> FullParams<'a, 'b> {
    // create a params object
    // note that currently the only implemented strategy is Greedy, BeamSearch is a WIP
    // n_past defaults to 0
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // edit things as needed
    // here we set the number of threads to use to 1
    params.set_n_threads(1);
    // we also enable translation
    // params.set_translate(true);
    // and set the language to translate to to japanese
    params.set_language(Some("ja"));
    // we also explicitly disable anything that prints to stdout
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params
}

fn get_audio_from_ffmpeg(raw: &[u8]) -> Option<Vec<f32>> {
    let mut decoder = decoder::new()
        .open_as(decoder::find(codec::Id::H264))
        .unwrap()
        .audio()
        .unwrap();

    let mut resample_context = ResampleContext::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        Sample::F32(ffmpeg::format::sample::Type::Packed),
        ChannelLayout::MONO,
        decoder.rate(),
    ).unwrap();

    let packet = Packet::copy(raw);
    decoder.send_packet(&packet).unwrap();

    let mut audio = Audio::empty();
    let mut audio_converted = Audio::empty();
    if decoder.receive_frame(&mut audio).is_ok() {
        resample_context.run(&audio, &mut audio_converted).unwrap();

        let mut data: Vec<u8> = vec![];
        for i in 0..audio_converted.samples() {
            data.extend_from_slice(audio_converted.data(i));
        }

        Some(convert_u8_to_f32(&data))
    } else {
        None
    }
}

fn convert_u8_to_f32(input: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0; input.len() / 4];
    for (i, chunk) in input.chunks(4).enumerate() {
        output[i] = LittleEndian::read_f32(chunk);
    }
    output
}

fn process(ctx: &WhisperContext, state: &WhisperState, audio_data: Vec<u8>) -> Result<(), &'static str> {
    let params = get_params();
    let audio_data = get_audio_from_ffmpeg(&audio_data).unwrap();

    // now we can run the model
    // note the key we use here is the one we created above
    ctx.full(state, params, &audio_data[..])
        .expect("failed to run model");

    // fetch the results
    let num_segments = ctx
        .full_n_segments(state)
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = ctx
            .full_get_segment_text(state, i)
            .expect("failed to get segment");
        let start_timestamp = ctx
            .full_get_segment_t0(state, i)
            .expect("failed to get segment start timestamp");
        let end_timestamp = ctx
            .full_get_segment_t1(state, i)
            .expect("failed to get segment end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }

    Ok(())
}
