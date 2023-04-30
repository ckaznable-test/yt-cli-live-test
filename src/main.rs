
use mpeg2ts::{ts::{TsPacketReader, ReadTsPacket, TsPayload}, es::StreamType};
use symphonia::core::{io::MediaSourceStream, probe::Hint, meta::MetadataOptions, formats::FormatOptions, codecs::{DecoderOptions, CODEC_TYPE_NULL}, errors::Error, audio::AudioBuffer};
use std::{io::{BufRead, BufReader, Cursor}, process::{Command, Stdio}, time::{Instant, Duration}};
use wait_timeout::ChildExt;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy, WhisperState};

fn main() -> std::io::Result<()> {
    run_yt().unwrap();
    Ok(())
}

fn run_yt() -> std::io::Result<()> {
    let mut cmd = Command::new("yt-dlp");
    const URL: &str = "https://www.youtube.com/watch?v=_WtAKwsdxaY";
    cmd.arg(URL)
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
    let ctx = WhisperContext::new("./ggml-tiny.bin").expect("failed to load model");
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
            process(&ctx, &state, &buffer).expect("failed to process");
            last_processed = Instant::now();
            buffer.clear();
        }

        reader.consume(len);
    }

    if !buffer.is_empty() {
        process(&ctx, &state, &buffer).expect("failed to process");
    }

    child.wait_timeout(Duration::from_secs(3)).expect("failed to wait on yt-dlp");
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

fn get_ts_audio(raw: &[u8]) -> Vec<u8> {
    let cursor = Cursor::new(raw);
    let mut reader = TsPacketReader::new(cursor);

    let mut data: Vec<u8> = vec![];
    let mut audio_pid: u16 = 0;

    while let Ok(Some(packet)) = reader.read_ts_packet() {
        use TsPayload::*;

        let pid = packet.header.pid.as_u16();
        let is_audio_pid = pid == audio_pid;

        if let Some(payload) = packet.payload {
            match payload {
                Pmt(pmt) => {
                    if let Some(el) = pmt.table.iter().find(|el| el.stream_type == StreamType::AdtsAac) {
                        audio_pid = el.elementary_pid.as_u16();
                    }
                }
                Pes(pes) => {
                    if pes.header.stream_id.is_audio() && is_audio_pid {
                        data.extend_from_slice(&pes.data);
                    }
                }
                Raw(bytes) => {
                    if is_audio_pid {
                        data.extend_from_slice(&bytes);
                    }
                },
                _ => (),
            }
        }
    }

    data
}

fn get_mono_f32(raw: Vec<u8>) -> Vec<f32> {
    let src = Cursor::new(raw);
    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    // Create a probe hint using the file's extension. [Optional]
    let mut hint = Hint::new();
    hint.with_extension("aac");

    // Use the default options for metadata and format readers.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    let dec_opts: DecoderOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .expect("unsupported format");

    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format.tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut data: Vec<f32> = vec![];

    // The decode loop.
    loop {
        // Get the next packet from the media format.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::ResetRequired) => {
                unimplemented!();
            }
            Err(_) => {
                break;
            }
        };

        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            // Pop the old head of the metadata queue.
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let mut buf = AudioBuffer::<f32>::new(packet.dur, *audio_buf.spec());
                audio_buf.convert(&mut buf);

                let planes = buf.planes();
                let planes = planes.planes();
                data.extend_from_slice(planes[0]);
            }
            Err(Error::DecodeError(_)) => (),
            _ => {
                break;
            }
        }
    }

    data
}

fn process(ctx: &WhisperContext, state: &WhisperState, audio_data: &[u8]) -> Result<(), &'static str> {
    let params = get_params();
    let audio_data = get_mono_f32(
        get_ts_audio(audio_data)
    );

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
