
use mpeg2ts::{ts::{TsPacketReader, ReadTsPacket, TsPayload}, es::StreamType};
use std::{io::{BufRead, BufReader, Cursor, Write}, process::{Command, Stdio}, time::{Instant, Duration}, fs::File};
use wait_timeout::ChildExt;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy, WhisperState};

fn main() -> std::io::Result<()> {
    run_yt().unwrap();
    Ok(())
}

fn run_yt() -> std::io::Result<()> {
    let mut cmd = Command::new("yt-dlp");
    const URL: &str = "https://www.youtube.com/watch?v=-5OCkK_yIDc";
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
    let target_time = Duration::from_secs(10);

    // load a context and model
    // let ctx = WhisperContext::new("./ggml-tiny.bin").expect("failed to load model");
    // make a state
    // let state = ctx.create_state().expect("failed to create state");

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        buffer.extend_from_slice(buf);

        if last_processed.elapsed() >= target_time {
            get_ts_audio(&buffer);
            // process(&ctx, &state, audio_data).expect("failed to process");
            last_processed = Instant::now();
            buffer.clear();
        }

        reader.consume(len);
    }

    if !buffer.is_empty() {
        // process(&ctx, &state, buffer.to_vec()).expect("failed to process");
        get_ts_audio(&buffer);
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

    // let mut file = File::create("example.aac").unwrap(); // 創建檔案
    // file.write_all(&data).unwrap();
    println!("done");
    data
}

fn get_mono_f32(raw: &[u8]) -> Vec<f32> {
    vec![]
}

fn process(ctx: &WhisperContext, state: &WhisperState, audio_data: Vec<u8>) -> Result<(), &'static str> {
    let params = get_params();
    let audio_data = get_mono_f32(&audio_data);

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
