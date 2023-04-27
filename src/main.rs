use std::{io::{BufRead, BufReader}, process::{Command, Stdio}};

fn main() -> std::io::Result<()> {
    let mut cmd = Command::new("yt-dlp");
    cmd.arg("https://youtu.be/Xxg-9jN3_-0").args(["-f", "wa/*wa"]).args(["-o", "-"]);
    let mut child = cmd.stdout(Stdio::piped())
        .spawn()
        .expect("failed to execute yt-dlp");

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut size: usize = 0;

    loop {
        let buf = reader.fill_buf()?;

        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        size += len;
        reader.consume(len);
    }

    child.wait().expect("failed to wait on yt-dlp");
    println!("{}", size / 1024);
    Ok(())
}
