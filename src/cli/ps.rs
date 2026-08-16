use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use hypura::server::ollama_types::PsResponse;

pub fn run(host: &str, port: u16) -> anyhow::Result<()> {
    let addr_str = format!("{host}:{port}");
    let addrs: Vec<SocketAddr> = match addr_str.to_socket_addrs() {
        Ok(a) => a.collect(),
        Err(e) => {
            eprintln!("Invalid address '{addr_str}': {e}");
            std::process::exit(1);
        }
    };

    let mut stream = None;
    for addr in addrs {
        if let Ok(s) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            stream = Some(s);
            break;
        }
    }

    let mut stream = match stream {
        Some(s) => s,
        None => {
            eprintln!();
            eprintln!("Error: Could not connect to Hypura server at http://{host}:{port}");
            eprintln!();
            eprintln!("To start the server, run:");
            eprintln!("  hypura serve --port {port}");
            eprintln!();
            std::process::exit(1);
        }
    };

    stream.set_read_timeout(Some(Duration::from_secs(4)))?;
    stream.set_write_timeout(Some(Duration::from_secs(4)))?;

    let req = format!(
        "GET /api/ps HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: hypura-cli\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes())?;

    let mut response_bytes = Vec::new();
    let _ = stream.read_to_end(&mut response_bytes);

    let response_str = String::from_utf8_lossy(&response_bytes);
    let (headers, body) = match response_str.split_once("\r\n\r\n") {
        Some((h, b)) => (h, b),
        None => ("", response_str.as_ref()),
    };

    let first_line = headers.lines().next().unwrap_or("");
    if !first_line.contains("200") {
        eprintln!();
        eprintln!("Server responded with: {first_line}");
        eprintln!("If you recently updated Hypura, restart the server to enable /api/ps.");
        eprintln!();
        std::process::exit(1);
    }

    let ps: PsResponse = match serde_json::from_str(body.trim()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error parsing server response: {e}");
            std::process::exit(1);
        }
    };

    println!();
    println!("Hypura Active Processes (http://{host}:{port})");
    println!("─────────────────────────────────────────────────────────────────────────────");

    if ps.models.is_empty() {
        println!("  No models currently loaded in memory (daemon idle).");
        println!();
        println!("  Models are loaded on-demand when client requests arrive.");
        println!("  Run 'hypura list' to view available models.");
        println!();
        return Ok(());
    }

    println!(
        "  {:<32} {:<16} {:<10} {:<10} {:<8} {:<8}",
        "NAME", "PROCESSOR", "SIZE", "CONTEXT", "PARAMS", "QUANT"
    );
    println!("  {}", "─".repeat(88));

    for model in ps.models {
        let size_str = format_bytes(model.size);
        println!(
            "  {:<32} {:<16} {:<10} {:<10} {:<8} {:<8}",
            model.name,
            "100% Metal/GPU",
            size_str,
            format!("{} tokens", model.context_size),
            model.details.parameter_size,
            model.details.quantization_level
        );
    }

    println!();
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MB", bytes as f64 / (1 << 20) as f64)
    } else {
        format!("{bytes} B")
    }
}
