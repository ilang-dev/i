mod ast;
mod backend;
mod block;
mod graph;
mod lowerer;
mod parser;
mod tokenizer;

use backend::cuda::CudaBackend;

use crate::backend::rust::RustBackend;
use crate::backend::Render;
use crate::graph::Graph;
use crate::lowerer::Lowerer;
use crate::parser::Parser;

use std::io::Read;
use std::{env, fs, io, process::Command};

// Formats Rust code using rustfmt
fn format_rust_code(code: String) -> String {
    let path = "/tmp/tmp.rs";
    fs::write(&path, code).unwrap();
    Command::new("rustfmt").arg(&path).status().unwrap();
    fs::read_to_string(&path).unwrap()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments
    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut source = "i";
    let mut target = "rust";

    let mut iter = args.iter().skip(1); // Skip the program name
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-s" | "--source" => {
                source = iter.next().ok_or("Error: Missing value for --source")?;
            }
            "-t" | "--target" => {
                target = iter.next().ok_or("Error: Missing value for --target")?;
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other if input_path.is_none() => input_path = Some(other.to_string()),
            other if output_path.is_none() => output_path = Some(other.to_string()),
            _ => return Err("Error: Too many arguments".to_string()),
        }
    }

    // Validate the source type
    if !(source == "i" || source == "ir") {
        return Err(format!("Error: Unsupported source '{}'", source));
    }

    // Validate the target platform
    if !(target == "rust" || target == "ir" || target == "cuda") {
        return Err(format!("Error: Unsupported target '{}'", target));
    }

    // Read input
    let input = if let Some(path) = input_path {
        if path == "-" {
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .map_err(|e| format!("Failed to read from STDIN: {}", e))?;
            buffer
        } else {
            fs::read_to_string(path).map_err(|e| format!("Failed to read input file: {}", e))?
        }
    } else {
        return Err("Error: Missing input file".to_string());
    };

    // Process the input
    let block = match source {
        "i" => {
            let (_ast, expr_bank) = Parser::new(&input)?.parse().unwrap();
            let graph = Graph::from_expr_bank(&expr_bank);

            // get IndexExpr
            let crate::ast::Expr::Index(_) = expr_bank.0[0] else {
                panic!("expression is not of variant Index")
            };

            // lower
            Lowerer::new().lower(&graph)
        }
        "ir" => block::parser::parse(&input),
        &_ => unreachable!(),
    };

    let formatted_code = match target {
        "rust" => format_rust_code(RustBackend::render(&block)),
        "ir" => BlockBackend::render(&block),
        "cuda" => CudaBackend::render(&block),
        &_ => unreachable!(),
    };

    // Write output
    if let Some(path) = output_path {
        if path == "-" {
            println!("{}", formatted_code);
        } else {
            fs::write(path, formatted_code)
                .map_err(|e| format!("Failed to write output file: {}", e))?;
        }
    } else {
        println!("{}", formatted_code);
    }

    Ok(())
}

// Prints the help message
fn print_help() {
    println!(
        r#"Usage: ic [OPTIONS] [INPUT] [OUTPUT]

Options:
  -t, --target <TARGET>  Specify the target platform (default: rust)
  -h, --help             Print this help message

Arguments:
  INPUT                  Path to the input file (use '-' for STDIN)
  OUTPUT                 Path to the output file (use '-' for STDOUT)"#
    );
}
