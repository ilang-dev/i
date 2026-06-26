use std::path::PathBuf;
use std::process::Command;

use compiler::ir::component::Component;
use compiler::ir::module::Module;

use super::{dylib_ext, load_program, unique_string, Device, Program};

pub fn compile(component: &Component) -> Result<Program, String> {
    let source = render(component)?;
    let dylib_path = build(&source)?;
    unsafe { load_program(dylib_path, Device::I_DEVICE_CPU) }
}

pub fn render(component: &Component) -> Result<String, String> {
    let module = lower_to_module(component)?;
    Ok(compiler::backends::c::render(&module))
}

fn lower_to_module(component: &Component) -> Result<Module, String> {
    let graph =
        compiler::lower::lower_component_to_graph(component).map_err(|err| format!("{err:?}"))?;
    let stages = compiler::lower::lower_node_graph_to_stage_program(&graph)
        .map_err(|err| format!("{err:?}"))?;
    let kernels = compiler::lower::lower_stage_program_to_kernel_program(&stages)
        .map_err(|err| format!("{err:?}"))?;
    let plan = compiler::lower::lower_kernel_program_to_exec_plan(&kernels)
        .map_err(|err| format!("{err:?}"))?;
    compiler::lower::lower_exec_plan_to_module(&plan).map_err(|err| format!("{err:?}"))
}

fn build(source: &str) -> Result<PathBuf, String> {
    let stem = format!("ilang_{}", unique_string());
    let source_path = std::env::temp_dir().join(format!("{stem}.c"));
    let dylib_path = std::env::temp_dir().join(format!("{stem}.{}", dylib_ext()));

    std::fs::write(&source_path, source).map_err(|err| err.to_string())?;
    let exit = Command::new("cc")
        .args(["-O3", "-shared", "-fPIC"])
        .arg(&source_path)
        .arg("-o")
        .arg(&dylib_path)
        .arg("-lm")
        .status()
        .map_err(|err| err.to_string())?;

    let _ = std::fs::remove_file(&source_path);
    if !exit.success() {
        return Err(format!("cc failed with status {exit}"));
    }

    Ok(dylib_path)
}
