pub(super) fn indent(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("  {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn comma_lines(items: impl Iterator<Item = String>) -> String {
    items.collect::<Vec<_>>().join(",\n")
}

pub(super) fn initializer(prefix: &str, items: impl Iterator<Item = String>) -> String {
    format!("{prefix}{{\n{}\n}}", indent(&comma_lines(items)))
}
