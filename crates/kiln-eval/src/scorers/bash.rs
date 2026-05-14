//! Bash-call introspection.
//!
//! Tool calls that invoke `bash` or `shell` almost universally pack their
//! actual semantics into a single command string. This module pries that
//! string apart so downstream scorers can compare the *meaningful* parts
//! instead of the raw concatenated command.
//!
//! Two layers:
//!
//! 1. `tokenize` — a small POSIX-ish lexer that handles single quotes,
//!    double quotes, and backslash escapes. Not a full shell parser
//!    (no $vars, no $(subshells)), but enough to split `python3 -c 'print(1)'`
//!    into the right pieces. Pipes, `&&`, and `;` are surfaced as separate
//!    `Token::Op` entries so callers can recognize the first command.
//! 2. `introspect` — runs the tokenizer over a candidate string and
//!    classifies the leading command into a `BashIntrospection` (program
//!    name, optional inline-language script, remaining args).
//!
//! The classifier handles the languages we see most in agent trajectories:
//! python (`python -c`, `python3 -c`, `uv run python -c`), node (`node -e`),
//! bash (`bash -c`), zsh (`zsh -c`), ruby (`ruby -e`), perl (`perl -e`),
//! plus generic `<lang> script.<ext>` runs that extract the script path
//! and language inference from the suffix.

use serde::{Deserialize, Serialize};

/// A single shell token. Operators (`|`, `&&`, `;`, `||`) are surfaced
/// separately so callers can pluck the leading command without losing
/// context.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Word(String),
    Op(&'static str),
}

/// POSIX-flavored tokenizer. Handles:
///
/// - Single quotes (verbatim until the next single quote)
/// - Double quotes (`\\\"` → `"`, `\\\\` → `\\`, everything else verbatim)
/// - Backslash escapes outside quotes
/// - Whitespace as separator
/// - `|`, `&&`, `||`, `;` as operator tokens
///
/// Unbalanced quotes return a partial parse (everything up to the EOF)
/// so we never fail at the call site.
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut iter = input.chars().peekable();
    let flush = |buf: &mut String, out: &mut Vec<Token>| {
        if !buf.is_empty() {
            out.push(Token::Word(std::mem::take(buf)));
        }
    };
    while let Some(c) = iter.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                buf.push(c);
            }
            continue;
        }
        if in_double {
            if c == '"' {
                in_double = false;
            } else if c == '\\' {
                if let Some(&next) = iter.peek() {
                    if matches!(next, '"' | '\\' | '$' | '`' | '\n') {
                        iter.next();
                        buf.push(next);
                        continue;
                    }
                }
                buf.push(c);
            } else {
                buf.push(c);
            }
            continue;
        }
        match c {
            '\'' => in_single = true,
            '"' => in_double = true,
            '\\' => {
                if let Some(next) = iter.next() {
                    buf.push(next);
                }
            }
            ' ' | '\t' | '\n' | '\r' => {
                flush(&mut buf, &mut out);
            }
            '|' => {
                flush(&mut buf, &mut out);
                if matches!(iter.peek(), Some(&'|')) {
                    iter.next();
                    out.push(Token::Op("||"));
                } else {
                    out.push(Token::Op("|"));
                }
            }
            '&' => {
                flush(&mut buf, &mut out);
                if matches!(iter.peek(), Some(&'&')) {
                    iter.next();
                    out.push(Token::Op("&&"));
                } else {
                    out.push(Token::Op("&"));
                }
            }
            ';' => {
                flush(&mut buf, &mut out);
                out.push(Token::Op(";"));
            }
            _ => buf.push(c),
        }
    }
    if !buf.is_empty() {
        out.push(Token::Word(buf));
    }
    out
}

/// Structured view of a bash invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BashIntrospection {
    /// First command word (`python3`, `cargo`, `bash`, …). Empty when the
    /// command string was unparseable.
    pub program: String,
    /// When the leading command wraps an inline program (`python -c`,
    /// `node -e`, etc.), the source goes here and `inline_language`
    /// names the language for the downstream code scorer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_code: Option<String>,
    /// When the leading command runs a script (e.g. `python script.py`)
    /// and we recognize the language from the extension, the language
    /// goes here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_language: Option<String>,
    /// Path of the script being run, when the introspector found one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_path: Option<String>,
    /// Argument words following the leading program (excluding `-c`-style
    /// flags that introduce inline programs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tail: Vec<String>,
    /// True when the command contains pipes / `&&` / `;` — useful for
    /// scorers that want to weight "trivial single-call" vs "compound
    /// pipeline" matches.
    #[serde(default)]
    pub is_pipeline: bool,
}

impl BashIntrospection {
    /// Identifier used by scoring / stats to group similar invocations.
    /// `python_inline`, `python_script`, `node_inline`, `bash`,
    /// `cargo`, `pip_install`, etc. Falls back to the program name.
    pub fn classification(&self) -> String {
        if let Some(lang) = self.inline_language.as_deref() {
            return format!("{lang}_inline");
        }
        if let Some(lang) = self.script_language.as_deref() {
            return format!("{lang}_script");
        }
        if self.program == "pip" && self.tail.first().map(|s| s.as_str()) == Some("install") {
            return "pip_install".to_string();
        }
        if self.program == "git" {
            return self
                .tail
                .first()
                .map(|t| format!("git_{t}"))
                .unwrap_or_else(|| "git".to_string());
        }
        if self.program.is_empty() {
            "unknown".into()
        } else {
            self.program.clone()
        }
    }
}

/// Introspect a bash command string.
pub fn introspect(command: &str) -> BashIntrospection {
    let tokens = tokenize(command);
    if tokens.is_empty() {
        return BashIntrospection::default();
    }
    // Take everything up to the first operator as the "leading command".
    let leading: Vec<&Token> = tokens
        .iter()
        .take_while(|t| !matches!(t, Token::Op(_)))
        .collect();
    let is_pipeline = tokens.iter().any(|t| matches!(t, Token::Op(_)));
    if leading.is_empty() {
        let mut out = BashIntrospection::default();
        out.is_pipeline = is_pipeline;
        return out;
    }
    let words: Vec<&str> = leading
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.as_str()),
            _ => None,
        })
        .collect();

    // Strip leading env assignments (`FOO=bar baz arg`) so `program` lands
    // on the real binary.
    let mut idx = 0;
    while idx < words.len() && words[idx].contains('=') && !words[idx].starts_with('-') {
        idx += 1;
    }
    if idx >= words.len() {
        return BashIntrospection {
            program: String::new(),
            is_pipeline,
            ..Default::default()
        };
    }
    let program = words[idx].to_string();
    let rest = &words[idx + 1..];

    // Handle `uv run python ...` / `uv run python3 ...` by recursing into
    // the inner command.
    if (program == "uv" || program == "uvx") && rest.first().map(|s| *s) == Some("run") {
        let inner: String = rest[1..].join(" ");
        let mut inner_intro = introspect(&inner);
        inner_intro.is_pipeline = inner_intro.is_pipeline || is_pipeline;
        return inner_intro;
    }

    let inline_language = inline_language_for(&program, rest);
    let (inline_code, tail) = if let Some(lang) = inline_language.as_deref() {
        extract_inline_program(rest, lang)
    } else {
        (None, rest.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    };

    let (script_language, script_path) = if inline_code.is_none() {
        script_info(&program, rest)
    } else {
        (None, None)
    };

    BashIntrospection {
        program,
        inline_language,
        inline_code,
        script_language,
        script_path,
        tail,
        is_pipeline,
    }
}

fn inline_language_for(program: &str, rest: &[&str]) -> Option<String> {
    let flag = match program {
        "python" | "python3" | "python3.11" | "python3.12" | "python3.13" => "-c",
        "node" | "nodejs" => "-e",
        "ruby" => "-e",
        "perl" => "-e",
        "bash" | "sh" | "zsh" | "dash" => "-c",
        "php" => "-r",
        _ => return None,
    };
    if rest.iter().any(|t| *t == flag) {
        let lang = match program {
            "python" | "python3" | "python3.11" | "python3.12" | "python3.13" => "python",
            "node" | "nodejs" => "node",
            "ruby" => "ruby",
            "perl" => "perl",
            "bash" | "sh" | "zsh" | "dash" => "bash",
            "php" => "php",
            _ => program,
        };
        Some(lang.to_string())
    } else {
        None
    }
}

fn extract_inline_program(rest: &[&str], lang: &str) -> (Option<String>, Vec<String>) {
    let flag = match lang {
        "python" | "ruby" | "perl" => match lang {
            "python" => "-c",
            _ => "-e",
        },
        "node" => "-e",
        "bash" => "-c",
        "php" => "-r",
        _ => "-c",
    };
    let mut iter = rest.iter().enumerate();
    while let Some((i, tok)) = iter.next() {
        if *tok == flag && i + 1 < rest.len() {
            let code = rest[i + 1].to_string();
            let mut tail: Vec<String> = Vec::new();
            for (j, t) in rest.iter().enumerate() {
                if j == i || j == i + 1 {
                    continue;
                }
                tail.push(t.to_string());
            }
            return (Some(code), tail);
        }
    }
    (None, rest.iter().map(|s| s.to_string()).collect())
}

fn script_info(program: &str, rest: &[&str]) -> (Option<String>, Option<String>) {
    let lang = match program {
        "python" | "python3" | "python3.11" | "python3.12" | "python3.13" => Some("python"),
        "node" | "nodejs" => Some("node"),
        "ruby" => Some("ruby"),
        "go" => Some("go"),
        "java" => Some("java"),
        _ => None,
    };
    if let Some(lang) = lang {
        for tok in rest {
            if !tok.starts_with('-') && !tok.contains('=') {
                return (Some(lang.to_string()), Some(tok.to_string()));
            }
        }
    }
    // Direct script run via shebang (./run.sh, ./build.py).
    if program.starts_with("./") || program.contains('/') {
        let suffix = program.rsplit('.').next().unwrap_or("");
        let lang = match suffix {
            "py" => Some("python"),
            "rs" => Some("rust"),
            "ts" | "js" | "mjs" | "cjs" => Some("node"),
            "rb" => Some("ruby"),
            "sh" | "bash" => Some("bash"),
            _ => None,
        };
        return (lang.map(str::to_string), Some(program.to_string()));
    }
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        let toks = tokenize("ls -la /tmp");
        assert_eq!(
            toks,
            vec![
                Token::Word("ls".into()),
                Token::Word("-la".into()),
                Token::Word("/tmp".into()),
            ]
        );
    }

    #[test]
    fn tokenize_handles_quotes() {
        let toks = tokenize(r#"python3 -c 'print("hello world")'"#);
        assert_eq!(
            toks,
            vec![
                Token::Word("python3".into()),
                Token::Word("-c".into()),
                Token::Word("print(\"hello world\")".into()),
            ]
        );
    }

    #[test]
    fn tokenize_recognizes_operators() {
        let toks = tokenize("cat foo | grep bar && echo ok");
        assert!(toks.iter().any(|t| matches!(t, Token::Op("|"))));
        assert!(toks.iter().any(|t| matches!(t, Token::Op("&&"))));
    }

    #[test]
    fn introspect_recognizes_python_inline() {
        let intro = introspect("python3 -c 'import os; print(os.getcwd())'");
        assert_eq!(intro.program, "python3");
        assert_eq!(intro.inline_language.as_deref(), Some("python"));
        assert!(intro.inline_code.as_deref().unwrap().contains("os.getcwd"));
        assert_eq!(intro.classification(), "python_inline");
    }

    #[test]
    fn introspect_recognizes_uv_run_python() {
        let intro = introspect("uv run python -c 'print(1)'");
        assert_eq!(intro.program, "python");
        assert_eq!(intro.inline_language.as_deref(), Some("python"));
        assert_eq!(intro.classification(), "python_inline");
    }

    #[test]
    fn introspect_recognizes_node_inline() {
        let intro = introspect(r#"node -e "console.log(42)""#);
        assert_eq!(intro.program, "node");
        assert_eq!(intro.inline_language.as_deref(), Some("node"));
        assert!(intro.inline_code.as_deref().unwrap().contains("42"));
    }

    #[test]
    fn introspect_recognizes_bash_inline() {
        let intro = introspect(r#"bash -c "echo hi; ls""#);
        assert_eq!(intro.inline_language.as_deref(), Some("bash"));
    }

    #[test]
    fn introspect_recognizes_python_script() {
        let intro = introspect("python3 scripts/run.py --flag");
        assert_eq!(intro.script_language.as_deref(), Some("python"));
        assert_eq!(intro.script_path.as_deref(), Some("scripts/run.py"));
        assert_eq!(intro.classification(), "python_script");
    }

    #[test]
    fn introspect_marks_pipelines() {
        let intro = introspect("ls -la | grep foo");
        assert!(intro.is_pipeline);
        assert_eq!(intro.program, "ls");
    }

    #[test]
    fn introspect_strips_env_prefix() {
        let intro = introspect("FOO=bar PYTHONPATH=. python3 main.py");
        assert_eq!(intro.program, "python3");
        assert_eq!(intro.script_path.as_deref(), Some("main.py"));
    }

    #[test]
    fn introspect_handles_pip_install() {
        let intro = introspect("pip install requests==2.31");
        assert_eq!(intro.classification(), "pip_install");
    }

    #[test]
    fn introspect_handles_git_subcommand() {
        let intro = introspect("git commit -m 'wip'");
        assert_eq!(intro.classification(), "git_commit");
    }

    #[test]
    fn introspect_empty_string_is_graceful() {
        let intro = introspect("");
        assert_eq!(intro.program, "");
        assert!(intro.inline_language.is_none());
    }

    #[test]
    fn introspect_unbalanced_quote_falls_through() {
        let intro = introspect(r#"python -c 'unbalanced"#);
        assert_eq!(intro.program, "python");
        // We still extract a partial inline code via the unclosed single quote.
        assert!(intro.inline_code.is_some());
    }
}
