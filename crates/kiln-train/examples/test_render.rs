// Debug: render and encode a prompt via the same path opd_train uses.
use kiln_core::tokenizer::{ChatMessage, ChatTemplateOptions, KilnTokenizer};
use serde_json::Value;

fn main() {
    let tok_path = "/workspace/kiln/Qwen3.5-4B/tokenizer.json";
    let tpl_path = "/workspace/kiln/Qwen3.5-4B/chat_template.jinja";
    let mut tok = KilnTokenizer::from_file(tok_path).unwrap();
    let tpl = std::fs::read_to_string(tpl_path).unwrap();
    tok = tok.with_chat_template(tpl);

    let msgs = vec![
        ChatMessage { role: "system".to_string(), content: "You are helpful.".to_string(), ..Default::default() },
        ChatMessage { role: "user".to_string(), content: "Say hi".to_string(), ..Default::default() },
    ];

    let mut kwargs = serde_json::Map::new();
    kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
    let opts = ChatTemplateOptions { template_kwargs: kwargs };

    let text = tok.apply_chat_template_full_with_options(msgs.as_slice(), None, None, opts).unwrap();
    println!("RENDERED TEXT:\n{}", text);
    println!("---END---\n");
    let ids = tok.encode(&text).unwrap();
    println!("TOKEN IDS ({}):", ids.len());
    for (i, id) in ids.iter().enumerate() {
        println!("  [{}] {}", i, id);
    }
}
