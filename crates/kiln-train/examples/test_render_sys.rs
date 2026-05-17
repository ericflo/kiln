use kiln_core::tokenizer::{ChatMessage, ChatTemplateOptions, KilnTokenizer};
use serde_json::Value;
fn main() {
    let tok_path = "/workspace/kiln/Qwen3.5-4B/tokenizer.json";
    let tpl_path = "/workspace/kiln/Qwen3.5-4B/chat_template.jinja";
    let mut tok = KilnTokenizer::from_file(tok_path).unwrap();
    let tpl = std::fs::read_to_string(tpl_path).unwrap();
    tok = tok.with_chat_template(tpl);
    let combined = vec![
        ChatMessage { role: "system".to_string(), content: "Few-shot exemplars.".to_string(), ..Default::default() },
        ChatMessage { role: "system".to_string(), content: "You emit JSON.".to_string(), ..Default::default() },
        ChatMessage { role: "user".to_string(), content: "Read x.rs".to_string(), ..Default::default() },
    ];
    let user_only = vec![
        ChatMessage { role: "system".to_string(), content: "You emit JSON.".to_string(), ..Default::default() },
        ChatMessage { role: "user".to_string(), content: "Read x.rs".to_string(), ..Default::default() },
    ];
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
    let opts = ChatTemplateOptions { template_kwargs: kwargs.clone() };
    
    println!("=== combined (extras + user_only) ===");
    let t2 = tok.apply_chat_template_full_with_options(combined.as_slice(), None, None, opts.clone()).unwrap();
    let i2 = tok.encode(&t2).unwrap();
    println!("text:\n{}", t2);
    println!("\nntoks: {}\n", i2.len());
    
    println!("=== user_only ===");
    let t3 = tok.apply_chat_template_full_with_options(user_only.as_slice(), None, None, opts).unwrap();
    let i3 = tok.encode(&t3).unwrap();
    println!("text:\n{}", t3);
    println!("\nntoks: {}\n", i3.len());
    
    println!("=== analysis ===");
    println!("len(combined) - len(user_only) = {}", i2.len() - i3.len());
    println!("Is user_only a SUFFIX of combined? {}", &i2[i2.len() - i3.len()..] == i3.as_slice());
}
