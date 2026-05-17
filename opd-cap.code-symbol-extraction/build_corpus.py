"""Generate prompt corpus for code-symbol-extraction.

Bigger snippets (30–80 lines, 3–10 top-level defined symbols) than the
faithful-summarization corpus — designed so the 4B baseline can drift
on both recall (miss less-common symbol kinds) and precision (list
variables / imports as symbols).

Produces:
  datasets/eval.jsonl       — blind eval set (DO NOT read after generation)
  datasets/train.opd.jsonl  — training prompts (visible)

Each line:
  {
    "id": str,
    "code": str,
    "ground_truth": ["symbol_a", "symbol_b", ...],
    "messages": [system, user, dummy_assistant],
  }
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

SYSTEM_PROMPT = (
    "You are a precise code analyzer. When given a code snippet, list every "
    "top-level DEFINED symbol — functions, classes, structs, enums, traits, "
    "type aliases — ONE PER LINE, nothing else. No prose, no markdown bullets, "
    "no commentary. Just the names."
)


def _prompt_for(code: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"List the top-level defined symbols in this code:\n\n```\n{code}```"},
        {"role": "assistant", "content": ""},
    ]


# ---------------------------------------------------------------------------
# Python snippets (multi-symbol)
# ---------------------------------------------------------------------------

def py_service(rng: random.Random) -> tuple[str, list[str]]:
    svc = rng.choice(["UserService", "OrderService", "EmailService", "PaymentService"])
    repo = svc.replace("Service", "Repository")
    code = f"""from dataclasses import dataclass
import functools


@dataclass
class {svc}Config:
    timeout: int = 30
    retries: int = 3


class {repo}:
    def __init__(self, db):
        self.db = db

    def find(self, id):
        return self.db.query(id)

    def save(self, item):
        return self.db.upsert(item)


class {svc}:
    def __init__(self, repo, config):
        self.repo = repo
        self.config = config

    def process(self, id):
        item = self.repo.find(id)
        return self._transform(item)

    def _transform(self, item):
        return item


def make_{svc.lower()}(db):
    return {svc}({repo}(db), {svc}Config())
"""
    return code, [f"{svc}Config", repo, svc, f"make_{svc.lower()}"]


def py_module(rng: random.Random) -> tuple[str, list[str]]:
    code = """from typing import Optional, Iterable
import json


CACHE_KEY_PREFIX = "v2:"


def normalize_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def parse_int_safe(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def chunk(iterable: Iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


class ValidationError(Exception):
    pass


class ParseError(ValidationError):
    pass


def validate_payload(payload: dict) -> None:
    if "id" not in payload:
        raise ValidationError("missing id")
"""
    return code, ["normalize_key", "parse_int_safe", "chunk", "ValidationError",
                  "ParseError", "validate_payload"]


def py_async_set(rng: random.Random) -> tuple[str, list[str]]:
    code = """import asyncio
import aiohttp
from contextlib import asynccontextmanager


class Connection:
    def __init__(self, url):
        self.url = url

    async def open(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        await self.session.close()


@asynccontextmanager
async def connection_ctx(url):
    conn = Connection(url)
    await conn.open()
    try:
        yield conn
    finally:
        await conn.close()


async def fetch_json(url):
    async with connection_ctx(url) as conn:
        async with conn.session.get(url) as resp:
            return await resp.json()


async def fetch_many(urls):
    return await asyncio.gather(*(fetch_json(u) for u in urls))
"""
    return code, ["Connection", "connection_ctx", "fetch_json", "fetch_many"]


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

def rust_module(rng: random.Random) -> tuple[str, list[str]]:
    code = """use std::collections::HashMap;
use std::io::{self, Read};


pub trait Parser {
    fn parse(&self, input: &str) -> Result<Token, ParseError>;
}


#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
}


#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Identifier,
    Number,
    String,
    Symbol,
}


#[derive(Debug)]
pub enum ParseError {
    Unexpected(char),
    Eof,
    InvalidNumber(String),
}


pub struct JsonParser {
    pos: usize,
}


impl Parser for JsonParser {
    fn parse(&self, input: &str) -> Result<Token, ParseError> {
        Ok(Token { kind: TokenKind::String, text: input.into() })
    }
}


pub fn tokenize(input: &str) -> Vec<Token> {
    vec![]
}


pub type ParserMap = HashMap<String, Box<dyn Parser>>;
"""
    return code, ["Parser", "Token", "TokenKind", "ParseError", "JsonParser",
                  "tokenize", "ParserMap"]


def rust_data_layer(rng: random.Random) -> tuple[str, list[str]]:
    code = """use std::sync::Arc;
use tokio::sync::RwLock;


#[derive(Debug, Clone)]
pub struct Record {
    pub id: u64,
    pub data: Vec<u8>,
}


#[derive(Default)]
pub struct Storage {
    inner: Arc<RwLock<Vec<Record>>>,
}


impl Storage {
    pub fn new() -> Self {
        Storage::default()
    }

    pub async fn put(&self, r: Record) -> u64 {
        let mut g = self.inner.write().await;
        g.push(r);
        g.len() as u64
    }

    pub async fn get(&self, id: u64) -> Option<Record> {
        let g = self.inner.read().await;
        g.iter().find(|r| r.id == id).cloned()
    }
}


pub enum StorageError {
    NotFound,
    Conflict,
    IoError(std::io::Error),
}


pub trait Backed {
    fn flush(&self) -> Result<(), StorageError>;
}
"""
    return code, ["Record", "Storage", "StorageError", "Backed"]


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def go_module(rng: random.Random) -> tuple[str, list[str]]:
    code = """package server

import (
    "context"
    "fmt"
    "net/http"
)


type Server struct {
    addr string
    mux  *http.ServeMux
}


type Handler interface {
    HandleRequest(ctx context.Context, req *Request) (*Response, error)
}


type Request struct {
    Path string
    Body []byte
}


type Response struct {
    Status int
    Body   []byte
}


func NewServer(addr string) *Server {
    return &Server{addr: addr, mux: http.NewServeMux()}
}


func (s *Server) Register(path string, h Handler) {
    // ...
}


func (s *Server) ListenAndServe() error {
    return http.ListenAndServe(s.addr, s.mux)
}


func DefaultHandler(req *Request) *Response {
    return &Response{Status: 200, Body: []byte("ok")}
}
"""
    return code, ["Server", "Handler", "Request", "Response", "NewServer",
                  "DefaultHandler"]


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

def ts_module(rng: random.Random) -> tuple[str, list[str]]:
    code = """import { EventEmitter } from "events";


export interface SessionOptions {
    timeout: number;
    retries: number;
}


export class Session extends EventEmitter {
    private opts: SessionOptions;
    constructor(opts: SessionOptions) {
        super();
        this.opts = opts;
    }

    async open(): Promise<void> {
        this.emit("open");
    }

    async close(): Promise<void> {
        this.emit("close");
    }
}


export type SessionFactory = (opts: SessionOptions) => Session;


export const makeSession: SessionFactory = (opts) => new Session(opts);


export function withRetry<T>(fn: () => Promise<T>, retries: number): Promise<T> {
    return fn();
}


export class TimeoutError extends Error {}
"""
    return code, ["SessionOptions", "Session", "SessionFactory", "makeSession",
                  "withRetry", "TimeoutError"]


def js_class_set(rng: random.Random) -> tuple[str, list[str]]:
    code = """class EventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, fn) {
        if (!this.listeners.has(event)) this.listeners.set(event, []);
        this.listeners.get(event).push(fn);
    }

    emit(event, ...args) {
        const fns = this.listeners.get(event) || [];
        for (const fn of fns) fn(...args);
    }
}


class Logger {
    constructor(prefix) {
        this.prefix = prefix;
    }

    log(msg) {
        console.log(`${this.prefix}: ${msg}`);
    }
}


function createBus() {
    return new EventBus();
}


const defaultLogger = new Logger("app");


function logEvent(event, payload) {
    defaultLogger.log(`${event}: ${JSON.stringify(payload)}`);
}
"""
    return code, ["EventBus", "Logger", "createBus", "logEvent"]


GENERATORS: list[Callable[[random.Random], tuple[str, list[str]]]] = [
    py_service, py_module, py_async_set,
    rust_module, rust_data_layer,
    go_module,
    ts_module, js_class_set,
]


def main() -> None:
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)

    # Eval — seed 42, 50 prompts
    rng = random.Random(42)
    eval_rows = []
    for i in range(50):
        gen = rng.choice(GENERATORS)
        code, symbols = gen(rng)
        eval_rows.append({
            "id": f"eval-{gen.__name__}-{i:03d}",
            "code": code,
            "ground_truth": symbols,
            "messages": _prompt_for(code),
        })

    # Train — seed 4242, 40 prompts
    rng = random.Random(4242)
    train_rows = []
    for i in range(40):
        gen = rng.choice(GENERATORS)
        code, symbols = gen(rng)
        train_rows.append({
            "id": f"train-{gen.__name__}-{i:03d}",
            "code": code,
            "ground_truth": symbols,
            "messages": _prompt_for(code),
        })

    with open(out_dir / "eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "train.opd.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    n_gt_eval = sum(len(r["ground_truth"]) for r in eval_rows)
    n_gt_train = sum(len(r["ground_truth"]) for r in train_rows)
    print(f"wrote {len(eval_rows)} eval prompts ({n_gt_eval} total symbols) → datasets/eval.jsonl")
    print(f"wrote {len(train_rows)} train prompts ({n_gt_train} total symbols) → datasets/train.opd.jsonl")
    print()
    print("REMINDER: do not read datasets/eval.jsonl from this point on.")


if __name__ == "__main__":
    main()
