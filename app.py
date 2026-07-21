import asyncio
import contextlib
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mcp_server import mcp
from transcription_service import (
    ADMIN_TOKEN,
    BackendTranscriptionError,
    BackendUnavailableError,
    CudaOOMTranscriptionError,
    InputValidationError,
    get_health_payload,
    guess_audio_suffix,
    maybe_preload_runtime,
    reload_model_backend,
    shutdown_backend,
    transcribe_input_bytes,
)


def _env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in ("0", "false", "no", "off")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    preload_task: Optional[asyncio.Task] = None
    async with mcp.session_manager.run():
        if _env_flag("PRELOAD_MODEL", "1"):
            preload_task = asyncio.create_task(maybe_preload_runtime())
        try:
            yield
        finally:
            if preload_task and not preload_task.done():
                preload_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await preload_task
            await shutdown_backend()


app = FastAPI(lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
app.mount("/mcp", mcp.streamable_http_app(), name="mcp")

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="theme-color" content="#0b1224"/>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg?v=sentence-ts-20260721"/>
  <link rel="apple-touch-icon" href="/static/logo.svg?v=sentence-ts-20260721"/>
  <title>Qwen3-ASR</title>
  <style>
    :root{
      --bg0:#070b16;
      --bg1:#0b1224;
      --panel:rgba(15, 23, 42, .62);
      --panel2:rgba(15, 23, 42, .76);
      --border:rgba(148, 163, 184, .16);
      --border2:rgba(148, 163, 184, .24);
      --text:rgba(226, 232, 240, .92);
      --muted:rgba(148, 163, 184, .86);
      --muted2:rgba(148, 163, 184, .66);
      --accent:#22c55e;
      --accent2:#60a5fa;
      --danger:#ef4444;
      --shadow:0 12px 34px rgba(0,0,0,.30);
      --radius:8px;
      --radius2:8px;
      --ring:0 0 0 4px rgba(96,165,250,.18);
    }
    *{box-sizing:border-box}
    *{
      scrollbar-width:thin;
      scrollbar-color:rgba(148,163,184,.34) transparent;
    }
    *::-webkit-scrollbar{width:10px; height:10px}
    *::-webkit-scrollbar-track{background:transparent}
    *::-webkit-scrollbar-thumb{
      background:rgba(148,163,184,.34);
      border:2px solid transparent;
      border-radius:999px;
      background-clip:padding-box;
    }
    *::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,.48); background-clip:padding-box}
    html{min-height:100%; background:var(--bg0)}
    body{min-height:100%}
    body{
      margin:0;
      color:var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background:linear-gradient(180deg, #0a1021 0%, #090f1d 46%, #070b16 100%);
      overflow-x:hidden;
      position:relative;
    }
    body::before{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      z-index:0;
      background:
        radial-gradient(960px 560px at 18% -12%, rgba(96,165,250,.13), transparent 64%),
        radial-gradient(820px 520px at 92% 8%, rgba(34,197,94,.11), transparent 62%);
      background-repeat:no-repeat;
    }
    a{color:inherit; text-decoration:none}
    .muted{color:var(--muted)}
    .muted2{color:var(--muted2)}
    code{
      background:rgba(2,6,23,.55);
      border:1px solid var(--border);
      padding:2px 8px;
      border-radius:4px;
      color:rgba(226,232,240,.92);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size:.92em;
    }

    .app{position:relative; z-index:1; display:flex; min-height:100vh}
    .sidebar{
      width:270px;
      padding:18px 16px;
      border-right:1px solid var(--border);
      background:rgba(2,6,23,.58);
      backdrop-filter: blur(10px);
    }
    .brand{
      display:flex;
      gap:12px;
      align-items:center;
      padding:10px 10px 14px;
    }
    .mark{
      width:34px; height:34px; border-radius:0;
      object-fit:contain;
      display:block;
    }
    .brand-title{font-weight:700; letter-spacing:.2px}
    .brand-sub{font-size:12px; color:var(--muted2); margin-top:2px}
    .nav{margin-top:6px; display:flex; flex-direction:column; gap:6px}
    .nav a{
      display:flex; align-items:center; gap:10px;
      padding:10px 12px;
      border-radius:6px;
      color:var(--muted);
      border:1px solid transparent;
      background:transparent;
      transition: background .15s ease, border-color .15s ease, color .15s ease;
    }
    .nav a:hover{background:rgba(148,163,184,.08)}
    .nav a.active{
      background:rgba(34,197,94,.10);
      border-color:rgba(34,197,94,.22);
      color:rgba(226,232,240,.96);
    }
    .icon-defs{
      position:absolute;
      width:0;
      height:0;
      overflow:hidden;
    }
    .icon{
      width:16px;
      height:16px;
      flex:0 0 16px;
      color:currentColor;
      fill:none;
      stroke:currentColor;
      stroke-width:1.8;
      stroke-linecap:square;
      stroke-linejoin:miter;
      opacity:.82;
    }
    .nav .icon{
      color:var(--muted2);
      opacity:.78;
    }
    .nav a.active .icon{
      color:var(--accent);
      opacity:.95;
    }
    .sidebar-footer{
      margin-top:18px;
      padding:12px;
      border-radius:var(--radius);
      border:1px solid var(--border);
      background:rgba(15,23,42,.45);
    }
    .kv{display:flex; justify-content:space-between; gap:10px; padding:6px 0}
    .kv .k{color:var(--muted2); font-size:12px}
    .kv .v{font-size:12px; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
    .kv .v.long{font-size:11.5px}

    .main{flex:1; display:flex; flex-direction:column}
    .topbar{
      position:sticky; top:0; z-index:10;
      display:flex; align-items:center; justify-content:space-between;
      padding:12px 18px;
      border-bottom:1px solid var(--border);
      background:rgba(2,6,23,.52);
      backdrop-filter: blur(10px);
    }
    .crumbs{display:flex; align-items:center; gap:10px; font-size:13px}
    .sep{opacity:.55}
    .top-actions{display:flex; align-items:center; gap:10px}

    .chip{
      padding:8px 12px;
      border-radius:999px;
      border:1px solid var(--border);
      background:rgba(15,23,42,.50);
      font-size:12px;
      font-weight:650;
      color:var(--muted);
      white-space:nowrap;
    }
    .chip.ok{
      border-color:rgba(34,197,94,.28);
      background:rgba(34,197,94,.10);
      color:rgba(187,247,208,.92);
    }
    .chip.warn{
      border-color:rgba(239,68,68,.26);
      background:rgba(239,68,68,.10);
      color:rgba(254,202,202,.92);
    }

    .btn{
      appearance:none;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:8px;
      border:1px solid var(--border);
      background:rgba(15,23,42,.48);
      color:rgba(226,232,240,.92);
      padding:10px 12px;
      border-radius:6px;
      font-weight:600;
      font-size:13px;
      line-height:1;
      cursor:pointer;
      transition: background .15s ease, border-color .15s ease, transform .05s ease;
      user-select:none;
    }
    .btn:hover{background:rgba(148,163,184,.10)}
    .btn:active{transform:translateY(1px)}
    .btn:disabled{opacity:.55; cursor:not-allowed}
    .btn.primary{
      border-color:rgba(34,197,94,.26);
      background:linear-gradient(180deg, rgba(34,197,94,.22), rgba(34,197,94,.14));
    }
    .btn.primary:hover{background:linear-gradient(180deg, rgba(34,197,94,.28), rgba(34,197,94,.16))}
    .btn.ghost{background:transparent}

    .content{padding:18px 16px 34px}
    .section{max-width:1160px; margin:0 auto}
    .section-head{margin:2px 0 12px}
    h1{margin:0; font-size:22px; letter-spacing:.2px}
    .section-head p{margin:8px 0 0; font-size:13px}
    .loading-banner{
      display:flex;
      align-items:center;
      gap:10px;
      margin:0 0 10px;
      padding:10px 12px;
      border:1px solid rgba(96,165,250,.22);
      border-radius:8px;
      background:rgba(30,41,59,.44);
      color:rgba(226,232,240,.92);
      font-size:13px;
    }
    .loading-banner[hidden]{display:none}
    .loading-dot{
      width:10px;
      height:10px;
      border-radius:999px;
      background:var(--accent2);
      box-shadow:0 0 0 5px rgba(96,165,250,.12);
      animation:pulse 1.2s ease-in-out infinite;
      flex:0 0 10px;
    }
    .loading-title{font-weight:720}
    .loading-text{color:var(--muted); margin-left:2px}
    @keyframes pulse{
      0%,100%{opacity:.42; transform:scale(.82)}
      50%{opacity:1; transform:scale(1)}
    }

    .grid{
      display:grid;
      grid-template-columns: 1.12fr .88fr;
      gap:10px;
    }
    .card{
      border:1px solid var(--border);
      border-radius:var(--radius2);
      background:var(--panel);
      box-shadow:var(--shadow);
      overflow:hidden;
    }
    .card-header{
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:11px 14px;
      border-bottom:1px solid rgba(148,163,184,.20);
      background:linear-gradient(180deg, rgba(30,41,59,.54), rgba(15,23,42,.26));
    }
    .card-title{font-weight:760; font-size:13px; letter-spacing:.24px; color:rgba(241,245,249,.96)}
    .card-body{padding:12px 14px}

    .form{display:flex; flex-direction:column; gap:10px}
    .row{display:grid; grid-template-columns: 1fr 1fr; gap:10px}
    .field{display:flex; flex-direction:column; gap:6px}
    .label{font-size:12px; color:var(--muted2)}
    .mode-group{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:8px;
    }
    .mode-option{
      position:relative;
      min-height:42px;
    }
    .mode-option input{
      position:absolute;
      opacity:0;
      pointer-events:none;
    }
    .mode-option span{
      display:flex;
      align-items:center;
      justify-content:center;
      width:100%;
      height:42px;
      border-radius:6px;
      border:1px solid var(--border);
      background:rgba(2,6,23,.28);
      color:var(--muted);
      font-size:13px;
      font-weight:650;
      cursor:pointer;
      transition:background .15s ease, border-color .15s ease, color .15s ease;
    }
    .mode-option input:checked + span{
      border-color:rgba(34,197,94,.30);
      background:rgba(34,197,94,.12);
      color:rgba(226,232,240,.96);
    }
    .mode-option.disabled span{
      opacity:.48;
      cursor:not-allowed;
    }
    input[type="text"], textarea{
      width:100%;
      padding:10px 11px;
      border-radius:6px;
      border:1px solid var(--border);
      background:rgba(2,6,23,.38);
      color:rgba(226,232,240,.92);
      outline:none;
      transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
    }
    textarea{min-height:260px; resize:vertical}
    input[type="text"]::placeholder, textarea::placeholder{color:rgba(148,163,184,.55)}
    input[type="text"]:focus, textarea:focus{
      border-color:rgba(96,165,250,.42);
      box-shadow:var(--ring);
      background:rgba(2,6,23,.46);
    }

    .drop{
      position:relative;
      border-radius:8px;
      border:1px dashed rgba(148,163,184,.28);
      background:rgba(2,6,23,.30);
      padding:12px;
      transition: border-color .15s ease, background .15s ease;
    }
    .drop.drag{
      border-color:rgba(96,165,250,.52);
      background:rgba(96,165,250,.10);
    }
    .drop input[type="file"]{
      position:absolute;
      inset:0;
      opacity:0;
      cursor:pointer;
    }
    .drop-title{font-weight:700; font-size:13px}
    .drop-meta{font-size:12px; margin-top:6px}

    .actions{
      display:flex;
      align-items:center;
      gap:10px;
      margin-top:2px;
      flex-wrap:wrap;
    }
    .hint{font-size:12px}
    .spinner{
      width:14px; height:14px;
      border-radius:999px;
      border:2px solid rgba(226,232,240,.22);
      border-top-color:rgba(226,232,240,.72);
      display:none;
      animation:spin .8s linear infinite;
      margin:0;
      vertical-align:0;
    }
    .spinner.on{display:inline-block}
    @keyframes spin{to{transform:rotate(360deg)}}

    .output-actions{
      display:flex;
      justify-content:flex-end;
      gap:8px;
      padding:10px 14px 12px;
      border-top:1px solid rgba(148,163,184,.12);
      background:transparent;
    }
    .sentences-panel{
      padding:0 14px 12px;
    }
    .sentences-panel[hidden]{display:none}
    .sentences-wrap{
      max-height:260px;
      overflow:auto;
      border:1px solid var(--border);
      border-radius:6px;
      background:rgba(2,6,23,.24);
    }
    .sentences-table{
      width:100%;
      border-collapse:collapse;
      table-layout:fixed;
      font-size:12px;
    }
    .sentences-table th,
    .sentences-table td{
      padding:9px 10px;
      border-bottom:1px solid rgba(148,163,184,.12);
      text-align:left;
      vertical-align:top;
    }
    .sentences-table th{
      color:var(--muted2);
      font-weight:650;
      background:rgba(15,23,42,.42);
      position:sticky;
      top:0;
      z-index:1;
    }
    .sentences-table tr:last-child td{border-bottom:0}
    .time-cell{
      width:74px;
      color:rgba(187,247,208,.92);
      font-variant-numeric:tabular-nums;
      white-space:nowrap;
    }
    .sentence-cell{
      color:rgba(226,232,240,.92);
      word-break:break-word;
      line-height:1.55;
    }

    .code{
      border:1px solid rgba(148,163,184,.14);
      background:rgba(2,6,23,.36);
      border-radius:6px;
      padding:10px 12px;
      overflow:hidden;
      text-align:left;
    }
    .code code{
      display:block;
      margin:0;
      padding:0;
      border:0;
      border-radius:0;
      background:transparent;
      color:rgba(226,232,240,.94);
      font-size:12px;
      line-height:1.55;
      white-space:pre-wrap;
      overflow-wrap:anywhere;
      word-break:break-word;
    }
    .stack{display:flex; flex-direction:column; gap:10px}
    .info-grid{
      display:grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap:10px;
      margin-top:0;
    }
    .info-box{
      border:1px solid rgba(148,163,184,.12);
      background:rgba(2,6,23,.16);
      border-radius:6px;
      padding:10px 11px;
    }
    .info-box .label{
      display:block;
      margin-bottom:8px;
    }
    .plain-list{
      margin:0;
      padding-left:18px;
      color:var(--muted);
      font-size:13px;
      line-height:1.7;
    }
    .plain-list li + li{margin-top:4px}
    .note{
      border-left:3px solid rgba(96,165,250,.38);
      padding-left:12px;
      color:var(--muted);
      font-size:13px;
      line-height:1.7;
    }
    .small{font-size:12px; margin-top:8px}

    @media (max-width: 980px){
      .sidebar{display:none}
      .grid{grid-template-columns: 1fr}
      textarea{min-height:220px}
      .row{grid-template-columns: 1fr}
    }
    @media (prefers-reduced-motion: reduce){
      *{transition:none !important}
      .spinner{animation:none}
      .loading-dot{animation:none}
    }
  </style>
</head>
<body>
  <svg class="icon-defs" aria-hidden="true" focusable="false">
    <symbol id="i-transcribe" viewBox="0 0 24 24">
      <path d="M4 13h3l2-6 4 12 2-6h5"/>
    </symbol>
    <symbol id="i-link" viewBox="0 0 24 24">
      <path d="M8 8h8v7H8z"/>
      <path d="M12 15v4"/>
      <path d="M7 19h10"/>
      <path d="M10 5v3"/>
      <path d="M14 5v3"/>
    </symbol>
    <symbol id="i-doc" viewBox="0 0 24 24">
      <path d="M7 4h7l4 4v12H7z"/>
      <path d="M14 4v4h4"/>
      <path d="M10 12h6"/>
      <path d="M10 16h5"/>
    </symbol>
    <symbol id="i-health" viewBox="0 0 24 24">
      <path d="M4 13h4l2-5 4 10 2-5h4"/>
    </symbol>
    <symbol id="i-clear" viewBox="0 0 24 24">
      <path d="M7 7l10 10"/>
      <path d="M17 7L7 17"/>
    </symbol>
    <symbol id="i-copy" viewBox="0 0 24 24">
      <path d="M9 9h9v11H9z"/>
      <path d="M6 16H5V5h11v1"/>
    </symbol>
    <symbol id="i-code" viewBox="0 0 24 24">
      <path d="M9 7l-5 5 5 5"/>
      <path d="M15 7l5 5-5 5"/>
    </symbol>
  </svg>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <img class="mark" src="/static/logo.svg?v=sentence-ts-20260721" alt="Qwen3-ASR"/>
        <div>
          <div class="brand-title">Qwen3-ASR</div>
          <div class="brand-sub">OpenAI 兼容接口 · Web 上传</div>
        </div>
      </div>

      <nav class="nav" aria-label="导航">
        <a class="active" href="#upload"><svg class="icon" aria-hidden="true"><use href="#i-transcribe"></use></svg> 转写</a>
        <a href="#mcp"><svg class="icon" aria-hidden="true"><use href="#i-link"></use></svg> MCP 说明</a>
        <a href="/docs" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg> API 文档</a>
        <a href="/health" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-health"></use></svg> 健康检查</a>
      </nav>

      <div class="sidebar-footer" aria-label="运行信息">
        <div class="kv"><span class="k">Endpoint</span><span class="v">/v1/audio/transcriptions</span></div>
        <div class="kv"><span class="k">MCP</span><span class="v">/mcp</span></div>
        <div class="kv"><span class="k">Model</span><span class="v" id="model_id">—</span></div>
        <div class="kv"><span class="k">Aligner</span><span class="v long" id="aligner_id">—</span></div>
        <div class="kv"><span class="k">Device</span><span class="v" id="device_map">—</span></div>
        <div class="kv"><span class="k">DType</span><span class="v" id="dtype">—</span></div>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div class="crumbs">
          <span class="muted2">Services</span>
          <span class="sep muted2">/</span>
          <span>Qwen3-ASR</span>
        </div>
        <div class="top-actions">
          <span class="chip" id="model_chip">Model: checking…</span>
          <span class="chip" id="timestamp_chip">Timestamps: checking…</span>
          <a class="btn ghost" href="/docs" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg>Docs</a>
          <a class="btn ghost" href="/redoc" target="_blank" rel="noreferrer"><svg class="icon" aria-hidden="true"><use href="#i-doc"></use></svg>Redoc</a>
        </div>
      </header>

      <main class="content">
        <section class="section" id="upload">
          <div class="section-head">
            <h1>音频/视频转写</h1>
            <p class="muted">上传文件即可转写为文本；服务端会自动抽取音频并转换为 16k 单声道 wav。</p>
          </div>
          <div class="loading-banner" id="loading_banner" hidden>
            <span class="loading-dot" aria-hidden="true"></span>
            <span class="loading-title">模型加载中</span>
            <span class="loading-text" id="loading_text">Web 服务已启动，ASR 模型正在准备。</span>
          </div>

          <div class="grid">
            <div class="card">
              <div class="card-header">
                <div class="card-title">Upload</div>
                <span class="muted2" style="font-size:12px;">POST <code>/v1/audio/transcriptions</code></span>
              </div>
              <div class="card-body">
                <form id="f" class="form">
                  <label class="field">
                    <span class="label">文件</span>
                    <div class="drop" id="drop">
                      <input id="file" type="file" name="file" required/>
                      <div>
                        <div class="drop-title">拖拽文件到此处，或点击选择</div>
                        <div class="drop-meta muted" id="file_meta">支持音频/视频；会自动转码处理</div>
                      </div>
                    </div>
                  </label>

                  <div class="row">
                    <label class="field">
                      <span class="label">language（可选）</span>
                      <input type="text" name="language" placeholder="zh / en"/>
                    </label>
                    <label class="field">
                      <span class="label">prompt（可选）</span>
                      <input type="text" name="prompt" placeholder="专有名词 / 上下文"/>
                    </label>
                  </div>

                  <div class="field">
                    <span class="label">模式</span>
                    <div class="mode-group" role="radiogroup" aria-label="转写模式">
                      <label class="mode-option">
                        <input id="mode_text" type="radio" name="transcription_mode" value="text" checked/>
                        <span>纯文本</span>
                      </label>
                      <label class="mode-option" id="mode_timestamps_option">
                        <input id="mode_timestamps" type="radio" name="transcription_mode" value="timestamps"/>
                        <span>句子时间戳</span>
                      </label>
                    </div>
                  </div>

                  <div class="actions">
                    <button class="btn primary" type="submit" id="submit_btn"><svg class="icon" aria-hidden="true"><use href="#i-transcribe"></use></svg><span class="spinner" id="spinner" aria-hidden="true"></span>开始转写</button>
                    <button class="btn ghost" type="button" id="clear_btn"><svg class="icon" aria-hidden="true"><use href="#i-clear"></use></svg>清空结果</button>
                    <span class="hint muted" id="hint">—</span>
                  </div>
                </form>
              </div>
            </div>

            <div class="card">
              <div class="card-header">
                <div class="card-title">Result</div>
                <span class="muted2" style="font-size:12px;">JSON <code>{"text": "...", "sentences": [...]}</code></span>
              </div>
              <div class="card-body">
                <textarea id="out" placeholder="这里会显示转写结果..." readonly></textarea>
              </div>
              <div class="sentences-panel" id="sentences_panel" hidden>
                <div class="sentences-wrap">
                  <table class="sentences-table">
                    <thead>
                      <tr>
                        <th class="time-cell">开始</th>
                        <th class="time-cell">结束</th>
                        <th>句子</th>
                      </tr>
                    </thead>
                    <tbody id="sentences_body"></tbody>
                  </table>
                </div>
              </div>
              <div class="output-actions">
                <button class="btn ghost" type="button" id="copy_btn"><svg class="icon" aria-hidden="true"><use href="#i-copy"></use></svg>复制</button>
                <a class="btn ghost" href="#api"><svg class="icon" aria-hidden="true"><use href="#i-code"></use></svg>接口示例</a>
              </div>
            </div>

            <div class="card" id="api" style="grid-column: 1 / -1;">
              <div class="card-header">
                <div class="card-title">API 示例</div>
                <span class="muted2" style="font-size:12px;">curl</span>
              </div>
              <div class="card-body">
                <div class="code">
                  <code>curl -X POST http://localhost:12301/v1/audio/transcriptions -F file=@audio.mp3 -F language=zh -F response_format=verbose_json -F 'timestamp_granularities[]=sentence'</code>
                </div>
                <div class="muted small">提示：也可以直接打开 <a href="/docs" target="_blank" rel="noreferrer" class="muted">/docs</a> 使用 Swagger 调用。</div>
              </div>
            </div>

            <div class="card" id="mcp" style="grid-column: 1 / -1;">
              <div class="card-header">
                <div class="card-title">MCP 接入说明</div>
                <span class="muted2" style="font-size:12px;">Streamable HTTP <code>/mcp</code></span>
              </div>
              <div class="card-body">
                <div class="stack">
                  <div class="note">
                    当客户端支持 MCP 时，优先通过 <code>/mcp</code> 暴露的 Streamable HTTP 接入；
                    如果是大文件上传或只需要简单 HTTP 表单调用，继续使用 <code>/v1/audio/transcriptions</code>。
                  </div>

                  <div class="info-grid">
                    <div class="info-box">
                      <span class="label">入口地址</span>
                      <code>http://localhost:12301/mcp</code>
                    </div>
                    <div class="info-box">
                      <span class="label">可用 Tool</span>
                      <code>transcribe_audio</code>
                    </div>
                    <div class="info-box">
                      <span class="label">输入上限</span>
                      <span class="muted"><span id="mcp_limit">检查中...</span>（按 base64 解码后的原始字节）</span>
                    </div>
                  </div>

                  <div class="info-grid">
                    <div class="info-box">
                      <span class="label">Tool 参数</span>
                      <ul class="plain-list">
                        <li><code>audio_base64</code>：必填，支持纯 base64 或 data URL</li>
                        <li><code>filename</code>：可选，用于推断临时文件后缀</li>
                        <li><code>mime_type</code>：可选，如 <code>audio/mpeg</code>、<code>video/mp4</code></li>
                        <li><code>language</code>、<code>prompt</code>：语种与上下文提示</li>
                        <li><code>response_format</code>、<code>timestamp_granularities</code>：句子时间戳模式</li>
                      </ul>
                    </div>
                    <div class="info-box">
                      <span class="label">可读资源</span>
                      <ul class="plain-list">
                        <li><code>qwen3asr://health</code>：模型、aligner、device、dtype、切片参数</li>
                        <li><code>qwen3asr://usage</code>：MCP 调用说明与返回格式</li>
                        <li><code>transcribe_audio_workflow</code>：转写调用提示词</li>
                        <li><code>transcript_cleanup_workflow</code>：整理转写文本提示词</li>
                      </ul>
                    </div>
                  </div>

                  <div class="code">
                    <code>{"method":"tools/call","params":{"name":"transcribe_audio","arguments":{"audio_base64":"data:audio/mp3;base64,...","filename":"audio.mp3","language":"zh","response_format":"verbose_json","timestamp_granularities":["sentence"]}}}</code>
                  </div>
                  <div class="muted small">提示：MCP 更适合 Agent / IDE / 桌面客户端接入；超大音频或视频文件建议走 HTTP 上传接口，避免 base64 体积膨胀。</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const form = document.getElementById('f');
    const out = document.getElementById('out');
    const submitBtn = document.getElementById('submit_btn');
    const spinner = document.getElementById('spinner');
    const hint = document.getElementById('hint');
    const clearBtn = document.getElementById('clear_btn');
    const copyBtn = document.getElementById('copy_btn');
    const fileInput = document.getElementById('file');
    const fileMeta = document.getElementById('file_meta');
    const drop = document.getElementById('drop');
    const modeText = document.getElementById('mode_text');
    const modeTimestamps = document.getElementById('mode_timestamps');
    const modeTimestampsOption = document.getElementById('mode_timestamps_option');
    const timestampChip = document.getElementById('timestamp_chip');
    const sentencesPanel = document.getElementById('sentences_panel');
    const sentencesBody = document.getElementById('sentences_body');
    const loadingBanner = document.getElementById('loading_banner');
    const loadingText = document.getElementById('loading_text');
    let backendLoading = false;
    let isSubmitting = false;

    function humanSize(bytes){
      if (!Number.isFinite(bytes)) return '';
      const units = ['B','KB','MB','GB'];
      let v = bytes;
      let i = 0;
      while (v >= 1024 && i < units.length - 1){ v /= 1024; i++; }
      const fixed = i === 0 ? 0 : 2;
      return v.toFixed(fixed) + ' ' + units[i];
    }

    function formatSeconds(value){
      const n = Number(value);
      if (!Number.isFinite(n)) return '—';
      return n.toFixed(3);
    }

    function clearSentences(){
      sentencesBody.replaceChildren();
      sentencesPanel.hidden = true;
    }

    function renderSentences(sentences){
      clearSentences();
      if (!Array.isArray(sentences) || sentences.length === 0) return;

      const fragment = document.createDocumentFragment();
      for (const sentence of sentences){
        const tr = document.createElement('tr');
        const start = document.createElement('td');
        const end = document.createElement('td');
        const text = document.createElement('td');
        start.className = 'time-cell';
        end.className = 'time-cell';
        text.className = 'sentence-cell';
        start.textContent = formatSeconds(sentence && sentence.start);
        end.textContent = formatSeconds(sentence && sentence.end);
        text.textContent = (sentence && sentence.text) ? String(sentence.text) : '';
        tr.append(start, end, text);
        fragment.appendChild(tr);
      }
      sentencesBody.appendChild(fragment);
      sentencesPanel.hidden = false;
    }

    function applyTimestampAvailability(health){
      const enabled = !!(health && health.sentence_timestamps_enabled);
      const loaded = !!(health && health.forced_aligner_loaded);
      modeTimestamps.disabled = !enabled;
      modeTimestampsOption.classList.toggle('disabled', !enabled);
      if (!enabled && modeTimestamps.checked){
        modeText.checked = true;
      }
      timestampChip.textContent = enabled
        ? (loaded ? 'Timestamps: ready' : 'Timestamps: configured')
        : 'Timestamps: off';
      timestampChip.classList.toggle('ok', enabled);
      timestampChip.classList.toggle('warn', !enabled);
    }

    async function refreshHealth(){
      const chip = document.getElementById('model_chip');
      try{
        const resp = await fetch('/health', { cache: 'no-store' });
        const j = await resp.json();
        const modelId = j.model_id || '—';
        const alignerId = j.forced_aligner_model_id || '—';
        document.getElementById('model_id').textContent = modelId;
        document.getElementById('model_id').title = modelId;
        document.getElementById('aligner_id').textContent = alignerId;
        document.getElementById('aligner_id').title = alignerId;
        document.getElementById('device_map').textContent = j.device_map || '—';
        document.getElementById('dtype').textContent = j.dtype || '—';
        document.getElementById('mcp_limit').textContent = j.mcp_max_input_bytes ? humanSize(j.mcp_max_input_bytes) : '—';
        const loaded = !!j.model_loaded;
        const backendState = String(j.backend_state || (loaded ? 'ready' : 'stopped'));
        backendLoading = !loaded && (
          backendState === 'starting' ||
          !!j.model_loading ||
          !!j.backend_process_alive
        );
        chip.textContent = loaded
          ? 'Model: loaded'
          : (backendLoading ? 'Model: loading...' : 'Model: not loaded');
        chip.classList.toggle('ok', loaded);
        chip.classList.toggle('warn', !loaded);
        loadingBanner.hidden = !backendLoading;
        if (backendLoading){
          const count = Number(j.backend_replica_count || 0);
          const ready = Number(j.backend_ready_count || 0);
          loadingText.textContent = count > 1
            ? `Web 服务已启动，ASR worker 正在加载模型（${ready}/${count} ready）。`
            : 'Web 服务已启动，ASR 模型正在加载，请稍候。';
        }
        if (!isSubmitting){
          submitBtn.disabled = backendLoading;
        }
        applyTimestampAvailability(j);
      }catch(e){
        document.getElementById('mcp_limit').textContent = '不可用';
        document.getElementById('aligner_id').textContent = '—';
        document.getElementById('aligner_id').title = '';
        chip.textContent = 'Health: unavailable';
        chip.classList.remove('ok');
        chip.classList.add('warn');
        backendLoading = false;
        loadingBanner.hidden = true;
        if (!isSubmitting){
          submitBtn.disabled = false;
        }
        applyTimestampAvailability(null);
      }
    }

    refreshHealth();
    setInterval(refreshHealth, 8000);

    fileInput.addEventListener('change', () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f){
        fileMeta.textContent = '支持音频/视频；会自动转码处理';
        return;
      }
      fileMeta.textContent = `${f.name} · ${humanSize(f.size)}`;
    });

    drop.addEventListener('dragover', (e) => {
      e.preventDefault();
      drop.classList.add('drag');
    });
    drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('drag');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length){
        const dt = new DataTransfer();
        for (const f of files) dt.items.add(f);
        fileInput.files = dt.files;
        fileInput.dispatchEvent(new Event('change'));
      }
    });

    clearBtn.addEventListener('click', () => {
      out.value = '';
      clearSentences();
      hint.textContent = '—';
    });

    copyBtn.addEventListener('click', async () => {
      const text = out.value || '';
      if (!text){
        hint.textContent = '没有可复制内容';
        return;
      }
      try{
        await navigator.clipboard.writeText(text);
        hint.textContent = '已复制';
      }catch(e){
        hint.textContent = '复制失败（浏览器限制）';
      }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (backendLoading){
        hint.textContent = '模型仍在加载中';
        return;
      }
      isSubmitting = true;
      hint.textContent = '上传中…';
      out.value = '';
      clearSentences();
      submitBtn.disabled = true;
      spinner.classList.add('on');

      try{
        const fd = new FormData(form);
        const wantTimestamps = modeTimestamps.checked && !modeTimestamps.disabled;
        fd.delete('transcription_mode');
        fd.delete('response_format');
        fd.delete('timestamp_granularities');
        fd.delete('timestamp_granularities[]');
        if (wantTimestamps){
          fd.set('response_format', 'verbose_json');
          fd.append('timestamp_granularities[]', 'sentence');
        }else{
          fd.set('response_format', 'json');
        }
        const resp = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
        let j = null;
        let rawText = '';
        try{
          j = await resp.json();
        }catch(_){
          try{ rawText = await resp.text(); }catch(__){}
        }

        if (!resp.ok){
          const detail = (j && j.detail) ? j.detail : (j ?? rawText);
          out.value = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
          hint.textContent = `失败：HTTP ${resp.status}`;
          return;
        }

        out.value = (j && j.text) ? j.text : JSON.stringify(j, null, 2);
        renderSentences(j && j.sentences);
        hint.textContent = '完成';
      }catch(err){
        out.value = String(err);
        hint.textContent = '请求异常';
      }finally{
        isSubmitting = false;
        submitBtn.disabled = backendLoading;
        spinner.classList.remove('on');
        refreshHealth();
      }
    });
  </script>
</body>
</html>
"""

@app.get("/health")
def health():
    return get_health_payload()

@app.post("/admin/reload")
async def admin_reload(request: Request):
    token = request.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    body = await request.json()
    current_health = get_health_payload()
    model_id = body.get("model_id", current_health["model_id"])
    revision = body.get("revision", current_health["revision"])
    payload = await reload_model_backend(model_id, revision)
    return {"status": "reloaded", "model_id": model_id, "revision": revision, "health": payload}

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    timestamp_granularities: Optional[list[str]] = Form(None),
    timestamp_granularities_bracket: Optional[list[str]] = Form(
        None, alias="timestamp_granularities[]"
    ),
    temperature: Optional[float] = Form(None),
):
    del temperature

    filename = file.filename
    content_type = file.content_type
    try:
        payload = await file.read()
    finally:
        await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        result = await transcribe_input_bytes(
            payload,
            suffix=guess_audio_suffix(filename, content_type),
            language=language,
            prompt=prompt,
            response_format=response_format,
            timestamp_granularities=[
                *(timestamp_granularities or []),
                *(timestamp_granularities_bracket or []),
            ],
        )
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CudaOOMTranscriptionError as exc:
        raise HTTPException(status_code=507, detail=exc.detail) from exc
    except BackendUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "backend_unavailable", "exception": str(exc)},
        ) from exc
    except BackendTranscriptionError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "transcribe_failed", "exception": str(exc)},
        ) from exc

    return JSONResponse(result)
