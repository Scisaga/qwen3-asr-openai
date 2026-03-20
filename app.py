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
    CudaOOMTranscriptionError,
    get_health_payload,
    guess_audio_suffix,
    load_model,
    transcribe_input_bytes,
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


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
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg"/>
  <link rel="apple-touch-icon" href="/static/logo.svg"/>
  <title>Qwen3-ASR</title>
  <style>
    :root{
      --bg0:#070b16;
      --bg1:#0b1224;
      --panel:rgba(15, 23, 42, .68);
      --panel2:rgba(15, 23, 42, .78);
      --border:rgba(148, 163, 184, .16);
      --border2:rgba(148, 163, 184, .24);
      --text:rgba(226, 232, 240, .92);
      --muted:rgba(148, 163, 184, .86);
      --muted2:rgba(148, 163, 184, .66);
      --accent:#22c55e;
      --accent2:#60a5fa;
      --danger:#ef4444;
      --shadow:0 18px 60px rgba(0,0,0,.45);
      --radius:14px;
      --radius2:18px;
      --ring:0 0 0 4px rgba(96,165,250,.18);
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0;
      color:var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background:
        radial-gradient(1200px 800px at 18% -10%, rgba(96,165,250,.18), transparent 60%),
        radial-gradient(900px 650px at 92% 10%, rgba(34,197,94,.14), transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      overflow-x:hidden;
    }
    a{color:inherit; text-decoration:none}
    .muted{color:var(--muted)}
    .muted2{color:var(--muted2)}
    code{
      background:rgba(2,6,23,.55);
      border:1px solid var(--border);
      padding:2px 8px;
      border-radius:10px;
      color:rgba(226,232,240,.92);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size:.92em;
    }

    .app{display:flex; min-height:100vh}
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
      width:34px; height:34px; border-radius:12px;
      box-shadow: 0 12px 30px rgba(34,197,94,.14), 0 12px 30px rgba(96,165,250,.16);
      display:block;
    }
    .brand-title{font-weight:700; letter-spacing:.2px}
    .brand-sub{font-size:12px; color:var(--muted2); margin-top:2px}
    .nav{margin-top:6px; display:flex; flex-direction:column; gap:6px}
    .nav a{
      display:flex; align-items:center; gap:10px;
      padding:10px 12px;
      border-radius:12px;
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
    .nav .dot{
      width:8px; height:8px; border-radius:999px;
      background:rgba(148,163,184,.55);
      box-shadow:0 0 0 4px rgba(148,163,184,.08);
    }
    .nav a.active .dot{
      background:var(--accent);
      box-shadow:0 0 0 4px rgba(34,197,94,.16);
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
      padding:8px 10px;
      border-radius:999px;
      border:1px solid var(--border);
      background:rgba(15,23,42,.50);
      font-size:12px;
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
      border:1px solid var(--border);
      background:rgba(15,23,42,.48);
      color:rgba(226,232,240,.92);
      padding:10px 12px;
      border-radius:12px;
      font-weight:600;
      font-size:13px;
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

    .content{padding:22px 18px 40px}
    .section{max-width:1160px; margin:0 auto}
    .section-head{margin:6px 0 16px}
    h1{margin:0; font-size:22px; letter-spacing:.2px}
    .section-head p{margin:8px 0 0; font-size:13px}

    .grid{
      display:grid;
      grid-template-columns: 1.12fr .88fr;
      gap:14px;
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
      padding:14px 16px;
      border-bottom:1px solid var(--border);
      background:rgba(2,6,23,.16);
    }
    .card-title{font-weight:700; font-size:13px; letter-spacing:.24px; color:rgba(226,232,240,.92)}
    .card-body{padding:14px 16px}

    .form{display:flex; flex-direction:column; gap:12px}
    .row{display:grid; grid-template-columns: 1fr 1fr; gap:12px}
    .field{display:flex; flex-direction:column; gap:6px}
    .label{font-size:12px; color:var(--muted2)}
    input[type="text"], textarea{
      width:100%;
      padding:12px 12px;
      border-radius:12px;
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
      border-radius:16px;
      border:1px dashed rgba(148,163,184,.28);
      background:rgba(2,6,23,.30);
      padding:14px;
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
      margin-right:8px;
      vertical-align:-2px;
    }
    .spinner.on{display:inline-block}
    @keyframes spin{to{transform:rotate(360deg)}}

    .output-actions{
      display:flex;
      justify-content:flex-end;
      gap:10px;
      padding:12px 16px 14px;
      border-top:1px solid var(--border);
      background:rgba(2,6,23,.12);
    }

    .code{
      border:1px solid var(--border);
      background:rgba(2,6,23,.34);
      border-radius:14px;
      padding:12px;
      overflow:auto;
    }
    .small{font-size:12px; margin-top:10px}

    @media (max-width: 980px){
      .sidebar{display:none}
      .grid{grid-template-columns: 1fr}
      textarea{min-height:220px}
      .row{grid-template-columns: 1fr}
    }
    @media (prefers-reduced-motion: reduce){
      *{transition:none !important}
      .spinner{animation:none}
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <img class="mark" src="/static/logo.svg" alt="Qwen3-ASR"/>
        <div>
          <div class="brand-title">Qwen3-ASR</div>
          <div class="brand-sub">OpenAI 兼容接口 · Web 上传</div>
        </div>
      </div>

      <nav class="nav" aria-label="导航">
        <a class="active" href="#upload"><span class="dot" aria-hidden="true"></span> 转写</a>
        <a href="/docs" target="_blank" rel="noreferrer"><span class="dot" aria-hidden="true"></span> API 文档</a>
        <a href="/health" target="_blank" rel="noreferrer"><span class="dot" aria-hidden="true"></span> 健康检查</a>
      </nav>

      <div class="sidebar-footer" aria-label="运行信息">
        <div class="kv"><span class="k">Endpoint</span><span class="v">/v1/audio/transcriptions</span></div>
        <div class="kv"><span class="k">Model</span><span class="v" id="model_id">—</span></div>
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
          <a class="btn ghost" href="/docs" target="_blank" rel="noreferrer">Docs</a>
          <a class="btn ghost" href="/redoc" target="_blank" rel="noreferrer">Redoc</a>
        </div>
      </header>

      <main class="content">
        <section class="section" id="upload">
          <div class="section-head">
            <h1>音频/视频转写</h1>
            <p class="muted">上传文件即可转写为文本；服务端会自动抽取音频并转换为 16k 单声道 wav。</p>
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

                  <div class="actions">
                    <button class="btn primary" type="submit" id="submit_btn"><span class="spinner" id="spinner" aria-hidden="true"></span>开始转写</button>
                    <button class="btn ghost" type="button" id="clear_btn">清空结果</button>
                    <span class="hint muted" id="hint">—</span>
                  </div>
                </form>
              </div>
            </div>

            <div class="card">
              <div class="card-header">
                <div class="card-title">Result</div>
                <span class="muted2" style="font-size:12px;">JSON <code>{"text": "...", "language": "Chinese"}</code></span>
              </div>
              <div class="card-body">
                <textarea id="out" placeholder="这里会显示转写结果..." readonly></textarea>
              </div>
              <div class="output-actions">
                <button class="btn ghost" type="button" id="copy_btn">复制</button>
                <a class="btn ghost" href="#api">接口示例</a>
              </div>
            </div>

            <div class="card" id="api" style="grid-column: 1 / -1;">
              <div class="card-header">
                <div class="card-title">API 示例</div>
                <span class="muted2" style="font-size:12px;">curl</span>
              </div>
              <div class="card-body">
                <div class="code">
                  <code>curl -X POST http://localhost:12301/v1/audio/transcriptions -F file=@audio.mp3 -F language=zh</code>
                </div>
                <div class="muted small">提示：也可以直接打开 <a href="/docs" target="_blank" rel="noreferrer" class="muted">/docs</a> 使用 Swagger 调用。</div>
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

    function humanSize(bytes){
      if (!Number.isFinite(bytes)) return '';
      const units = ['B','KB','MB','GB'];
      let v = bytes;
      let i = 0;
      while (v >= 1024 && i < units.length - 1){ v /= 1024; i++; }
      const fixed = i === 0 ? 0 : 2;
      return v.toFixed(fixed) + ' ' + units[i];
    }

    async function refreshHealth(){
      const chip = document.getElementById('model_chip');
      try{
        const resp = await fetch('/health', { cache: 'no-store' });
        const j = await resp.json();
        document.getElementById('model_id').textContent = j.model_id || '—';
        document.getElementById('device_map').textContent = j.device_map || '—';
        document.getElementById('dtype').textContent = j.dtype || '—';
        const loaded = !!j.model_loaded;
        chip.textContent = loaded ? 'Model: loaded' : 'Model: not loaded';
        chip.classList.toggle('ok', loaded);
        chip.classList.toggle('warn', !loaded);
      }catch(e){
        chip.textContent = 'Health: unavailable';
        chip.classList.remove('ok');
        chip.classList.add('warn');
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
      hint.textContent = '上传中…';
      out.value = '';
      submitBtn.disabled = true;
      spinner.classList.add('on');

      try{
        const fd = new FormData(form);
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
        hint.textContent = '完成';
      }catch(err){
        out.value = String(err);
        hint.textContent = '请求异常';
      }finally{
        submitBtn.disabled = false;
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
    load_model(model_id, revision)
    return {"status": "reloaded", "model_id": model_id, "revision": revision}

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    del temperature

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        result = await transcribe_input_bytes(
            payload,
            suffix=guess_audio_suffix(file.filename, file.content_type),
            language=language,
            prompt=prompt,
        )
    except CudaOOMTranscriptionError as exc:
        raise HTTPException(status_code=507, detail=exc.detail) from exc
    except BackendTranscriptionError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "transcribe_failed", "exception": str(exc)},
        ) from exc

    return JSONResponse(result)
