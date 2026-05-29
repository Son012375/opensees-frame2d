/**
 * Chat Widget — Phase A.4
 * ============================================================================
 * Floating AI assistant for V2 Editor. Talks to /api/v2/chat/* and uses
 * window.EditorV2ChatBridge for editor context.
 *
 * Loaded last (after editor3d_v2.js) so it can rely on the editor's
 * globals being defined. The whole module is wrapped in try/catch — if
 * anything blows up at init, the FAB stays hidden and the editor
 * continues working without it (Codex Phase A.4 invariant: chat widget
 * MUST NOT break the editor).
 *
 * Wire contract (NDJSON, one event per line):
 *   {type: "status",      message, provider}
 *   {type: "tool_call",   round, tool, arguments}
 *   {type: "tool_result", round, tool, result, ms}
 *   {type: "token",       text}
 *   {type: "error",       message, code}
 *   {type: "done",        rounds, total_tokens, ms_total}
 */
(function () {
    'use strict';

    const ENDPOINT_SESSIONS = '/api/v2/chat/sessions';
    const ENDPOINT_MESSAGES = '/api/v2/chat/messages';

    // ---------- module state ----------
    let sessionId = null;
    let busy = false;
    let assistantTarget = null;   // current streaming <div> for assistant text
    let assistantBuffer = '';
    let typingIndicator = null;
    const dom = {};

    // ---------- bridge helper ----------
    function getBridgeContext() {
        try {
            return window.EditorV2ChatBridge?.getContext?.() || {};
        } catch (_) {
            return {};
        }
    }

    // ---------- DOM scaffold ----------
    function buildDom() {
        const fab = document.createElement('button');
        fab.id = 'chat-fab';
        fab.title = 'AI Assistant (Phase A.4)';
        fab.textContent = '💬';

        const win = document.createElement('div');
        win.id = 'chat-window';
        win.className = 'hidden';
        win.innerHTML = `
            <div class="chat-header" data-role="drag-handle">
                <span class="chat-title">
                    <span class="chat-status-dot" data-role="status-dot"></span>
                    AI Assistant
                </span>
                <div class="chat-controls">
                    <button data-action="minimize" title="최소화">─</button>
                    <button data-action="close" title="닫기">×</button>
                </div>
            </div>
            <div class="chat-body" data-role="body"></div>
            <div class="chat-input-area">
                <textarea data-role="input" rows="2"
                          placeholder="메시지 입력 (Enter 전송, Shift+Enter 줄바꿈)"></textarea>
                <button data-action="send" title="전송">➤</button>
            </div>
            <div class="chat-resize-handle" data-role="resize"></div>
        `;

        document.body.appendChild(fab);
        document.body.appendChild(win);

        dom.fab = fab;
        dom.win = win;
        dom.dragHandle = win.querySelector('[data-role="drag-handle"]');
        dom.statusDot = win.querySelector('[data-role="status-dot"]');
        dom.body = win.querySelector('[data-role="body"]');
        dom.input = win.querySelector('[data-role="input"]');
        dom.sendBtn = win.querySelector('[data-action="send"]');
        dom.minBtn = win.querySelector('[data-action="minimize"]');
        dom.closeBtn = win.querySelector('[data-action="close"]');
        dom.resize = win.querySelector('[data-role="resize"]');
    }

    // ---------- show / hide / minimize ----------
    function openWindow() {
        dom.win.classList.remove('hidden', 'minimized');
        dom.fab.classList.add('hidden');
        setTimeout(() => dom.input.focus(), 50);
    }

    function closeWindow() {
        dom.win.classList.add('hidden');
        dom.fab.classList.remove('hidden');
    }

    function toggleMinimize() {
        // The CSS minimized rule collapses the window to its header, but a
        // user who already resized the panel has an inline ``style.height``
        // that overrides the rule. Stash and clear inline height on the
        // way into minimised state, then restore it on the way back so
        // the user's custom size survives a minimise/restore round trip.
        // (Codex P2 on 7b18d40.)
        const win = dom.win;
        if (win.classList.contains('minimized')) {
            win.classList.remove('minimized');
            const saved = win.dataset.preMinimizeHeight;
            if (saved) {
                win.style.height = saved;
                delete win.dataset.preMinimizeHeight;
            }
        } else {
            if (win.style.height) {
                win.dataset.preMinimizeHeight = win.style.height;
                win.style.height = '';
            }
            win.classList.add('minimized');
        }
    }

    // ---------- session ----------
    async function ensureSession() {
        if (sessionId) return sessionId;
        const ctx = getBridgeContext();
        const body = ctx.analysis_id ? { analysis_id: ctx.analysis_id } : {};
        const r = await fetch(ENDPOINT_SESSIONS, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            throw new Error(`session create failed (${r.status})`);
        }
        const data = await r.json();
        sessionId = data.session_id;
        if (data.provider && data.provider !== 'noop') {
            appendStatusMessage(`provider=${data.provider}`);
        } else if (data.provider === 'noop') {
            appendStatusMessage('provider=noop (LLM 미설정)');
        }
        return sessionId;
    }

    // ---------- NDJSON line reader ----------
    async function* readNdjsonLines(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
            const { done, value } = await reader.read();
            if (done) {
                // Final flush — if the server's last chunk ended in the
                // middle of a UTF-8 sequence, the bytes are still inside
                // the decoder's internal buffer. Without a no-arg
                // decode() call they would be silently dropped, clipping
                // the last line (Codex P3 on 7b18d40).
                buffer += decoder.decode();
                if (buffer.trim()) yield buffer;
                return;
            }
            buffer += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 1);
                if (line.trim()) yield line;
            }
        }
    }

    // ---------- DOM appenders ----------
    function scrollToBottom() {
        dom.body.scrollTop = dom.body.scrollHeight;
    }

    function appendUserMessage(text) {
        const el = document.createElement('div');
        el.className = 'chat-msg chat-msg-user';
        el.textContent = text;
        dom.body.appendChild(el);
        scrollToBottom();
    }

    function appendStatusMessage(text) {
        const el = document.createElement('div');
        el.className = 'chat-msg chat-msg-status';
        el.textContent = text;
        dom.body.appendChild(el);
        scrollToBottom();
    }

    function appendErrorMessage(text) {
        const el = document.createElement('div');
        el.className = 'chat-msg chat-msg-error';
        el.textContent = text;
        dom.body.appendChild(el);
        scrollToBottom();
    }

    function appendToolCall(event) {
        const el = document.createElement('div');
        el.className = 'chat-tool-call';
        const args = JSON.stringify(event.arguments || {});
        el.textContent = `[round ${event.round}] ${event.tool}(${args})`;
        dom.body.appendChild(el);
        scrollToBottom();
    }

    function appendToolResult(event) {
        const el = document.createElement('div');
        el.className = 'chat-tool-result';
        const result = event.result || {};
        const hasError = result.error || result.code === 'tool_blocked' || result.code === 'tool_crash';
        if (hasError) el.classList.add('has-error');
        let summary = `[round ${event.round}] ${event.tool} →`;
        if (result.error) summary += ` ${result.error}`;
        else summary += ` ${summariseResult(result)}`;
        const meta = document.createElement('span');
        meta.className = 'chat-tool-meta';
        meta.textContent = ` (${event.ms}ms)`;
        el.textContent = summary;
        el.appendChild(meta);
        dom.body.appendChild(el);
        scrollToBottom();
    }

    // Phase B — dispatch UI-side actions a tool may request via
    // ``result.ui_action`` (e.g. open the rec-diff modal for a chat-
    // staged section change). The chat stream itself can't carry the
    // heavy payload (streaming.FORBIDDEN_KEYS), so the action receives
    // an opaque ``ui_payload`` (typically ``{preview_id, source}``)
    // and the bridge fetches the rest over plain HTTP. Unknown
    // actions are ignored — forward-compat for future tool tiers.
    function dispatchUiAction(result) {
        if (!result || typeof result !== 'object') return;
        const action = result.ui_action;
        if (!action) return;
        const payload = result.ui_payload || {};
        try {
            if (action === 'open_diff_preview') {
                const bridge = window.EditorV2ChatBridge;
                if (bridge && typeof bridge.openDiffPreview === 'function') {
                    Promise.resolve(bridge.openDiffPreview(payload))
                        .catch((e) => {
                            appendErrorMessage(
                                `미리보기 모달을 열지 못했습니다: ${e?.message || e}`,
                            );
                        });
                } else {
                    appendStatusMessage(
                        '미리보기 모달 브릿지를 찾지 못했습니다 (EditorV2ChatBridge.openDiffPreview).',
                    );
                }
                return;
            }
            // Unknown action — log once at debug, ignore for the user.
            console.debug('[ChatWidget] unknown ui_action:', action);
        } catch (e) {
            appendErrorMessage(`ui_action 실행 실패: ${e?.message || e}`);
        }
    }

    function summariseResult(result) {
        // Compact one-line preview of the tool result. Full payload is in
        // the chat session history; the UI just shows enough to confirm
        // the call did the right thing — and importantly, to let the user
        // spot when the LLM's narration disagrees with the actual tool
        // output (e.g. tool returned story=3 but bot said "1층").
        if (Array.isArray(result.elements)) {
            const n = result.elements.length;
            if (n === 0) return '0 elements';
            // Surface the first element's identifying fields so a "어느 층?"
            // mis-answer is visible in the box itself.
            const e0 = result.elements[0] || {};
            const info = e0.info || {};
            const ratios = e0.ratios || {};
            const parts = [`${n} element(s)`];
            if (e0.element_id != null) parts.push(`id=${e0.element_id}`);
            if (info.member_id != null) parts.push(`member=${info.member_id}`);
            if (info.story != null) parts.push(`story=${info.story}`);
            if (info.section) parts.push(`sec=${info.section}`);
            if (ratios.status) parts.push(ratios.status);
            return parts.join(', ');
        }
        if (result.summary) {
            const s = result.summary;
            const parts = [];
            if (s.ng_count != null) parts.push(`NG=${s.ng_count}`);
            if (s.max_drift?.x != null) parts.push(`maxDriftX=${s.max_drift.x}`);
            if (s.num_stories) parts.push(`stories=${s.num_stories}`);
            return parts.join(', ') || 'ok';
        }
        const keys = Object.keys(result);
        return keys.length ? keys.slice(0, 3).join(',') : 'ok';
    }

    function ensureAssistantTarget() {
        if (assistantTarget) return assistantTarget;
        removeTypingIndicator();
        const el = document.createElement('div');
        el.className = 'chat-msg chat-msg-assistant';
        el.textContent = '';
        dom.body.appendChild(el);
        assistantTarget = el;
        assistantBuffer = '';
        return el;
    }

    function appendToken(text) {
        const target = ensureAssistantTarget();
        assistantBuffer += text;
        target.textContent = assistantBuffer;
        scrollToBottom();
    }

    function appendCollapsibleBlock(summaryLabel, bodyText) {
        // The text-token side of the message lives inside ``assistantTarget``
        // (a single <div> whose textContent is updated as tokens stream).
        // The collapsible block is a sibling <details> rendered AFTER that
        // text — keeping it outside the streaming <div> means later tokens
        // (rare but possible) don't disturb the toggle's DOM state. We
        // close the current assistant text target so any subsequent token
        // event starts a fresh paragraph below the toggle, not inside it.
        removeTypingIndicator();
        if (!assistantTarget) {
            // No prose preceded — synthesise an empty assistant message
            // wrapper so the toggle still anchors to the chat thread.
            ensureAssistantTarget();
        }
        const details = document.createElement('details');
        details.className = 'chat-collapsible';
        const summary = document.createElement('summary');
        summary.textContent = summaryLabel;
        const body = document.createElement('div');
        body.className = 'chat-collapsible-body';
        // textContent (NOT innerHTML) — the body may contain user-controlled
        // strings via tool results; an HTML injection in a KDS quote could
        // break out of the toggle. The CSS keeps newlines visible via
        // white-space: pre-wrap.
        body.textContent = bodyText;
        details.appendChild(summary);
        details.appendChild(body);
        // Append to the parent of the current text target so the toggle
        // sits next to (not inside) the streaming text div.
        const parent = assistantTarget && assistantTarget.parentNode
            ? assistantTarget.parentNode
            : dom.body;
        parent.appendChild(details);
        // Reset the text target so any later token event starts a new
        // assistant text div below the toggle.
        assistantTarget = null;
        assistantBuffer = '';
        scrollToBottom();
    }

    function showTypingIndicator() {
        removeTypingIndicator();
        const el = document.createElement('div');
        el.className = 'chat-typing';
        el.innerHTML = '<span></span><span></span><span></span>';
        dom.body.appendChild(el);
        typingIndicator = el;
        scrollToBottom();
    }

    function removeTypingIndicator() {
        if (typingIndicator && typingIndicator.parentNode) {
            typingIndicator.remove();
        }
        typingIndicator = null;
    }

    // ---------- event dispatch ----------
    function handleEvent(event) {
        switch (event.type) {
            case 'status':
                // ``thinking`` is the only status the A.1 orchestrator
                // emits and the typing indicator already covers it. Phase B
                // will stream evaluation progress ("evaluating 3/12", "tool
                // queued", etc.) as status events — surface those as a
                // compact line so the user sees the orchestrator is alive
                // even when no tokens are flowing (Codex P2 on 7b18d40).
                if (event.message && event.message !== 'thinking') {
                    appendStatusMessage(event.message);
                }
                showTypingIndicator();
                break;
            case 'tool_call':
                removeTypingIndicator();
                appendToolCall(event);
                showTypingIndicator();
                break;
            case 'tool_result':
                removeTypingIndicator();
                appendToolResult(event);
                dispatchUiAction(event.result);
                showTypingIndicator();
                break;
            case 'token':
                removeTypingIndicator();
                appendToken(event.text || '');
                break;
            case 'collapsible':
                // Server attached a long deterministic block (e.g. KDS
                // evidence quotes from explain_member_compliance) that
                // belongs to the current assistant message but should
                // stay collapsed until the user clicks. Renders as a
                // native HTML5 <details> so no JS framework dependency.
                removeTypingIndicator();
                appendCollapsibleBlock(
                    event.summary_label || '자세히 보기',
                    event.text || '',
                );
                break;
            case 'error':
                removeTypingIndicator();
                appendErrorMessage(`[${event.code || 'error'}] ${event.message || ''}`);
                break;
            case 'done':
                removeTypingIndicator();
                assistantTarget = null;
                assistantBuffer = '';
                break;
            default:
                // unknown event type — surface so the user can tell
                // something is up, but don't crash the stream loop
                appendStatusMessage(`unknown event: ${event.type}`);
                break;
        }
    }

    // ---------- send turn ----------
    async function sendMessage(text) {
        if (busy || !text) return;
        busy = true;
        dom.sendBtn.disabled = true;
        dom.statusDot.classList.add('busy');
        dom.statusDot.classList.remove('error');

        try {
            await ensureSession();
            appendUserMessage(text);
            dom.input.value = '';
            const ctx = getBridgeContext();

            const r = await fetch(ENDPOINT_MESSAGES, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: text,
                    ui_context: ctx,
                }),
            });

            if (!r.ok) {
                // 404 = session expired → drop the id and let the next
                // turn create a fresh one
                if (r.status === 404) sessionId = null;
                appendErrorMessage(`HTTP ${r.status}: ${await r.text().catch(() => '')}`);
                dom.statusDot.classList.add('error');
                return;
            }

            for await (const line of readNdjsonLines(r)) {
                let event;
                try { event = JSON.parse(line); }
                catch (_) {
                    appendErrorMessage(`malformed event line: ${line.slice(0, 80)}`);
                    continue;
                }
                handleEvent(event);
            }
        } catch (e) {
            appendErrorMessage(`network: ${e?.message || e}`);
            dom.statusDot.classList.add('error');
        } finally {
            busy = false;
            dom.sendBtn.disabled = false;
            dom.statusDot.classList.remove('busy');
            removeTypingIndicator();
            assistantTarget = null;
            assistantBuffer = '';
            dom.input.focus();
        }
    }

    // ---------- input handling ----------
    function attachInputHandlers() {
        dom.input.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
                ev.preventDefault();
                const text = dom.input.value.trim();
                if (text) sendMessage(text);
            }
            if (ev.key === 'Escape') closeWindow();
        });
        dom.sendBtn.addEventListener('click', () => {
            const text = dom.input.value.trim();
            if (text) sendMessage(text);
        });
    }

    // ---------- drag (header) ----------
    function attachDragHandlers() {
        let start = null;
        dom.dragHandle.addEventListener('mousedown', (ev) => {
            // Ignore clicks on the header buttons
            if (ev.target.closest('button')) return;
            const rect = dom.win.getBoundingClientRect();
            start = {
                x: ev.clientX,
                y: ev.clientY,
                left: rect.left,
                top: rect.top,
            };
            // Switch from `right/bottom` anchor to `left/top` so dragging
            // does what the user expects.
            dom.win.style.left = `${rect.left}px`;
            dom.win.style.top = `${rect.top}px`;
            dom.win.style.right = 'auto';
            dom.win.style.bottom = 'auto';
            ev.preventDefault();
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp, { once: true });
        });

        function onMove(ev) {
            if (!start) return;
            const dx = ev.clientX - start.x;
            const dy = ev.clientY - start.y;
            const left = Math.max(0, Math.min(window.innerWidth - 100, start.left + dx));
            const top = Math.max(0, Math.min(window.innerHeight - 60, start.top + dy));
            dom.win.style.left = `${left}px`;
            dom.win.style.top = `${top}px`;
        }

        function onUp() {
            start = null;
            document.removeEventListener('mousemove', onMove);
        }
    }

    // ---------- resize (bottom-right handle) ----------
    function attachResizeHandlers() {
        let start = null;
        dom.resize.addEventListener('mousedown', (ev) => {
            const rect = dom.win.getBoundingClientRect();
            start = { x: ev.clientX, y: ev.clientY, w: rect.width, h: rect.height };
            // The CSS anchors the window at bottom/right by default. Only
            // changing width/height keeps the right edge fixed, so the
            // bottom-right handle would expand the panel UP and LEFT —
            // backwards from where the cursor is going. Mirror the drag
            // setup: pin left/top from the current rect and release the
            // right/bottom anchors so growth tracks the cursor. (Codex P1
            // on 7b18d40.)
            dom.win.style.left = `${rect.left}px`;
            dom.win.style.top = `${rect.top}px`;
            dom.win.style.right = 'auto';
            dom.win.style.bottom = 'auto';
            ev.preventDefault();
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp, { once: true });
        });

        function onMove(ev) {
            if (!start) return;
            const dx = ev.clientX - start.x;
            const dy = ev.clientY - start.y;
            const w = Math.max(320, Math.min(window.innerWidth * 0.9, start.w + dx));
            const h = Math.max(280, Math.min(window.innerHeight * 0.9, start.h + dy));
            dom.win.style.width = `${w}px`;
            dom.win.style.height = `${h}px`;
        }

        function onUp() {
            start = null;
            document.removeEventListener('mousemove', onMove);
        }
    }

    // ---------- init ----------
    function init() {
        try {
            buildDom();
            dom.fab.addEventListener('click', openWindow);
            dom.closeBtn.addEventListener('click', closeWindow);
            dom.minBtn.addEventListener('click', toggleMinimize);
            attachInputHandlers();
            attachDragHandlers();
            attachResizeHandlers();
            // Expose a tiny surface for the editor / future tests
            window.ChatWidget = {
                open: openWindow,
                close: closeWindow,
                send: (text) => sendMessage(String(text || '')),
                version: '0.1.0',
            };
        } catch (e) {
            console.error('[ChatWidget] init failed:', e);
            // Best-effort cleanup so a half-built widget doesn't shadow
            // the editor
            try { document.getElementById('chat-fab')?.remove(); } catch (_) {}
            try { document.getElementById('chat-window')?.remove(); } catch (_) {}
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // editor3d_v2.js has already run — safe to mount immediately
        init();
    }
})();
