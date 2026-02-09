/**
 * Continuous Beam Analysis Form Handler
 * Supports both Natural Language (Claude) and Direct Input modes
 */

// Store parsed data from Claude
let parsedData = null;

document.addEventListener('DOMContentLoaded', () => {
    // Check Claude API status
    checkClaudeStatus();

    // Initialize supports based on initial spans
    updateSupportsUI();

    // Setup direct input form
    const form = document.getElementById('analysis-form');
    const submitBtn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        submitBtn.disabled = true;
        submitBtn.textContent = 'Submitting...';

        try {
            const data = buildInputData();
            console.log('Submitting:', data);
            await submitAnalysis(data);
        } catch (error) {
            console.error('Submit error:', error);
            alert('Error: ' + error.message);
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Analysis';
        }
    });

    // Update load span options when spans change
    updateLoadSpanOptions();
});


/**
 * Check if Claude API is available
 */
async function checkClaudeStatus() {
    try {
        const response = await fetch('/api/claude/status');
        const result = await response.json();

        if (!result.available) {
            document.getElementById('api-warning').style.display = 'block';
            document.getElementById('natural-submit-btn').disabled = true;
            document.getElementById('parse-btn').disabled = true;
        }
    } catch (error) {
        console.error('Failed to check Claude status:', error);
    }
}


/**
 * Switch between input modes
 */
function switchMode(mode) {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    document.getElementById('natural-mode').classList.toggle('active', mode === 'natural');
    document.getElementById('direct-mode').classList.toggle('active', mode === 'direct');
}


/**
 * Fill example prompt
 */
function fillExample(num) {
    const examples = {
        1: "3경간 연속보, 각 경간 6m, 전체에 등분포하중 15kN/m 작용, H-400x200 단면, SS275 재료",
        2: "2경간 연속보 (8m, 10m), 각 경간 중앙에 집중하중 50kN, H-500x200 단면",
        3: "4경간 연속보, 경간 5m-6m-6m-5m, 1번째와 4번째 경간에만 등분포하중 25kN/m"
    };

    document.getElementById('natural-input').value = examples[num] || '';
    document.getElementById('parsed-preview').style.display = 'none';
    document.getElementById('parse-error').style.display = 'none';
}


/**
 * Parse natural language input using Claude API
 */
async function parseNaturalLanguage() {
    const input = document.getElementById('natural-input').value.trim();
    if (!input) {
        alert('연속보 설명을 입력해주세요.');
        return;
    }

    const parseBtn = document.getElementById('parse-btn');
    const errorDiv = document.getElementById('parse-error');
    const previewDiv = document.getElementById('parsed-preview');

    parseBtn.disabled = true;
    parseBtn.textContent = 'Claude 변환 중...';
    errorDiv.style.display = 'none';
    previewDiv.style.display = 'none';

    try {
        const response = await fetch('/api/claude/parse-continuous-beam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });

        const result = await response.json();

        if (result.success) {
            parsedData = result.data;
            document.getElementById('parsed-json').textContent = JSON.stringify(parsedData, null, 2);
            previewDiv.style.display = 'block';
        } else {
            errorDiv.textContent = result.error;
            errorDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Parse error:', error);
        errorDiv.textContent = '네트워크 오류가 발생했습니다.';
        errorDiv.style.display = 'block';
    } finally {
        parseBtn.disabled = false;
        parseBtn.textContent = 'Claude로 변환';
    }
}


/**
 * Submit analysis with parsed natural language data
 */
async function submitNatural() {
    const input = document.getElementById('natural-input').value.trim();
    if (!input) {
        alert('연속보 설명을 입력해주세요.');
        return;
    }

    const submitBtn = document.getElementById('natural-submit-btn');
    submitBtn.disabled = true;
    submitBtn.textContent = '처리 중...';

    try {
        // If not already parsed, parse first
        if (!parsedData) {
            const parseResponse = await fetch('/api/claude/parse-continuous-beam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input })
            });

            const parseResult = await parseResponse.json();

            if (!parseResult.success) {
                alert('변환 실패: ' + parseResult.error);
                submitBtn.disabled = false;
                submitBtn.textContent = '해석 실행';
                return;
            }

            parsedData = parseResult.data;
        }

        // Submit the parsed data
        await submitAnalysis(parsedData);
    } catch (error) {
        console.error('Submit error:', error);
        alert('오류가 발생했습니다: ' + error.message);
        submitBtn.disabled = false;
        submitBtn.textContent = '해석 실행';
    }
}


/**
 * Submit analysis to API
 */
async function submitAnalysis(data) {
    const response = await fetch('/api/continuous-beam/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (response.ok) {
        window.location.href = `/continuous-beam/jobs/${result.job_id}/status`;
    } else {
        alert(`Error: ${result.detail || 'Failed to create job'}`);
    }
}


/**
 * Add a span row
 */
function addSpan() {
    const container = document.getElementById('spans-container');
    const index = container.children.length;

    const row = document.createElement('div');
    row.className = 'span-row';
    row.dataset.index = index;
    row.innerHTML = `
        <label>경간 ${index + 1}</label>
        <input type="number" class="span-input" value="6.0" step="0.1" min="0.5" max="50" required>
        <span>m</span>
        <button type="button" class="btn-remove" onclick="removeSpan(this)">삭제</button>
    `;
    container.appendChild(row);

    updateSpanLabels();
    updateSupportsUI();
    updateLoadSpanOptions();
}


/**
 * Remove a span row
 */
function removeSpan(btn) {
    const container = document.getElementById('spans-container');
    if (container.children.length <= 2) {
        alert('최소 2개의 경간이 필요합니다.');
        return;
    }

    btn.closest('.span-row').remove();
    updateSpanLabels();
    updateSupportsUI();
    updateLoadSpanOptions();
}


/**
 * Update span labels after add/remove
 */
function updateSpanLabels() {
    const rows = document.querySelectorAll('.span-row');
    rows.forEach((row, i) => {
        row.dataset.index = i;
        row.querySelector('label').textContent = `경간 ${i + 1}`;
        // Show delete button only if more than 2 spans
        const btn = row.querySelector('.btn-remove');
        btn.style.display = rows.length > 2 ? 'inline-block' : 'none';
    });
}


/**
 * Update supports UI based on number of spans
 */
function updateSupportsUI() {
    const spanCount = document.querySelectorAll('.span-row').length;
    const supportCount = spanCount + 1;
    const container = document.getElementById('supports-container');

    container.innerHTML = '';

    for (let i = 0; i < supportCount; i++) {
        const label = i === 0 ? 'A (좌단)' : i === supportCount - 1 ? `${String.fromCharCode(65 + i)} (우단)` : String.fromCharCode(65 + i);
        const div = document.createElement('div');
        div.className = 'support-item';
        div.innerHTML = `
            <label>지점 ${label}</label>
            <select class="support-select">
                <option value="pin" ${i === 0 ? 'selected' : ''}>핀 (Pin)</option>
                <option value="roller" ${i > 0 && i < supportCount - 1 ? 'selected' : ''}>롤러 (Roller)</option>
                <option value="fixed">고정 (Fixed)</option>
                <option value="free">자유 (Free)</option>
            </select>
            <label class="hinge-label">
                <input type="checkbox" class="hinge-check"> 힌지
            </label>
        `;
        container.appendChild(div);
    }
}


/**
 * Update load span select options
 */
function updateLoadSpanOptions() {
    const spanCount = document.querySelectorAll('.span-row').length;
    const selects = document.querySelectorAll('.load-span-select');

    selects.forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="all">전체 경간</option>';
        for (let i = 0; i < spanCount; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `경간 ${i + 1}`;
            select.appendChild(option);
        }
        // Restore selection if valid
        if (currentValue === 'all' || parseInt(currentValue) < spanCount) {
            select.value = currentValue;
        }
    });
}


/**
 * Add a load card
 */
function addLoad() {
    const container = document.getElementById('loads-container');
    const index = container.children.length;

    const card = document.createElement('div');
    card.className = 'load-card';
    card.dataset.index = index;
    card.innerHTML = `
        <div class="load-card-header">
            <span>하중 ${index + 1}</span>
            <button type="button" class="btn-remove" onclick="removeLoad(this)">삭제</button>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>적용 경간</label>
                <select class="load-span-select">
                    <option value="all" selected>전체 경간</option>
                </select>
            </div>
            <div class="form-group">
                <label>하중 타입</label>
                <select class="load-type-select" onchange="updateLoadInputs(this)">
                    <option value="uniform" selected>등분포하중</option>
                    <option value="point">집중하중</option>
                    <option value="triangular">삼각형 하중</option>
                    <option value="partial_uniform">부분 등분포</option>
                </select>
            </div>
        </div>
        <div class="load-value-inputs">
            <div class="form-row uniform-inputs">
                <div class="form-group">
                    <label>하중 값 (kN/m)</label>
                    <input type="number" class="load-value" value="20" step="0.1">
                </div>
            </div>
        </div>
    `;
    container.appendChild(card);
    updateLoadSpanOptions();
    updateLoadLabels();
}


/**
 * Remove a load card
 */
function removeLoad(btn) {
    const container = document.getElementById('loads-container');
    if (container.children.length <= 1) {
        alert('최소 1개의 하중이 필요합니다.');
        return;
    }
    btn.closest('.load-card').remove();
    updateLoadLabels();
}


/**
 * Update load card labels
 */
function updateLoadLabels() {
    const cards = document.querySelectorAll('.load-card');
    cards.forEach((card, i) => {
        card.dataset.index = i;
        card.querySelector('.load-card-header span').textContent = `하중 ${i + 1}`;
        const btn = card.querySelector('.btn-remove');
        btn.style.display = cards.length > 1 ? 'inline-block' : 'none';
    });
}


/**
 * Update load input fields based on load type
 */
function updateLoadInputs(select) {
    const card = select.closest('.load-card');
    const inputsDiv = card.querySelector('.load-value-inputs');
    const loadType = select.value;

    let html = '';
    if (loadType === 'uniform') {
        html = `
            <div class="form-row uniform-inputs">
                <div class="form-group">
                    <label>하중 값 (kN/m)</label>
                    <input type="number" class="load-value" value="20" step="0.1">
                </div>
            </div>
        `;
    } else if (loadType === 'point') {
        html = `
            <div class="form-row point-inputs">
                <div class="form-group">
                    <label>하중 값 (kN)</label>
                    <input type="number" class="load-value" value="50" step="0.1">
                </div>
                <div class="form-group">
                    <label>위치 (경간 내 m)</label>
                    <input type="number" class="load-location" value="3.0" step="0.1">
                </div>
            </div>
        `;
    } else if (loadType === 'triangular') {
        html = `
            <div class="form-row triangular-inputs">
                <div class="form-group">
                    <label>시작값 (kN/m)</label>
                    <input type="number" class="load-value" value="0" step="0.1">
                </div>
                <div class="form-group">
                    <label>끝값 (kN/m)</label>
                    <input type="number" class="load-value-end" value="20" step="0.1">
                </div>
            </div>
        `;
    } else if (loadType === 'partial_uniform') {
        html = `
            <div class="form-row partial-inputs">
                <div class="form-group">
                    <label>하중 값 (kN/m)</label>
                    <input type="number" class="load-value" value="20" step="0.1">
                </div>
                <div class="form-group">
                    <label>시작 (m)</label>
                    <input type="number" class="load-start" value="1.0" step="0.1">
                </div>
                <div class="form-group">
                    <label>끝 (m)</label>
                    <input type="number" class="load-end" value="5.0" step="0.1">
                </div>
            </div>
        `;
    }

    inputsDiv.innerHTML = html;
}


/**
 * Build input data from form
 */
function buildInputData() {
    // Collect spans
    const spans = [];
    document.querySelectorAll('.span-input').forEach(input => {
        spans.push(parseFloat(input.value));
    });

    if (spans.length < 2) {
        throw new Error('최소 2개의 경간이 필요합니다.');
    }

    // Collect supports
    const supports = [];
    document.querySelectorAll('.support-select').forEach(select => {
        supports.push(select.value);
    });

    // Collect hinges
    const hinges = [];
    document.querySelectorAll('.hinge-check').forEach((check, i) => {
        if (check.checked) hinges.push(i);
    });

    // Collect loads
    const loads = [];
    document.querySelectorAll('.load-card').forEach(card => {
        const spanSelect = card.querySelector('.load-span-select');
        const typeSelect = card.querySelector('.load-type-select');
        const loadType = typeSelect.value;

        const load = {
            type: loadType
        };

        // Set span_index if not "all"
        if (spanSelect.value !== 'all') {
            load.span_index = parseInt(spanSelect.value);
        }

        // Get values based on type
        if (loadType === 'uniform') {
            load.value = parseFloat(card.querySelector('.load-value').value);
        } else if (loadType === 'point') {
            load.value = parseFloat(card.querySelector('.load-value').value);
            load.location = parseFloat(card.querySelector('.load-location').value);
        } else if (loadType === 'triangular') {
            load.value = parseFloat(card.querySelector('.load-value').value);
            load.value_end = parseFloat(card.querySelector('.load-value-end').value);
        } else if (loadType === 'partial_uniform') {
            load.value = parseFloat(card.querySelector('.load-value').value);
            load.start = parseFloat(card.querySelector('.load-start').value);
            load.end = parseFloat(card.querySelector('.load-end').value);
        }

        loads.push(load);
    });

    return {
        spans: spans,
        loads: loads,
        supports: supports,
        hinges: hinges.length > 0 ? hinges : null,
        section_name: document.getElementById('section_name').value,
        material_name: document.getElementById('material_name').value,
        num_elements_per_span: parseInt(document.getElementById('num_elements').value),
        deflection_limit: parseInt(document.getElementById('deflection_limit').value)
    };
}


/**
 * Edit parsed data in direct mode
 */
function editParsed() {
    if (!parsedData) return;

    // Clear current spans and rebuild
    const spansContainer = document.getElementById('spans-container');
    spansContainer.innerHTML = '';

    if (parsedData.spans && parsedData.spans.length > 0) {
        parsedData.spans.forEach((span, i) => {
            const row = document.createElement('div');
            row.className = 'span-row';
            row.dataset.index = i;
            row.innerHTML = `
                <label>경간 ${i + 1}</label>
                <input type="number" class="span-input" value="${span}" step="0.1" min="0.5" max="50" required>
                <span>m</span>
                <button type="button" class="btn-remove" onclick="removeSpan(this)" ${parsedData.spans.length <= 2 ? 'style="display:none;"' : ''}>삭제</button>
            `;
            spansContainer.appendChild(row);
        });
    }

    updateSupportsUI();
    updateLoadSpanOptions();

    // Fill other fields
    if (parsedData.section_name) document.getElementById('section_name').value = parsedData.section_name;
    if (parsedData.material_name) document.getElementById('material_name').value = parsedData.material_name;
    if (parsedData.num_elements_per_span) document.getElementById('num_elements').value = parsedData.num_elements_per_span;
    if (parsedData.deflection_limit) document.getElementById('deflection_limit').value = parsedData.deflection_limit;

    // Switch to direct mode
    switchMode('direct');
}
