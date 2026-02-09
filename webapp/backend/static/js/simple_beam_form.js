/**
 * Simple Beam Analysis Form Handler
 * Supports both Natural Language (Claude) and Direct Input modes
 */

// Store parsed data from Claude
let parsedData = null;

document.addEventListener('DOMContentLoaded', () => {
    // Check Claude API status
    checkClaudeStatus();

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
            alert('Network error. Please try again.');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Analysis';
        }
    });

    // Initialize load input visibility
    updateLoadInputs();
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
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Update content
    document.getElementById('natural-mode').classList.toggle('active', mode === 'natural');
    document.getElementById('direct-mode').classList.toggle('active', mode === 'direct');
}


/**
 * Fill example prompt
 */
function fillExample(num) {
    const examples = {
        1: "길이 8m 단순지지 보에 등분포하중 15kN/m가 작용합니다. H-400x200 단면, SS275 재료를 사용합니다.",
        2: "5m 캔틸레버 보의 끝단에 집중하중 50kN이 작용합니다. H-500x200 단면을 사용하고 처짐 제한은 L/300입니다.",
        3: "10m 양단고정 보에 중앙 집중하중 100kN이 작용합니다. SS400 강재, H-600x200 단면입니다."
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
        alert('보 설명을 입력해주세요.');
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
        const response = await fetch('/api/claude/parse-beam', {
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
        alert('보 설명을 입력해주세요.');
        return;
    }

    const submitBtn = document.getElementById('natural-submit-btn');
    submitBtn.disabled = true;
    submitBtn.textContent = '처리 중...';

    try {
        // If not already parsed, parse first
        if (!parsedData) {
            const parseResponse = await fetch('/api/claude/parse-beam', {
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
 * Edit parsed data in direct mode
 */
function editParsed() {
    if (!parsedData) return;

    // Fill the form with parsed data
    if (parsedData.span) document.getElementById('span').value = parsedData.span;
    if (parsedData.support_type) document.getElementById('support_type').value = parsedData.support_type;
    if (parsedData.section_name) document.getElementById('section_name').value = parsedData.section_name;
    if (parsedData.material_name) document.getElementById('material_name').value = parsedData.material_name;
    if (parsedData.load_type) {
        document.getElementById('load_type').value = parsedData.load_type;
        updateLoadInputs();
    }
    if (parsedData.load_value) document.getElementById('load_value').value = parsedData.load_value;
    if (parsedData.num_elements) document.getElementById('num_elements').value = parsedData.num_elements;
    if (parsedData.deflection_limit) document.getElementById('deflection_limit').value = parsedData.deflection_limit;

    // Point load specific
    if (parsedData.point_location) document.getElementById('point_location').value = parsedData.point_location;
    if (parsedData.point_load_value) document.getElementById('point_load_value').value = parsedData.point_load_value;

    // Partial load specific
    if (parsedData.load_start) document.getElementById('load_start').value = parsedData.load_start;
    if (parsedData.load_end) document.getElementById('load_end').value = parsedData.load_end;

    // Switch to direct mode
    switchMode('direct');
}


/**
 * Submit analysis to API
 */
async function submitAnalysis(data) {
    const response = await fetch('/api/simple-beam/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (response.ok) {
        window.location.href = `/simple-beam/jobs/${result.job_id}/status`;
    } else {
        alert(`Error: ${result.detail || 'Failed to create job'}`);
    }
}


/**
 * Build input data from form
 */
function buildInputData() {
    const span = parseFloat(document.getElementById('span').value);
    const supportType = document.getElementById('support_type').value;
    const sectionName = document.getElementById('section_name').value;
    const materialName = document.getElementById('material_name').value;
    const numElements = parseInt(document.getElementById('num_elements').value);
    const deflectionLimit = parseInt(document.getElementById('deflection_limit').value);
    const loadType = document.getElementById('load_type').value;

    const data = {
        span: span,
        support_type: supportType,
        section_name: sectionName,
        material_name: materialName,
        num_elements: numElements,
        deflection_limit: deflectionLimit,
        load_type: loadType
    };

    // Add load-specific values
    if (loadType === 'uniform' || loadType === 'triangular') {
        data.load_value = parseFloat(document.getElementById('load_value').value);
    } else if (loadType === 'point_center') {
        data.load_value = parseFloat(document.getElementById('point_load_value').value);
    } else if (loadType === 'point') {
        data.load_value = parseFloat(document.getElementById('point_load_value').value);
        data.point_location = parseFloat(document.getElementById('point_location').value);
    } else if (loadType === 'partial_uniform') {
        data.load_value = parseFloat(document.getElementById('partial_load_value').value);
        data.load_start = parseFloat(document.getElementById('load_start').value);
        data.load_end = parseFloat(document.getElementById('load_end').value);
    }

    return data;
}


/**
 * Update load input fields based on load type
 */
function updateLoadInputs() {
    const loadType = document.getElementById('load_type').value;

    // Hide all specific inputs
    document.getElementById('uniform-inputs').style.display = 'none';
    document.getElementById('point-inputs').style.display = 'none';
    document.getElementById('partial-inputs').style.display = 'none';

    // Show relevant inputs
    if (loadType === 'uniform' || loadType === 'triangular') {
        document.getElementById('uniform-inputs').style.display = 'flex';
    } else if (loadType === 'point_center' || loadType === 'point') {
        document.getElementById('point-inputs').style.display = 'flex';
        // Hide location for center point
        document.querySelector('#point-inputs .form-group:last-child').style.display =
            loadType === 'point' ? 'block' : 'none';
    } else if (loadType === 'partial_uniform') {
        document.getElementById('partial-inputs').style.display = 'flex';
    }
}
