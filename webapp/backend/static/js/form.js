/**
 * Frame2D Analysis Form Handler
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
        1: "3층 2경간 철골 건물입니다. 층고는 3.5m이고 경간은 6m입니다. 바닥 고정하중은 각 층에 20kN/m를 적용합니다.",
        2: "5층짜리 단일경간 건물을 해석하려고 합니다. 층고 3.2m, 경간 8m입니다. 지진하중은 1층부터 50, 60, 70, 80, 100kN을 적용합니다. 하중조합은 1.2DL+1.0EQX로 합니다.",
        3: "2층 3경간 건물, 층고 4m, 경간은 5m, 6m, 5m입니다. 기둥은 H-400x400, 보는 H-500x200을 사용하고 고정단으로 설계합니다. DL은 25kN/m입니다."
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
        alert('구조물 설명을 입력해주세요.');
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
        const response = await fetch('/api/claude/parse', {
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
        alert('구조물 설명을 입력해주세요.');
        return;
    }

    const submitBtn = document.getElementById('natural-submit-btn');
    submitBtn.disabled = true;
    submitBtn.textContent = '처리 중...';

    try {
        // If not already parsed, parse first
        if (!parsedData) {
            const parseResponse = await fetch('/api/claude/parse', {
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
    document.getElementById('stories').value = parsedData.stories.join(', ');
    document.getElementById('bays').value = parsedData.bays.join(', ');

    if (parsedData.column_section) {
        document.getElementById('column_section').value = parsedData.column_section;
    }
    if (parsedData.beam_section) {
        document.getElementById('beam_section').value = parsedData.beam_section;
    }
    if (parsedData.material_name) {
        document.getElementById('material_name').value = parsedData.material_name;
    }
    if (parsedData.supports) {
        document.getElementById('supports').value = parsedData.supports;
    }
    if (parsedData.num_elements_per_member) {
        document.getElementById('num_elements').value = parsedData.num_elements_per_member;
    }

    // Switch to direct mode
    switchMode('direct');
}


/**
 * Submit analysis to API
 */
async function submitAnalysis(data) {
    const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (response.ok) {
        window.location.href = `/jobs/${result.job_id}/status`;
    } else {
        alert(`Error: ${result.detail || 'Failed to create job'}`);
    }
}


/**
 * Build input data from form
 */
function buildInputData() {
    const storiesInput = document.getElementById('stories').value;
    const baysInput = document.getElementById('bays').value;

    const stories = storiesInput.split(',').map(s => parseFloat(s.trim())).filter(x => !isNaN(x));
    const bays = baysInput.split(',').map(s => parseFloat(s.trim())).filter(x => !isNaN(x));

    const columnSection = document.getElementById('column_section').value;
    const beamSection = document.getElementById('beam_section').value;
    const materialName = document.getElementById('material_name').value;
    const supports = document.getElementById('supports').value;
    const numElements = parseInt(document.getElementById('num_elements').value);

    const loadCases = {};
    const loadCaseCards = document.querySelectorAll('.load-case-card');

    loadCaseCards.forEach(card => {
        const caseName = card.dataset.case;
        const loads = [];

        const loadItems = card.querySelectorAll('.load-item');
        loadItems.forEach(item => {
            const type = item.querySelector('.load-type').value;
            const story = parseInt(item.querySelector('.load-story').value);
            const value = parseFloat(item.querySelector('.load-value').value);

            if (!isNaN(story) && !isNaN(value)) {
                if (type === 'floor') {
                    loads.push({ type: 'floor', story: story, value: value });
                } else if (type === 'lateral') {
                    loads.push({ type: 'lateral', story: story, fx: value });
                }
            }
        });

        if (loads.length > 0) {
            loadCases[caseName] = loads;
        }
    });

    const loadCombinations = {};
    const comboItems = document.querySelectorAll('.combo-item');

    comboItems.forEach(item => {
        const name = item.querySelector('.combo-name').value.trim();
        const formula = item.querySelector('.combo-formula').value.trim();

        if (name && formula) {
            const factors = {};
            formula.split(',').forEach(part => {
                const [caseName, factorStr] = part.split(':').map(s => s.trim());
                if (caseName && factorStr) {
                    factors[caseName] = parseFloat(factorStr);
                }
            });
            if (Object.keys(factors).length > 0) {
                loadCombinations[name] = factors;
            }
        }
    });

    return {
        stories,
        bays,
        column_section: columnSection,
        beam_section: beamSection,
        material_name: materialName,
        supports,
        num_elements_per_member: numElements,
        load_cases: loadCases,
        load_combinations: Object.keys(loadCombinations).length > 0 ? loadCombinations : null
    };
}


/**
 * Add a new load to a case
 */
function addLoad(caseName) {
    const container = document.getElementById(`loads-${caseName}`);
    const newItem = document.createElement('div');
    newItem.className = 'load-item';
    newItem.innerHTML = `
        <select class="load-type">
            <option value="floor">Floor Load</option>
            <option value="lateral">Lateral Load</option>
        </select>
        <input type="number" class="load-story" placeholder="Story" value="1" min="1">
        <input type="number" class="load-value" placeholder="Value" value="10" step="0.1">
        <button type="button" class="btn-remove" onclick="removeLoad(this)">X</button>
    `;
    container.appendChild(newItem);
}


/**
 * Remove a load item
 */
function removeLoad(btn) {
    const item = btn.closest('.load-item');
    const container = item.parentElement;

    if (container.querySelectorAll('.load-item').length > 1) {
        item.remove();
    }
}


/**
 * Add a new load case
 */
let caseCounter = 0;
function addLoadCase() {
    caseCounter++;
    const caseName = `CASE${caseCounter}`;
    const container = document.getElementById('load-cases-container');

    const card = document.createElement('div');
    card.className = 'load-case-card';
    card.dataset.case = caseName;
    card.innerHTML = `
        <div class="load-case-header">
            <h3>${caseName}</h3>
            <button type="button" class="btn-remove" onclick="removeLoadCase(this)">Remove</button>
        </div>
        <div class="load-items" id="loads-${caseName}">
            <div class="load-item">
                <select class="load-type">
                    <option value="floor" selected>Floor Load</option>
                    <option value="lateral">Lateral Load</option>
                </select>
                <input type="number" class="load-story" placeholder="Story" value="1" min="1">
                <input type="number" class="load-value" placeholder="Value" value="10" step="0.1">
                <button type="button" class="btn-remove" onclick="removeLoad(this)">X</button>
            </div>
        </div>
        <button type="button" class="btn-secondary btn-sm" onclick="addLoad('${caseName}')">+ Add Load</button>
    `;
    container.appendChild(card);
}


/**
 * Remove a load case
 */
function removeLoadCase(btn) {
    const card = btn.closest('.load-case-card');
    const container = document.getElementById('load-cases-container');

    if (container.querySelectorAll('.load-case-card').length > 1) {
        card.remove();
    }
}


/**
 * Add a new combination
 */
function addCombo() {
    const container = document.getElementById('combinations-container');
    const item = document.createElement('div');
    item.className = 'combo-item';
    item.innerHTML = `
        <input type="text" class="combo-name" placeholder="Name">
        <input type="text" class="combo-formula" placeholder="Formula: DL:1.4">
        <button type="button" class="btn-remove" onclick="removeCombo(this)">X</button>
    `;
    container.appendChild(item);
}


/**
 * Remove a combination
 */
function removeCombo(btn) {
    btn.closest('.combo-item').remove();
}
