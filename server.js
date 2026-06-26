require('dotenv').config();

const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const app = express();
const { execSync } = require('child_process');
const pty = require('node-pty');
const { Server } = require('socket.io');

const PORT = 3000;

app.use(express.json());
app.use('/static', express.static('static'));

const server = http.createServer(app);
const io = new Server(server);

// --- DATABASE SIMULATION ---
const DB_FILE = 'things-database.json';
if (!fs.existsSync(DB_FILE)) fs.writeFileSync(DB_FILE, JSON.stringify([], null, 2));

function saveToDatabase(thingDescription) {
    const db = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    db.push({ id: thingDescription.id || `urn:thing:${Date.now()}`, ...thingDescription, savedAt: new Date() });
    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
}

// --- VALIDATION FUNCTION (Direct API, no shell commands) ---
async function validateTD(tdJson, tempFilePath = 'temp-td.json') {
    try {
        // Write the TD to a temporary file
        fs.writeFileSync(tempFilePath, JSON.stringify(tdJson, null, 2));
        // Find the correct path to the CLI tool
        let cliPath;
        const possiblePaths = [
            'node_modules/@thing-description-playground/cli/index.js',
            './node_modules/@thing-description-playground/cli/index.js',
            '../node_modules/@thing-description-playground/cli/index.js',
        ];
        for (const p of possiblePaths) {
            if (fs.existsSync(p)) {
                cliPath = p;
                break;
            }
        }
        if (!cliPath) {
            // Try to find it using npm root
            try {
                const npmRoot = execSync('npm root -g', { encoding: 'utf8' }).trim();
                const globalCliPath = path.join(npmRoot, '@thing-description-playground/cli/index.js');
                if (fs.existsSync(globalCliPath)) {
                    cliPath = globalCliPath;
                }
            } catch(e) {}
        }
        
        if (!cliPath) {
            console.warn('CLI tool not found, using basic validation');
            return basicValidation(tdJson);
        }
        
        // Execute the validation command that you know works
        const command = `node "${cliPath}" -i "${tempFilePath}" --no-jsonld`;
        try {
            // Run validation
            const stdout = execSync(command, { 
                encoding: 'utf8',
                stdio: 'pipe'
            });
            // Clean up temp file
            fs.unlinkSync(tempFilePath);
            // Check if validation passed
            // The command returns exit code 0 on success
            return { 
                valid: true, 
                errors: [],
                output: stdout
            };
        } catch (validationError) {
            // Validation failed - capture the error output
            const stderr = validationError.stderr?.toString() || '';
            const stdout = validationError.stdout?.toString() || '';
            const errorMessage = stderr || stdout || validationError.message;
            
            // Clean up temp file
            if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
            }
            
            // Parse the validation errors from the output
            const errors = parseValidationErrors(errorMessage);
            
            return { 
                valid: false, 
                errors: errors,
                rawOutput: errorMessage
            };
        }
        
    } catch (err) {
        // Clean up on error
        if (fs.existsSync(tempFilePath)) {
            fs.unlinkSync(tempFilePath);
        }
        console.warn('\nValidation error:', err.message);
        return { 
            valid: false, 
            errors: [`Validation system error: ${err.message}`],
            isFallback: true
        };
    }
}

// --- Helper: Parse validation errors from CLI output ---
function parseValidationErrors(output) {
    const errors = [];
    
    // Look for common error patterns in the CLI output
    const errorLines = output.split('\n').filter(line => 
        line.includes('error') || 
        line.includes('Error') || 
        line.includes('ERROR') ||
        line.includes('missing') ||
        line.includes('required') ||
        line.includes('invalid')
    );
    
    if (errorLines.length > 0) {
        errors.push(...errorLines.slice(0, 10)); // Limit to 10 errors
    } else {
        errors.push(output.substring(0, 500)); // Return first 500 chars if no clear errors
    }
    
    return errors;
}

// --- Basic validation fallback (still useful) ---
function basicValidation(tdJson) {
    const errors = [];
    
    // Check required fields according to WoT TD spec
    const requiredFields = ['@context', 'id', 'title', 'security', 'securityDefinitions'];
    for (const field of requiredFields) {
        if (!tdJson[field]) {
            errors.push(`Missing required field: "${field}"`);
        }
    }
    
    // Check @context value
    if (tdJson['@context'] && typeof tdJson['@context'] === 'string') {
        if (!tdJson['@context'].includes('https://www.w3.org/2022/wot/td')) {
            errors.push('@context should be a valid W3C TD context URL (e.g., https://www.w3.org/2022/wot/td/v1.1)');
        }
    }
    
    // Check security structure
    if (tdJson.security && !Array.isArray(tdJson.security)) {
        errors.push('"security" field must be an array');
    }
    
    if (tdJson.securityDefinitions && typeof tdJson.securityDefinitions !== 'object') {
        errors.push('"securityDefinitions" must be an object');
    }
    
    return { 
        valid: errors.length === 0, 
        errors: errors,
        isBasicValidation: true 
    };
}

// --- Install check helper ---
function checkCLIInstallation() {
    try {
        const npmRoot = execSync('npm root -g', { encoding: 'utf8' }).trim();
        const cliPath = path.join(npmRoot, '@thing-description-playground/cli/index.js');
        
        if (fs.existsSync(cliPath)) {
            console.log(`\nCLI found at: ${cliPath}`);
            return true;
        }
        
        // Check local installation
        const localCliPath = 'node_modules/@thing-description-playground/cli/index.js';
        if (fs.existsSync(localCliPath)) {
            console.log(`\nCLI found locally at: ${localCliPath}`);
            return true;
        }
        
        console.warn('\nCLI not found, will use basic validation');
        return false;
    } catch (err) {
        console.warn('\nCould not check CLI installation:', err.message);
        return false;
    }
}

// --- LLM ROUTER ---
async function callLLM(model, systemPrompt, userPrompt) {
    // 1. Resolve API Key based on the model
    let apiKey;
    if (model === 'gemini') {
        apiKey = process.env.GEMINI_API_KEY;
    } else if (model === 'openai') {
        apiKey = process.env.OPENAI_API_KEY;
    } else if (model === 'deepseek') {
        apiKey = process.env.OPENROUTER_API_KEY;
    } else if (model === 'ollama') {
        apiKey = process.env.OLLAMA_API_KEY; 
    } else if (model === 'mistral') {
        apiKey = process.env.MISTRAL_API_KEY;
    }

    if (!apiKey) {
        throw new Error(`Missing API Key for ${model}. Please check your .env file.`);
    }

    // 2. Handle Gemini API
    if (model === 'gemini') {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{ parts: [{ text: `${systemPrompt}\n\nUser Input: ${userPrompt}` }] }],
                generationConfig: { responseMimeType: "application/json" }
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Gemini API HTTP ${res.status}: ${errorText}`);
        }

        const data = await res.json();
        if (!data.candidates || data.candidates.length === 0) {
            throw new Error(`Gemini returned empty candidates. Response: ${JSON.stringify(data)}`);
        }
        return data.candidates[0].content.parts[0].text;

    // 3. Handle ChatGPT (OpenAI) API
    } else if (model === 'openai') {
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4o', 
                response_format: { type: "json_object" },
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`ChatGPT API HTTP ${res.status}: ${errorText}`);
        }

        const data = await res.json();
        if (!data.choices || data.choices.length === 0) {
            throw new Error(`ChatGPT returned empty choices. Response: ${JSON.stringify(data)}`);
        }
        return data.choices[0].message.content;

    // 4. Handle DeepSeek API
    } else if (model === 'deepseek') {
        const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'openrouter/free', 
                response_format: { type: "json_object" }, 
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            })
        });

        const data = await res.json();
        if (!data.choices || data.choices.length === 0) {
            throw new Error(`Deepseek returned empty choices. Response: ${JSON.stringify(data)}`);
        }
        return data.choices[0].message.content;

    // 5. Handle Online/Cloud Ollama API
    } else if (model === 'ollama') {
        const res = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'llama3.1', // Change this to your locally pulled model (e.g., 'deepseek-r1')
                format: 'json',  // Forces structured JSON output matching your other providers
                stream: false,   // Ensures we get a single complete response instead of a stream
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Ollama Local API HTTP ${res.status}: ${errorText}`);
        }

        const data = await res.json();
        if (!data.message || !data.message.content) {
            throw new Error(`Ollama returned empty content. Response: ${JSON.stringify(data)}`);
        }
        return data.message.content;
    // 6. Handle Mistral API
    } else if (model === 'mistral') {
        const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json', 
                'Authorization': `Bearer ${apiKey}` 
            },
            body: JSON.stringify({
                model: 'mistral-large-latest',
                response_format: { type: "json_object" },
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Mistral API HTTP ${res.status}: ${errorText}`);
        }

        const data = await res.json();
        if (!data.choices || data.choices.length === 0) {
            throw new Error(`Mistral returned empty choices. Response: ${JSON.stringify(data)}`);
        }
        return data.choices[0].message.content;
        
    } else {
        throw new Error(`Unsupported model provider: ${model}`);
    }
}

// --- CORE AGENTIC LOOP ---
app.post('/api/generate', async (req, res) => {
    const { prompt, model } = req.body;
    let systemPrompt = `You are a W3C Thing Description (TD) expert. Output ONLY a valid JSON object matching the W3C WoT TD standard version 1.1.
    
Required fields:
- "@context": Must be "https://www.w3.org/2022/wot/td/v1.1"
- "id": Unique identifier (URN or URL)
- "title": Human-readable title
- "security": Array of security scheme names (e.g., ["no_sc"])
- "securityDefinitions": Object defining security schemes

Do NOT wrap in markdown code blocks. Output pure JSON only.`;

    let currentPrompt = prompt;
    let attempts = 0;
    const maxAttempts = 10;
    const logs = [];

    while (attempts < maxAttempts) {
        attempts++;
        logs.push(`[Attempt ${attempts}/${maxAttempts}] Generating with ${model}...`);

        try {
            const llmOutput = await callLLM(model, systemPrompt, currentPrompt);

            // Improved JSON extraction - handles multiple formats
            let jsonString = llmOutput;

            // Remove markdown code blocks if present
            jsonString = jsonString.replace(/```json\s*/g, '').replace(/```\s*/g, '');

            // Find the first complete JSON object
            const jsonMatch = jsonString.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
                throw new Error('No JSON object found in response');
            }
            jsonString = jsonMatch[0];

            // Parse JSON
            let parsedJson;
            try {
                parsedJson = JSON.parse(jsonString);
            } catch (parseErr) {
                // Try to fix common JSON issues
                jsonString = jsonString.replace(/(['"])?([a-zA-Z0-9_]+)(['"])?:/g, '"$2":');
                parsedJson = JSON.parse(jsonString);
            }

            logs.push(`[Attempt ${attempts}] JSON parsed successfully, validating...`);

            // Validate the TD
            const validation = await validateTD(parsedJson);
            if (validation.valid) {
                saveToDatabase(parsedJson);
                logs.push(`[Success] Valid TD generated!`);
                return res.json({ success: true, logs, data: parsedJson });
            } else {
                logs.push(`Validation failed: ${validation.errors.join(', ')}`);
                currentPrompt = `Your TD failed validation. Errors:\n${validation.errors.join('\n')}\n\nPlease fix and return a valid TD.`;
            }

        } catch (err) {
            logs.push(`[Attempt ${attempts} Error] ${err.message}`);

            // Send parsing errors back to LLM
            currentPrompt = `Your previous output failed to parse as valid JSON. Error: ${err.message}\n\nPlease generate a valid JSON object only, no markdown or extra text. Include the required fields.`;
        }
    }

    logs.push(`Failed to generate valid TD after ${maxAttempts} attempts`);
    return res.status(422).json({ success: false, logs, message: "Failed to generate valid TD within limits." });
});

// --- FRONTEND ---
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

app.get('/saved-things', (req, res) => {
    res.sendFile(path.join(__dirname, 'static', 'saved-things.html')); 
    // Adjust 'static' if your HTML files live in a different folder
});

app.get('/api/things', (req, res) => {
    const filePath = path.join(__dirname, 'things-database.json');
    
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            // If the file doesn't exist yet, return an empty array instead of crashing
            if (err.code === 'ENOENT') {
                return res.json([]);
            }
            return res.status(500).json({ error: 'Failed to read database' });
        }
        try {
            res.json(JSON.parse(data));
        } catch (parseErr) {
            res.status(500).json({ error: 'Database file corrupted' });
        }
    });
});

app.get('/search-terminal', (req, res) => {
    res.sendFile(path.join(__dirname, 'static', 'search-terminal.html'));
});

io.on('connection', (socket) => {
    const shell = 'powershell.exe';
    const args = [
        '-NoProfile', 
        '-ExecutionPolicy', 'Bypass', 
        '-Command', 
        `& C:\\Python312\\python.exe wot-rag.py`
    ];

    const pyTerminal = pty.spawn(shell, args, {
        name: 'xterm-color',
        cols: 80,
        rows: 24,
        cwd: __dirname, // Sets working dir to the application directory
        env: process.env
    });

    // Stream everything the script logs directly to the user's web browser
    pyTerminal.onData((data) => {
        socket.emit('terminal-output', data);
    });

    // Capture incoming keystrokes from the web-UI
    socket.on('terminal-input', (data) => {
        pyTerminal.write(data);
    });

    // Clean up when page closes
    socket.on('disconnect', () => {
        pyTerminal.kill();
    });
});

app.delete('/api/things/:index', (req, res) => {
    const index = parseInt(req.params.index, 10);
    const filePath = path.join(__dirname, 'things-database.json');
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            return res.status(500).json({ success: false, error: 'Failed to read database' });
        }
        try {
            let things = JSON.parse(data);
            
            if (index < 0 || index >= things.length) {
                return res.status(400).json({ success: false, error: 'Invalid item index' });
            }
            // Remove the targeted item from the array
            things.splice(index, 1);
            // Write the updated array back to things-database.json
            fs.writeFile(filePath, JSON.stringify(things, null, 2), (writeErr) => {
                if (writeErr) {
                    return res.status(500).json({ success: false, error: 'Failed to save updates' });
                }
                res.json({ success: true, message: 'Thing description deleted successfully' });
            });
        } catch (parseErr) {
            res.status(500).json({ success: false, error: 'Database file corrupted' });
        }
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`Database file: ${DB_FILE}`);
});