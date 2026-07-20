document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const promptInput = document.getElementById("prompt-input");
    const rangeTemp = document.getElementById("range-temperature");
    const valTemp = document.getElementById("val-temperature");
    const rangeTopK = document.getElementById("range-top-k");
    const valTopK = document.getElementById("val-top-k");
    const rangeMaxTokens = document.getElementById("range-max-tokens");
    const valMaxTokens = document.getElementById("val-max-tokens");
    
    const btnGenerate = document.getElementById("btn-generate");
    const btnClear = document.getElementById("btn-clear");
    const btnCopy = document.getElementById("btn-copy");
    const btnSpinner = document.getElementById("btn-spinner");
    const outputDisplay = document.getElementById("output-display");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    // Stats Elements
    const statParams = document.getElementById("stat-params");
    const statEmbed = document.getElementById("stat-embed");
    const statArchitecture = document.getElementById("stat-architecture");
    const statBlock = document.getElementById("stat-block");
    // Presets
    const presetButtons = document.querySelectorAll(".preset-btn");

    // Update range value displays dynamically
    rangeTemp.addEventListener("input", () => { valTemp.textContent = rangeTemp.value; });
    rangeTopK.addEventListener("input", () => { valTopK.textContent = rangeTopK.value; });
    rangeMaxTokens.addEventListener("input", () => { valMaxTokens.textContent = rangeMaxTokens.value; });

    // Handle suggested presets
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            promptInput.value = btn.getAttribute("data-prompt");
            promptInput.focus();
        });
    });

    // Clear handler
    btnClear.addEventListener("click", () => {
        promptInput.value = "";
        outputDisplay.textContent = "Press generate to start writing...";
        outputDisplay.classList.add("placeholder");
    });

    // Copy to clipboard handler
    btnCopy.addEventListener("click", () => {
        if (outputDisplay.classList.contains("placeholder")) return;
        
        navigator.clipboard.writeText(outputDisplay.textContent).then(() => {
            const originalText = btnCopy.textContent;
            btnCopy.textContent = "Copied!";
            btnCopy.style.borderColor = "var(--accent-emerald)";
            btnCopy.style.color = "var(--accent-emerald)";
            
            setTimeout(() => {
                btnCopy.textContent = originalText;
                btnCopy.style.borderColor = "";
                btnCopy.style.color = "";
            }, 1500);
        }).catch((error) => {
            console.error("Unable to copy generated text:", error);
        });
    });

    const selectCheckpoint = document.getElementById("select-checkpoint");

    // Fetch and populate checkpoints dropdown list
    async function loadCheckpointsList() {
        try {
            const response = await fetch("/api/checkpoints");
            const data = await response.json();
            
            if (data.checkpoints && data.checkpoints.length > 0) {
                selectCheckpoint.innerHTML = "";
                data.checkpoints.forEach(cp => {
                    const opt = document.createElement("option");
                    opt.value = cp;
                    opt.textContent = cp;
                    selectCheckpoint.appendChild(opt);
                });
            } else {
                selectCheckpoint.innerHTML = '<option value="">No checkpoints found</option>';
            }
        } catch (error) {
            console.error("Failed to load checkpoints list:", error);
        }
    }

    // Handle changing the active checkpoint
    selectCheckpoint.addEventListener("change", async () => {
        const selectedCp = selectCheckpoint.value;
        if (!selectedCp) return;

        btnGenerate.disabled = true;
        statusDot.className = "status-dot orange";
        statusText.textContent = "Loading model...";
        
        try {
            const response = await fetch("/api/load-checkpoint", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ checkpoint: selectedCp })
            });
            const data = await response.json();
            
            if (response.ok) {
                statusDot.className = "status-dot green";
                statusText.textContent = "Ready";
                
                // Update model stats grid dynamically
                const paramsM = (data.total_params / 1000000).toFixed(2) + "M";
                statParams.textContent = paramsM;
                btnGenerate.disabled = false;
                
                // Refresh model specs as well
                loadModelInfo();
            } else {
                statusDot.className = "status-dot orange";
                statusText.textContent = "Error Loading";
                alert(`Failed to load checkpoint: ${data.detail}`);
            }
        } catch (error) {
            console.error("Error loading checkpoint:", error);
            statusDot.className = "status-dot orange";
            statusText.textContent = "Server Offline";
        }
    });

    // Fetch model info on load
    async function loadModelInfo() {
        try {
            const response = await fetch("/api/model-info");
            const data = await response.json();
            
            if (data.status === "ready") {
                statusDot.className = "status-dot green";
                statusText.textContent = "Ready";
                
                // Format parameter counts (e.g. 11.13M)
                const paramsM = (data.total_params / 1000000).toFixed(2) + "M";
                statParams.textContent = paramsM;
                statEmbed.textContent = data.embedding_dimension;
                statArchitecture.textContent = `${data.number_of_layers} / ${data.number_of_heads}`;
                statBlock.textContent = data.block_size;
                btnGenerate.disabled = false;
            } else {
                statusDot.className = "status-dot orange";
                statusText.textContent = "Not Trained";
                statParams.textContent = "—";
                statEmbed.textContent = "—";
                statArchitecture.textContent = "—";
                statBlock.textContent = "—";
                outputDisplay.textContent = "Warning: GPT-2 model checkpoint is missing. Please run 'train.py' to train the model first before generating.";
                outputDisplay.classList.remove("placeholder");
                btnGenerate.disabled = true;
            }
        } catch (error) {
            console.error("Failed to load model details:", error);
            statusDot.className = "status-dot orange";
            statusText.textContent = "Server Offline";
        }
    }

    // Load initial info and checkpoints list on startup
    loadModelInfo();
    loadCheckpointsList();

    // Streaming/Typewriter text effect
    let typingInterval = null;
    function typeText(fullText, promptText) {
        // Clear any active typing interval
        if (typingInterval) clearInterval(typingInterval);
        
        outputDisplay.innerHTML = "";
        outputDisplay.classList.remove("placeholder");
        
        // Highlight prompt in bold/color if it is at the start of generated text
        let promptSpan = null;
        let restText = fullText;
        if (fullText.startsWith(promptText)) {
            promptSpan = document.createElement("span");
            promptSpan.style.color = "var(--text-secondary)";
            promptSpan.style.fontWeight = "600";
            outputDisplay.appendChild(promptSpan);
            restText = fullText.slice(promptText.length);
        }

        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            if (promptSpan) promptSpan.textContent = promptText;
            outputDisplay.appendChild(document.createTextNode(restText));
            return;
        }

        let charIdx = 0;
        let promptIdx = 0;

        function step() {
            if (promptSpan && promptIdx < promptText.length) {
                promptSpan.textContent += promptText[promptIdx];
                promptIdx++;
            } else if (charIdx < restText.length) {
                outputDisplay.appendChild(document.createTextNode(restText[charIdx]));
                charIdx++;
            } else {
                clearInterval(typingInterval);
                typingInterval = null;
            }
            // Auto scroll container
            outputDisplay.scrollTop = outputDisplay.scrollHeight;
        }

        typingInterval = setInterval(step, 10);
    }

    // Generate handler
    btnGenerate.addEventListener("click", async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert("Please type a generation prompt first!");
            return;
        }

        // Disable UI controls
        btnGenerate.disabled = true;
        btnGenerate.setAttribute("aria-busy", "true");
        btnSpinner.style.display = "block";
        outputDisplay.textContent = "Thinking...";
        outputDisplay.classList.remove("placeholder");

        const payload = {
            prompt: prompt,
            max_tokens: parseInt(rangeMaxTokens.value),
            method: "top_k",
            temperature: parseFloat(rangeTemp.value),
            top_k: parseInt(rangeTopK.value)
        };

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server error");
            }

            const data = await response.json();
            typeText(data.generated, data.prompt);
        } catch (error) {
            console.error("Text generation failed:", error);
            outputDisplay.textContent = `Error: ${error.message}`;
            outputDisplay.classList.remove("placeholder");
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.setAttribute("aria-busy", "false");
            btnSpinner.style.display = "none";
        }
    });
});
