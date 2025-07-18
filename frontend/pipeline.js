// Updated API endpoints for single-service deployment
const API_ENDPOINTS = {
    // All requests now go through the main service (port 3000)
    target: '/api/target',
    query: '/api/query',
    poseOverlay: '/api/pose-overlay',
    health: '/health',
    upload: '/upload',
    process: '/process'
};

// Your existing API key
const API_KEY = '84fee1d4-c83b-46b6-9338-191036b7ec6c';

class PoseEstimationPipeline {
    constructor() {
        this.uploadedFiles = {
            ply: null,
            texture: null,
            rgb: null,
            mask: null,
            depth: null,
            intrinsics: null
        };

        this.currentPipeline = 'target';
        this.currentStep = 1;
        this.processing = false;
        this.results = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUploads();
        this.initializeInterface();
        this.checkServiceHealth();
    }

    // New method to check service health
    async checkServiceHealth() {
        try {
            const response = await fetch(API_ENDPOINTS.health, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${API_KEY}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const healthData = await response.json();
                this.addLog('success', `Service health check passed: ${healthData.status}`);
            } else {
                this.addLog('warning', 'Service health check failed, but continuing...');
            }
        } catch (error) {
            this.addLog('warning', `Health check error: ${error.message}`);
        }
    }

    setupEventListeners() {
        // Process button
        document.getElementById('processBtn').addEventListener('click', () => {
            this.startProcessing();
        });

        // Pipeline selector
        document.querySelectorAll('.pipeline-option').forEach(option => {
            option.addEventListener('click', (e) => {
                this.selectPipeline(e.target.closest('.pipeline-option').dataset.pipeline);
            });
        });

        // Download buttons
        document.getElementById('downloadPose').addEventListener('click', () => {
            this.downloadFile('pose_data.json', 'pose');
        });

        document.getElementById('downloadVisualization').addEventListener('click', () => {
            this.downloadFile('visualization.png', 'visualization');
        });

        document.getElementById('downloadDescriptors').addEventListener('click', () => {
            this.downloadFile('descriptors.npy', 'descriptors');
        });

        document.getElementById('downloadAll').addEventListener('click', () => {
            this.downloadFile('complete_results.zip', 'all');
        });
    }

    setupFileUploads() {
        const fileInputs = {
            ply: { id: 'plyFile', zone: 'plyUpload', status: 'plyStatus' },
            texture: { id: 'textureFile', zone: 'textureUpload', status: 'textureStatus' },
            rgb: { id: 'rgbFile', zone: 'rgbUpload', status: 'rgbStatus' },
            mask: { id: 'maskFile', zone: 'maskUpload', status: 'maskStatus' },
            depth: { id: 'depthFile', zone: 'depthUpload', status: 'depthStatus' },
            intrinsics: { id: 'intrinsicsFile', zone: 'intrinsicsUpload', status: 'intrinsicsStatus' }
        };

        Object.entries(fileInputs).forEach(([type, config]) => {
            const input = document.getElementById(config.id);
            const zone = document.getElementById(config.zone);
            const status = document.getElementById(config.status);

            // Click to upload
            zone.addEventListener('click', () => {
                if (type === 'texture') {
                    const toggle = document.getElementById('textureToggle');
                    if (toggle.checked) return;
                }
                input.click();
            });

            // File selection
            input.addEventListener('change', (e) => {
                this.handleFileUpload(type, e.target.files[0], zone, status);
            });

            // Drag and drop
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.style.borderColor = 'var(--cyber-secondary)';
            });

            zone.addEventListener('dragleave', () => {
                zone.style.borderColor = 'var(--cyber-primary)';
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.style.borderColor = 'var(--cyber-primary)';

                if (type === 'texture') {
                    const toggle = document.getElementById('textureToggle');
                    if (toggle.checked) return;
                }

                const file = e.dataTransfer.files[0];
                this.handleFileUpload(type, file, zone, status);
            });
        });

        // Texture toggle
        document.getElementById('textureToggle').addEventListener('change', (e) => {
            const textureZone = document.getElementById('textureUpload');
            const textureStatus = document.getElementById('textureStatus');

            if (e.target.checked) {
                textureZone.style.opacity = '0.5';
                textureZone.style.pointerEvents = 'none';
                this.uploadedFiles.texture = 'skipped';
                textureStatus.textContent = 'Texture embedded in PLY model';
                textureStatus.className = 'upload-status success';
            } else {
                textureZone.style.opacity = '1';
                textureZone.style.pointerEvents = 'auto';
                this.uploadedFiles.texture = null;
                textureStatus.textContent = '';
                textureStatus.className = 'upload-status';
            }

            this.updateProcessButton();
        });
    }

    async handleFileUpload(type, file, zone, status) {
        if (!file) return;

        // Validate file type
        const validTypes = {
            ply: ['.ply'],
            texture: ['.jpg', '.jpeg', '.png', '.bmp'],
            rgb: ['.zip'],
            mask: ['.zip'],
            depth: ['.zip'],
            intrinsics: ['.json']
        };

        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes[type].includes(fileExtension)) {
            status.textContent = `Invalid file type. Expected: ${validTypes[type].join(', ')}`;
            status.className = 'upload-status error';
            return;
        }

        // Store file locally
        this.uploadedFiles[type] = file;

        // Update UI immediately
        zone.classList.add('uploaded');
        status.textContent = `✓ ${file.name} (${this.formatFileSize(file.size)})`;
        status.className = 'upload-status success';

        // Update process button
        this.updateProcessButton();

        // Log upload
        this.addLog('info', `Uploaded ${type}: ${file.name}`);

        // Optional: Upload to server immediately
        try {
            await this.uploadFileToServer(type, file);
        } catch (error) {
            this.addLog('warning', `Upload to server failed: ${error.message}`);
        }
    }

    // New method to upload files to server
    async uploadFileToServer(type, file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);

        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${API_KEY}`
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        return await response.json();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    updateProcessButton() {
        const required = ['ply', 'rgb', 'mask', 'depth', 'intrinsics'];
        const processBtn = document.getElementById('processBtn');

        const hasRequired = required.every(type => this.uploadedFiles[type] !== null);
        const hasTexture = this.uploadedFiles.texture !== null;

        processBtn.disabled = !(hasRequired && hasTexture);

        if (processBtn.disabled) {
            processBtn.querySelector('.btn-text').textContent = 'MISSING REQUIRED FILES';
        } else {
            processBtn.querySelector('.btn-text').textContent = 'INITIATE PROCESSING';
        }
    }

    selectPipeline(pipeline) {
        this.currentPipeline = pipeline;

        // Update UI
        document.querySelectorAll('.pipeline-option').forEach(option => {
            option.classList.remove('active');
        });

        document.querySelector(`[data-pipeline="${pipeline}"]`).classList.add('active');

        this.addLog('info', `Selected pipeline: ${pipeline.toUpperCase()}`);
    }

    async startProcessing() {
        if (this.processing) return;

        this.processing = true;
        this.currentStep = 1;

        // Show progress section
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('progressSection').scrollIntoView({ behavior: 'smooth' });

        // Disable process button
        document.getElementById('processBtn').disabled = true;

        this.addLog('info', 'Starting 6D pose estimation pipeline...');

        try {
            // Send processing request to server
            const response = await fetch(API_ENDPOINTS.process, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_KEY}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pipeline: this.currentPipeline,
                    files: Object.keys(this.uploadedFiles).filter(key => this.uploadedFiles[key] !== null)
                })
            });

            if (response.ok) {
                await this.runPipelineSteps();
                this.addLog('success', 'Pipeline completed successfully!');
                this.showResults();
            } else {
                throw new Error(`Processing failed: ${response.statusText}`);
            }
        } catch (error) {
            this.addLog('error', `Pipeline failed: ${error.message}`);
            this.handleError(error);
        } finally {
            this.processing = false;
            document.getElementById('processBtn').disabled = false;
        }
    }

    async runPipelineSteps() {
        const steps = [
            { name: 'Data Preprocessing', duration: 3000, modules: ['File Validation', 'Data Parsing'] },
            { name: 'CNOS Processing', duration: 5000, modules: ['Object Detection', 'Mask Generation'] },
            { name: 'DINOv2 Features', duration: 8000, modules: ['Feature Extraction', 'Descriptor Generation'] },
            { name: 'GeDi Processing', duration: 6000, modules: ['Geometric Analysis', '3D Features'] },
            { name: 'Pose Estimation', duration: 4000, modules: ['RANSAC', 'Pose Refinement'] }
        ];

        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            this.currentStep = i + 1;

            this.updateCurrentStep(step.name, `Processing with ${step.modules.join(' and ')}...`);
            this.setStepStatus(i + 1, 'active');

            await this.simulateProcessing(step.duration, step.modules);

            this.setStepStatus(i + 1, 'completed');
            this.addLog('success', `Completed: ${step.name}`);
        }
    }

    async simulateProcessing(duration, modules) {
        const startTime = Date.now();
        const interval = 100;

        return new Promise((resolve) => {
            const timer = setInterval(() => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min((elapsed / duration) * 100, 100);

                this.updateProgress(progress);

                // Simulate module progress
                if (elapsed > duration * 0.3 && elapsed < duration * 0.7) {
                    const currentModule = modules[Math.floor(Math.random() * modules.length)];
                    this.addLog('info', `Processing with ${currentModule}...`);
                }

                if (elapsed >= duration) {
                    clearInterval(timer);
                    this.updateProgress(100);
                    resolve();
                }
            }, interval);
        });
    }

    updateCurrentStep(title, description) {
        document.getElementById('currentStepTitle').textContent = title;
        document.getElementById('currentStepDescription').textContent = description;
        document.querySelector('.step-number').textContent = this.currentStep;
    }

    updateProgress(percentage) {
        document.getElementById('stepProgress').style.width = percentage + '%';
        document.getElementById('progressText').textContent = Math.round(percentage) + '%';
    }

    setStepStatus(stepNumber, status) {
        const stepCard = document.querySelector(`[data-step="${stepNumber}"]`);
        const stepStatus = stepCard.querySelector('.step-status');

        // Remove all status classes
        stepCard.classList.remove('active', 'completed', 'error');
        stepStatus.classList.remove('pending', 'active', 'completed', 'error');

        // Add new status
        stepCard.classList.add(status);
        stepStatus.classList.add(status);

        // Update icon
        const icon = stepStatus.querySelector('i');
        if (status === 'active') {
            icon.className = 'fas fa-spinner fa-spin';
        } else if (status === 'completed') {
            icon.className = 'fas fa-check';
        } else if (status === 'error') {
            icon.className = 'fas fa-times';
        }
    }

    addLog(level, message) {
        const logsWindow = document.getElementById('logsWindow');
        const timestamp = new Date().toLocaleTimeString();

        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';

        logEntry.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="log-level ${level}">${level.toUpperCase()}</span>
            <span class="log-message">${message}</span>
        `;

        logsWindow.appendChild(logEntry);
        logsWindow.scrollTop = logsWindow.scrollHeight;
    }

    showResults() {
        // Generate mock results
        this.results = {
            processingTime: '12.34',
            detectedObjects: '3',
            confidence: '94.2%',
            poseData: this.generateMockPoseData(),
            visualization: this.generateMockVisualization(),
            descriptors: this.generateMockDescriptors()
        };

        // Update results display
        document.getElementById('processingTime').textContent = this.results.processingTime + 's';
        document.getElementById('detectedObjects').textContent = this.results.detectedObjects;
        document.getElementById('confidenceScore').textContent = this.results.confidence;

        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

        // Update visualization
        this.updateVisualization();
    }

    generateMockPoseData() {
        return {
            objects: [
                {
                    id: 1,
                    class: 'object_1',
                    pose: {
                        translation: [0.15, -0.08, 0.45],
                        rotation: [0.92, 0.15, -0.36, 0.08]
                    },
                    confidence: 0.94
                },
                {
                    id: 2,
                    class: 'object_2',
                    pose: {
                        translation: [-0.22, 0.12, 0.38],
                        rotation: [0.85, -0.25, 0.42, 0.15]
                    },
                    confidence: 0.87
                }
            ],
            camera_params: {
                fx: 525.0,
                fy: 525.0,
                cx: 319.5,
                cy: 239.5
            }
        };
    }

    generateMockVisualization() {
        // In a real implementation, this would be actual visualization data
        return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
    }

    generateMockDescriptors() {
        // Mock descriptor data
        return new Float32Array(1000).fill(0).map(() => Math.random());
    }

    updateVisualization() {
        const container = document.getElementById('visualizationContainer');
        container.innerHTML = `
            <div style="width: 100%; height: 100%; background: linear-gradient(45deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1)); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: var(--cyber-primary); font-size: 1.2rem;">
                <div style="text-align: center;">
                    <i class="fas fa-cube" style="font-size: 3rem; margin-bottom: 1rem; animation: rotation 3s linear infinite;"></i>
                    <p>3D Pose Visualization</p>
                    <p style="font-size: 0.9rem; opacity: 0.8;">Objects: ${this.results.detectedObjects} | Confidence: ${this.results.confidence}</p>
                </div>
            </div>
            <style>
                @keyframes rotation {
                    0% { transform: rotateY(0deg); }
                    100% { transform: rotateY(360deg); }
                }
            </style>
        `;
    }

    downloadFile(filename, type) {
        let data, mimeType;

        switch (type) {
            case 'pose':
                data = JSON.stringify(this.results.poseData, null, 2);
                mimeType = 'application/json';
                break;
            case 'visualization':
                data = this.results.visualization;
                mimeType = 'image/png';
                break;
            case 'descriptors':
                data = this.results.descriptors;
                mimeType = 'application/octet-stream';
                break;
            case 'all':
                data = JSON.stringify({
                    pose_data: this.results.poseData,
                    processing_time: this.results.processingTime,
                    confidence: this.results.confidence,
                    detected_objects: this.results.detectedObjects
                }, null, 2);
                mimeType = 'application/json';
                filename = 'complete_results.json';
                break;
        }

        const blob = new Blob([data], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.addLog('info', `Downloaded: ${filename}`);
    }

    handleError(error) {
        const errorStep = this.currentStep;
        this.setStepStatus(errorStep, 'error');

        // Show error details
        const errorMessage = `
            <div style="color: var(--cyber-danger); background: rgba(255,0,64,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>❌ Processing Error</h4>
                <p><strong>Step:</strong> ${errorStep}</p>
                <p><strong>Error:</strong> ${error.message}</p>
                <p><strong>Pipeline:</strong> ${this.currentPipeline.toUpperCase()}</p>
            </div>
        `;

        document.getElementById('logsWindow').innerHTML += errorMessage;
    }

    initializeInterface() {
        // Add initial log
        this.addLog('info', 'System initialized. Ready for file uploads.');

        // Set default values
        this.updateProcessButton();

        // Add some cyberpunk flair
        this.addCyberpunkEffects();
    }

    addCyberpunkEffects() {
        // Add random glitch effects
        setInterval(() => {
            const elements = document.querySelectorAll('.cyber-button, .step-card, .upload-card');
            const randomElement = elements[Math.floor(Math.random() * elements.length)];

            randomElement.style.animation = 'glitch 0.1s ease-in-out';
            setTimeout(() => {
                randomElement.style.animation = '';
            }, 100);
        }, 15000);

        // Add data flow animations
        setInterval(() => {
            const lines = document.querySelectorAll('.cyber-line');
            lines.forEach(line => {
                line.style.animation = 'none';
                setTimeout(() => {
                    line.style.animation = 'lineMove 4s linear infinite';
                }, 10);
            });
        }, 8000);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new PoseEstimationPipeline();
});
