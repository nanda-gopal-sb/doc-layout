import JSZip from 'jszip';

const API_URL = "https://nandagopalsb-document-layout-analysis.hf.space/analyze";

const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const status = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const resultImg = document.getElementById('resultImage');
const jsonOutput = document.getElementById('jsonOutput');

analyzeBtn.addEventListener('click', async () => {
    if (!fileInput.files[0]) return alert("Please select an image first!");

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        analyzeBtn.disabled = true;
        status.innerText = "⏳ Processing... (This may take a minute)";
        resultsDiv.style.display = "none";

        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API Error: " + response.statusText);

        // Handle the ZIP response
        const blob = await response.blob();
        const zip = await JSZip.loadAsync(blob);
        
        // Find the files in the zip
        const files = Object.keys(zip.files);
        const jsonFile = files.find(f => f.endsWith('.json'));
        const imgFile = files.find(f => !f.endsWith('.json'));

        // Extract JSON
        const jsonData = await zip.file(jsonFile).async("string");
        jsonOutput.innerText = JSON.stringify(JSON.parse(jsonData), null, 2);

        // Extract Image
        const imgBlob = await zip.file(imgFile).async("blob");
        resultImg.src = URL.createObjectURL(imgBlob);

        status.innerText = "✅ Analysis Complete!";
        resultsDiv.style.display = "grid";
    } catch (err) {
        status.innerText = "❌ Error: " + err.message;
        console.error(err);
    } finally {
        analyzeBtn.disabled = false;
    }
});